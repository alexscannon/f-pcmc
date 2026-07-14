"""Composed paradigm: kNN-gallery detection + vMF clustering of the OOD stream.

Phase B of the score-fix / oracle-matrix / composition study. Phase C showed the
two single paradigms have complementary bottlenecks: kNN owns detection (AUROC
0.925) but its DP-means clustering is the bottleneck; vMF owns novel-class
discovery (34/49 no-oracle) but its detector caps it. This paradigm composes the
strong halves:

  - **Detection** is the kNN gallery from ``knn_dpmeans`` (kth-neighbor cosine
    distance vs a frozen tau). Named IND classes live in the gallery, NOT as vMF
    components.
  - **Novel-class discovery** is a streaming vMF Dirichlet-process mixture over
    the OOD-flagged stream, restricted to **unnamed** components only (the novel
    side of ``vmf_dpmm``). Promotion uses vMF's size + kappa gate.
  - The structurally important coupling: when an unnamed component promotes, its
    centroid + all members are appended to the **kNN gallery**, so the detector
    grows with discovery (exactly as ``knn_dpmeans`` grows its gallery).

Documented divergences:
  - vMF is NOT initialized from warmup labels (``components`` starts empty); it
    only ever models discovered novel structure. IND lives in the gallery.
  - alpha is calibrated like ``vmf_dpmm`` (warmup-as-stream binary search) but
    against ``alpha_calibration_target_novel`` (the expected novel-class count,
    default 50) rather than the full warmup class count.
  - For diagnostics, a FROZEN set of IND vMF components is fit at warmup (the same
    fit ``vmf_dpmm`` does) purely to log "what vMF's own detection signal would
    have said" per example (the Phase A winner ``score_entropy``), in ``extras``.
    These never route, promote, or update — they only produce the coherence
    diagnostic against the kNN score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from scipy.special import gammaln

from config import ContinualConfig
from paradigms.base import ClusterEventInfo, PromotionRecord, StepResult
from paradigms.knn_dpmeans import (
    _l2_normalize_rows,
    _l2_normalize_vec,
    _mean_pairwise_cosine,
)
from paradigms.vmf_dpmm import (
    VMFDPMMParadigm,
    estimate_kappa,
    log_C_D,
    validate_log_bessel,
)

EPS = 1e-12


@dataclass
class _NovelComponent:
    """An unnamed (novel) vMF component over the OOD stream.

    Tracks per-member normalized embeddings (not just the sufficient statistic)
    so that on promotion the members can be appended to the kNN gallery.
    """

    mu: np.ndarray
    kappa: float
    n: int
    r_sum: np.ndarray
    log_C: float
    class_idx: int                                   # -1 until promoted
    spawn_t: int
    members_t: list[int] = field(default_factory=list)
    member_norms: list[np.ndarray] = field(default_factory=list)


class KNNVMFParadigm:
    """kNN detection + vMF (unnamed-only) clustering of the OOD stream."""

    def __init__(self) -> None:
        self.config: ContinualConfig | None = None
        self.oracle_mode: str = "none"
        self.D: int = 0

        # ----- kNN detection state (mirrors knn_dpmeans) -----
        self.gallery_embeddings: np.ndarray | None = None  # (N, D) normalized
        self.gallery_labels: np.ndarray | None = None      # (N,) int
        self._class_names: list[str] = []
        self._next_class_idx: int = 0
        self.tau: float = 0.0
        self.k: int = 20
        self.distance_variant: str = "kth"

        # ----- vMF (unnamed-only) clustering state -----
        self.components: list[_NovelComponent] = []
        self.alpha: float = 1.0
        self.log_p_base: float = 0.0
        self.kappa_init: float = 100.0
        self._promotion_counter: int = 0
        # clustering-oracle: true_subclass -> component
        self._oracle_map: dict[str, _NovelComponent] = {}

        # ----- frozen IND vMF components (diagnostic only) -----
        self._diag_mus: np.ndarray | None = None     # (C, D)
        self._diag_kappas: np.ndarray | None = None  # (C,)
        self._diag_logC: np.ndarray | None = None    # (C,)
        self._diag_logn: np.ndarray | None = None    # (C,)

    # ----- lifecycle -----
    def warmup(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        class_names: list[str],
        config: ContinualConfig,
    ) -> None:
        self.config = config
        self.oracle_mode = config.oracle_mode
        self.k = config.knn_k
        self.distance_variant = config.knn_distance
        self.kappa_init = config.vmf_kappa_init

        X = _l2_normalize_rows(embeddings.astype(np.float64))
        D = X.shape[1]
        self.D = D
        self.log_p_base = -np.log(2.0) - (D / 2.0) * np.log(np.pi) + gammaln(D / 2.0)
        validate_log_bessel(D)

        # kNN gallery + tau (named IND classes live here, not as vMF components).
        self.gallery_embeddings = X
        self.gallery_labels = labels.astype(int).copy()
        self._class_names = list(class_names)
        self._next_class_idx = len(class_names)
        self.tau = self._calibrate_tau()

        # Frozen IND vMF fit — diagnostic detection signal only (never routes).
        self._fit_diagnostic_ind(X, labels, len(class_names), D, config)

        # alpha calibrated to the *novel* target (documented divergence).
        self.alpha = self._calibrate_alpha(X, config)

        logger.info(
            f"kNN-vMF warmup: gallery={self.gallery_embeddings.shape[0]}, k={self.k}, "
            f"distance={self.distance_variant}, tau={self.tau:.4f}, "
            f"alpha={self.alpha:.3e} (novel target={config.knn_vmf_alpha_target_novel}), "
            f"D={D}"
        )

    def _calibrate_tau(self) -> float:
        """kth-neighbor cosine distance percentile over the gallery (exclude self)."""
        G = self.gallery_embeddings
        n = G.shape[0]
        k = self.k
        kth = np.empty(n, dtype=np.float64)
        chunk = 1000
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            dists = 1.0 - G[start:end] @ G.T
            for r in range(end - start):
                dists[r, start + r] = np.inf
            part = np.partition(dists, k - 1, axis=1)[:, :k]
            kth[start:end] = (
                part.mean(axis=1) if self.distance_variant == "mean_k" else part[:, k - 1]
            )
        return float(np.percentile(kth, self.config.threshold_percentile))

    def _fit_diagnostic_ind(
        self, X: np.ndarray, labels: np.ndarray, n_classes: int, D: int, config: ContinualConfig
    ) -> None:
        """Fit one frozen vMF component per IND class (diagnostic detection signal)."""
        mus, kappas, logC, logn = [], [], [], []
        for c in range(n_classes):
            mask = labels == c
            n_c = int(mask.sum())
            if n_c == 0:
                continue
            r_sum = X[mask].sum(axis=0)
            mu = r_sum / max(float(np.linalg.norm(r_sum)), EPS)
            kappa = estimate_kappa(r_sum, n_c, D, config.vmf_kappa_min, config.vmf_kappa_max)
            mus.append(mu); kappas.append(kappa)
            logC.append(log_C_D(kappa, D)); logn.append(np.log(n_c))
        self._diag_mus = np.stack(mus)
        self._diag_kappas = np.array(kappas)
        self._diag_logC = np.array(logC)
        self._diag_logn = np.array(logn)

    def _calibrate_alpha(self, X: np.ndarray, config: ContinualConfig) -> float:
        """Binary search over log-alpha so warmup-as-stream yields ~novel-target comps.

        Same mechanism as vmf_dpmm (coarse, fixed-seed subsample); target is the
        expected novel-class count, not the warmup class count.
        """
        target = config.knn_vmf_alpha_target_novel
        rng = np.random.default_rng(config.random_seed)
        n_sub = min(5000, X.shape[0])
        sub = X[rng.permutation(X.shape[0])[:n_sub]]
        lo, hi = np.log(config.vmf_alpha_search_lo), np.log(config.vmf_alpha_search_hi)
        sub_target = max(2, int(round(target * n_sub / X.shape[0])))
        best = np.exp((lo + hi) / 2)
        log_C = log_C_D(self.kappa_init, self.D)
        for _ in range(config.vmf_alpha_search_max_iters):
            mid = (lo + hi) / 2
            n_comp = self._simulate_clustering(sub, np.exp(mid), log_C)
            if n_comp > sub_target:
                hi = mid
            else:
                lo = mid
            best = np.exp((lo + hi) / 2)
            if abs(n_comp - sub_target) < max(1, 0.05 * sub_target):
                break
        return float(best)

    def _simulate_clustering(self, X: np.ndarray, alpha: float, log_C: float) -> int:
        """Coarse sequential hard-assignment clustering with fixed kappa_init."""
        log_alpha = np.log(alpha)
        mus: list[np.ndarray] = []
        ns: list[int] = []
        r_sums: list[np.ndarray] = []
        for z in X:
            if not mus:
                mus.append(z.copy()); ns.append(1); r_sums.append(z.copy()); continue
            M = np.stack(mus)
            log_r = np.log(np.asarray(ns, dtype=float)) + log_C + self.kappa_init * (M @ z)
            if log_alpha + self.log_p_base > float(log_r.max()):
                mus.append(z.copy()); ns.append(1); r_sums.append(z.copy())
            else:
                k = int(log_r.argmax())
                r_sums[k] = r_sums[k] + z
                ns[k] += 1
                mus[k] = r_sums[k] / max(float(np.linalg.norm(r_sums[k])), EPS)
        return len(mus)

    # ----- per-step -----
    def step(self, z: np.ndarray, t: int) -> StepResult:
        return self._process(z, t, forced_is_ood=None, true_subclass=None)

    def step_oracle(
        self, z: np.ndarray, t: int, true_is_ood: bool, true_subclass: str
    ) -> StepResult:
        forced = true_is_ood if self.oracle_mode in {"detection", "both"} else None
        return self._process(z, t, forced_is_ood=forced, true_subclass=true_subclass)

    def _diag_vmf_score(self, z_n: np.ndarray) -> float:
        """Frozen-IND vMF score_entropy for this example (diagnostic only)."""
        dots = self._diag_mus @ z_n
        log_r = self._diag_logn + self._diag_logC + self._diag_kappas * dots
        named = np.ones(log_r.shape[0], dtype=bool)
        return VMFDPMMParadigm._score_variants(
            log_r, 0.0, self._diag_logC, self._diag_kappas, dots, named
        )["score_entropy"]

    def _process(
        self,
        z: np.ndarray,
        t: int,
        forced_is_ood: bool | None,
        true_subclass: str | None,
    ) -> StepResult:
        z_n = _l2_normalize_vec(z.astype(np.float64))

        # kNN detection (always computed/logged, even under detection oracle).
        dists = 1.0 - self.gallery_embeddings @ z_n
        k = self.k
        part = np.partition(dists, k - 1)[:k]
        score = float(part.mean()) if self.distance_variant == "mean_k" else float(part[k - 1])
        model_is_ood = score >= self.tau
        is_ood = forced_is_ood if forced_is_ood is not None else model_is_ood

        extras = {"vmf_score_entropy": self._diag_vmf_score(z_n)}

        if not is_ood:
            nearest = int(np.argmin(dists))
            return StepResult(
                score=score, is_ood=False,
                predicted_class=int(self.gallery_labels[nearest]),
                n_ind_classes=len(self._class_names),
                n_ood_buffer=self.ood_buffer_size,
                promotions_this_step=[], cluster_event=None, extras=extras,
            )

        # ----- OOD path: assign into the unnamed vMF mixture -----
        clustering_oracle = self.oracle_mode in {"clustering", "both"}
        if clustering_oracle:
            comp = self._oracle_map.get(true_subclass)
            if comp is None:
                comp = self._spawn_component(z_n, t)
                self._oracle_map[true_subclass] = comp
            else:
                self._update_component(comp, z_n, t)
        else:
            comp = self._assign_geometric(z_n, t)

        n_ind_snapshot = len(self._class_names)
        n_buf_snapshot = self.ood_buffer_size

        promotions: list[PromotionRecord] = []
        cluster_event: ClusterEventInfo | None = None
        promo = self._maybe_promote(comp)
        if promo is not None:
            promotions = [promo]
            cluster_event = ClusterEventInfo(
                buffer_size=n_buf_snapshot, sweep_counts={}, n_dedup_candidates=1
            )

        if (
            self.config.vmf_prune_singletons
            and t > 0
            and t % self.config.vmf_prune_interval == 0
        ):
            self._prune(t)

        return StepResult(
            score=score, is_ood=True, predicted_class=-1,
            n_ind_classes=n_ind_snapshot, n_ood_buffer=n_buf_snapshot,
            promotions_this_step=promotions, cluster_event=cluster_event, extras=extras,
        )

    # ----- vMF mechanics (unnamed components only) -----
    def _assign_geometric(self, z_n: np.ndarray, t: int) -> _NovelComponent:
        """Choose among existing unnamed components + a new one (vMF spawn rule)."""
        if not self.components:
            return self._spawn_component(z_n, t)
        mus = np.stack([c.mu for c in self.components])
        kappas = np.array([c.kappa for c in self.components])
        log_n = np.log(np.array([c.n for c in self.components], dtype=float))
        log_C = np.array([c.log_C for c in self.components])
        log_r = log_n + log_C + kappas * (mus @ z_n)
        log_r_new = np.log(self.alpha) + self.log_p_base
        best = int(np.argmax(log_r))
        if log_r_new > log_r[best]:
            return self._spawn_component(z_n, t)
        comp = self.components[best]
        self._update_component(comp, z_n, t)
        return comp

    def _spawn_component(self, z_n: np.ndarray, t: int) -> _NovelComponent:
        comp = _NovelComponent(
            mu=z_n.copy(), kappa=self.kappa_init, n=1, r_sum=z_n.copy(),
            log_C=log_C_D(self.kappa_init, self.D), class_idx=-1, spawn_t=t,
            members_t=[t], member_norms=[z_n.copy()],
        )
        self.components.append(comp)
        return comp

    def _update_component(self, comp: _NovelComponent, z_n: np.ndarray, t: int) -> None:
        comp.r_sum = comp.r_sum + z_n
        comp.n += 1
        comp.mu = comp.r_sum / max(float(np.linalg.norm(comp.r_sum)), EPS)
        comp.kappa = estimate_kappa(
            comp.r_sum, comp.n, self.D, self.config.vmf_kappa_min, self.config.vmf_kappa_max
        )
        comp.log_C = log_C_D(comp.kappa, self.D)
        comp.members_t.append(t)
        comp.member_norms.append(z_n.copy())

    def _maybe_promote(self, comp: _NovelComponent) -> PromotionRecord | None:
        if comp.class_idx != -1:
            return None
        if comp.n < self.config.min_promote_size:
            return None
        if comp.kappa < self.config.vmf_kappa_promote_min:
            return None

        self._promotion_counter += 1
        cid = f"promoted_{self._promotion_counter:03d}"
        cls_id = self._next_class_idx
        self._next_class_idx += 1
        self._class_names.append(cid)
        comp.class_idx = cls_id

        # Grow the kNN gallery with the centroid + all members (the coupling step).
        member_norm = np.stack(comp.member_norms)
        self.gallery_embeddings = np.vstack(
            [self.gallery_embeddings, member_norm, comp.mu[None, :]]
        )
        self.gallery_labels = np.concatenate(
            [self.gallery_labels, np.full(len(comp.member_norms) + 1, cls_id, dtype=int)]
        )
        intra = _mean_pairwise_cosine(member_norm)
        members_t = list(comp.members_t)

        # Remove the promoted component from the unnamed mixture.
        self.components = [c for c in self.components if c is not comp]

        return PromotionRecord(
            cid=cid,
            member_stream_indices=members_t,
            n_members=len(members_t),
            intra_cosine_sim=intra,
            mean_soft_prob=np.nan,
            min_cluster_size_used=None,
            centroid=comp.mu.copy(),
        )

    def _prune(self, t: int) -> None:
        age = self.config.vmf_prune_age
        kept, n_pruned = [], 0
        for c in self.components:
            if c.class_idx == -1 and c.n == 1 and c.spawn_t < t - age:
                n_pruned += 1
                continue
            kept.append(c)
        if n_pruned:
            self.components = kept
            logger.debug(f"kNN-vMF prune @ t={t}: dropped {n_pruned} stale singletons")

    # ----- end-of-stream -----
    def drain(self) -> list[tuple[int, int]]:
        """Nearest-classify residual unnamed members against the expanded gallery."""
        out: list[tuple[int, int]] = []
        for c in self.components:
            for z_n, st in zip(c.member_norms, c.members_t):
                nearest = int(np.argmin(1.0 - self.gallery_embeddings @ z_n))
                out.append((st, int(self.gallery_labels[nearest])))
        return out

    # ----- introspection -----
    @property
    def n_ind_classes(self) -> int:
        return len(self._class_names)

    @property
    def ood_buffer_size(self) -> int:
        return sum(c.n for c in self.components)

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    @property
    def detection_threshold(self) -> float | None:
        return self.tau

    @property
    def extra_score_names(self) -> list[str]:
        return ["vmf_score_entropy"]

    @property
    def ood_buffer_stream_indices(self) -> list[int]:
        out: list[int] = []
        for c in self.components:
            out.extend(c.members_t)
        return out
