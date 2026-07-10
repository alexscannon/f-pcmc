"""Streaming von Mises-Fisher Dirichlet-process mixture (vMF-DPMM).

Collapses OOD detection and novel-class clustering into a single online model: a
DP mixture of vMF distributions over L2-normalized embeddings. Each stream
example is processed once and either assigned to an existing component (named
IND/promoted, or an unnamed novel component) or spawns a new component. There is
no separate detection threshold and no batched clustering trigger; promotions
happen inline when an unnamed component grows tight and large enough.

Conforms to the Paradigm interface. main.py keeps all GT/metric bookkeeping; a
``cluster_event`` is emitted on (and only on) promotion steps so the existing
promotion bookkeeping + ClusteringEvent assembly fire.

Numerical note (important): with ResNet50 embeddings D=2048 the vMF order is
v = D/2 - 1 = 1023. scipy.special.ive(v, kappa) underflows to 0 (log = -inf) for
kappa below ~1000, which is squarely in the range the Banerjee estimator
produces. Therefore the PRODUCTION log-Bessel uses the Abramowitz & Stegun 9.7.7
uniform asymptotic expansion (finite and accurate for large order), and ive is
used as a VALIDATION ORACLE: at warmup we cross-check the two agree in the regime
where ive is finite. (This swaps the production/oracle roles relative to a
hypothetical small-D setting, because ive is simply unusable as production here.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from scipy.special import gammaln, ive

from config import ContinualConfig
from paradigms.base import (
    ClusterEventInfo,
    PromotionRecord,
    StepResult,
)

EPS = 1e-12


# ----------------------------------------------------------------------------
# Numerical helpers (Phase 1.6)
# ----------------------------------------------------------------------------

def estimate_kappa(
    r_sum: np.ndarray, n: int, D: int, kappa_min: float, kappa_max: float
) -> float:
    """Banerjee et al. 2005 concentration point estimate."""
    if n == 0:
        return kappa_min
    r_bar = float(np.linalg.norm(r_sum)) / n
    r_bar = min(r_bar, 1.0 - 1e-10)
    kappa = r_bar * (D - r_bar * r_bar) / (1.0 - r_bar * r_bar)
    return float(np.clip(kappa, kappa_min, kappa_max))


def log_iv_uniform(v: float, kappa: float) -> float:
    """log I_v(kappa) via the A&S 9.7.7 large-order uniform asymptotic.

    Stable (finite) for large v across the kappa range of interest; this is the
    production path. z = kappa / v.
    """
    z = kappa / v
    s = np.sqrt(1.0 + z * z)
    eta = s + np.log(z / (1.0 + s))
    return -0.5 * np.log(2.0 * np.pi * v) + v * eta - 0.25 * np.log(1.0 + z * z)


def log_iv_ive(v: float, kappa: float) -> float:
    """log I_v(kappa) via scipy.special.ive (validation oracle; underflows low kappa)."""
    val = ive(v, kappa)
    if val <= 0.0:
        return -np.inf
    return float(np.log(val) + kappa)


def log_C_D(kappa: float, D: int) -> float:
    """log normalizing constant of vMF: (v)log k - (D/2)log(2pi) - log I_v(k)."""
    v = D / 2.0 - 1.0
    if kappa < 1e-8:
        # near-uniform; return a constant that makes density ~ uniform
        return 0.0
    return v * np.log(kappa) - (D / 2.0) * np.log(2.0 * np.pi) - log_iv_uniform(v, kappa)


def validate_log_bessel(D: int) -> None:
    """Cross-check the uniform asymptotic against ive where ive is finite.

    Asserts the production log-Bessel is finite across the expected kappa range
    and agrees with ive (the oracle) where ive does not underflow. Logs a warning
    on any non-finite production value or disagreement.
    """
    v = D / 2.0 - 1.0
    bad = []
    for kappa in [1.0, 10.0, 100.0, 1e3, 2e3, 5e3, 1e4, 2e4]:
        prod = log_iv_uniform(v, kappa)
        if not np.isfinite(prod):
            bad.append(f"non-finite production log_iv at kappa={kappa}")
            continue
        oracle = log_iv_ive(v, kappa)
        if np.isfinite(oracle):
            rel = abs(prod - oracle) / max(abs(oracle), 1.0)
            if rel > 1e-3:
                bad.append(
                    f"prod/oracle disagree at kappa={kappa}: "
                    f"{prod:.4f} vs {oracle:.4f} (rel={rel:.2e})"
                )
    if bad:
        for b in bad:
            logger.warning(f"[vMF log-Bessel validation] {b}")
    else:
        logger.info(
            f"[vMF log-Bessel validation] OK at D={D} "
            f"(uniform asymptotic finite; matches ive where finite)"
        )


# ----------------------------------------------------------------------------
# Component state
# ----------------------------------------------------------------------------

@dataclass
class Component:
    mu: np.ndarray            # (D,) unit mean direction
    kappa: float
    n: int
    r_sum: np.ndarray         # (D,) running sum of normalized embeddings
    log_C: float              # cached log C_D(kappa)
    is_named: bool
    class_idx: int            # -1 if unnamed
    spawn_t: int
    members_t: list[int] = field(default_factory=list)  # stream indices (unnamed only)


# ----------------------------------------------------------------------------
# Paradigm
# ----------------------------------------------------------------------------

class VMFDPMMParadigm:
    """Streaming vMF Dirichlet-process mixture."""

    def __init__(self) -> None:
        self.config: ContinualConfig | None = None
        self.oracle_mode: str = "none"
        self.D: int = 0

        self.components: list[Component] = []
        self._class_names: list[str] = []
        self._next_class_idx: int = 0

        self.alpha: float = 1.0
        self.log_p_base: float = 0.0

        self.kappa_init: float = 100.0
        self._promotion_counter: int = 0
        # Which candidate score populates the primary `score` field / column.
        self._primary_score: str = "score_current"

        # clustering-oracle: true_subclass -> component
        self._oracle_map: dict[str, Component] = {}

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
        X = embeddings.astype(np.float64)
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)
        D = X.shape[1]
        self.D = D
        self._class_names = list(class_names)
        self._next_class_idx = len(class_names)
        self.kappa_init = config.vmf_kappa_init

        valid_scores = set(self.extra_score_names)
        if config.vmf_primary_score not in valid_scores:
            logger.warning(
                f"vmf_dpmm.primary_score={config.vmf_primary_score!r} is not one of "
                f"{sorted(valid_scores)}; falling back to 'score_current'."
            )
            self._primary_score = "score_current"
        else:
            self._primary_score = config.vmf_primary_score

        # log of inverse surface area of S^{D-1}
        self.log_p_base = -np.log(2.0) - (D / 2.0) * np.log(np.pi) + gammaln(D / 2.0)

        # Validate the production log-Bessel before fitting anything.
        validate_log_bessel(D)

        # Fit one named component per class.
        warmup_kappas = []
        for c in range(len(class_names)):
            mask = labels == c
            n_c = int(mask.sum())
            if n_c == 0:
                continue
            r_sum = X[mask].sum(axis=0)
            mu = r_sum / max(float(np.linalg.norm(r_sum)), EPS)
            kappa = estimate_kappa(r_sum, n_c, D, config.vmf_kappa_min, config.vmf_kappa_max)
            warmup_kappas.append(kappa)
            self.components.append(
                Component(
                    mu=mu, kappa=kappa, n=n_c, r_sum=r_sum,
                    log_C=log_C_D(kappa, D), is_named=True, class_idx=c, spawn_t=-1,
                )
            )

        # Calibrate alpha (warmup-as-stream → ~target components).
        self.alpha = self._calibrate_alpha(X, config)

        # Warn about ignored config fields.
        logger.warning(
            "vMF-DPMM ignores threshold_percentile, cluster_interval, "
            "min_ood_for_clustering, and min_soft_prob (no detection threshold / "
            "no batched clustering)."
        )
        logger.info(
            f"vMF-DPMM warmup: {len(self.components)} named components, D={D}, "
            f"kappa median={np.median(warmup_kappas):.1f}, alpha={self.alpha:.3e}, "
            f"log_p_base={self.log_p_base:.2f}"
        )

    def _calibrate_alpha(self, X: np.ndarray, config: ContinualConfig) -> float:
        """Binary search over log-alpha so warmup-as-stream yields ~target comps.

        Coarse, on a fixed-seed subsample (the actual init uses labels, so alpha
        only governs stream new-component creation — precision is not critical).
        """
        target = config.vmf_alpha_calibration_target
        rng = np.random.default_rng(config.random_seed)
        n_sub = min(5000, X.shape[0])
        sub = X[rng.permutation(X.shape[0])[:n_sub]]
        lo, hi = np.log(config.vmf_alpha_search_lo), np.log(config.vmf_alpha_search_hi)
        # scale target to the subsample size
        sub_target = max(2, int(round(target * n_sub / X.shape[0])))
        best = np.exp((lo + hi) / 2)
        for _ in range(config.vmf_alpha_search_max_iters):
            mid = (lo + hi) / 2
            alpha = np.exp(mid)
            n_comp = self._simulate_warmup_clustering(sub, alpha)
            if n_comp > sub_target:
                hi = mid
            else:
                lo = mid
            best = np.exp((lo + hi) / 2)
            if abs(n_comp - sub_target) < max(1, 0.05 * sub_target):
                break
        return float(best)

    def _simulate_warmup_clustering(self, X: np.ndarray, alpha: float) -> int:
        """Coarse sequential hard-assignment clustering with fixed kappa_init."""
        log_alpha = np.log(alpha)
        log_C = log_C_D(self.kappa_init, self.D)
        mus: list[np.ndarray] = []
        ns: list[int] = []
        r_sums: list[np.ndarray] = []
        for z in X:
            if not mus:
                mus.append(z.copy()); ns.append(1); r_sums.append(z.copy())
                continue
            M = np.stack(mus)
            log_r = np.log(np.asarray(ns, dtype=float)) + log_C + self.kappa_init * (M @ z)
            best = float(log_r.max())
            if log_alpha + self.log_p_base > best:
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

    def _process(
        self,
        z: np.ndarray,
        t: int,
        forced_is_ood: bool | None,
        true_subclass: str | None,
    ) -> StepResult:
        z_n = z.astype(np.float64)
        z_n = z_n / max(float(np.linalg.norm(z_n)), EPS)

        comps = self.components
        mus = np.stack([c.mu for c in comps])
        kappas = np.array([c.kappa for c in comps])
        log_n = np.log(np.array([c.n for c in comps], dtype=float))
        log_C = np.array([c.log_C for c in comps])
        named = np.array([c.is_named for c in comps])
        dots = mus @ z_n
        log_r = log_n + log_C + kappas * dots  # log N_k + log p(z | comp_k)
        log_r_new = np.log(self.alpha) + self.log_p_base

        # Candidate detection scores (higher = more OOD); the primary `score`
        # column is selected by config.vmf_primary_score (default score_current,
        # which reproduces the original -max log-responsibility readout).
        extras = self._score_variants(log_r, float(log_r_new), log_C, kappas, dots, named)
        score = extras[self._primary_score]

        # Geometric routing.
        best_k = int(np.argmax(log_r))
        geo_spawn = log_r_new > log_r[best_k]
        geo_is_ood = geo_spawn or (not named[best_k])

        clustering_oracle = self.oracle_mode in {"clustering", "both"}
        is_ood = forced_is_ood if forced_is_ood is not None else geo_is_ood

        n_ind_snapshot = sum(1 for c in comps if c.is_named)
        n_buf_snapshot = sum(c.n for c in comps if not c.is_named)

        if not is_ood:
            # Assign to the best NAMED component (forced-IND or geometric-IND).
            named_idx = np.where(named)[0]
            k = int(named_idx[np.argmax(log_r[named_idx])])
            self._update_component(comps[k], z_n)
            return StepResult(
                score=score, is_ood=False, predicted_class=comps[k].class_idx,
                n_ind_classes=n_ind_snapshot, n_ood_buffer=n_buf_snapshot,
                promotions_this_step=[], cluster_event=None, extras=extras,
            )

        # ----- OOD path -----
        if clustering_oracle:
            comp = self._oracle_map.get(true_subclass)
            if comp is None or comp.is_named:
                comp = self._spawn_component(z_n, t)
                self._oracle_map[true_subclass] = comp
            else:
                self._update_component(comp, z_n)
                comp.members_t.append(t)
        else:
            # Choose among unnamed components + new.
            unnamed_idx = np.where(~named)[0]
            spawn = True
            chosen = None
            if len(unnamed_idx) > 0:
                ku = int(unnamed_idx[np.argmax(log_r[unnamed_idx])])
                if log_r_new <= log_r[ku]:
                    spawn = False
                    chosen = comps[ku]
            if spawn:
                comp = self._spawn_component(z_n, t)
            else:
                comp = chosen
                self._update_component(comp, z_n)
                comp.members_t.append(t)

        # Promotion check for the unnamed component just touched.
        promotions: list[PromotionRecord] = []
        cluster_event: ClusterEventInfo | None = None
        promo = self._maybe_promote(comp, t)
        if promo is not None:
            promotions = [promo]
            cluster_event = ClusterEventInfo(
                buffer_size=n_buf_snapshot,
                sweep_counts={},
                n_dedup_candidates=1,
            )

        # Periodic pruning of stale singletons.
        if self.config.vmf_prune_singletons and t > 0 and t % self.config.vmf_prune_interval == 0:
            self._prune(t)

        return StepResult(
            score=score, is_ood=True, predicted_class=-1,
            n_ind_classes=n_ind_snapshot, n_ood_buffer=n_buf_snapshot,
            promotions_this_step=promotions, cluster_event=cluster_event,
            extras=extras,
        )

    @staticmethod
    def _score_variants(
        log_r: np.ndarray,
        log_r_new: float,
        log_C: np.ndarray,
        kappas: np.ndarray,
        dots: np.ndarray,
        named: np.ndarray,
    ) -> dict[str, float]:
        """Four candidate detection scores over NAMED components (higher = more OOD).

        - score_logratio: log α + log p(z|base) − max_k(log N_k + log p(z|comp_k)).
          Directly answers "should this have routed to a new component?"
        - score_entropy:  entropy of the normalized named responsibilities; high
          when no IND class strongly claims the point.
        - score_density:  −max_k log p(z|comp_k) (drops the log N_k size prior);
          isolates raw vMF density from component-mass weighting.
        - score_current:  −max_k(log N_k + log p(z|comp_k)); the original readout.
        """
        if named.any():
            lr = log_r[named]
            r_max = float(np.max(lr))
            dens = (log_C + kappas * dots)[named]  # raw log-density, no log N_k
            d_max = float(np.max(dens))
            m = lr.max()
            w = np.exp(lr - m)
            p = w / w.sum()
            entropy = float(-np.sum(p * np.log(p + EPS)))
            return {
                "score_logratio": float(log_r_new - r_max),
                "score_entropy": entropy,
                "score_density": float(-d_max),
                "score_current": float(-r_max),
            }
        # Degenerate: no named components (does not occur post-warmup, which
        # seeds one named component per IND class).
        return {
            "score_logratio": 0.0,
            "score_entropy": 0.0,
            "score_density": float(-log_r_new),
            "score_current": float(-log_r_new),
        }

    # ----- component mechanics -----
    def _spawn_component(self, z_n: np.ndarray, t: int) -> Component:
        comp = Component(
            mu=z_n.copy(), kappa=self.kappa_init, n=1, r_sum=z_n.copy(),
            log_C=log_C_D(self.kappa_init, self.D), is_named=False,
            class_idx=-1, spawn_t=t, members_t=[t],
        )
        self.components.append(comp)
        return comp

    def _update_component(self, comp: Component, z_n: np.ndarray) -> None:
        comp.r_sum = comp.r_sum + z_n
        comp.n += 1
        comp.mu = comp.r_sum / max(float(np.linalg.norm(comp.r_sum)), EPS)
        comp.kappa = estimate_kappa(
            comp.r_sum, comp.n, self.D, self.config.vmf_kappa_min, self.config.vmf_kappa_max
        )
        comp.log_C = log_C_D(comp.kappa, self.D)

    def _maybe_promote(self, comp: Component, t: int) -> PromotionRecord | None:
        if comp.is_named:
            return None
        if comp.n < self.config.min_promote_size:
            return None
        if comp.kappa < self.config.vmf_kappa_promote_min:
            return None

        self._promotion_counter += 1
        cid = f"promoted_{self._promotion_counter:03d}"
        comp.is_named = True
        comp.class_idx = self._next_class_idx
        self._next_class_idx += 1
        self._class_names.append(cid)

        # Exact mean pairwise cosine sim from sufficient stats:
        # (||r_sum||^2 - n) / (n(n-1)).
        n = comp.n
        intra = float((float(comp.r_sum @ comp.r_sum) - n) / (n * (n - 1))) if n > 1 else 1.0
        members = list(comp.members_t)
        comp.members_t = []  # named components no longer track members

        return PromotionRecord(
            cid=cid,
            member_stream_indices=members,
            n_members=n,
            intra_cosine_sim=intra,
            mean_soft_prob=np.nan,        # vMF is hard MAP assignment
            min_cluster_size_used=None,
            centroid=comp.mu.copy(),      # unit mean direction
        )

    def _prune(self, t: int) -> None:
        age = self.config.vmf_prune_age
        kept = []
        n_pruned = 0
        for c in self.components:
            if (not c.is_named) and c.n == 1 and c.spawn_t < t - age:
                n_pruned += 1
                continue
            kept.append(c)
        if n_pruned:
            self.components = kept
            logger.debug(f"vMF prune @ t={t}: dropped {n_pruned} stale singletons")

    # ----- end-of-stream -----
    def drain(self) -> list[tuple[int, int]]:
        """Assign each leftover unnamed component's members to the nearest named class."""
        named = [c for c in self.components if c.is_named]
        if not named:
            return []
        named_mu = np.stack([c.mu for c in named])
        named_cls = [c.class_idx for c in named]
        out: list[tuple[int, int]] = []
        for c in self.components:
            if c.is_named or not c.members_t:
                continue
            nearest = int(np.argmax(named_mu @ c.mu))
            cls_idx = named_cls[nearest]
            for st in c.members_t:
                out.append((st, cls_idx))
        return out

    # ----- introspection -----
    @property
    def n_ind_classes(self) -> int:
        return sum(1 for c in self.components if c.is_named)

    @property
    def ood_buffer_size(self) -> int:
        return sum(c.n for c in self.components if not c.is_named)

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    @property
    def detection_threshold(self) -> float | None:
        return None  # no scalar threshold

    @property
    def extra_score_names(self) -> list[str]:
        """Auxiliary detection-score columns this paradigm logs in ``extras``."""
        return ["score_logratio", "score_entropy", "score_density", "score_current"]

    @property
    def ood_buffer_stream_indices(self) -> list[int]:
        out: list[int] = []
        for c in self.components:
            if not c.is_named:
                out.extend(c.members_t)
        return out
