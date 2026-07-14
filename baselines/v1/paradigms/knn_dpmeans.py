"""kNN-gallery OOD detection + streaming DP-means clustering.

Replaces Mahalanobis scoring with k-nearest-neighbor cosine distance against a
gallery of L2-normalized embeddings, and replaces periodic HDBSCAN sweeps with
DP-means (Kulis & Jordan 2012) run in batch over the OOD buffer on the existing
trigger. Conforms to the Paradigm interface; main.py keeps all GT/metric
bookkeeping.

Design choices (documented divergences from the Mahalanobis baseline):
  - Gallery = original IND warmup embeddings + ALL members of promoted clusters
    (not just centroids). Stream examples classified into existing classes do not
    enter the gallery.
  - Distance is cosine on L2-normalized vectors throughout (kNN scoring, DP-means
    assignment, intra-cluster similarity).
  - tau (kth-neighbor threshold) and lambda (DP-means spawn threshold) are both
    calibrated from config.threshold_percentile, and frozen after warmup — same
    as the Mahalanobis tau.
  - DP-means is hard-assignment, so there is no soft membership: PromotionRecord
    carries mean_soft_prob = NaN and the min_soft_prob promotion check is skipped.
    Promotion is gated on min_promote_size + min_intra_cosine_sim only.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from clustering import form_oracle_clusters
from config import ContinualConfig
from paradigms.base import (
    ClusterEventInfo,
    PromotionRecord,
    StepResult,
)

EPS = 1e-12


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, EPS)


def _l2_normalize_vec(z: np.ndarray) -> np.ndarray:
    return z / max(float(np.linalg.norm(z)), EPS)


def _mean_pairwise_cosine(normalized_members: np.ndarray) -> float:
    """Mean pairwise cosine similarity of already-L2-normalized rows."""
    m = normalized_members.shape[0]
    if m < 2:
        return 1.0
    sim = normalized_members @ normalized_members.T
    # exclude the m diagonal ones (== 1.0)
    total = float(sim.sum() - m)
    return total / (m * (m - 1))


class KNNDPMeansParadigm:
    """kNN-gallery detection + streaming DP-means clustering."""

    def __init__(self) -> None:
        self.config: ContinualConfig | None = None
        self.oracle_mode: str = "none"

        # kNN gallery (normalized embeddings + class-index labels).
        self.gallery_embeddings: np.ndarray | None = None  # (N, D) normalized
        self.gallery_labels: np.ndarray | None = None      # (N,) int
        self._class_names: list[str] = []
        self._next_class_idx: int = 0

        # Calibrated thresholds (frozen post-warmup).
        self.tau: float = 0.0
        self.lambda_: float = 0.0

        # DP-means persistent centroids (normalized); members recomputed per event.
        self._dpmeans_centroids: list[np.ndarray] = []

        # OOD buffer: normalized embeddings, raw embeddings, stream indices, and
        # true subclass (only retained under a clustering oracle).
        self._buf_norm: list[np.ndarray] = []
        self._buf_raw: list[np.ndarray] = []
        self._buf_idx: list[int] = []
        self._buf_true: list[str] = []

        self._n_new_ood_since_cluster: int = 0
        self._promotion_counter: int = 0

        self.k: int = 20
        self.distance_variant: str = "kth"

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

        self.gallery_embeddings = _l2_normalize_rows(embeddings.astype(np.float64))
        self.gallery_labels = labels.astype(int).copy()
        self._class_names = list(class_names)
        self._next_class_idx = len(class_names)

        self.tau = self._calibrate_tau()
        self.lambda_ = self._calibrate_lambda()

        logger.info(
            f"kNN-DPmeans warmup: gallery={self.gallery_embeddings.shape[0]}, "
            f"k={self.k}, distance={self.distance_variant}, "
            f"tau={self.tau:.4f}, lambda={self.lambda_:.4f}"
        )

    def _calibrate_tau(self) -> float:
        """kth-neighbor cosine distance percentile over the gallery (exclude self)."""
        G = self.gallery_embeddings
        n = G.shape[0]
        k = self.k
        kth_dists = np.empty(n, dtype=np.float64)
        chunk = 1000
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            sims = G[start:end] @ G.T               # (b, n) cosine similarity
            dists = 1.0 - sims                       # cosine distance
            # exclude self: row r corresponds to gallery index start+r
            for r in range(end - start):
                dists[r, start + r] = np.inf
            # kth smallest distance per row (k-th neighbor, 0-indexed k-1)
            part = np.partition(dists, k - 1, axis=1)[:, :k]
            if self.distance_variant == "mean_k":
                kth_dists[start:end] = part.mean(axis=1)
            else:
                kth_dists[start:end] = part[:, k - 1]
        return float(np.percentile(kth_dists, self.config.threshold_percentile))

    def _calibrate_lambda(self) -> float:
        """Nearest-class-mean cosine distance percentile (DP-means spawn threshold)."""
        G = self.gallery_embeddings
        labels = self.gallery_labels
        n_classes = len(self._class_names)
        D = G.shape[1]
        means = np.zeros((n_classes, D), dtype=np.float64)
        for c in range(n_classes):
            mask = labels == c
            if not np.any(mask):
                continue
            means[c] = G[mask].mean(axis=0)
        means = _l2_normalize_rows(means)
        # nearest class-mean cosine distance for every warmup embedding
        nearest = np.empty(G.shape[0], dtype=np.float64)
        chunk = 4000
        for start in range(0, G.shape[0], chunk):
            end = min(start + chunk, G.shape[0])
            sims = G[start:end] @ means.T            # (b, C)
            nearest[start:end] = 1.0 - sims.max(axis=1)
        return float(np.percentile(nearest, self.config.threshold_percentile))

    # ----- per-step -----
    def step(self, z: np.ndarray, t: int) -> StepResult:
        return self._process(z, t, forced_is_ood=None, true_subclass=None)

    def step_oracle(
        self, z: np.ndarray, t: int, true_is_ood: bool, true_subclass: str
    ) -> StepResult:
        forced = (
            true_is_ood if self.oracle_mode in {"detection", "both"} else None
        )
        return self._process(z, t, forced_is_ood=forced, true_subclass=true_subclass)

    def _process(
        self,
        z: np.ndarray,
        t: int,
        forced_is_ood: bool | None,
        true_subclass: str | None,
    ) -> StepResult:
        z_n = _l2_normalize_vec(z.astype(np.float64))
        dists = 1.0 - self.gallery_embeddings @ z_n  # (N,) cosine distance

        k = self.k
        part = np.partition(dists, k - 1)[:k]
        score = float(part.mean()) if self.distance_variant == "mean_k" else float(
            np.partition(dists, k - 1)[k - 1]
        )
        model_is_ood = score >= self.tau
        is_ood = forced_is_ood if forced_is_ood is not None else model_is_ood

        if not is_ood:
            nearest = int(np.argmin(dists))
            predicted_class = int(self.gallery_labels[nearest])
            return StepResult(
                score=score,
                is_ood=False,
                predicted_class=predicted_class,
                n_ind_classes=len(self._class_names),
                n_ood_buffer=len(self._buf_norm),
                promotions_this_step=[],
                cluster_event=None,
            )

        # OOD: append to buffer.
        self._buf_norm.append(z_n)
        self._buf_raw.append(z.astype(np.float64))
        self._buf_idx.append(t)
        if self.oracle_mode in {"clustering", "both"}:
            self._buf_true.append(true_subclass)
        self._n_new_ood_since_cluster += 1

        n_ind_snapshot = len(self._class_names)
        n_buf_snapshot = len(self._buf_norm)

        promotions: list[PromotionRecord] = []
        cluster_event: ClusterEventInfo | None = None
        if (
            self._n_new_ood_since_cluster >= self.config.cluster_interval
            and len(self._buf_norm) >= self.config.min_ood_for_clustering
        ):
            promotions, cluster_event = self._run_clustering()
            self._n_new_ood_since_cluster = 0

        return StepResult(
            score=score,
            is_ood=True,
            predicted_class=-1,
            n_ind_classes=n_ind_snapshot,
            n_ood_buffer=n_buf_snapshot,
            promotions_this_step=promotions,
            cluster_event=cluster_event,
        )

    # ----- clustering + promotion -----
    def _run_clustering(self) -> tuple[list[PromotionRecord], ClusterEventInfo]:
        if self.oracle_mode in {"clustering", "both"}:
            return self._run_clustering_oracle()
        return self._run_clustering_dpmeans()

    def _dpmeans(self, X: np.ndarray) -> tuple[list[np.ndarray], list[list[int]]]:
        """Hard-assignment DP-means over normalized rows, warm-started from state."""
        centroids = [c.copy() for c in self._dpmeans_centroids]
        member_indices: list[list[int]] = [[] for _ in centroids]
        max_iters = self.config.dpmeans_max_iters
        tol = self.config.dpmeans_convergence_tol

        for _ in range(max(1, max_iters)):
            member_indices = [[] for _ in centroids]
            for i in range(X.shape[0]):
                z = X[i]
                if not centroids:
                    centroids.append(z.copy())
                    member_indices.append([i])
                    continue
                C = np.stack(centroids)
                d = 1.0 - C @ z
                k_min = int(np.argmin(d))
                if float(d[k_min]) > self.lambda_:
                    centroids.append(z.copy())
                    member_indices.append([i])
                else:
                    member_indices[k_min].append(i)

            # M-step: drop empty centroids; update as normalized mean of members.
            new_centroids: list[np.ndarray] = []
            new_members: list[list[int]] = []
            for members in member_indices:
                if not members:
                    continue
                mean = X[members].mean(axis=0)
                new_centroids.append(mean / max(float(np.linalg.norm(mean)), EPS))
                new_members.append(members)

            converged = len(new_centroids) == len(centroids) and all(
                float(np.max(np.abs(nc - oc))) < tol
                for nc, oc in zip(new_centroids, centroids)
            )
            centroids, member_indices = new_centroids, new_members
            if converged:
                break

        return centroids, member_indices

    def _run_clustering_dpmeans(self) -> tuple[list[PromotionRecord], ClusterEventInfo]:
        X = np.stack(self._buf_norm)
        centroids, member_indices = self._dpmeans(X)

        # Evaluate promotions (min_promote_size + min_intra_cosine_sim; soft-prob
        # check bypassed for hard assignment).
        promoted_centroid_idxs: set[int] = set()
        promotable: list[tuple[int, list[int]]] = []
        for ci, members in enumerate(member_indices):
            if len(members) < self.config.min_promote_size:
                continue
            member_norm = X[members]
            intra = _mean_pairwise_cosine(member_norm)
            if intra < self.config.min_intra_cosine_sim:
                continue
            promotable.append((ci, members))

        promotions, indices_to_remove = self._apply_promotions(
            [(members, None) for _, members in promotable]
        )
        promoted_centroid_idxs = {ci for ci, _ in promotable}

        # Persist surviving centroids (drop promoted ones).
        self._dpmeans_centroids = [
            c for j, c in enumerate(centroids) if j not in promoted_centroid_idxs
        ]

        cluster_event = ClusterEventInfo(
            buffer_size=X.shape[0],
            sweep_counts={},  # no mcs sweep in DP-means
            n_dedup_candidates=len(centroids),
        )
        self._remove_from_buffer(indices_to_remove)
        self._log_event(X.shape[0], len(centroids), len(promotions))
        return promotions, cluster_event

    def _run_clustering_oracle(self) -> tuple[list[PromotionRecord], ClusterEventInfo]:
        buffer_raw = np.stack(self._buf_raw)
        candidates = form_oracle_clusters(buffer_raw, list(self._buf_true), self.config)

        promotable: list[tuple[list[int], np.ndarray]] = []
        for cand in candidates:
            if cand.n_members < self.config.min_promote_size:
                continue
            # oracle clusters are 100% pure (intra==1.0); gate is min_promote_size
            promotable.append((cand.member_indices, cand.centroid_raw))

        promotions, indices_to_remove = self._apply_promotions(promotable)

        cluster_event = ClusterEventInfo(
            buffer_size=buffer_raw.shape[0],
            sweep_counts={},
            n_dedup_candidates=len(candidates),
        )
        self._remove_from_buffer(indices_to_remove)
        self._log_event(buffer_raw.shape[0], len(candidates), len(promotions))
        return promotions, cluster_event

    def _apply_promotions(
        self, promotable: list[tuple[list[int], np.ndarray | None]]
    ) -> tuple[list[PromotionRecord], set[int]]:
        """Append cluster members to the gallery and emit PromotionRecords.

        Each entry is (member_buffer_indices, centroid_raw_or_None). When the
        centroid is None it is computed as the raw mean of the members.
        """
        promotions: list[PromotionRecord] = []
        indices_to_remove: set[int] = set()
        for members, centroid_raw in promotable:
            self._promotion_counter += 1
            cid = f"promoted_{self._promotion_counter:03d}"
            cls_id = self._next_class_idx
            self._next_class_idx += 1
            self._class_names.append(cid)

            member_norm = np.stack([self._buf_norm[i] for i in members])
            self.gallery_embeddings = np.vstack([self.gallery_embeddings, member_norm])
            self.gallery_labels = np.concatenate(
                [self.gallery_labels, np.full(len(members), cls_id, dtype=int)]
            )

            if centroid_raw is None:
                centroid_raw = np.stack(
                    [self._buf_raw[i] for i in members]
                ).mean(axis=0)
            intra = _mean_pairwise_cosine(member_norm)
            member_stream_indices = [self._buf_idx[i] for i in members]

            promotions.append(
                PromotionRecord(
                    cid=cid,
                    member_stream_indices=member_stream_indices,
                    n_members=len(members),
                    intra_cosine_sim=intra,
                    mean_soft_prob=np.nan,  # hard assignment: no soft membership
                    min_cluster_size_used=None,
                    centroid=centroid_raw,
                )
            )
            indices_to_remove.update(members)
        return promotions, indices_to_remove

    def _remove_from_buffer(self, indices_to_remove: set[int]) -> None:
        if not indices_to_remove:
            return
        keep = [i for i in range(len(self._buf_norm)) if i not in indices_to_remove]
        self._buf_norm = [self._buf_norm[i] for i in keep]
        self._buf_raw = [self._buf_raw[i] for i in keep]
        self._buf_idx = [self._buf_idx[i] for i in keep]
        if self.oracle_mode in {"clustering", "both"}:
            self._buf_true = [self._buf_true[i] for i in keep]

    def _log_event(self, buffer_size: int, n_candidates: int, n_promoted: int) -> None:
        logger.info(
            f"DP-means clustering — buffer={buffer_size}, "
            f"clusters={n_candidates}, promoted={n_promoted}, "
            f"buffer_after={len(self._buf_norm)}, gallery={self.gallery_embeddings.shape[0]}"
        )

    # ----- end-of-stream -----
    def drain(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for z_n, stream_idx in zip(self._buf_norm, self._buf_idx):
            dists = 1.0 - self.gallery_embeddings @ z_n
            nearest = int(np.argmin(dists))
            out.append((stream_idx, int(self.gallery_labels[nearest])))
        return out

    # ----- introspection -----
    @property
    def n_ind_classes(self) -> int:
        return len(self._class_names)

    @property
    def ood_buffer_size(self) -> int:
        return len(self._buf_norm)

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    @property
    def detection_threshold(self) -> float | None:
        return self.tau

    @property
    def ood_buffer_stream_indices(self) -> list[int]:
        return list(self._buf_idx)
