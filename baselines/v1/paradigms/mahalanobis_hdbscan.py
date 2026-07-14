"""Mahalanobis detection + periodic HDBSCAN clustering, behind the Paradigm interface.

This wraps the existing ``ind_model`` and ``clustering`` functions without
changing their numerics — it is the regression baseline. The stream loop in
``main.py`` retains all ground-truth/metric bookkeeping; this class owns only the
GT-agnostic mechanics (scoring, the OOD buffer, the HDBSCAN sweep, promotion
mechanics, and drain).

Ordering invariant (critical for bit-identical regression): ``step`` mirrors the
original loop exactly — detect → (if OOD) append to buffer → snapshot
``n_ind_classes`` / ``n_ood_buffer`` → (on trigger) sweep/promote/remove. The
snapshots are taken *before* promotion mutates the model/buffer, because the
original loop logged the per-step record before its clustering block ran.
Promotions are emitted in ``evaluate_promotion`` iteration order, with ``cid``
assignment and buffer removal following that same order.
"""

from __future__ import annotations

import numpy as np

from clustering import (
    evaluate_promotion,
    form_oracle_clusters,
    run_sweep,
)
from config import ContinualConfig
from data_loader import TrainData
from ind_model import add_promoted_class, initialize_ind_model, score
from paradigms.base import (
    ClusterEventInfo,
    PromotionRecord,
    StepResult,
)


class MahalanobisHDBSCANParadigm:
    """Baseline paradigm: Mahalanobis OOD detection + periodic HDBSCAN sweeps."""

    def __init__(self) -> None:
        self.model = None  # INDModel, set in warmup
        self.config: ContinualConfig | None = None
        self.oracle_mode: str = "none"

        # OOD buffer (raw embeddings + their stream indices; true_subclass only
        # retained when a clustering oracle is active).
        self._buf_emb: list[np.ndarray] = []
        self._buf_idx: list[int] = []
        self._buf_true: list[str] = []

        self._n_new_ood_since_cluster = 0
        self._promotion_counter = 0

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

        # Reconstruct the original TrainData (subclass_names per example, in the
        # original row order) and reuse initialize_ind_model verbatim. labels
        # index class_names (sorted unique), so class_names[labels] reproduces the
        # original subclass-name array exactly, and the per-class mean/covariance
        # sums are computed over the same rows in the same order -> bit-identical.
        subclass_names = np.asarray(class_names)[labels]
        train_data = TrainData(embeddings=embeddings, subclass_names=subclass_names)
        self.model = initialize_ind_model(train_data, config)

    # ----- per-step -----
    def step(self, z: np.ndarray, t: int) -> StepResult:
        return self._process(z, t, forced_is_ood=None, true_subclass=None)

    def step_oracle(
        self, z: np.ndarray, t: int, true_is_ood: bool, true_subclass: str
    ) -> StepResult:
        # Detection oracle overrides is_ood only when the mode includes detection.
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
        score_t, model_is_ood, nearest_idx = score(self.model, z)
        is_ood = forced_is_ood if forced_is_ood is not None else model_is_ood

        if not is_ood:
            # IND: classify by nearest prototype; no buffer/model mutation.
            return StepResult(
                score=score_t,
                is_ood=False,
                predicted_class=nearest_idx,
                n_ind_classes=len(self.model.class_names),
                n_ood_buffer=len(self._buf_emb),
                promotions_this_step=[],
                cluster_event=None,
            )

        # OOD: append to buffer.
        self._buf_emb.append(z)
        self._buf_idx.append(t)
        if self.oracle_mode in {"clustering", "both"}:
            self._buf_true.append(true_subclass)
        self._n_new_ood_since_cluster += 1

        # Snapshot BEFORE any clustering removal/promotion (matches original log order).
        n_ind_snapshot = len(self.model.class_names)
        n_buf_snapshot = len(self._buf_emb)

        promotions: list[PromotionRecord] = []
        cluster_event: ClusterEventInfo | None = None
        if (
            self._n_new_ood_since_cluster >= self.config.cluster_interval
            and len(self._buf_emb) >= self.config.min_ood_for_clustering
        ):
            promotions, cluster_event = self._run_clustering()
            self._n_new_ood_since_cluster = 0

        return StepResult(
            score=score_t,
            is_ood=True,
            predicted_class=-1,
            n_ind_classes=n_ind_snapshot,
            n_ood_buffer=n_buf_snapshot,
            promotions_this_step=promotions,
            cluster_event=cluster_event,
        )

    def _run_clustering(self) -> tuple[list[PromotionRecord], ClusterEventInfo]:
        buffer_embs = np.stack(self._buf_emb)

        if self.oracle_mode in {"clustering", "both"}:
            candidates = form_oracle_clusters(
                buffer_embs, list(self._buf_true), self.config
            )
        else:
            candidates = run_sweep(buffer_embs, buffer_embs, self.config)

        promotable = evaluate_promotion(candidates, self.config)

        promotions: list[PromotionRecord] = []
        indices_to_remove: set[int] = set()
        for cluster in promotable:
            self._promotion_counter += 1
            cid = f"promoted_{self._promotion_counter:03d}"
            add_promoted_class(self.model, cluster.centroid_raw, cid)
            indices_to_remove.update(cluster.member_indices)
            member_stream_indices = [
                self._buf_idx[i] for i in cluster.member_indices
            ]
            promotions.append(
                PromotionRecord(
                    cid=cid,
                    member_stream_indices=member_stream_indices,
                    n_members=cluster.n_members,
                    intra_cosine_sim=cluster.intra_cosine_sim,
                    mean_soft_prob=cluster.mean_soft_prob,
                    min_cluster_size_used=cluster.min_cluster_size_used,
                    centroid=cluster.centroid_raw,
                )
            )

        # sweep_counts over ALL candidates grouped by mcs (matches original main).
        sweep_counts: dict[int, int] = {}
        for c in candidates:
            mcs = c.min_cluster_size_used
            sweep_counts[mcs] = sweep_counts.get(mcs, 0) + 1

        cluster_event = ClusterEventInfo(
            buffer_size=buffer_embs.shape[0],
            sweep_counts=sweep_counts,
            n_dedup_candidates=len(candidates),
        )

        # Remove promoted members from the buffer, preserving order.
        if indices_to_remove:
            self._buf_emb = [
                e for i, e in enumerate(self._buf_emb) if i not in indices_to_remove
            ]
            self._buf_idx = [
                s for i, s in enumerate(self._buf_idx) if i not in indices_to_remove
            ]
            if self.oracle_mode in {"clustering", "both"}:
                self._buf_true = [
                    x
                    for i, x in enumerate(self._buf_true)
                    if i not in indices_to_remove
                ]

        return promotions, cluster_event

    # ----- end-of-stream -----
    def drain(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for emb, stream_idx in zip(self._buf_emb, self._buf_idx):
            _, _, nearest_idx = score(self.model, emb)
            out.append((stream_idx, nearest_idx))
        return out

    # ----- introspection -----
    @property
    def n_ind_classes(self) -> int:
        return len(self.model.class_names)

    @property
    def ood_buffer_size(self) -> int:
        return len(self._buf_emb)

    @property
    def class_names(self) -> list[str]:
        return self.model.class_names

    @property
    def detection_threshold(self) -> float | None:
        return self.model.threshold

    @property
    def ood_buffer_stream_indices(self) -> list[int]:
        return list(self._buf_idx)
