"""Paradigm interface: the contract every continual-learning paradigm satisfies.

The stream loop in ``main.py`` owns all ground-truth / metric bookkeeping
(majority labels, purity, cluster_label_map, promoted_subclasses, snapshots,
detection-accuracy metrics, etc.). A paradigm owns only the GT-agnostic
mechanics: detection-model state, the OOD buffer, clustering, promotion
*mechanics* (assigning a cluster id, appending the class to its model, removing
members from its buffer), and the end-of-stream drain.

The bridge between the two is ``StepResult``: on a promotion step the paradigm
returns ``PromotionRecord`` objects carrying exactly what ``main.py`` needs to
reproduce its bookkeeping (member stream indices + the assigned cluster id), and
a ``ClusterEventInfo`` carrying the mechanical bits for assembling a
``ClusteringEvent``.

Oracle modes: ``main.py`` keeps the detection-oracle ``promoted_subclasses``
logic and passes the resolved ``true_is_ood`` into ``step_oracle``. The
clustering-oracle substitution (``form_oracle_clusters``) lives inside each
paradigm's ``step_oracle``; a paradigm reads ``config.oracle_mode`` at warmup
and retains each buffered item's ``true_subclass`` iff the mode is
``clustering`` or ``both``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np

from config import ContinualConfig
from data_loader import TrainData


@dataclass
class PromotionRecord:
    """One promoted cluster, as surfaced to ``main.py`` for GT bookkeeping.

    ``member_stream_indices`` are stream ``t`` values (not buffer indices), so
    ``main.py`` can look up ground-truth labels directly. ``cid`` is assigned by
    the paradigm (e.g. ``"promoted_001"``). For hard-assignment paradigms that
    have no soft membership, ``mean_soft_prob`` is ``np.nan`` and
    ``min_cluster_size_used`` is ``None``.
    """

    cid: str
    member_stream_indices: list[int]
    n_members: int
    intra_cosine_sim: float
    mean_soft_prob: float | None
    min_cluster_size_used: int | None
    centroid: np.ndarray  # raw or normalized depending on paradigm; for t-SNE


@dataclass
class ClusterEventInfo:
    """Mechanical bits for assembling a ``ClusteringEvent`` in ``main.py``.

    ``sweep_counts`` maps min_cluster_size -> n_clusters for sweep-based
    paradigms; it is ``{}`` for paradigms with no mcs sweep.
    """

    buffer_size: int
    sweep_counts: dict[int, int]
    n_dedup_candidates: int


class StepResult(NamedTuple):
    """Result of processing a single stream example.

    ``predicted_class`` is an index into ``Paradigm.class_names`` (or -1 when the
    example is OOD and no assignment was made). ``main.py`` maps it back to a
    name. ``cluster_event`` is set only on a clustering-trigger step.

    ``n_ind_classes`` and ``n_ood_buffer`` are **snapshots as of record-build
    time** — i.e. after this item's detection/classification but *before* any
    promotion this step mutates them. They are returned (rather than read from
    the introspection properties after ``step``) so ``main.py`` reproduces the
    original logging order bit-for-bit: in the unrefactored loop the per-step
    record is logged before the clustering/promotion block runs.

    ``extras`` carries paradigm-specific auxiliary signals (e.g. alternative
    detection-score variants) keyed by name. It defaults to ``{}`` and is empty
    for every paradigm except those that opt in (vMF logs candidate scores), so
    the per-step CSV schema and ``results_summary.json`` are unchanged for
    paradigms that don't use it. The default dict is read-only — paradigms build
    a fresh dict per step and never mutate the shared default.
    """

    score: float
    is_ood: bool
    predicted_class: int
    n_ind_classes: int
    n_ood_buffer: int
    promotions_this_step: list[PromotionRecord]
    cluster_event: ClusterEventInfo | None
    extras: dict[str, float] = {}


@dataclass
class WarmupData:
    """Inputs for ``Paradigm.warmup``.

    ``train_data`` carries the original ``TrainData`` so paradigms that need
    bit-identical reuse of existing fitting code (the Mahalanobis baseline) can
    call it verbatim rather than re-deriving from ``embeddings`` / ``labels``.
    """

    embeddings: np.ndarray   # (N, D)
    labels: np.ndarray       # (N,) int, indexes class_names
    class_names: list[str]   # sorted unique
    train_data: TrainData


@runtime_checkable
class Paradigm(Protocol):
    """Common surface for all continual-learning paradigms."""

    # ----- lifecycle -----
    def warmup(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        class_names: list[str],
        config: ContinualConfig,
    ) -> None: ...

    # ----- per-step -----
    def step(self, z: np.ndarray, t: int) -> StepResult: ...

    def step_oracle(
        self, z: np.ndarray, t: int, true_is_ood: bool, true_subclass: str
    ) -> StepResult: ...

    # ----- end-of-stream -----
    def drain(self) -> list[tuple[int, int]]:
        """Force-classify the residual buffer.

        Returns ``(stream_idx, predicted_class_idx)`` per residual item in buffer
        insertion order. No score is carried (drain rows log score=0.0).
        """
        ...

    # ----- introspection (for final report / status logs) -----
    @property
    def n_ind_classes(self) -> int: ...
    @property
    def ood_buffer_size(self) -> int: ...
    @property
    def class_names(self) -> list[str]: ...
    @property
    def detection_threshold(self) -> float | None:
        """Scalar detection threshold for metadata / score-distribution plot.

        ``None`` for paradigms with no scalar threshold (e.g. vMF-DPMM).
        """
        ...
    @property
    def ood_buffer_stream_indices(self) -> list[int]:
        """Stream ``t`` values of items remaining in the residual buffer.

        Used by ``main.py`` at end-of-stream for residual-buffer cluster-quality
        metrics. Order matches ``drain()``.
        """
        ...
