"""B2 baseline: batch knn_vmf detection at each protocol checkpoint (PRD §7.4).

The "no memory management" reference: the source paradigm's STATIC batch
detector — a frozen T0-train gallery, growth disabled, scored offline over the
stream prefix at each P1/P2 checkpoint. No batch source exists to port
(docs/ASSETS.md §2): this module reimplements the detection statistic with
citations, restructured batch-over-queries.

Statistic (lib/evaluation/continual/paradigms/knn_vmf.py, blob 7ef929c1 —
``_calibrate_tau`` lines 148-164 and ``_process`` detection lines 261-263;
also ``scripts/batch_knn_vmf_static.py``, blob f17c4194, the run behind the
``t6_m1_gate`` pin):

    score(z) = kth-neighbor cosine distance to the gallery
             = np.partition(1 - G @ z, k-1)[k-1]          (variant "kth")
    "mean_k" variant: mean of the k smallest distances.

k = 20 and variant = "kth" mirror the source ``config.yaml`` ``knn:`` block
(the values behind every pinned batch number). Scoring is blocked GEMM over
queries — mathematically identical per query to the source's per-query matvec
(each query's distance vector is an independent row); the T14 tolerance is
±0.005, not bitwise.

Everything here is detection-only, mirroring the existing batch artifact; the
classification-side B2 story (if any) is T16's concern.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from eval.gt import _POOL_OOD_KIND
from eval.metrics import ood_metrics

# Source config.yaml `knn:` defaults (blob 33154bf4) — the pinned statistic.
KNN_K = 20
KNN_DISTANCE = "kth"


def build_gallery(
    ind_reference, t0_classes: Sequence[str] | None = None
) -> np.ndarray:
    """Frozen detection gallery from the ind_reference (T0 train) pool.

    ``t0_classes`` restricts the gallery to the protocol's T0 classes (P2's
    80-class split); None keeps all rows (P1: T0 = all 100 classes). Rows are
    float64 unit vectors, matching the source's ``_l2_normalize_rows(
    embeddings.astype(np.float64))`` gallery (fpcmc.data pools are already
    unit-norm, so the cast is the only remaining step).
    """
    x = np.asarray(ind_reference.x, dtype=np.float64)
    if t0_classes is None:
        return x
    keep = np.isin(ind_reference.subclass_names, np.asarray(list(t0_classes)))
    return x[keep]


def knn_scores(
    gallery: np.ndarray,
    queries: np.ndarray,
    *,
    k: int = KNN_K,
    variant: str = KNN_DISTANCE,
    block: int = 512,
) -> np.ndarray:
    """kth-NN (or mean-k) cosine distance of each query to the gallery."""
    if variant not in ("kth", "mean_k"):
        raise ValueError(f"variant must be 'kth' or 'mean_k', got {variant!r}")
    gallery = np.asarray(gallery, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    n = queries.shape[0]
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, block):
        end = min(start + block, n)
        # knn_vmf.py:261-263 per query: dists = 1 - G @ z; partition to k.
        dists = 1.0 - queries[start:end] @ gallery.T
        part = np.partition(dists, k - 1, axis=1)[:, :k]
        out[start:end] = part.mean(axis=1) if variant == "mean_k" else part[:, k - 1]
    return out


def _stratified(scores: np.ndarray, kind: np.ndarray) -> dict:
    """all/near/far detection blocks present iff their populations are."""
    ind = scores[kind == "ind"]
    out: dict = {}
    ood_all = scores[kind != "ind"]
    if len(ind) and len(ood_all):
        out["all_ood"] = ood_metrics(ind, ood_all)
        for stratum in ("near", "far"):
            sel = scores[kind == stratum]
            if len(sel):
                out[f"{stratum}_ood"] = ood_metrics(ind, sel)
    return out


def evaluate_batch_checkpoints(
    protocol,
    pools: Mapping[str, object] | None = None,
    *,
    k: int = KNN_K,
    variant: str = KNN_DISTANCE,
) -> dict:
    """Score a T12 ``ProtocolStream`` with the static batch detector.

    Scores every stream embedding once against the frozen T0 gallery, then
    reports stratified detection metrics over the stream PREFIX at each
    protocol checkpoint, always including end-of-stream (P1 declares no
    checkpoints, so its report is the end-of-stream block alone — which on P1
    is exactly the ``t6_m1_gate`` static-batch population: warmup rows are
    ind_test and belong to the pinned run's 10,250-example IND side).
    """
    if pools is None:
        from fpcmc.data import load_all_pools  # deferred: touches real data

        pools = load_all_pools()
    gallery = build_gallery(pools["ind_reference"], protocol.t0_classes)
    scores = knn_scores(gallery, protocol.x, k=k, variant=variant)
    kind = np.array(
        [_POOL_OOD_KIND[p] for p in protocol.manifest.pool.tolist()], dtype=str
    )

    n = int(protocol.x.shape[0])
    steps = sorted(set(protocol.checkpoint_steps) | {n - 1})
    all_steps = np.arange(n)
    checkpoints = [
        {
            "step": int(s),
            "detection": _stratified(scores[all_steps <= s], kind[all_steps <= s]),
        }
        for s in steps
    ]
    return {
        "n_steps": n,
        "gallery_size": int(gallery.shape[0]),
        "k": int(k),
        "variant": variant,
        "checkpoint_steps": [int(s) for s in steps],
        "checkpoints": checkpoints,
        "scores": scores,
    }


__all__ = [
    "KNN_DISTANCE",
    "KNN_K",
    "build_gallery",
    "evaluate_batch_checkpoints",
    "knn_scores",
]
