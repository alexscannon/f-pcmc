"""Concept data structure (T2 stub — completed in T3; PRD FR-1).

At T2 this carries exactly what the scorers (fpcmc/scorers.py) consume:
identity, geometry (centroid, ref_set), the per-scorer acceptance thresholds,
and the cached vMF concentration. T3 adds the full FR-1 field set
(match/window bookkeeping, reservoir counters, provenance, status, lineage)
plus `add_observation`/`seed` dynamics; the fields below keep their meaning
unchanged there.

Approved deviation (owner, 2026-07-10): FR-1 declares a single `tau: float`,
but FR-4.3's composed scorer accepts "under its respective per-concept
threshold" and the two sub-scorers live on incommensurable scales (knn_ref:
cosine distance in [0, 2]; vmf: negative log-likelihood, large-magnitude and
negative at D=1024). The concept therefore carries `tau` (the knn_ref
threshold — also what single-scorer configs and FR-3.2 singleton seeding use)
and `tau_vmf` (the vmf sub-scorer's threshold). T4 computes both via
leave-one-out percentiles under the respective sub-scorer, and the FR-5.3
global prior likewise becomes per-sub-scorer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Concept:
    """Scoring-relevant slice of PRD FR-1's Concept (see module docstring).

    centroid: (D,) L2-normalized mean direction.
    ref_set:  (K, D) L2-normalized reference embeddings, K >= 1.
    tau:      knn_ref acceptance threshold (accept iff score <= tau, FR-9).
    tau_vmf:  vmf acceptance threshold; NaN is fine while the vmf branch is
              unused (e.g. knn_ref-only configs, or ref_set < n_vmf_min where
              FR-4.2 falls back to knn_ref against `tau`).
    kappa:    cached Banerjee vMF concentration estimate from ref_set (FR-4.2);
              maintained by the concept's owner (T3 dynamics, T6 init).
    """

    concept_id: str
    centroid: np.ndarray
    ref_set: np.ndarray
    tau: float
    kappa: float
    tau_vmf: float = float("nan")
