"""LTM initialization (T6; PRD FR-2).

``initialize_ltm(pool, labels, config)`` builds the production ConceptStore
from a T0 training split: one LTM concept per T0 class (FR-2.1), then the
frozen FR-5.3 global prior, then per-concept thresholds. The TASKS post-T5
build order is load-bearing — the store needs the prior at construction, so
concepts are built first, the prior computed over them, taus recomputed, and
only then is the ConceptStore assembled.

Per-concept construction (FR-2.1):
  - centroid: normalized class mean over ALL class embeddings (not just the
    reservoir).
  - ref_set: reservoir sample of K_max class embeddings. Owner-approved T6
    decision (Q&A 2026-07-11, docs/CHANGES.md T6): the literal FR-1.1 draw
    discipline is replayed over the class pool directly — append while below
    K_max, then exactly one ``rng.random(2)`` pair per further item, replace
    iff ``u < K_max/ref_count_seen`` at slot ``int(v*K_max)`` — using the
    per-concept substream ``make_rng(config.seed, f"reservoir/{cid}")``.
    ``add_observation`` is deliberately NOT used: it would count init
    material in ``match_count``/``match_windows``, which are post-seed match
    statistics (T3 decision 2). ``ref_count_seen`` ends at the full class
    count, so stream-time replacement probabilities continue exactly where
    init left off, on the same Generator.
  - kappa: FR-4.2 Banerjee estimate from the ref_set (the cache the frozen
    VmfScorer reads; self-maintained by add_observation thereafter).
  - tau / tau_vmf: pure FR-5.1 percentiles for both sub-scorers via the
    status-sensitive ``recompute_thresholds`` (LTM branch), after the global
    prior exists.
  - ids: ``ltm_{i:03d}`` in np.unique(labels) order (owner-approved T6
    decision — sorted class names for real data and fixtures alike; id order
    feeds the FR-4.3 lexicographic tie-break, so it must be canonical).

No pooled covariance, no precision matrix (FR-2.2). Read-only scoring sweeps
over an initialized store (the M1 gate) must go through the frozen scorer,
never ``ConceptStore.route()``, which mutates (T5 as-built note).
"""

from __future__ import annotations

import numpy as np

from fpcmc.concepts import Concept, ConceptStore
from fpcmc.config import FPCMCConfig
from fpcmc.rng import make_rng
from fpcmc.scorers import estimate_kappa
from fpcmc.thresholds import compute_global_prior, recompute_thresholds

_EPS = 1e-12

#: ltm ids are zero-padded to 3 digits (PRD FR-1 literal; T5 decision 8 —
#: widening would break the lexicographic ordering the tie-break relies on).
_MAX_LTM_CLASSES = 1000


def _reservoir_sample(x: np.ndarray, k_max: int, rng: np.random.Generator) -> np.ndarray:
    """Exact FR-1.1 reservoir replay over a class pool.

    Mirrors Concept.add_observation's draw discipline (module docstring
    there): items 0..k_max-1 append without randomness; item i (0-based,
    ref_count_seen = i+1) consumes exactly one ``rng.random(2)`` pair and
    replaces slot ``int(v * k_max)`` iff ``u < k_max / (i+1)``. Returns an
    owned (min(n, k_max), D) float64 array.
    """
    n = x.shape[0]
    ref = np.array(x[: min(n, k_max)], dtype=np.float64)
    for i in range(k_max, n):
        u, v = rng.random(2)
        if u < k_max / (i + 1):
            ref[int(v * k_max)] = x[i]
    return ref


def initialize_ltm(pool: np.ndarray, labels: np.ndarray, config: FPCMCConfig) -> ConceptStore:
    """FR-2: build the T0 ConceptStore — one LTM concept per class in `pool`.

    pool:   (N, D) L2-normalized embeddings (FR-1.2 contract, as the data
            layer and fixture world both provide).
    labels: (N,) parallel class labels; classes are np.unique(labels) and
            ltm ids follow that (sorted) order.
    """
    pool = np.asarray(pool)
    labels = np.asarray(labels)
    if pool.ndim != 2 or pool.shape[0] != labels.shape[0]:
        raise ValueError(
            f"pool/labels misaligned: pool {pool.shape}, labels {labels.shape}"
        )
    classes = np.unique(labels)
    if classes.size == 0:
        raise ValueError("initialize_ltm needs at least one class (FR-2.1)")
    if classes.size > _MAX_LTM_CLASSES:
        raise ValueError(
            f"{classes.size} classes exceed the ltm_{{:03d}} id space "
            f"({_MAX_LTM_CLASSES}); widening would break the FR-4.3 tie-break ordering"
        )

    concepts: list[Concept] = []
    for i, name in enumerate(classes):
        cid = f"ltm_{i:03d}"
        x_c = np.asarray(pool[labels == name], dtype=np.float64)

        centroid = x_c.mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), _EPS)

        # The reservoir substream continues into stream time on this same
        # Generator (T3 decision 3: the concept owns it thereafter).
        rng = make_rng(config.seed, f"reservoir/{cid}")
        ref_set = _reservoir_sample(x_c, config.K_max_refset, rng)

        concepts.append(
            Concept(
                concept_id=cid,
                centroid=centroid,
                ref_set=ref_set,
                tau=float("nan"),  # both taus land below, after the prior
                kappa=estimate_kappa(ref_set),
                tau_vmf=float("nan"),
                status="LTM",
                ref_count_seen=int(x_c.shape[0]),
                created_at=0,
                last_matched_at=0,
                provenance="initial",
                window_W=config.window_W,
                k_max=config.K_max_refset,
                alpha_ema=config.alpha_stm_ema,
                rng=rng,
            )
        )

    # FR-5.3 prior over the finished T0 concepts, then pure FR-5.1 taus
    # (recompute_thresholds' LTM branch) — the TASKS post-T5 build order.
    prior = compute_global_prior(concepts, config)
    for concept in concepts:
        recompute_thresholds(concept, config, prior)

    return ConceptStore(config, prior, concepts)
