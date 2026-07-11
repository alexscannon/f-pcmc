"""Per-concept adaptive thresholds (T4; PRD FR-5.1–5.4).

The core replacement for v1's global static threshold: every concept carries
its own acceptance boundary, computed from leave-one-out scores of its own
reference-set members and shrunk toward a frozen global prior while the
concept is a small STM candidate. Per the owner-approved T2 threshold split
(PRD FR-5 note, docs/CHANGES.md T2), everything here runs per sub-scorer
under `scorer=knn_vmf`: the LOO/percentile/shrinkage/prior machinery is
applied under knn_ref (feeding `concept.tau`) and under vmf (feeding
`concept.tau_vmf`), and the FR-5.3 global prior is a per-sub-scorer pair.

Owner-approved T4 decisions (Q&A 2026-07-10, recorded in docs/CHANGES.md T4):

  1. vmf LOO semantics — a ref member's LOO vmf score is computed against the
     concept's cached (centroid, kappa) as-is (FR-5.1's "against the
     concept", exactly the decision function runtime queries face; the
     kappa cache is self-maintained per T3 decision 1). No per-member
     parameter re-fit: "leave-one-out" bites only where self-exclusion
     matters, i.e. the knn_ref pairwise distances.
  2. tau_emp below the computable floor holds the prior — knn_ref LOO needs
     ref_set >= 2 (a singleton's LOO view is empty), so at n=1 a recompute
     sets tau = tau_prior exactly; tau_vmf is recomputed only once
     ref_set >= n_vmf_min (below that VmfScorer is in knn_ref-fallback and
     never reads tau_vmf, which keeps its seeded prior).
  3. Lazy trigger — `Concept.refset_changes_since_tau` counts actual ref_set
     mutations (appends and reservoir replacements; post-fill skipped draws
     don't count, matching FR-5.1's "ref_set has changed"). The trigger fires
     when the counter reaches >= REFSET_CHANGE_FRACTION (25%) of the current
     ref_set size, evaluated only when `maybe_recompute` is called (lazy
     check-on-call; T5's store picks the call site). Every recompute resets
     the counter. The trigger governs tau/tau_vmf only — kappa is
     self-maintained per observation (T3 decision 1).

Threshold rules by status (FR-5.1 vs FR-5.2): LTM concepts get the pure
q-th-percentile tau; STM concepts get the shrinkage estimate
tau = w*tau_emp + (1-w)*tau_prior with w = n/(n + n_shrink), n = ref_set
size. `recompute_on_promotion` (FR-5.4) applies the pure FR-5.1 rule to the
full ref_set unconditionally — that recompute is what makes routing
promotion-aware.

Percentile method: np.percentile with the default linear interpolation,
mirroring the source batch pipeline's tau calibration
(lib/evaluation/continual/paradigms/knn_vmf.py::_calibrate_tau, read-only
lib snapshot) — as is the LOO discipline of masking the self-distance to
+inf before the k-smallest partition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fpcmc.concepts import Concept
from fpcmc.config import FPCMCConfig
from fpcmc.scorers import log_C_D

#: FR-5.1: recompute lazily once the ref_set has changed by >= 25%.
REFSET_CHANGE_FRACTION = 0.25


@dataclass(frozen=True)
class GlobalPrior:
    """FR-5.3 global prior pair — the q-th percentile of pooled T0 LOO scores.

    Computed once at initialization and frozen thereafter (this dataclass is
    immutable and stores plain floats, so later concept mutation cannot reach
    it). Used only as the FR-5.2 shrinkage target and for FR-3.2 singleton
    seeding — never as a decision boundary by itself.

    tau:     knn_ref-scale prior (cosine-distance scale).
    tau_vmf: vmf-scale prior (negative log-likelihood scale).
    """

    tau: float
    tau_vmf: float


# ---------------------------------------------------------- LOO scores (FR-5.1)


def loo_knn_scores(ref_set: np.ndarray, k_ref: int) -> np.ndarray:
    """(K,) leave-one-out knn_ref scores of every ref_set member.

    Member i's score is its FR-4.1 knn_ref score against the concept with
    itself excluded: mean cosine distance to its min(k_ref, K-1) nearest
    *other* members. Self-exclusion masks the diagonal to +inf (duplicate
    rows still legitimately score 0 against their twins), mirroring
    lib/.../knn_vmf.py::_calibrate_tau. Requires K >= 2.
    """
    n = ref_set.shape[0]
    if ref_set.ndim != 2 or n < 2:
        raise ValueError(
            f"knn_ref LOO needs a (K>=2, D) ref_set (a singleton has an empty "
            f"LOO view), got shape {ref_set.shape}"
        )
    dists = 1.0 - ref_set @ ref_set.T
    np.fill_diagonal(dists, np.inf)
    k = min(int(k_ref), n - 1)
    return np.partition(dists, k - 1, axis=1)[:, :k].mean(axis=1)


def loo_vmf_scores(concept: Concept) -> np.ndarray:
    """(K,) vmf scores of every ref_set member against the cached (mu, kappa).

    Owner decision 1 (module docstring): members are scored under the
    concept's cached parameters — the same -(log C_D(kappa) + kappa*mu.z)
    the frozen VmfScorer computes for runtime queries — with no per-member
    re-fit, so no self-exclusion applies.
    """
    kappa = concept.kappa
    if not np.isfinite(kappa):
        raise ValueError(
            f"concept {concept.concept_id!r}: kappa={kappa} is not finite — "
            "the cached Banerjee estimate must be maintained before computing "
            "vmf LOO scores (FR-4.2; T3 per-observation cadence)"
        )
    d = concept.centroid.shape[0]
    return -(log_C_D(kappa, d) + kappa * (concept.ref_set @ concept.centroid))


def tau_empirical(scores: np.ndarray, q: float) -> float:
    """q-th percentile of LOO scores — np.percentile, default linear method
    (pinned by mirroring lib/.../knn_vmf.py::_calibrate_tau)."""
    return float(np.percentile(scores, q))


# ------------------------------------------------------------ shrinkage (FR-5.2)


def shrinkage_weight(n: int, n_shrink: int) -> float:
    """FR-5.2 empirical weight w = n / (n + n_shrink)."""
    return n / (n + n_shrink)


def shrink(tau_emp: float, tau_prior: float, n: int, n_shrink: int) -> float:
    """FR-5.2 shrinkage estimate tau = w*tau_emp + (1-w)*tau_prior."""
    if n == 0:
        return float(tau_prior)  # exact, even against a NaN tau_emp
    w = shrinkage_weight(n, n_shrink)
    return w * float(tau_emp) + (1.0 - w) * float(tau_prior)


# --------------------------------------------------------- global prior (FR-5.3)


def compute_global_prior(concepts: Sequence[Concept], config: FPCMCConfig) -> GlobalPrior:
    """FR-5.3: per-sub-scorer q-th percentiles of LOO scores pooled over all
    T0 concepts. Computed once at initialization; the returned pair is frozen.

    Every T0 concept must be well-populated (ref_set >= 2 for the knn pool;
    LTM initialization gives K_max members, so also >= n_vmf_min for a
    trustworthy vmf pool) — this is an init-time computation, not a
    streaming one.
    """
    if len(concepts) == 0:
        raise ValueError("global prior needs at least one T0 concept (FR-5.3)")
    knn_pool = np.concatenate([loo_knn_scores(c.ref_set, config.k_ref) for c in concepts])
    vmf_pool = np.concatenate([loo_vmf_scores(c) for c in concepts])
    return GlobalPrior(
        tau=tau_empirical(knn_pool, config.tau_percentile_q),
        tau_vmf=tau_empirical(vmf_pool, config.tau_percentile_q),
    )


# ------------------------------------------------- recomputation (FR-5.1/5.2/5.4)


def recompute_thresholds(concept: Concept, config: FPCMCConfig, prior: GlobalPrior) -> None:
    """Recompute the concept's tau (and, when computable, tau_vmf) in place.

    LTM: pure FR-5.1 percentile. STM: FR-5.2 shrinkage toward the global
    prior with n = ref_set size. Below the per-sub-scorer floors (owner
    decision 2): tau at n=1 is set to the prior exactly; tau_vmf below
    n_vmf_min is left untouched (unread in VmfScorer's fallback mode).
    Resets the FR-5.1 dirty counter.
    """
    n = concept.ref_set.shape[0]
    q = config.tau_percentile_q

    if n >= 2:
        emp = tau_empirical(loo_knn_scores(concept.ref_set, config.k_ref), q)
        concept.tau = emp if concept.status == "LTM" else shrink(emp, prior.tau, n, config.n_shrink)
    else:
        concept.tau = float(prior.tau)

    if n >= config.n_vmf_min:
        emp_vmf = tau_empirical(loo_vmf_scores(concept), q)
        concept.tau_vmf = (
            emp_vmf
            if concept.status == "LTM"
            else shrink(emp_vmf, prior.tau_vmf, n, config.n_shrink)
        )

    concept.refset_changes_since_tau = 0


def maybe_recompute(concept: Concept, config: FPCMCConfig, prior: GlobalPrior) -> bool:
    """FR-5.1 lazy trigger: recompute iff the ref_set has changed by >= 25%
    (mutation count vs current ref_set size) since the last computation.

    Evaluated on call (owner decision 3) — T5's store decides when to check.
    Returns True iff a recompute fired.
    """
    if concept.refset_changes_since_tau >= REFSET_CHANGE_FRACTION * concept.ref_set.shape[0]:
        recompute_thresholds(concept, config, prior)
        return True
    return False


def recompute_on_promotion(concept: Concept, config: FPCMCConfig) -> None:
    """FR-5.4 promotion hook: both taus recomputed from the full ref_set under
    the pure FR-5.1 percentile rule — no shrinkage, regardless of the status
    field's current value (T8's atomic promotion flips status around this
    call). This calibrated boundary is what makes routing promotion-aware.
    """
    n = concept.ref_set.shape[0]
    if n < 2:
        raise ValueError(
            f"concept {concept.concept_id!r}: promotion recompute needs "
            f"ref_set >= 2, got {n} (FR-7.1 size criterion admits no such candidate)"
        )
    q = config.tau_percentile_q
    concept.tau = tau_empirical(loo_knn_scores(concept.ref_set, config.k_ref), q)
    if n >= config.n_vmf_min:
        concept.tau_vmf = tau_empirical(loo_vmf_scores(concept), q)
    concept.refset_changes_since_tau = 0
