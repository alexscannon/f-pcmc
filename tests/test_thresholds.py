"""T4 [U] tests — per-concept adaptive thresholds (TASKS Task 4; PRD FR-5.1–5.4).

All tests are synthetic/hand-built (2-D exact-arithmetic cases or the vMF
fixture world); no real data; all randomness through fpcmc.rng.make_rng.

Owner-approved T4 semantics (Q&A 2026-07-10, recorded in docs/CHANGES.md T4):
  - vmf LOO scores each ref member against the concept's cached
    (centroid, kappa) as-is — "against the concept" as runtime queries face
    it; leave-one-out bites only where self-exclusion matters (knn_ref).
  - Below the computable floor, thresholds hold the prior: tau needs
    ref_set >= 2 (at n=1 a recompute sets tau = tau_prior exactly); tau_vmf
    is recomputed only once ref_set >= n_vmf_min (below that VmfScorer is in
    knn_ref-fallback and never reads it — it keeps its seeded prior).
  - The FR-5.1 dirty counter counts actual ref_set mutations (append or
    reservoir replacement; post-fill skipped draws don't count); the trigger
    is counter >= 0.25 * current ref_set size, evaluated only when
    maybe_recompute() is called (lazy check-on-call; T5 picks the call site);
    the counter resets on every recompute.
Percentile method mirrors the source batch pipeline (lib/.../knn_vmf.py::
_calibrate_tau): np.percentile with the default linear interpolation.
"""

import dataclasses

import numpy as np
import pytest

import fpcmc.thresholds as thresholds
from fpcmc.concepts import Concept
from fpcmc.config import FPCMCConfig
from fpcmc.rng import make_rng
from fpcmc.scorers import estimate_kappa, make_scorer
from fpcmc.thresholds import (
    GlobalPrior,
    compute_global_prior,
    loo_knn_scores,
    loo_vmf_scores,
    maybe_recompute,
    recompute_on_promotion,
    recompute_thresholds,
    shrink,
    shrinkage_weight,
    tau_empirical,
)
from tests.fixtures.vmf_world import VMFWorld

SEED = 401


def _unit_rows(n: int, d: int, stream: str) -> np.ndarray:
    x = make_rng(SEED, stream).standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _concept_from(ref_set: np.ndarray, status: str, **kw) -> Concept:
    """A consistent Concept over an owned copy of ref_set (kappa cached)."""
    ref = np.array(ref_set, dtype=np.float64)
    centroid = ref.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    return Concept(
        concept_id=kw.pop("concept_id", "stm_0000"),
        centroid=centroid,
        ref_set=ref,
        tau=kw.pop("tau", 0.5),
        kappa=estimate_kappa(ref),
        status=status,
        **kw,
    )


# ------------------------------------------------------- LOO percentile (FR-5.1)


def test_loo_hand_case():
    """5-point 2-D ref_set, k_ref=1: LOO scores and percentile match hand math.

    Points on the unit circle: p0=[1,0], p1=[.8,.6], p2=[.6,.8], p3=[0,1],
    p4=[-1,0]. Pairwise cosines are tiny exact-ish products, so every LOO
    nearest-neighbour distance is hand-computable:
      p0: NN p1, 1-0.8;  p1: NN p2, 1-0.96;  p2: NN p1, 1-0.96;
      p3: NN p2, 1-0.8;  p4: NN p3, 1-0.0 = 1.0.
    Assertions are exact up to one ulp (atol 1e-15 absorbs BLAS dot-product
    ordering/FMA in the pairwise matmul; all values are O(0.04..1)).
    """
    ref = np.array(
        [[1.0, 0.0], [0.8, 0.6], [0.6, 0.8], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float64
    )
    hand = np.array([1.0 - 0.8, 1.0 - 0.96, 1.0 - 0.96, 1.0 - 0.8, 1.0])

    scores = loo_knn_scores(ref, k_ref=1)
    np.testing.assert_allclose(scores, hand, rtol=0.0, atol=1e-15)

    # q=75 lands exactly on an order statistic ((n-1)*0.75 = 3.0): no
    # interpolation, so the value is the 4th sorted score, 1-0.8, exactly.
    np.testing.assert_allclose(
        tau_empirical(scores, q=75), 1.0 - 0.8, rtol=0.0, atol=1e-15
    )
    # Default q=95 interpolates between sorted[3] and sorted[4]; the module
    # must agree with np.percentile (default linear method, mirroring
    # lib/.../knn_vmf.py::_calibrate_tau) applied to the hand scores.
    np.testing.assert_allclose(
        tau_empirical(scores, q=95), np.percentile(hand, 95), rtol=0.0, atol=1e-15
    )


def test_loo_excludes_self():
    """LOO never scores a member against itself (the trivial zero self-distance).

    ref_set [a, a, b] with a duplicated and b at 60 degrees from a:
      - k_ref=1: the duplicated members legitimately score 0 (their twin is a
        genuinely distinct row at distance 0); b must score 1-cos(60)=0.5 —
        a self-matching bug would give 0.
      - k_ref=2: b's two nearest non-self members are the two a's, so its
        score is 0.5; self-inclusion would average in a zero and give 0.25.
        The a's score mean(0, 0.5)/2 = 0.25 each.
    """
    a = np.array([1.0, 0.0])
    b = np.array([0.5, np.sqrt(3.0) / 2.0])  # 60 degrees from a
    ref = np.stack([a, a, b])

    s1 = loo_knn_scores(ref, k_ref=1)
    np.testing.assert_allclose(s1[:2], [0.0, 0.0], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(s1[2], 0.5, rtol=0.0, atol=1e-15)
    assert s1[2] > 0.0, "self-match bug: b's LOO score collapsed to its self-distance"

    s2 = loo_knn_scores(ref, k_ref=2)
    np.testing.assert_allclose(s2, [0.25, 0.25, 0.5], rtol=0.0, atol=1e-15)

    # k_ref clips to K-1 non-self members: with K=3, k_ref=99 averages the
    # two allowed neighbours, never the self row.
    s_clip = loo_knn_scores(ref, k_ref=99)
    np.testing.assert_allclose(s_clip[2], 0.5, rtol=0.0, atol=1e-15)

    # A singleton has no LOO view at all.
    with pytest.raises(ValueError, match="2"):
        loo_knn_scores(a[None, :], k_ref=1)


# ------------------------------------------------------------ shrinkage (FR-5.2)


def test_shrinkage_limits():
    """n=0 => tau_prior exactly; n=10,000 => within 1e-3*tau_emp; w(n_shrink)=0.5."""
    tau_emp, tau_prior, n_shrink = 0.40, 0.60, 10

    assert shrink(tau_emp, tau_prior, n=0, n_shrink=n_shrink) == tau_prior

    tau_big = shrink(tau_emp, tau_prior, n=10_000, n_shrink=n_shrink)
    assert abs(tau_big - tau_emp) < 1e-3 * tau_emp

    assert shrinkage_weight(n=n_shrink, n_shrink=n_shrink) == 0.5
    # w=0.5 means the exact midpoint.
    assert shrink(tau_emp, tau_prior, n=n_shrink, n_shrink=n_shrink) == pytest.approx(
        0.5 * (tau_emp + tau_prior), abs=1e-15
    )


# ---------------------------------------------------------- global prior (FR-5.3)


def test_prior_fixed_after_t0():
    """The stored global prior is frozen: mutating concepts never changes it."""
    config = FPCMCConfig()
    world = VMFWorld(seed=5, k_known=3, k_novel=1, kappa_known=150.0)
    pool = world.t0_pool(n_per_class=40)
    concepts = [
        _concept_from(
            pool.x[pool.labels == name],
            status="LTM",
            provenance="initial",
            concept_id=f"ltm_{i:03d}",
        )
        for i, name in enumerate(world.known_names)
    ]

    prior = compute_global_prior(concepts, config)
    tau_before, tau_vmf_before = prior.tau, prior.tau_vmf
    assert np.isfinite(tau_before) and tau_before > 0.0
    assert np.isfinite(tau_vmf_before)

    # Mutate every concept hard: off-class observations churn ref_set,
    # kappa, and (for STM) would move centroids; recompute their taus too.
    novel = world.novel_pool(n_per_class=50)
    for i, concept in enumerate(concepts):
        concept.rng = make_rng(SEED, f"t4/prior/reservoir/{i}")
        for step, z in enumerate(novel.x):
            concept.add_observation(z, step=step)
        recompute_thresholds(concept, config, prior)

    assert prior.tau == tau_before
    assert prior.tau_vmf == tau_vmf_before

    # And the object is structurally immutable (frozen dataclass).
    with pytest.raises(dataclasses.FrozenInstanceError):
        prior.tau = 0.0

    # Determinism: recomputing from identically rebuilt T0 concepts gives the
    # identical pair (the mutated concepts above must not be the source).
    rebuilt = [
        _concept_from(
            pool.x[pool.labels == name],
            status="LTM",
            provenance="initial",
            concept_id=f"ltm_{i:03d}",
        )
        for i, name in enumerate(world.known_names)
    ]
    prior2 = compute_global_prior(rebuilt, config)
    assert (prior2.tau, prior2.tau_vmf) == (tau_before, tau_vmf_before)


# ------------------------------------------------------- lazy recompute (FR-5.1)


def test_lazy_recompute_trigger(mocker):
    """K_max=64: 15 mutations don't reach 25% of the ref_set; 17 do; counter resets.

    The concept starts below capacity (47 rows), so every observation appends
    (a real ref_set mutation, no reservoir draws — fully deterministic). The
    trigger is evaluated lazily, only when maybe_recompute() is called:
    after 15 appends the check sees 15 < 0.25*62; after two more it sees
    17 >= 0.25*64 and fires exactly one recompute, resetting the counter.
    """
    config = FPCMCConfig()  # K_max_refset=64, n_vmf_min=10, q=95, n_shrink=10
    prior = GlobalPrior(tau=0.5, tau_vmf=0.0)
    rows = _unit_rows(64, 8, "t4/lazy/rows")
    concept = _concept_from(
        rows[:47],
        status="STM",
        provenance="seeded",
        k_max=64,
        rng=make_rng(SEED, "t4/lazy/reservoir"),
    )
    assert concept.refset_changes_since_tau == 0

    spy = mocker.spy(thresholds, "recompute_thresholds")

    for i in range(15):
        concept.add_observation(rows[47 + i], step=i)
    assert concept.refset_changes_since_tau == 15
    assert maybe_recompute(concept, config, prior) is False
    assert spy.call_count == 0, "recompute fired below the 25% change threshold"

    for i in range(15, 17):
        concept.add_observation(rows[47 + i], step=i)
    assert concept.refset_changes_since_tau == 17
    assert maybe_recompute(concept, config, prior) is True
    assert spy.call_count == 1
    assert concept.refset_changes_since_tau == 0, "counter must reset on recompute"

    # Immediately after the reset there is nothing dirty: no second fire.
    assert maybe_recompute(concept, config, prior) is False
    assert spy.call_count == 1


# ------------------------------------------------- semantic correctness (FR-5 stack)


def test_threshold_separates_fixture():
    """Well-separated fixture class (kappa=200, NN class 60 deg away):
    >=90% same-class acceptance, >=99% other-class rejection under the
    computed taus — the semantic test of the whole FR-5 stack."""
    config = FPCMCConfig()
    world = VMFWorld(
        seed=13, k_known=2, k_novel=0, separation_deg=60.0, kappa_known=200.0
    )
    pool = world.t0_pool(n_per_class=64)
    concepts = [
        _concept_from(
            pool.x[pool.labels == name],
            status="LTM",
            provenance="initial",
            concept_id=f"ltm_{i:03d}",
        )
        for i, name in enumerate(world.known_names)
    ]

    # Full FR-5 stack: FR-5.3 pooled T0 prior, then FR-5.1 per-concept taus.
    prior = compute_global_prior(concepts, config)
    for concept in concepts:
        recompute_thresholds(concept, config, prior)

    target = concepts[0]
    assert np.isfinite(target.tau) and target.tau > 0.0
    assert np.isfinite(target.tau_vmf)

    scorer = make_scorer(config)  # knn_vmf default
    same = world.sample_class("known_00", 500, stream="t4/separates/same")
    other = world.sample_class("known_01", 500, stream="t4/separates/other")

    accept_same = np.mean([scorer.accepts(z, target) for z in same])
    accept_other = np.mean([scorer.accepts(z, target) for z in other])

    assert accept_same >= 0.90, f"same-class acceptance {accept_same:.3f} < 0.90"
    assert 1.0 - accept_other >= 0.99, f"other-class rejection {1 - accept_other:.3f} < 0.99"


# ----------------------------------------------------- promotion recompute (FR-5.4)


def test_promotion_recompute():
    """The promotion hook recomputes both taus from the full ref_set under the
    pure FR-5.1 rule — the value moves away from the shrunk STM tau."""
    config = FPCMCConfig()
    world = VMFWorld(seed=17, k_known=1, k_novel=0, kappa_known=150.0)
    ref = world.sample_class("known_00", 32, stream="t4/promotion/refs")
    concept = _concept_from(ref, status="STM", provenance="seeded")

    # A prior deliberately far from the empirical percentile makes the FR-5.2
    # shrinkage visible: with n=32, w = 32/42, the STM tau is pulled well away
    # from tau_emp.
    prior = GlobalPrior(tau=1.5, tau_vmf=0.0)
    recompute_thresholds(concept, config, prior)
    tau_stm, tau_vmf_stm = concept.tau, concept.tau_vmf

    tau_emp = tau_empirical(loo_knn_scores(concept.ref_set, config.k_ref), config.tau_percentile_q)
    tau_vmf_emp = tau_empirical(loo_vmf_scores(concept), config.tau_percentile_q)
    assert tau_stm != tau_emp, "crafted case must actually shrink the STM tau"
    assert tau_vmf_stm != tau_vmf_emp

    # Dirty the counter to prove the hook also resets it.
    concept.refset_changes_since_tau = 7

    recompute_on_promotion(concept, config)
    assert concept.tau == tau_emp, "promoted tau must be the pure FR-5.1 percentile"
    assert concept.tau_vmf == tau_vmf_emp
    assert concept.tau != tau_stm
    assert concept.tau_vmf != tau_vmf_stm
    assert concept.refset_changes_since_tau == 0
