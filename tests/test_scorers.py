"""T2 [U] tests — per-concept scorers (TASKS Task 2; PRD FR-4.1–4.3).

All tests run on the synthetic vMF fixture world (D=32) or hand-built
concepts; no real data, no RNG outside fpcmc.rng.make_rng substreams.
"""

import numpy as np
import pytest

from fpcmc.concepts import Concept
from fpcmc.config import FPCMCConfig
from fpcmc.rng import make_rng
from fpcmc.scorers import (
    KnnRefScorer,
    KnnVmfScorer,
    VmfScorer,
    estimate_kappa,
    make_scorer,
)
from tests.fixtures.vmf_world import VMFWorld, sample_vmf

D = 32


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _concept(
    cid: str,
    ref_set: np.ndarray,
    tau: float,
    tau_vmf: float = float("nan"),
    kappa: float = float("nan"),
) -> Concept:
    return Concept(
        concept_id=cid,
        centroid=_unit(ref_set.mean(axis=0)),
        ref_set=ref_set,
        tau=tau,
        kappa=kappa,
        tau_vmf=tau_vmf,
    )


def _scorers(k_ref: int = 5, n_vmf_min: int = 10):
    knn = KnnRefScorer(k_ref=k_ref)
    vmf = VmfScorer(n_vmf_min=n_vmf_min, fallback=knn)
    return knn, vmf, KnnVmfScorer(knn=knn, vmf=vmf)


# ---------------------------------------------------------------- kappa (FR-4.2)


@pytest.mark.parametrize("kappa_true", [20.0, 100.0, 500.0])
def test_kappa_recovery(kappa_true):
    """Banerjee estimate within 15% relative error, n=500, D=32, 5 seeds."""
    for seed in range(5):
        mu = _unit(make_rng(seed, "t2/kappa/mu").standard_normal(D))
        x = sample_vmf(mu, kappa_true, 500, make_rng(seed, f"t2/kappa/sample/{kappa_true}"))
        kappa_hat = estimate_kappa(x)
        rel_err = abs(kappa_hat - kappa_true) / kappa_true
        assert rel_err <= 0.15, (
            f"kappa={kappa_true} seed={seed}: estimate {kappa_hat:.1f} off by {rel_err:.1%}"
        )


def test_kappa_monotone():
    """Tighter fixture class => larger kappa estimate."""
    world = VMFWorld(seed=13, k_known=2, k_novel=0, kappa_known=[60.0, 300.0])
    k_loose = estimate_kappa(world.sample_class("known_00", 300, stream="t2/mono"))
    k_tight = estimate_kappa(world.sample_class("known_01", 300, stream="t2/mono"))
    assert k_tight > k_loose


# ------------------------------------------------------------- knn_ref (FR-4.1)


def test_knn_ref_monotonicity():
    """Score strictly increases along a geodesic away from the class mean."""
    world = VMFWorld(seed=17, k_known=1, k_novel=0)
    ref = world.sample_class("known_00", 100, stream="t2/knn_mono")
    concept = _concept("ltm_000", ref, tau=1.0)
    knn = KnnRefScorer(k_ref=5)

    mu = world.true_mean("known_00")
    v = make_rng(17, "t2/knn_mono/tangent").standard_normal(D)
    v = _unit(v - (v @ mu) * mu)  # unit tangent, exactly orthogonal to mu
    scores = []
    for theta_deg in (0.0, 15.0, 30.0, 45.0, 60.0):
        theta = np.radians(theta_deg)
        z = np.cos(theta) * mu + np.sin(theta) * v
        scores.append(knn.score(z, concept))
    assert all(b > a for a, b in zip(scores, scores[1:])), scores


def test_knn_ref_small_refset():
    """ref_set sizes 1..4 clip k_ref, raise nothing, and score sensibly."""
    world = VMFWorld(seed=19, k_known=1, k_novel=0)
    knn = KnnRefScorer(k_ref=5)
    z_near = world.sample_class("known_00", 1, stream="t2/small/probe")[0]
    z_far = -world.true_mean("known_00")  # antipode of the class mean

    for n in range(1, 5):
        ref = world.sample_class("known_00", n, stream="t2/small/ref")
        concept = _concept(f"stm_{n:04d}", ref, tau=1.0)
        s_near = knn.score(z_near, concept)
        s_far = knn.score(z_far, concept)
        assert np.isfinite(s_near) and s_near >= 0.0
        # k_ref clipped to n: the score is the mean over ALL n distances.
        expected = float(np.mean(1.0 - ref @ z_near))
        assert s_near == pytest.approx(expected, abs=1e-12)
        assert s_far > s_near

    # Size 1 degenerates to plain cosine distance to the single reference.
    ref1 = world.sample_class("known_00", 1, stream="t2/small/ref")
    c1 = _concept("stm_0001", ref1, tau=1.0)
    assert knn.score(z_near, c1) == pytest.approx(float(1.0 - ref1[0] @ z_near), abs=1e-12)


# ----------------------------------------------------------------- vmf (FR-4.2)


def test_vmf_fallback(mocker):
    """ref_set < n_vmf_min => VmfScorer delegates to knn_ref and flags it."""
    world = VMFWorld(seed=23, k_known=1, k_novel=0)
    knn, vmf, _ = _scorers(n_vmf_min=10)
    z = world.sample_class("known_00", 1, stream="t2/fallback/probe")[0]
    spy = mocker.spy(knn, "score_detail")

    # 9 refs (< n_vmf_min): fallback path. kappa/tau_vmf deliberately NaN —
    # the fallback must never touch them, and acceptance runs against `tau`.
    ref9 = world.sample_class("known_00", 9, stream="t2/fallback/ref")
    c9 = _concept("stm_0009", ref9, tau=1.0)
    detail = vmf.score_detail(z, c9)
    assert detail.fallback is True
    assert spy.call_count == 1
    assert detail.score == knn.score(z, c9)
    assert detail.accepted == (detail.score <= c9.tau)
    assert vmf.score(z, c9) == detail.score

    # 10 refs (== n_vmf_min): native vMF path, no delegation, no flag.
    ref10 = world.sample_class("known_00", 10, stream="t2/fallback/ref")
    c10 = _concept("stm_0010", ref10, tau=1.0, tau_vmf=0.0, kappa=estimate_kappa(ref10))
    spy.reset_mock()
    detail10 = vmf.score_detail(z, c10)
    assert detail10.fallback is False
    assert spy.call_count == 0
    assert np.isfinite(detail10.score)
    # Negative log-likelihood: -(log C_D(kappa) + kappa * <mu, z>).
    assert detail10.score != knn.score(z, c10)


def test_vmf_survives_high_dimension():
    """The A&S 9.7.7 log-Bessel path stays finite at D=1024 (T6's regime).

    Not in the TASKS T2 list, but cheap and load-bearing for the M1 gate:
    scipy.special.ive underflows to 0 here, so a naive log I_v would be -inf.
    """
    rng = make_rng(29, "t2/highdim")
    mu = _unit(rng.standard_normal(1024))
    ref = sample_vmf(mu, 800.0, 64, rng)
    kappa = estimate_kappa(ref)
    concept = _concept("ltm_000", ref, tau=1.0, tau_vmf=0.0, kappa=kappa)
    _, vmf, _ = _scorers()
    s = vmf.score(mu, concept)
    assert np.isfinite(kappa) and np.isfinite(s)


# ----------------------------------------------------------- composed (FR-4.3)


def _raw_sub_scores(z, ref):
    """(s_knn, s_vmf) for a probe against a throwaway concept over `ref`."""
    knn, vmf, _ = _scorers()
    c = _concept("tmp", ref, tau=0.0, tau_vmf=0.0, kappa=estimate_kappa(ref))
    return knn.score(z, c), vmf.score(z, c)


def test_composed_or_logic():
    """Composed accepts iff at least one sub-scorer accepts (OR, FR-4.3)."""
    world = VMFWorld(seed=31, k_known=1, k_novel=0)
    ref = world.sample_class("known_00", 40, stream="t2/or/ref")
    z = world.sample_class("known_00", 1, stream="t2/or/probe")[0]
    s_k, s_v = _raw_sub_scores(z, ref)
    kappa = estimate_kappa(ref)
    knn, vmf, composed = _scorers()

    # Only knn accepts (tau above s_k; tau_vmf below s_v).
    c_knn_only = _concept("a", ref, tau=s_k + 0.05, tau_vmf=s_v - 10.0, kappa=kappa)
    assert knn.accepts(z, c_knn_only) and not vmf.accepts(z, c_knn_only)
    assert composed.accepts(z, c_knn_only)

    # Only vmf accepts.
    c_vmf_only = _concept("b", ref, tau=s_k * 0.5, tau_vmf=s_v + 10.0, kappa=kappa)
    assert vmf.accepts(z, c_vmf_only) and not knn.accepts(z, c_vmf_only)
    assert composed.accepts(z, c_vmf_only)

    # Neither accepts.
    c_neither = _concept("c", ref, tau=s_k * 0.5, tau_vmf=s_v - 10.0, kappa=kappa)
    assert not knn.accepts(z, c_neither) and not vmf.accepts(z, c_neither)
    assert not composed.accepts(z, c_neither)

    # The composed scalar is the knn_ref sub-score (owner-approved 2026-07-10).
    assert composed.score(z, c_knn_only) == s_k


def test_composed_assignment_margin():
    """Best normalized margin wins; exact ties break lexicographically."""
    world = VMFWorld(seed=37, k_known=1, k_novel=0)
    ref = world.sample_class("known_00", 40, stream="t2/margin/ref")
    z = world.sample_class("known_00", 1, stream="t2/margin/probe")[0]
    s_k, s_v = _raw_sub_scores(z, ref)
    kappa = estimate_kappa(ref)
    _, _, composed = _scorers()

    # Both accept via knn; wider tau => larger normalized margin (tau - s)/|tau|.
    c_narrow = _concept("stm_0001", ref, tau=s_k + 0.02, tau_vmf=s_v - 10.0, kappa=kappa)
    c_wide = _concept("stm_0002", ref, tau=s_k + 0.50, tau_vmf=s_v - 10.0, kappa=kappa)
    assert composed.margin(z, c_wide) > composed.margin(z, c_narrow) > 0.0
    selected = composed.select(z, [c_narrow, c_wide])
    assert selected is not None and selected.concept.concept_id == "stm_0002"

    # Exact tie (identical geometry + thresholds, distinct ids): the
    # lexicographically smaller concept_id wins, regardless of input order.
    c_z = _concept("stm_z", ref, tau=s_k + 0.10, tau_vmf=s_v - 10.0, kappa=kappa)
    c_a = _concept("stm_a", ref, tau=s_k + 0.10, tau_vmf=s_v - 10.0, kappa=kappa)
    assert composed.margin(z, c_z) == composed.margin(z, c_a)  # tie is exact
    assert composed.select(z, [c_z, c_a]).concept.concept_id == "stm_a"
    assert composed.select(z, [c_a, c_z]).concept.concept_id == "stm_a"

    # No accepting concept => no assignment.
    c_reject = _concept("stm_r", ref, tau=s_k * 0.5, tau_vmf=s_v - 10.0, kappa=kappa)
    assert composed.select(z, [c_reject]) is None


def test_scorer_determinism():
    """Identical inputs across two fresh scorer instances => identical floats."""
    world = VMFWorld(seed=41, k_known=2, k_novel=0)
    ref = world.sample_class("known_00", 50, stream="t2/det/ref")
    kappa = estimate_kappa(ref)
    concept = _concept("ltm_000", ref, tau=0.4, tau_vmf=0.0, kappa=kappa)
    queries = world.sample_class("known_01", 50, stream="t2/det/probes")

    config = FPCMCConfig()
    for name in ("knn_ref", "vmf", "knn_vmf"):
        cfg = FPCMCConfig(scorer=name, seed=config.seed)
        a, b = make_scorer(cfg), make_scorer(cfg)  # two fresh instances
        sa = [a.score(z, concept) for z in queries]
        sb = [b.score(z, concept) for z in queries]
        assert sa == sb, f"{name}: scores differ across fresh instances"
        da = [a.score_detail(z, concept) for z in queries]
        db = [b.score_detail(z, concept) for z in queries]
        assert da == db, f"{name}: details differ across fresh instances"
