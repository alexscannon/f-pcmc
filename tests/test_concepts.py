"""T3 [U] tests — Concept dataclass, reservoir ref_sets, centroid dynamics
(TASKS Task 3; PRD FR-1.1–1.4, FR-3.2 seeding shape).

All tests are synthetic/hand-built (D=2 or the vMF fixture world); no real
data; all randomness through fpcmc.rng.make_rng named substreams.

Owner-approved T3 semantics (Q&A 2026-07-10, recorded in docs/CHANGES.md T3):
  - kappa is recomputed via fpcmc.scorers.estimate_kappa on every observation
    that changes ref_set (per-observation cadence).
  - The seeding embedding does NOT count toward match_count/match_windows
    (match_count=0, windows empty at seed); it does count in ref_count_seen
    (it occupies the reservoir) and sets last_matched_at=created_at=step.
  - The Concept carries its own reservoir Generator (per-concept named
    substream) plus the window_W/k_max/alpha_ema scalars, so
    add_observation(z, step) keeps its TASKS-stated signature.
  - provenance gains the value "seeded" for unpromoted STM candidates.
"""

import numpy as np
import pytest
from scipy.stats import chi2 as chi2_dist

from fpcmc.concepts import Concept
from fpcmc.rng import make_rng
from fpcmc.scorers import estimate_kappa

RESERVOIR_SEED = 97
K_MAX = 64
N_ITEMS = 1_000
N_TRIALS = 2_000


def _unit_items(n: int) -> np.ndarray:
    """(n, 2) distinct unit rows — item i is uniquely identifiable by angle."""
    theta = 2.0 * np.pi * np.arange(n) / n
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


def _stm(centroid: np.ndarray, ref_set: np.ndarray, **kw) -> Concept:
    return Concept(
        concept_id=kw.pop("concept_id", "stm_0000"),
        centroid=centroid,
        ref_set=ref_set,
        tau=kw.pop("tau", 0.5),
        kappa=kw.pop("kappa", float("nan")),
        **kw,
    )


# ------------------------------------------------------------ reservoir (FR-1.1)


def _simulate_trial(stream: str, n_items: int, k_max: int) -> np.ndarray:
    """Final slot->item map for one reservoir trial under the FR-1.1 rule.

    Replicates Concept.add_observation's draw discipline bit-exactly: the
    first k_max items fill slots 0..k_max-1 in order with no draws; every
    later item consumes exactly one rng.random(2) pair (u, v) and, iff
    u < k_max / ref_count_seen, replaces slot int(v * k_max). A Generator's
    random((m, 2)) emits the same double sequence as m successive random(2)
    calls, so this vectorized replay is exact — test_reservoir_uniformity
    asserts that equivalence against the real add_observation path on trial 0
    before trusting the statistics.
    """
    rng = make_rng(RESERVOIR_SEED, stream)
    m = n_items - k_max
    draws = rng.random((m, 2))
    n_vals = np.arange(k_max + 1, n_items + 1, dtype=np.float64)
    accepted = np.flatnonzero(draws[:, 0] < k_max / n_vals)
    slots = (draws[:, 1] * k_max).astype(np.int64)
    slot_item = np.arange(k_max, dtype=np.int64)
    for i in accepted:
        slot_item[slots[i]] = k_max + i
    return slot_item


def test_reservoir_uniformity():
    """FR-1.1 replacement rule: every item equally likely to end in the reservoir.

    2,000 trials of streaming 1,000 items into a K_max=64 reservoir. Primary
    assertion is the TASKS-stated chi-square goodness-of-fit criterion
    (p > 0.01) on per-item inclusion counts. The literal "every item within
    3 sigma" reading is statistically self-defeating (with 1,000 items a
    perfectly uniform reservoir leaves ~2.7 outside 3 sigma by chance, so it
    fails ~93% of the time); we bound the count of 3-sigma outliers by a
    generous ceiling on that expectation instead, which still catches any
    systematic per-item bias.
    """
    # Wiring check first: the vectorized replay must equal the real
    # add_observation reservoir bit-for-bit (trial 0, distinct items).
    x = _unit_items(N_ITEMS)
    concept = Concept.seed(
        x[0],
        step=0,
        tau_prior=0.5,
        concept_id="stm_0000",
        rng=make_rng(RESERVOIR_SEED, "t3/reservoir/trial/0000"),
        k_max=K_MAX,
    )
    for i in range(1, N_ITEMS):
        concept.add_observation(x[i], step=i)
    expected_rows = x[_simulate_trial("t3/reservoir/trial/0000", N_ITEMS, K_MAX)]
    assert np.array_equal(concept.ref_set, expected_rows), (
        "vectorized reservoir replay diverged from add_observation — "
        "the uniformity statistics below would not be testing the real rule"
    )

    counts = np.zeros(N_ITEMS)
    for t in range(N_TRIALS):
        counts[_simulate_trial(f"t3/reservoir/trial/{t:04d}", N_ITEMS, K_MAX)] += 1

    p_incl = K_MAX / N_ITEMS
    expected = N_TRIALS * p_incl  # 128 inclusions per item
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    p_value = float(chi2_dist.sf(chi2, df=N_ITEMS - 1))
    assert p_value > 0.01, f"reservoir not uniform: chi2={chi2:.1f}, p={p_value:.4g}"

    sigma = np.sqrt(N_TRIALS * p_incl * (1.0 - p_incl))
    n_outside = int((np.abs(counts - expected) > 3.0 * sigma).sum())
    # E[#outside 3 sigma] ~= 1000 * 0.0027 ~= 2.7 under uniformity; 15 is ~7x
    # that — far below what any real per-item bias produces, never flaky.
    assert n_outside <= 15, f"{n_outside} items outside 3 sigma of {expected}"


def test_reservoir_bound_and_count():
    """ref_set never exceeds K_max; ref_count_seen counts every observation."""
    k_max = 16
    x = _unit_items(500)
    concept = Concept.seed(
        x[0],
        step=0,
        tau_prior=0.5,
        concept_id="stm_0000",
        rng=make_rng(RESERVOIR_SEED, "t3/bound"),
        k_max=k_max,
    )
    assert concept.ref_count_seen == 1
    for i in range(1, 500):
        concept.add_observation(x[i], step=i)
        assert concept.ref_set.shape[0] <= k_max
        assert concept.ref_count_seen == i + 1
    assert concept.ref_set.shape[0] == k_max


# ----------------------------------------------------- centroid dynamics (FR-1.3)


def test_ema_stm_centroid():
    """3-step EMA (alpha=0.1) on 2-D unit vectors matches hand arithmetic.

    c0 = [1, 0]; z1 = [0, 1], z2 = [0, 1], z3 = [-1, 0];
    each step c <- normalize(0.9*c + 0.1*z).
    """
    c0 = np.array([1.0, 0.0])
    z1 = np.array([0.0, 1.0])
    z2 = np.array([0.0, 1.0])
    z3 = np.array([-1.0, 0.0])

    # Hand-computed chain (plain arithmetic, no shared code with fpcmc):
    e1 = np.array([0.9, 0.1]) / np.sqrt(0.9**2 + 0.1**2)
    e2_pre = 0.9 * e1 + 0.1 * z2
    e2 = e2_pre / np.sqrt(e2_pre[0] ** 2 + e2_pre[1] ** 2)
    e3_pre = 0.9 * e2 + 0.1 * z3
    e3 = e3_pre / np.sqrt(e3_pre[0] ** 2 + e3_pre[1] ** 2)

    concept = _stm(centroid=c0.copy(), ref_set=c0[None].copy(), status="STM", alpha_ema=0.10)
    for step, (z, expected) in enumerate(zip([z1, z2, z3], [e1, e2, e3])):
        concept.add_observation(z, step=step)
        np.testing.assert_allclose(concept.centroid, expected, atol=1e-8, rtol=0.0)
        assert abs(float(np.linalg.norm(concept.centroid)) - 1.0) < 1e-12, (
            f"centroid not re-normalized at step {step}"
        )


def test_ltm_centroid_frozen():
    """LTM centroid is bit-identical across 100 observations (FR-1.3)."""
    world_rng = make_rng(RESERVOIR_SEED, "t3/ltm")
    d = 8
    ref = world_rng.standard_normal((K_MAX, d))
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    ref0 = ref.copy()  # the Concept owns (and mutates) `ref` after this
    centroid = ref.mean(axis=0)
    centroid /= np.linalg.norm(centroid)

    concept = _stm(
        centroid=centroid,
        ref_set=ref,
        status="LTM",
        provenance="initial",
        rng=make_rng(RESERVOIR_SEED, "t3/ltm/reservoir"),
        k_max=K_MAX,
    )
    before = concept.centroid.tobytes()
    obs = world_rng.standard_normal((100, d))
    obs /= np.linalg.norm(obs, axis=1, keepdims=True)
    for i, z in enumerate(obs):
        concept.add_observation(z, step=i)
    assert concept.centroid.tobytes() == before, "LTM centroid must be bit-frozen"
    # The observations were really processed: bookkeeping + reservoir moved.
    assert concept.ref_count_seen == K_MAX + 100
    assert concept.match_count == 100
    assert not np.array_equal(concept.ref_set, ref0), (
        "reservoir should still churn on an LTM concept (FR-8.2 relies on it)"
    )


# ---------------------------------------------------------- bookkeeping (FR-1)


def test_match_windows():
    """Steps spanning windows {0,0,2,5} => match_windows {0,2,5}, match_count 4."""
    z = np.array([1.0, 0.0])
    concept = Concept.seed(
        z,
        step=10,
        tau_prior=0.5,
        concept_id="stm_0000",
        rng=make_rng(RESERVOIR_SEED, "t3/windows"),
        window_W=250,
    )
    assert concept.match_count == 0 and concept.match_windows == set()
    for step in (10, 240, 510, 1260):  # windows 0, 0, 2, 5 at W=250
        concept.add_observation(z, step=step)
    assert concept.match_windows == {0, 2, 5}
    assert concept.match_count == 4
    assert concept.last_matched_at == 1260


# ------------------------------------------------------------- seeding (FR-3.2)


def test_seed_singleton():
    """Seeded singleton: ref_set=[z], centroid=z, status STM, tau=tau_prior."""
    z = _unit_items(8)[3]
    concept = Concept.seed(
        z,
        step=42,
        tau_prior=0.37,
        tau_vmf_prior=-2400.0,
        concept_id="stm_0007",
        rng=make_rng(RESERVOIR_SEED, "t3/seed"),
    )
    assert concept.concept_id == "stm_0007"
    assert concept.ref_set.shape == (1, 2)
    np.testing.assert_array_equal(concept.ref_set[0], z)
    np.testing.assert_array_equal(concept.centroid, z)
    assert concept.status == "STM"
    assert concept.tau == 0.37
    assert concept.tau_vmf == -2400.0
    assert concept.provenance == "seeded"
    assert concept.created_at == 42
    assert concept.last_matched_at == 42
    assert concept.ref_count_seen == 1
    assert concept.match_count == 0
    assert concept.match_windows == set()
    assert concept.gt_majority_label is None
    assert concept.merged_from == []
    # kappa cache is valid from birth (per-observation cadence).
    assert np.isfinite(concept.kappa)
    assert concept.kappa == estimate_kappa(z[None, :])
    # The seed embedding is owned, not aliased: mutating the caller's copy
    # must not reach into the concept.
    z[0] = 99.0
    assert concept.ref_set[0, 0] != 99.0 and concept.centroid[0] != 99.0

    # The pair of priors is optional in the knn_ref-only configuration:
    solo = Concept.seed(
        _unit_items(8)[1],
        step=0,
        tau_prior=0.5,
        concept_id="stm_0008",
        rng=make_rng(RESERVOIR_SEED, "t3/seed2"),
    )
    assert np.isnan(solo.tau_vmf)


# --------------------------------------------------------- immutability (FR-1.4)


def test_concept_id_immutable():
    """Mutating (or deleting) concept_id raises; other fields stay mutable."""
    z = np.array([1.0, 0.0])
    concept = _stm(centroid=z, ref_set=z[None])
    with pytest.raises(AttributeError, match="concept_id"):
        concept.concept_id = "stm_9999"
    with pytest.raises(AttributeError, match="concept_id"):
        del concept.concept_id
    assert concept.concept_id == "stm_0000"
    # status must remain assignable — T8 promotion flips it.
    concept.status = "LTM"
    assert concept.status == "LTM"
