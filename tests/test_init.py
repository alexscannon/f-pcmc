"""T6 tests — LTM initialization + M1 sanity gate (TASKS Task 6; PRD FR-2,
PRD §9 M1).

[U] tests run on the synthetic vMF fixture world; the two [I] tests run
against the real embedding pools resolved via roots.env (skipped with a clear
message when unavailable) and are marked slow.

Owner-approved T6 decisions (Q&A 2026-07-11, recorded in docs/CHANGES.md T6):
  - Init ref_set sampling is the exact FR-1.1 reservoir replay over all class
    embeddings (per-concept substream "reservoir/ltm_{i:03d}"), applied
    directly — never via add_observation, which would corrupt the post-seed
    match bookkeeping. ref_count_seen ends at the full class count, so
    stream-time replacement probabilities continue correctly.
  - Concept ids ltm_{i:03d} follow np.unique(labels) order (sorted class
    names for real data and the fixture world alike).
  - The M1 IND side is the pinned run's evaluated population: ind_test
    (10,000) + synthetic_ind (250) = 10,250, vs near+far OOD (3,076);
    stratified AUROCs reuse the same IND side vs each OOD pool alone.
  - The M1 read-only sweep batches across QUERIES per concept (one
    `ref_set @ X.T` GEMM per concept — never a stacked cross-concept GEMV),
    lives test-locally here, and is spot-checked against the frozen
    per-query scorer. The novelty statistic is the min-over-concepts
    composed knn_vmf scalar, which by the frozen T2 decision is the knn_ref
    sub-score.
"""

import math
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from fpcmc.config import FPCMCConfig
from fpcmc.data import embeddings_available, load_all_pools
from fpcmc.init import initialize_ltm
from fpcmc.rng import make_rng
from fpcmc.scorers import make_scorer
from tests.fixtures.vmf_world import VMFWorld

AVAILABLE, REASON = embeddings_available()

REFERENCE = yaml.safe_load(
    (Path(__file__).resolve().parent / "reference_numbers.yaml").read_text()
)["t6_m1_gate"]


# ------------------------------------------------------------------- helpers


# Fixture world for the [U] tests: 8 known classes (TASKS T6), default
# kappa=150 / n=200 per class puts the sample-mean direction well inside the
# 3-degree centroid assertion (expected error ~1.8 degrees at D=32).
_WORLD = VMFWorld(seed=61, k_known=8, k_novel=3)
_N_PER_CLASS = 200


def _fixture_store(config: FPCMCConfig):
    pool = _WORLD.t0_pool(n_per_class=_N_PER_CLASS)
    return initialize_ltm(pool.x, pool.labels, config)


def _assert_stores_identical(s1, s2) -> None:
    """Bitwise store equality (T5 pattern), including reservoir contents."""
    c1 = {c.concept_id: c for c in s1.concepts}
    c2 = {c.concept_id: c for c in s2.concepts}
    assert list(c1) == list(c2)
    for cid, a in c1.items():
        b = c2[cid]
        assert a.status == b.status
        assert a.provenance == b.provenance
        assert a.match_count == b.match_count
        assert a.ref_count_seen == b.ref_count_seen
        assert a.match_windows == b.match_windows
        assert (a.created_at, a.last_matched_at) == (b.created_at, b.last_matched_at)
        assert a.refset_changes_since_tau == b.refset_changes_since_tau
        assert a.tau == b.tau
        assert (math.isnan(a.tau_vmf) and math.isnan(b.tau_vmf)) or a.tau_vmf == b.tau_vmf
        assert a.kappa == b.kappa
        assert np.array_equal(a.centroid, b.centroid)
        assert np.array_equal(a.ref_set, b.ref_set)


def _min_composed_scores(concepts, x: np.ndarray, k_ref: int) -> np.ndarray:
    """(N,) min-over-concepts composed knn_vmf scalar, computed READ-ONLY.

    The composed scalar is the knn_ref sub-score (frozen T2 decision), so per
    concept this is the mean cosine distance to the min(k_ref, K) nearest
    ref_set members. Owner-approved T6 mechanics: batch across queries with
    one per-concept `ref_set @ x.T` GEMM (never a stacked cross-concept
    GEMV); nothing here mutates a concept — the M1 sweep must not go through
    ConceptStore.route(), which seeds/updates (T5 as-built note).
    """
    x = np.asarray(x, dtype=np.float64)
    best = np.full(x.shape[0], np.inf)
    for c in concepts:
        dists = 1.0 - c.ref_set @ x.T  # (K, N)
        k = min(int(k_ref), dists.shape[0])
        scores = np.partition(dists, k - 1, axis=0)[:k].mean(axis=0)
        np.minimum(best, scores, out=best)
    return best


def _auroc(ind_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """AUROC of the novelty statistic (higher = more novel; OOD = positive)."""
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.zeros(len(ind_scores)), np.ones(len(ood_scores))])
    return float(roc_auc_score(y, np.concatenate([ind_scores, ood_scores])))


# ------------------------------------------------------------- [U] fixture init


def test_init_fixture():
    """TASKS T6: 8-class fixture world -> 8 LTM concepts, centroids within
    3 degrees of the true class means, finite positive tau, provenance
    'initial' — plus the FR-2.1 structural contract."""
    config = FPCMCConfig()
    store = _fixture_store(config)

    assert len(store) == 8
    assert [c.concept_id for c in store.concepts] == [f"ltm_{i:03d}" for i in range(8)]

    # np.unique(labels) ordering decision: sorted class names.
    expected_names = sorted(_WORLD.known_names)
    for concept, name in zip(store.concepts, expected_names):
        angle = math.degrees(
            math.acos(min(1.0, max(-1.0, float(concept.centroid @ _WORLD.true_mean(name)))))
        )
        assert angle <= 3.0, f"{concept.concept_id} ({name}): centroid {angle:.2f} deg off"

        assert concept.status == "LTM"
        assert concept.provenance == "initial"
        assert np.isfinite(concept.tau) and concept.tau > 0.0
        assert np.isfinite(concept.tau_vmf)  # D=32, K=64 >= n_vmf_min: computed
        assert np.isfinite(concept.kappa) and concept.kappa > 0.0
        assert abs(float(np.linalg.norm(concept.centroid)) - 1.0) < 1e-12

        # FR-2.1 reservoir contract: K_max members drawn from the class pool,
        # ref_count_seen = the full class count (replay semantics).
        assert concept.ref_set.shape == (config.K_max_refset, _WORLD.d)
        assert concept.ref_count_seen == _N_PER_CLASS
        assert concept.match_count == 0 and concept.match_windows == set()
        assert concept.created_at == 0 and concept.last_matched_at == 0
        assert concept.refset_changes_since_tau == 0

    # The store's id allocator advanced past the registered ids.
    assert store.new_concept_id("ltm") == "ltm_008"
    assert store.new_concept_id("stm") == "stm_0000"


def test_init_fixture_refset_members_are_class_samples():
    """Every reservoir member is literally a row of its class's init pool
    (the replay never mixes classes or fabricates rows)."""
    config = FPCMCConfig()
    pool = _WORLD.t0_pool(n_per_class=_N_PER_CLASS)
    store = initialize_ltm(pool.x, pool.labels, config)

    for concept, name in zip(store.concepts, sorted(_WORLD.known_names)):
        class_rows = pool.x[pool.labels == name]
        for row in concept.ref_set:
            assert (class_rows == row).all(axis=1).any(), (
                f"{concept.concept_id}: reservoir row is not a {name} pool row"
            )


# ---------------------------------------------------------- [U] determinism


def test_init_determinism():
    """TASKS T6: two runs from the same config produce identical stores,
    including reservoir contents (all randomness via named substreams)."""
    config = FPCMCConfig()
    _assert_stores_identical(_fixture_store(config), _fixture_store(config))

    # Different seed => different reservoir draws (guards against a fixed
    # internal seed masquerading as determinism). Centroids/means are
    # seed-independent here (same pool), but reservoir contents must differ.
    other = _fixture_store(FPCMCConfig(seed=43))
    base = _fixture_store(config)
    assert any(
        not np.array_equal(a.ref_set, b.ref_set)
        for a, b in zip(base.concepts, other.concepts)
    )


# ----------------------------------------------------------- [I] M1 gate


@pytest.fixture(scope="module")
def real_pools():
    if not AVAILABLE:
        pytest.skip(REASON)
    return load_all_pools()


@pytest.mark.slow
def test_m1_gate(real_pools):
    """TASKS T6 / PRD §9 M1 gate: LTM-only routing must reproduce the pinned
    batch knn_vmf detection AUROC within ±0.01 (stratified ±0.015).

    A red gate means the implementation is wrong — do not proceed to T7.
    """
    config = FPCMCConfig()  # seed 42, PRD §8 defaults
    ref = real_pools["ind_reference"]
    store = initialize_ltm(ref.x, ref.subclass_names, config)
    assert len(store) == 100
    concepts = store.concepts

    # Read-only sweep (never route()): min-over-concepts composed scalar.
    ind_scores = np.concatenate(
        [
            _min_composed_scores(concepts, real_pools["ind_test"].x, config.k_ref),
            _min_composed_scores(concepts, real_pools["synthetic_ind"].x, config.k_ref),
        ]
    )
    near_scores = _min_composed_scores(concepts, real_pools["near_ood"].x, config.k_ref)
    far_scores = _min_composed_scores(concepts, real_pools["far_ood"].x, config.k_ref)
    assert ind_scores.shape == (10_250,)  # owner decision: the pin's population
    assert near_scores.shape == (500,) and far_scores.shape == (2_576,)

    # Spot-check the batched sweep against the frozen per-query scorer.
    scorer = make_scorer(config)
    x_ind = np.asarray(real_pools["ind_test"].x, dtype=np.float64)
    idx = make_rng(config.seed, "t6/m1/spotcheck").choice(x_ind.shape[0], size=50, replace=False)
    for j in idx:
        per_query = min(scorer.score(x_ind[j], c) for c in concepts)
        assert abs(per_query - ind_scores[j]) <= 1e-12

    # The sweep left the store untouched (read-only contract).
    assert len(store) == 100
    assert all(c.match_count == 0 and c.ref_count_seen == 500 for c in concepts)

    pins, tol = REFERENCE["metrics"], REFERENCE["tolerance"]
    auroc_all = _auroc(ind_scores, np.concatenate([near_scores, far_scores]))
    auroc_near = _auroc(ind_scores, near_scores)
    auroc_far = _auroc(ind_scores, far_scores)

    print(
        f"\nM1 gate: auroc_all={auroc_all:.6f} (pin {pins['auroc_all_ood']:.6f}), "
        f"near={auroc_near:.6f} (pin {pins['auroc_near_ood']:.6f}), "
        f"far={auroc_far:.6f} (pin {pins['auroc_far_ood']:.6f})"
    )
    assert abs(auroc_all - pins["auroc_all_ood"]) <= tol["all_ood"]
    assert abs(auroc_near - pins["auroc_near_ood"]) <= tol["near_ood"]
    assert abs(auroc_far - pins["auroc_far_ood"]) <= tol["far_ood"]


@pytest.mark.slow
def test_init_runtime(real_pools):
    """TASKS T6 / NFR-1: LTM initialization from 50k x 1024 completes < 60 s
    (data loading excluded — the pool arrives via the module fixture)."""
    config = FPCMCConfig()
    ref = real_pools["ind_reference"]

    start = time.perf_counter()
    store = initialize_ltm(ref.x, ref.subclass_names, config)
    elapsed = time.perf_counter() - start

    assert len(store) == 100
    print(f"\ninitialize_ltm(50k x 1024): {elapsed:.2f} s")
    assert elapsed < 60.0, f"LTM init took {elapsed:.2f} s (NFR-1 budget: 60 s)"
