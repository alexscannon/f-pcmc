"""T9 [U] tests — duplicate-cluster merging (TASKS Task 9; PRD FR-8.1–8.3).

Owner-approved T9 semantics (Q&A 2026-07-11, recorded in docs/CHANGES.md T9):
  - FR-8.1 condition 2: k = min(k_ref, available) throughout; within = pooled
    mean of both ref_sets' LOO knn scores (frozen loo_knn_scores discipline);
    cross = pooled mean over both directions; merge iff cross/within <= 1.1
    (hardcoded FR-8.1 literal) AND centroid sim >= merge_sim.
  - Sweep = PRD order (STM<->STM, then STM->LTM folds, then LTM<->LTM
    promoted pairs), each phase iterated to a deterministic fixpoint over
    ascending-id pairs, merges effective immediately; on_promotion runs the
    LTM<->LTM phase only.
  - FR-8.1 merges union the bookkeeping (match_count sum, match_windows
    union, last_matched_at max, ref_count_seen sum); STM survivor centroid =
    normalized mean of the FULL union (pre-subsample); FR-8.2 folds move
    ref_set + ref_count_seen only (a fold is not a match event).
  - Union > K_max is uniformly subsampled without replacement via the
    dedicated substream make_rng(seed, f"merge/{step}/{survivor}<-{absorbed}")
    — the survivor's own reservoir Generator is never touched.
  - Pairs with K < 2 on either side are not sweep-mergeable (the within
    structure is unobservable); T10's HDBSCAN path calls merge_pair directly.

Geometry: hand-crafted unit cones in D=8 where the two-condition rule needs
exact control (30-degree twins for the bimodal case, 60 degrees for the
centroid-sim case), the T8-style orthogonal vMF world for the fold test, and
the golden world for the promoted-promoted fragmentation case (TASKS literal).
"""

import numpy as np
import pytest

from fpcmc.concepts import Concept, ConceptStore
from fpcmc.config import FPCMCConfig
from fpcmc.init import initialize_ltm
from fpcmc.memory import MergeRecord, MergeSweeper
from fpcmc.rng import make_rng
from fpcmc.scorers import estimate_kappa
from fpcmc.thresholds import (
    GlobalPrior,
    compute_global_prior,
    loo_knn_scores,
    recompute_on_promotion,
    tau_empirical,
)
from tests.fixtures.golden_stream import make_golden_world

SEED = 901
D = 8


# ------------------------------------------------------------------- helpers


def _direction(angle_deg: float) -> np.ndarray:
    """Unit vector in the (e0, e1) plane at `angle_deg` from e0."""
    a = np.radians(angle_deg)
    d = np.zeros(D)
    d[0], d[1] = np.cos(a), np.sin(a)
    return d


def _cone(direction: np.ndarray, n: int, stream: str, spread: float) -> np.ndarray:
    """(n, D) unit rows in a deterministic cone around `direction`."""
    rng = make_rng(SEED, f"t9/cone/{stream}")
    x = direction[None, :] + spread * rng.standard_normal((n, D))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _stm(
    ref: np.ndarray,
    concept_id: str,
    *,
    match_count: int,
    windows: set[int] = frozenset(),
    merged_from: list[str] | None = None,
) -> Concept:
    centroid = ref.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    return Concept(
        concept_id=concept_id,
        centroid=centroid,
        ref_set=np.array(ref, dtype=np.float64),
        tau=0.30,
        kappa=estimate_kappa(ref),
        tau_vmf=0.0,
        status="STM",
        provenance="seeded",
        match_count=match_count,
        match_windows=set(windows),
        merged_from=list(merged_from or []),
        rng=make_rng(SEED, f"t9/reservoir/{concept_id}"),
    )


def _prior() -> GlobalPrior:
    return GlobalPrior(tau=0.30, tau_vmf=0.0)


def _bare_store(config: FPCMCConfig, concepts) -> ConceptStore:
    return ConceptStore(config, _prior(), concepts)


# -------------------------------------------------- two-condition rule (FR-8.1)


def test_merge_two_conditions():
    """(a) both conditions hold => merge; (b) similar centroids but bimodal
    ref_sets (cross/within > 1.1) => no merge (the PRD §11 near-OOD-collapse
    guard); (c) tight ref_sets but centroid sim < merge_sim => no merge."""
    config = FPCMCConfig()
    sweeper = MergeSweeper(config, _prior())

    # (a) Two halves of one moderate cone: sim ~1, cross ~ within.
    a1 = _stm(_cone(_direction(0), 20, "two_cond/a1", spread=0.12), "stm_0000", match_count=12)
    a2 = _stm(_cone(_direction(0), 20, "two_cond/a2", spread=0.12), "stm_0001", match_count=7)
    check = sweeper.check_pair(a1, a2)
    assert check.centroid_sim >= config.merge_sim, "premise: condition 1 must hold"
    assert check.cross_within_ratio <= 1.1, "premise: condition 2 must hold"
    assert check.compatible
    store = _bare_store(config, [a1, a2])
    sweeper.sweep(store, step=500)
    assert "stm_0001" not in store, "case (a): both conditions hold, must merge"
    assert "stm_0000" in store
    assert [(r.kind, r.survivor_id, r.absorbed_id) for r in sweeper.merge_log] == [
        ("stm_stm", "stm_0000", "stm_0001")
    ]

    # (b) Two TIGHT cones 30 degrees apart: centroid sim ~0.87 >= 0.80 but the
    # ref_sets are bimodal — cross/within explodes past 1.1. No merge.
    b1 = _stm(_cone(_direction(0), 20, "two_cond/b1", spread=0.02), "stm_0000", match_count=12)
    b2 = _stm(_cone(_direction(30), 20, "two_cond/b2", spread=0.02), "stm_0001", match_count=7)
    sweeper2 = MergeSweeper(config, _prior())
    check_b = sweeper2.check_pair(b1, b2)
    assert check_b.centroid_sim >= config.merge_sim, "premise: condition 1 must hold"
    assert check_b.cross_within_ratio > 1.1, "premise: condition 2 must fail"
    assert not check_b.compatible
    store_b = _bare_store(config, [b1, b2])
    sweeper2.sweep(store_b, step=500)
    assert "stm_0000" in store_b and "stm_0001" in store_b, "case (b): bimodal, must not merge"
    assert sweeper2.merge_log == []

    # (c) Two tight cones 60 degrees apart: sim ~0.5 < merge_sim. No merge.
    c1 = _stm(_cone(_direction(0), 20, "two_cond/c1", spread=0.02), "stm_0000", match_count=12)
    c2 = _stm(_cone(_direction(60), 20, "two_cond/c2", spread=0.02), "stm_0001", match_count=7)
    sweeper3 = MergeSweeper(config, _prior())
    check_c = sweeper3.check_pair(c1, c2)
    assert check_c.centroid_sim < config.merge_sim, "premise: condition 1 must fail"
    assert not check_c.compatible
    store_c = _bare_store(config, [c1, c2])
    sweeper3.sweep(store_c, step=500)
    assert "stm_0000" in store_c and "stm_0001" in store_c, "case (c): dissimilar centroids"
    assert sweeper3.merge_log == []


# ------------------------------------------------- survivor + lineage (FR-1.4)


def test_merge_survivor_and_lineage():
    """Survivor is the larger match_count; lineage {survivor: [absorbed]}
    accumulates across transitive merges (A<-B then A<-C, in one fixpoint
    sweep) and inherits the absorbed concept's own lineage; absorbed ids
    never reappear in routing and stay burned."""
    config = FPCMCConfig()
    ref_a = _cone(_direction(0), 20, "lineage/a", spread=0.12)
    ref_b = _cone(_direction(0), 20, "lineage/b", spread=0.12)
    ref_c = _cone(_direction(0), 20, "lineage/c", spread=0.12)
    a = _stm(ref_a, "stm_0000", match_count=50, windows={0, 1})
    b = _stm(ref_b, "stm_0001", match_count=30, windows={1, 2}, merged_from=["stm_0099"])
    c = _stm(ref_c, "stm_0002", match_count=10, windows={3})
    a.last_matched_at, b.last_matched_at, c.last_matched_at = 400, 450, 420
    store = _bare_store(config, [a, b, c])
    sweeper = MergeSweeper(config, _prior())
    sweeper.sweep(store, step=500)

    # A absorbed B (larger mc), then the fixpoint pass absorbed C too.
    assert "stm_0000" in store
    assert "stm_0001" not in store and "stm_0002" not in store
    assert a.match_count == 90, "match_count is summed (owner decision 18)"
    assert a.match_windows == {0, 1, 2, 3}, "match_windows are unioned"
    assert a.last_matched_at == 450, "last_matched_at is the max"
    assert a.ref_count_seen == 60, "ref_count_seen is summed"
    # Lineage accumulates transitively and inherits B's own absorbed ids.
    assert a.merged_from == ["stm_0001", "stm_0099", "stm_0002"]
    assert sweeper.lineage == {"stm_0000": ["stm_0001", "stm_0002"]}
    assert [r.kind for r in sweeper.merge_log] == ["stm_stm", "stm_stm"]

    # Absorbed ids never reappear in routing: queries land on the survivor...
    r = store.route(_cone(_direction(0), 1, "lineage/query", spread=0.02)[0], step=600)
    assert r.concept_id == "stm_0000"
    # ...and the ids stay burned (invariant 4).
    with pytest.raises(ValueError, match="stm_0001"):
        store.register(_stm(ref_b, "stm_0001", match_count=1))


# ------------------------------------------------------- STM->LTM fold (FR-8.2)


def _ltm_world(config: FPCMCConfig):
    from tests.fixtures.vmf_world import VMFWorld

    world = VMFWorld(seed=SEED, k_known=4, k_novel=3, separation_deg=90.0)
    pool = world.t0_pool(n_per_class=100)
    store = initialize_ltm(pool.x, pool.labels, config)
    prior = compute_global_prior(store.ltm, config)  # equals the store's (untouched T0)
    return world, store, prior


def test_stm_ltm_fold():
    """An STM candidate whose centroid is accepted by an LTM concept's taus is
    folded: ref_set unioned into the LTM reservoir (re-bounded to K_max),
    ref_count_seen increased, LTM centroid bit-identical, match bookkeeping
    untouched, candidate deleted; taus/kappa recomputed for the survivor."""
    config = FPCMCConfig()
    world, store, prior = _ltm_world(config)

    cand_ref = world.sample_class("known_00", 40, stream="t9/fold/cand")
    cand = _stm(cand_ref, "stm_0000", match_count=8, windows={1})
    store.register(cand)

    ltm = store.get("ltm_000")  # known_00 (np.unique order)
    centroid_before = ltm.centroid.tobytes()
    refs_before = ltm.ref_set.copy()
    stats_before = (ltm.match_count, set(ltm.match_windows), ltm.last_matched_at)
    assert ltm.ref_count_seen == 100

    sweeper = MergeSweeper(config, prior)
    sweeper.sweep(store, step=700)

    assert "stm_0000" not in store, "the folded candidate is deleted"
    assert ltm.centroid.tobytes() == centroid_before, "LTM centroid stays frozen (FR-8.2)"
    assert ltm.ref_count_seen == 140, "ref_count_seen increased by the candidate's count"
    assert (ltm.match_count, ltm.match_windows, ltm.last_matched_at) == stats_before, (
        "a fold is not a match event (owner decision 18)"
    )
    # Union (64 + 40) re-bounded to K_max; every row comes from the union.
    assert ltm.ref_set.shape == (config.K_max_refset, world.d)
    union_rows = {row.tobytes() for row in np.vstack([refs_before, cand.ref_set])}
    assert all(row.tobytes() in union_rows for row in ltm.ref_set)
    # kappa recomputed at the merge site (T4 rule), tau = pure FR-5.1 on the
    # new ref_set (LTM branch of the status-sensitive recompute).
    assert ltm.kappa == estimate_kappa(ltm.ref_set)
    assert ltm.tau == tau_empirical(
        loo_knn_scores(ltm.ref_set, config.k_ref), config.tau_percentile_q
    )
    assert ltm.merged_from == ["stm_0000"]
    assert [(r.kind, r.survivor_id, r.absorbed_id) for r in sweeper.merge_log] == [
        ("stm_ltm", "ltm_000", "stm_0000")
    ]


# --------------------------------------------- initial-initial guard (FR-8.3)


def test_initial_initial_never_merges():
    """Two provenance="initial" concepts moved artificially on top of each
    other: the sweep refuses even though both FR-8.1 conditions hold."""
    config = FPCMCConfig()
    world, store, prior = _ltm_world(config)

    a, b = store.get("ltm_000"), store.get("ltm_001")
    # Move b onto a: identical geometry, so both conditions trivially hold.
    b.centroid = a.centroid.copy()
    b.ref_set = a.ref_set.copy()
    b.kappa = a.kappa
    b.tau, b.tau_vmf = a.tau, a.tau_vmf

    sweeper = MergeSweeper(config, prior)
    check = sweeper.check_pair(a, b)
    assert check.compatible, "premise: the pair would merge on FR-8.1 grounds"

    n_ltm = len(store.ltm)
    sweeper.sweep(store, step=700)
    assert "ltm_000" in store and "ltm_001" in store
    assert len(store.ltm) == n_ltm
    assert sweeper.merge_log == []
    assert a.provenance == b.provenance == "initial"


# --------------------------------------------- promoted-promoted merge (FR-8.3)


def _promoted_fragment(x: np.ndarray, concept_id: str, config: FPCMCConfig, *,
                       match_count: int, windows: set[int]) -> Concept:
    """A manually promoted fragment (TASKS: 'force via manual promotion')."""
    c = _stm(x, concept_id, match_count=match_count, windows=windows)
    c.status = "LTM"
    c.provenance = "promoted"
    recompute_on_promotion(c, config)
    return c


def test_promoted_promoted_merge():
    """Two promoted fragments of the golden novel class: the LTM<->LTM sweep
    merges them and the fragmentation index for that class returns to 1."""
    config = FPCMCConfig()
    world = make_golden_world()
    t0 = world.t0_pool(50)
    store = initialize_ltm(t0.x, t0.labels, config)
    prior = compute_global_prior(store.ltm, config)

    a = _promoted_fragment(
        world.sample_class("novel_00", 36, stream="t9/promoted/a"),
        "stm_0000", config, match_count=40, windows={1, 2, 3},
    )
    b = _promoted_fragment(
        world.sample_class("novel_00", 36, stream="t9/promoted/b"),
        "stm_0001", config, match_count=31, windows={2, 3, 4},
    )
    store.register(a)
    store.register(b)
    centroid_before = a.centroid.tobytes()

    sweeper = MergeSweeper(config, prior)
    check = sweeper.check_pair(a, b)
    assert check.compatible, "premise: same-class fragments must pass FR-8.1"
    sweeper.sweep(store, step=1500)

    assert "stm_0000" in store and "stm_0001" not in store
    assert a.status == "LTM" and a.provenance == "promoted"
    assert a.centroid.tobytes() == centroid_before, "LTM survivor centroid stays frozen"
    assert a.match_count == 71
    assert a.match_windows == {1, 2, 3, 4}
    assert a.ref_set.shape[0] == config.K_max_refset  # 72-row union re-bounded
    assert a.merged_from == ["stm_0001"]
    assert [(r.kind, r.survivor_id, r.absorbed_id) for r in sweeper.merge_log] == [
        ("ltm_ltm", "stm_0000", "stm_0001")
    ]
    # Fragmentation index for novel_00 back to 1 promoted concept.
    promoted = [c for c in store.ltm if c.provenance == "promoted"]
    assert [c.concept_id for c in promoted] == ["stm_0000"]
    # The initial T0 concepts were never merge candidates.
    assert len([c for c in store.ltm if c.provenance == "initial"]) == 8


def test_on_promotion_check_merges_immediately():
    """The FR-8 on-promotion check runs just the LTM<->LTM phase: a freshly
    promoted duplicate of an earlier promoted concept merges without waiting
    for the periodic sweep."""
    config = FPCMCConfig()
    world = make_golden_world()
    t0 = world.t0_pool(50)
    store = initialize_ltm(t0.x, t0.labels, config)
    prior = compute_global_prior(store.ltm, config)

    first = _promoted_fragment(
        world.sample_class("novel_01", 36, stream="t9/onpromo/a"),
        "stm_0000", config, match_count=45, windows={2, 3, 4},
    )
    store.register(first)
    sweeper = MergeSweeper(config, prior)
    sweeper.on_promotion(store, step=1200)
    assert sweeper.merge_log == [], "a lone promoted concept has nothing to merge with"

    second = _promoted_fragment(
        world.sample_class("novel_01", 36, stream="t9/onpromo/b"),
        "stm_0001", config, match_count=32, windows={3, 4, 5},
    )
    store.register(second)
    sweeper.on_promotion(store, step=1300)
    assert "stm_0001" not in store
    assert [(r.kind, r.survivor_id, r.absorbed_id) for r in sweeper.merge_log] == [
        ("ltm_ltm", "stm_0000", "stm_0001")
    ]


# ------------------------------------------------------ bounded union (FR-8.1)


def test_merged_refset_bound():
    """Post-union ref_set is capped at K_max via the dedicated merge
    substream: every row comes from the union, the subsample is bitwise
    deterministic under the config seed, and a different seed draws a
    different subsample."""

    def _merged_refset(seed: int) -> np.ndarray:
        config = FPCMCConfig(seed=seed)
        x1 = _cone(_direction(0), 40, "bound/a", spread=0.12)
        x2 = _cone(_direction(0), 40, "bound/b", spread=0.12)
        a = _stm(x1, "stm_0000", match_count=20)
        b = _stm(x2, "stm_0001", match_count=5)
        store = _bare_store(config, [a, b])
        sweeper = MergeSweeper(config, _prior())
        sweeper.sweep(store, step=500)
        assert "stm_0001" not in store
        survivor = store.get("stm_0000")
        union_rows = {row.tobytes() for row in np.vstack([x1, x2])}
        assert survivor.ref_set.shape[0] == config.K_max_refset, "union of 80 capped at K_max"
        assert all(row.tobytes() in union_rows for row in survivor.ref_set)
        assert survivor.ref_count_seen == 80
        return survivor.ref_set

    ref_42a = _merged_refset(42)
    ref_42b = _merged_refset(42)
    ref_43 = _merged_refset(43)
    assert ref_42a.tobytes() == ref_42b.tobytes(), "same seed => bitwise-identical subsample"
    assert ref_42a.tobytes() != ref_43.tobytes(), "different seed => different subsample"
