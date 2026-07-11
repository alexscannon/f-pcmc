"""T7 [U] tests — STM dynamics: capacity, LRU eviction, maturity (TASKS
Task 7; PRD FR-3.1–3.3).

All tests are synthetic (crafted axis-aligned clusters in D=8); no real data;
all randomness through fpcmc.rng.make_rng. The sixth TASKS-T7 test,
test_capacity_invariant, lives in tests/test_invariants.py — it IS
cross-cutting invariant 3 ("STM capacity <= Δ at every step, T7+").

Owner-approved T7 semantics (Q&A 2026-07-11, recorded in docs/CHANGES.md T7):
  - EvictionRecord.size = match_count at eviction (PRD FR-7 criterion 1
    literally names match_count "Size"); age = step - created_at (lifetime).
  - Beyond the TASKS-literal four fields (id, size, age, step) the record
    carries created_at, last_matched_at and ref_count_seen so T11's JSONL
    `evict` records and T13's eviction-composition metric need no extra
    plumbing.
  - LRU victim order: min last_matched_at, ties to older created_at (TASKS
    literal), final tie to the smallest concept_id.
  - Capacity is enforced by a drain-while loop at the tier-3 seeding site
    (the only STM growth site): while |STM| >= Δ, evict the LRU victim, then
    seed — normally exactly one eviction per seed; a store hand-built above
    capacity converges back under Δ.
"""

import dataclasses

import numpy as np

from fpcmc.concepts import Concept, ConceptStore, EvictionRecord
from fpcmc.config import FPCMCConfig
from fpcmc.rng import make_rng
from fpcmc.scorers import estimate_kappa
from fpcmc.thresholds import GlobalPrior

SEED = 701
D = 8


# ------------------------------------------------------------------- helpers


def _cluster(axis: int, n: int, stream: str, spread: float = 0.05) -> np.ndarray:
    """(n, D) unit rows in a tight deterministic cone around basis axis `axis`."""
    rng = make_rng(SEED, f"t7/cluster/{stream}")
    e = np.zeros(D)
    e[axis] = 1.0
    x = e[None, :] + spread * rng.standard_normal((n, D))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _stm_concept(
    axis: int,
    concept_id: str,
    *,
    created_at: int,
    last_matched_at: int,
    match_count: int = 0,
    tau: float = 0.30,
    status: str = "STM",
    stream: str | None = None,
) -> Concept:
    """A 5-member STM candidate on basis axis `axis` with crafted LRU inputs.

    ref_set stays below n_vmf_min=10, so the composed knn_vmf scorer is in
    its knn_ref fallback and the NaN tau_vmf default is never read.
    """
    ref = _cluster(axis, 5, stream or f"refs/{concept_id}")
    centroid = ref.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    return Concept(
        concept_id=concept_id,
        centroid=centroid,
        ref_set=ref,
        tau=tau,
        kappa=estimate_kappa(ref),
        status=status,
        provenance="initial" if status == "LTM" else "seeded",
        match_count=match_count,
        created_at=created_at,
        last_matched_at=last_matched_at,
        rng=make_rng(SEED, f"t7/reservoir/{concept_id}"),
    )


def _prior() -> GlobalPrior:
    return GlobalPrior(tau=0.30, tau_vmf=0.0)


def _ids(store: ConceptStore) -> set[str]:
    return {c.concept_id for c in store.concepts}


# ------------------------------------------------------ LRU order (FR-3.1)


def test_lru_eviction_order():
    """At Δ=5, insertion #6 evicts exactly the least-recently-matched; a
    re-match rescues the next would-be victim."""
    config = FPCMCConfig(stm_capacity=5)
    # Axes 0..4, crafted last-match steps; the LRU victim is B (step 10).
    concepts = [
        _stm_concept(0, "stm_0000", created_at=0, last_matched_at=50),  # A
        _stm_concept(1, "stm_0001", created_at=1, last_matched_at=10),  # B
        _stm_concept(2, "stm_0002", created_at=2, last_matched_at=30),  # C
        _stm_concept(3, "stm_0003", created_at=3, last_matched_at=20),  # D
        _stm_concept(4, "stm_0004", created_at=4, last_matched_at=40),  # E
    ]
    store = ConceptStore(config, _prior(), concepts)

    # Insertion #6: a novel direction (axis 5) rejected everywhere seeds and
    # must evict exactly B.
    r = store.route(_cluster(5, 1, "order/novel_1")[0], step=100)
    assert r.tier == 3
    assert "stm_0001" not in store
    assert _ids(store) == {"stm_0000", "stm_0002", "stm_0003", "stm_0004", r.concept_id}
    assert [e.concept_id for e in store.eviction_log] == ["stm_0001"]
    assert len(store.stm) == config.stm_capacity

    # Rescue: D (step 20) is the next would-be victim; a re-match at step 101
    # refreshes it, so the next insertion evicts C (step 30) instead.
    r_match = store.route(_cluster(3, 1, "order/rescue")[0], step=101)
    assert r_match.tier == 2 and r_match.concept_id == "stm_0003"
    assert store.get("stm_0003").last_matched_at == 101

    r2 = store.route(_cluster(6, 1, "order/novel_2")[0], step=102)
    assert r2.tier == 3
    assert "stm_0002" not in store, "rescued D must not be evicted; C is next-LRU"
    assert "stm_0003" in store
    assert [e.concept_id for e in store.eviction_log] == ["stm_0001", "stm_0002"]
    assert len(store.stm) == config.stm_capacity


def test_lru_tiebreak():
    """Equal last_matched_at: the older created_at is evicted (TASKS literal);
    with created_at also equal, the smallest concept_id goes (owner-approved
    final tie-break, 2026-07-11)."""
    config = FPCMCConfig(stm_capacity=2)

    # created_at decides — deliberately anti-correlated with id order so the
    # test proves created_at (not id allocation order) is the tie-break.
    old = _stm_concept(0, "stm_0009", created_at=2, last_matched_at=10)
    young = _stm_concept(1, "stm_0001", created_at=7, last_matched_at=10)
    store = ConceptStore(config, _prior(), [old, young])
    r = store.route(_cluster(5, 1, "tie/novel_1")[0], step=50)
    assert r.tier == 3
    assert "stm_0009" not in store, "equal LRU: older created_at is evicted first"
    assert "stm_0001" in store
    assert [e.concept_id for e in store.eviction_log] == ["stm_0009"]

    # Full tie (last_matched_at AND created_at equal): smallest id goes.
    twin_a = _stm_concept(0, "stm_0005", created_at=3, last_matched_at=10)
    twin_b = _stm_concept(1, "stm_0002", created_at=3, last_matched_at=10)
    store2 = ConceptStore(config, _prior(), [twin_a, twin_b])
    r2 = store2.route(_cluster(5, 1, "tie/novel_2")[0], step=50)
    assert r2.tier == 3
    assert "stm_0002" not in store2, "full tie must evict the smallest concept_id"
    assert "stm_0005" in store2


# ------------------------------------------------------- log schema (FR-3.1)


def test_eviction_log_schema():
    """Every eviction produces a record with all fields correctly valued;
    the record count matches the eviction count."""
    config = FPCMCConfig(stm_capacity=3)
    concepts = [
        _stm_concept(0, "stm_0000", created_at=0, last_matched_at=5, match_count=3),
        _stm_concept(1, "stm_0001", created_at=1, last_matched_at=6, match_count=4),
        # match_count=9 >= n_mature: a mature-but-unpromoted candidate is
        # still LRU-evictable (FR-3.1 exempts nothing but LTM).
        _stm_concept(2, "stm_0002", created_at=2, last_matched_at=7, match_count=9),
    ]
    expected = {
        c.concept_id: (c.match_count, c.created_at, c.last_matched_at, c.ref_count_seen)
        for c in concepts
    }
    store = ConceptStore(config, _prior(), concepts)

    # Three seeds at capacity => three evictions, in last_matched_at order
    # (the fresh seeds are always more recently "matched" than the originals).
    steps = (100, 101, 102)
    for i, step in enumerate(steps):
        r = store.route(_cluster(5 + i, 1, f"schema/novel_{i}")[0], step=step)
        assert r.tier == 3

    log = store.eviction_log
    assert len(log) == 3, "record count must equal eviction count"
    assert [e.concept_id for e in log] == ["stm_0000", "stm_0001", "stm_0002"]

    field_names = [f.name for f in dataclasses.fields(EvictionRecord)]
    assert field_names == [
        "concept_id", "size", "age", "step",
        "created_at", "last_matched_at", "ref_count_seen",
    ]
    for record, step in zip(log, steps, strict=True):
        match_count, created_at, last_matched_at, ref_count_seen = expected[record.concept_id]
        assert isinstance(record, EvictionRecord)
        assert record.size == match_count, "size = match_count (owner decision)"
        assert record.age == step - created_at, "age = step - created_at (owner decision)"
        assert record.step == step
        assert record.created_at == created_at
        assert record.last_matched_at == last_matched_at
        assert record.ref_count_seen == ref_count_seen


# ------------------------------------------------------- maturity (FR-3.3)


def test_maturity_transition():
    """A concept at match_count = n_mature - 1 matches at tier 2; one more
    match makes it tier-1 on the very next route call."""
    config = FPCMCConfig()  # n_mature=5
    cand = _stm_concept(
        0, "stm_0000", created_at=0, last_matched_at=0, match_count=config.n_mature - 1
    )
    store = ConceptStore(config, _prior(), [cand])

    r1 = store.route(_cluster(0, 1, "maturity/q1")[0], step=10)
    assert r1.tier == 2
    assert r1.prediction == "unknown"
    assert r1.concept_id == "stm_0000"
    assert cand.match_count == config.n_mature

    r2 = store.route(_cluster(0, 1, "maturity/q2")[0], step=11)
    assert r2.tier == 1, "n_mature-th match must flip the concept to tier 1"
    assert r2.prediction == "stm_0000"


# ------------------------------------------------- LTM exemption (FR-3.1)


def test_ltm_never_evicted():
    """LTM concepts are exempt from capacity/eviction regardless of staleness:
    maximally stale LTMs survive while every STM candidate cycles out."""
    config = FPCMCConfig(stm_capacity=2)
    ltms = [
        _stm_concept(0, "ltm_000", created_at=0, last_matched_at=0, status="LTM"),
        _stm_concept(1, "ltm_001", created_at=0, last_matched_at=0, status="LTM"),
    ]
    stms = [
        _stm_concept(2, "stm_0000", created_at=10, last_matched_at=50),
        _stm_concept(3, "stm_0001", created_at=11, last_matched_at=60),
    ]
    store = ConceptStore(config, _prior(), ltms + stms)

    # Three seeds: evict stm_0000 (50), stm_0001 (60), then the first seed
    # itself — the stale (last_matched_at=0) LTMs are never candidates.
    seeded = []
    for i, step in enumerate((100, 101, 102)):
        r = store.route(_cluster(4 + i, 1, f"ltm_exempt/novel_{i}")[0], step=step)
        assert r.tier == 3
        seeded.append(r.concept_id)
        assert "ltm_000" in store and "ltm_001" in store
        assert len(store.stm) <= config.stm_capacity

    evicted = [e.concept_id for e in store.eviction_log]
    assert evicted == ["stm_0000", "stm_0001", seeded[0]]
    assert all(cid.startswith("stm_") for cid in evicted)
    assert len(store.ltm) == 2
