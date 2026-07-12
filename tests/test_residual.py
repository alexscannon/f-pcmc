"""T10 tests — residual clustering: identity-preserving consolidation
(TASKS Task 10; PRD FR-6.1–6.2).

Owner-approved T10 semantics (Q&A 2026-07-11, decisions 21–24; recorded in
docs/CHANGES.md T10):

  21. Clustering core = faithful port of the lib sweep: HDBSCAN over §8's
      min_cluster_sizes + Jaccard dedup at the source literal 0.80
      (msproject_misc @ e723f028 config.yaml jaccard_dedup_threshold).
  22. Live-predicate pool: a seeded candidate's SEED embedding enters at
      s + w_residual iff the candidate is alive and immature (match_count <
      n_mature — "singleton" read as seeded-as-singleton, so 1–4 matches
      still enter); evicted-early immature candidates enter at eviction time
      (owner ruling 2026-07-11); absorbed candidates never enter (and leave
      if already in — their seed lives in the survivor's union); candidates
      that mature after entry leave at the next hook; evicted-after-entry
      entries stay (a dead parent can never mature).
  23. Retention: pool rows that land in any final (post-dedup) cluster are
      consumed (removed); noise rows stay.
  24. Clusters drive merges only: a cluster containing >= 2 alive immature
      candidates merges them via the T9 merge_pair seam (kind="residual",
      identities preserved, no new concept ids); pool rows are density
      context only; pool-only and single-candidate clusters are no-ops.

Row-layout contract under test: clustering input = pool rows in entry order
(entered_at, concept_id) followed by alive immature-STM centroids ascending
concept_id — exposed as ResidualClusterer.clustering_membership(store).
"""

import dataclasses

import numpy as np
import pytest

from fpcmc.concepts import Concept, ConceptStore
from fpcmc.config import FPCMCConfig
from fpcmc.data import POOL_SPECS, embeddings_available, load_pool
from fpcmc.memory import MergeSweeper
from fpcmc.residual import RESIDUAL_POOL_MIN, ResidualClusterer
from fpcmc.rng import make_rng
from fpcmc.scorers import estimate_kappa
from fpcmc.thresholds import GlobalPrior

AVAILABLE, REASON = embeddings_available()

SEED = 1001
D = 32


# ------------------------------------------------------------------- helpers


def _config(**kw) -> FPCMCConfig:
    base = dict(w_residual=50, T_cluster=100, n_mature=5, stm_capacity=100, seed=SEED)
    base.update(kw)
    return FPCMCConfig(**base)


def _make(
    config: FPCMCConfig, tau_prior: float = 0.30
) -> tuple[ConceptStore, ResidualClusterer]:
    prior = GlobalPrior(tau=tau_prior, tau_vmf=0.0)
    store = ConceptStore(config, prior)
    rc = ResidualClusterer(config, MergeSweeper(config, prior))
    return store, rc


def _axis(i: int) -> np.ndarray:
    z = np.zeros(D)
    z[i] = 1.0
    return z


def _random_unit(rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal(D)
    return z / np.linalg.norm(z)


def _seed_singleton(store: ConceptStore, rc: ResidualClusterer, z: np.ndarray, step: int) -> str:
    """Route a novel embedding (must land at tier 3) and wire note_seed the
    way T11's runner will."""
    result = store.route(z, step)
    assert result.tier == 3, "test premise: embedding must seed a new singleton"
    rc.note_seed(result.concept_id, z, step)
    return result.concept_id


def _match(store: ConceptStore, cid: str, z: np.ndarray, step: int) -> None:
    result = store.route(z, step)
    assert result.tier in (1, 2) and result.concept_id == cid, (
        f"test premise: match must land on {cid}, got {result}"
    )


# ------------------------------------------------------- pool aging (FR-6.1)


def test_pool_aging():
    """TASKS T10: a singleton seeded at step s enters the pool exactly at
    s + w_residual if still immature; matured singletons never enter."""
    config = _config()  # w_residual=50
    store, rc = _make(config)

    cid = _seed_singleton(store, rc, _axis(0), 10)
    for step in range(11, 60):
        rc.hook(store, step)
        assert cid not in rc.pool_ids, f"entered early at step {step}"
    rc.hook(store, 60)  # 10 + w_residual: exact entry step
    assert rc.pool_ids == [cid]

    # A singleton that matures before aging never enters.
    z2 = _axis(1)
    cid2 = _seed_singleton(store, rc, z2, 100)
    for k in range(config.n_mature):  # 5 matches -> mature by step 105
        _match(store, cid2, z2, 101 + k)
    for step in range(101, 200):
        rc.hook(store, step)
    assert cid2 not in rc.pool_ids
    assert rc.pool_ids == [cid]


def test_pool_eviction_entry():
    """Owner ruling (2026-07-11): an immature singleton evicted before
    s + w_residual contributes its seed embedding at eviction time; a MATURE
    candidate evicted never enters."""
    config = _config(stm_capacity=2)
    store, rc = _make(config)

    cid_a = _seed_singleton(store, rc, _axis(0), 0)
    z_b = _axis(1)
    cid_b = _seed_singleton(store, rc, z_b, 1)
    cid_c = _seed_singleton(store, rc, _axis(2), 2)  # evicts A (LRU)
    assert cid_a not in store and store.eviction_log[-1].concept_id == cid_a

    rc.hook(store, 3)
    assert rc.pool_ids == [cid_a], "evicted-early immature seed enters at eviction time"

    # Mature B (5 matches), then evict it: matured singletons never enter.
    for k in range(config.n_mature):
        _match(store, cid_b, z_b, 3 + k)
    _seed_singleton(store, rc, _axis(3), 8)  # evicts C (LRU: last_matched 2)
    assert store.eviction_log[-1].concept_id == cid_c
    _seed_singleton(store, rc, _axis(4), 9)  # evicts B (mature, but STM is not exempt)
    assert store.eviction_log[-1].concept_id == cid_b

    for step in range(10, 100):
        rc.hook(store, step)
    assert cid_b not in rc.pool_ids, "mature-evicted candidate must never enter"
    assert rc.pool_ids[:2] == [cid_a, cid_c], "entry order = eviction order"


def test_pool_live_predicate():
    """Decision 22 edges: absorbed candidates never enter / leave the pool;
    candidates that mature after entry are removed at the next hook."""
    config = _config()  # w_residual=50
    store, rc = _make(config)
    sweeper = rc.sweeper

    cid_a = _seed_singleton(store, rc, _axis(0), 0)
    cid_b = _seed_singleton(store, rc, _axis(1), 1)
    sweeper.merge_pair(store, store.get(cid_a), store.get(cid_b), 5, kind="residual")
    assert cid_b not in store  # equal match_count: smaller id survives

    for step in range(6, 80):
        rc.hook(store, step)
    assert cid_b not in rc.pool_ids, "absorbed-pending seed must never enter"
    assert rc.pool_ids == [cid_a], "survivor (alive, immature) enters at s + w_residual"

    # A matures after entry -> removed at the next hook. Match on A's merged
    # centroid (normalized mean of the union), which A accepts.
    z_a = store.get(cid_a).centroid.copy()
    for k in range(config.n_mature):
        _match(store, cid_a, z_a, 80 + k)
    rc.hook(store, 85)
    assert rc.pool_ids == [], "matured-after-entry seed is removed"

    # Pooled-then-absorbed -> removed. D gets one match so it wins survivorship.
    cid_c = _seed_singleton(store, rc, _axis(2), 90)
    cid_d = _seed_singleton(store, rc, _axis(3), 91)
    _match(store, cid_d, _axis(3), 92)
    for step in range(93, 145):
        rc.hook(store, step)
    assert set(rc.pool_ids) == {cid_c, cid_d}  # both aged in (140 / 141)
    sweeper.merge_pair(store, store.get(cid_d), store.get(cid_c), 145, kind="residual")
    rc.hook(store, 146)
    assert rc.pool_ids == [cid_d], "absorbed-pooled seed is removed; survivor stays"


# ------------------------------------------------------------------- trigger


def test_trigger_conditions(mocker):
    """TASKS T10: no clustering run below RESIDUAL_POOL_MIN pool items or off
    the T_cluster schedule (spy on the clustering call)."""
    config = _config(w_residual=10, T_cluster=100)
    store, rc = _make(config)
    clustered = mocker.patch.object(ResidualClusterer, "_cluster", return_value=[])

    rng = make_rng(SEED, "t10/trigger-dirs")
    for _ in range(RESIDUAL_POOL_MIN - 1):  # 29 singletons, all seeded at step 0
        _seed_singleton(store, rc, _random_unit(rng), 0)

    rc.hook(store, 0)  # step 0 never triggers
    for step in range(1, 100):
        rc.hook(store, step)
    rc.hook(store, 100)  # on-schedule but pool = 29 < 30
    assert clustered.call_count == 0
    assert len(rc.pool_ids) == RESIDUAL_POOL_MIN - 1

    _seed_singleton(store, rc, _random_unit(rng), 101)  # pooled at 111
    rc.hook(store, 150)  # pool = 30 but off-schedule
    assert clustered.call_count == 0

    rc.hook(store, 200)  # on-schedule AND pool >= 30
    assert clustered.call_count == 1
    assert len(rc.run_log) == 1 and rc.run_log[0].step == 200


# ------------------------------------------------- identity-preserving merge


# Fixture geometry for the candidates below. Candidate i belongs to "class"
# i // 2 and sits at its own angular offset from that class axis, so candidates
# 0-1 are a genuine same-class pair and 2-3 are another, while the two classes
# are orthogonal. Matches land NEAR a candidate's center in a direction private
# to each match, so a ref_set acquires REAL internal spread.
_CLASS_AXIS = (0, 1)
_T_WITHIN = 0.36927   # sim(candidate_a, candidate_b) = 1/(1+t^2) = 0.88
_S_MATCH = 0.2295     # sim(center, match)  = 1/sqrt(1+s^2) = 0.975
_TAU_PRIOR = 0.10     # tight enough that a same-class sibling still SEEDS


def _candidate_center(i: int) -> np.ndarray:
    z = np.zeros(D)
    z[_CLASS_AXIS[i // 2]] = 1.0
    z[4 + i] = _T_WITHIN          # axes 4..7: one private offset per candidate
    return z / np.linalg.norm(z)


def _candidate_match(i: int, m: int) -> np.ndarray:
    z = _candidate_center(i)
    z = z.copy()
    z[12 + 4 * i + m] = _S_MATCH  # axes 12..: private per (candidate, match)
    return z / np.linalg.norm(z)


def _build_candidates_and_pool(store, rc, match_counts):
    """Four candidates in TWO genuine same-class pairs (0-1, 2-3) with the
    given match_counts, plus 30 pooled distractor seeds (random directions,
    seeded at step 20, pooled at 30).

    FIXTURE REDESIGN (2026-07-11 owner ruling; docs/CHANGES.md). The candidates
    used to sit on mutually ORTHOGONAL basis axes, and the merge test asserted
    that a mocked HDBSCAN grouping merged them anyway — which pinned the
    superseded contract that "clustering evidence stands in for the
    two-condition check". Merging orthogonal concepts is now exactly what the
    FR-6 admission guard exists to refuse, so that geometry can no longer stand.
    Every ASSERTION in the tests below is unchanged; only the geometry is now
    one a real cluster could actually have.

    Two properties make it a faithful miniature of the real dynamic:
      * each pair is genuinely same-class (centroid sim 0.88 > merge_sim), so
        consolidating them is the CORRECT outcome the guard must still allow;
      * they nonetheless SEED separately, because at seed time the host's tau
        is still tight — which is precisely why FR-6 consolidation has to exist
        at all, and precisely the case the guard must not throw away.
    Matches are spread around each center rather than being identical copies of
    the seed: a degenerate ref_set has LOO scores of 0, which drives tau BELOW
    the prior and yields a concept incapable of ever admitting a neighbour.
    """
    cand_ids = []
    for i, n_matches in enumerate(match_counts):
        z = _candidate_center(i)
        # Deliberately NOT note_seed'd: keeps the pool = the 30 distractor
        # seeds below (what note_seed sees is the runner's wiring choice;
        # pool membership is pinned by the aging tests above).
        result = store.route(z, i)
        assert result.tier == 3, (
            f"test premise: candidate {i} must SEED (a same-class sibling is "
            f"{1 - float(_candidate_center(i) @ _candidate_center(i - 1)):.3f} "
            f"away, tau_prior={_TAU_PRIOR}), got {result}"
        )
        cand_ids.append(result.concept_id)
        for m in range(n_matches):
            _match(store, cand_ids[-1], _candidate_match(i, m), 10 + 4 * m + i)

    # Premise: each mocked pair really is same-class-similar (so the guard
    # SHOULD admit it), and the two classes really are far apart.
    if len(cand_ids) == 4:
        for a, b in ((0, 1), (2, 3)):
            sim = float(store.get(cand_ids[a]).centroid @ store.get(cand_ids[b]).centroid)
            assert sim >= 0.8, f"test premise: pair ({a},{b}) sim {sim:.3f} < merge_sim"
        cross = float(store.get(cand_ids[0]).centroid @ store.get(cand_ids[2]).centroid)
        assert cross < 0.3, f"test premise: the two classes must be far apart, got {cross:.3f}"

    rng = make_rng(SEED, "t10/merge-pool-dirs")
    for _ in range(RESIDUAL_POOL_MIN):
        _seed_singleton(store, rc, _random_unit(rng), 20)
    rc.hook(store, 30)  # w_residual=10: distractor seeds enter the pool
    assert len(rc.pool_ids) == RESIDUAL_POOL_MIN
    return cand_ids


def test_identity_preserving_merge(mocker):
    """TASKS T10: mock HDBSCAN returning a known grouping over 4 immature
    candidates: they merge pairwise via the T9 merge path (lineage recorded,
    kind="residual"); no new concept_ids are created by this pathway."""
    config = _config(w_residual=10, T_cluster=100)
    store, rc = _make(config, tau_prior=_TAU_PRIOR)
    cand_ids = _build_candidates_and_pool(store, rc, match_counts=(3, 2, 1, 1))
    ids_before = {c.concept_id for c in store.concepts}

    pool_ids, immature_ids = rc.clustering_membership(store)
    assert pool_ids == rc.pool_ids and set(cand_ids) <= set(immature_ids)
    row = {cid: len(pool_ids) + immature_ids.index(cid) for cid in cand_ids}
    grouping = [
        [row[cand_ids[0]], row[cand_ids[1]], 0, 1],  # pool rows 0,1: consumed
        [row[cand_ids[2]], row[cand_ids[3]]],
    ]
    mocker.patch.object(ResidualClusterer, "_cluster", return_value=grouping)
    consumed_pool = pool_ids[:2]

    rc.hook(store, 100)

    log = rc.sweeper.merge_log
    assert [r.kind for r in log] == ["residual", "residual"]
    # Survivors by the T9 rule: larger match_count (3>2), tie -> smaller id.
    assert log[0].survivor_id == cand_ids[0] and log[0].absorbed_id == cand_ids[1]
    assert log[1].survivor_id == cand_ids[2] and log[1].absorbed_id == cand_ids[3]
    assert cand_ids[1] not in store and cand_ids[3] not in store
    assert store.get(cand_ids[0]).merged_from == [cand_ids[1]]
    assert store.get(cand_ids[2]).merged_from == [cand_ids[3]]
    assert rc.sweeper.lineage == {
        cand_ids[0]: [cand_ids[1]],
        cand_ids[2]: [cand_ids[3]],
    }
    # Identities preserved: this pathway creates no new concept ids.
    assert {c.concept_id for c in store.concepts} == ids_before - {cand_ids[1], cand_ids[3]}
    # Decision 23: clustered pool rows consumed; the rest untouched.
    assert rc.pool_ids == [p for p in pool_ids if p not in consumed_pool]
    assert rc.run_log[-1].n_merges == 2 and rc.run_log[-1].n_pool_consumed == 2


def test_noise_untouched(mocker):
    """TASKS T10: mock all-noise labels: candidates bitwise unchanged and
    still LRU-eligible; the pool is not consumed (decision 23)."""
    config = _config(w_residual=10, T_cluster=100, stm_capacity=32)
    store, rc = _make(config, tau_prior=_TAU_PRIOR)
    cand_ids = _build_candidates_and_pool(store, rc, match_counts=(0, 0))
    mocker.patch.object(ResidualClusterer, "_cluster", return_value=[])

    def snapshot(cid):
        c = store.get(cid)
        return (
            c.centroid.tobytes(),
            c.ref_set.tobytes(),
            c.tau,
            c.kappa,
            c.match_count,
            c.ref_count_seen,
            c.last_matched_at,
            list(c.merged_from),
        )

    before = {cid: snapshot(cid) for cid in cand_ids}
    pool_before = list(rc.pool_ids)

    rc.hook(store, 100)

    assert rc.run_log[-1].n_clusters == 0
    assert rc.sweeper.merge_log == []
    assert {cid: snapshot(cid) for cid in cand_ids} == before
    assert rc.pool_ids == pool_before, "noise pool rows stay (decision 23)"

    # Still LRU-eligible: at capacity 32 (2 candidates + 30 distractors), the
    # next seed evicts the least-recently-matched noise candidate.
    _seed_singleton(store, rc, _axis(10), 101)
    assert store.eviction_log[-1].concept_id == cand_ids[0]


# ----------------------------------------------- real clustering + determinism


def _two_cone_scenario(config: FPCMCConfig) -> tuple[ConceptStore, ResidualClusterer, list[str]]:
    """30 evicted-immature seeds from two tight cones (15 each) + 4 alive
    singleton candidates (2 per cone), via the public seams only. Total
    clustering input = 34 rows <= umap.dim, so the real _cluster path runs
    HDBSCAN without UMAP (fast, deterministic). The prior tau is tight
    (0.01, below the ~0.08 typical intra-cone pair distance) so every cone
    point seeds its own singleton instead of matching an earlier one."""
    store, rc = _make(config, tau_prior=0.01)
    rng = make_rng(SEED, "t10/cones")
    cones = []
    for axis in (0, 1):
        x = _axis(axis)[None, :] + 0.05 * rng.standard_normal((17, D))
        cones.append(x / np.linalg.norm(x, axis=1, keepdims=True))
    # 30 pool-fodder seeds, alternating cones; capacity 4 evicts them all
    # (immature -> pooled at eviction, owner ruling).
    for i in range(15):
        for cone in cones:
            _seed_singleton(store, rc, cone[i], 2 * i + (0 if cone is cones[0] else 1))
    # 4 candidates (2 per cone), seeded last so they are the alive survivors.
    cand_ids = []
    for k, step in ((15, 30), (16, 31)):
        for cone in cones:
            cand_ids.append(_seed_singleton(store, rc, cone[k], step))
    rc.hook(store, 34)  # off-schedule: reconciles evictions, no clustering
    assert len(store.stm) == 4 and len(rc.pool_ids) == 30
    return store, rc, cand_ids


def test_residual_consolidation_and_determinism():
    """Real (un-mocked) sweep + dedup on a crafted two-cone world: each
    cone's two candidates merge (correct pairing), the clustered pool rows
    are consumed, and the whole pass is deterministic across two identically
    built runs (CLAUDE.md determinism mandate)."""
    config = _config(w_residual=500, T_cluster=100, stm_capacity=4)

    def run():
        store, rc, cand_ids = _two_cone_scenario(config)
        rc.hook(store, 100)
        return store, rc, cand_ids

    store, rc, cand_ids = run()
    log = rc.sweeper.merge_log
    assert len(log) == 2 and all(r.kind == "residual" for r in log)
    # Pairing: candidates 0/2 are cone-0, 1/3 cone-1 (seed order alternated).
    survivors = [c for c in store.stm]
    assert len(survivors) == 2
    pair0 = {cand_ids[0], cand_ids[2]}
    pair1 = {cand_ids[1], cand_ids[3]}
    for rec in log:
        pair = {rec.survivor_id, rec.absorbed_id}
        assert pair == pair0 or pair == pair1, f"cross-cone merge: {rec}"
    assert rc.pool_ids == [], "all pool rows clustered -> consumed"
    assert rc.run_log[-1].n_merges == 2

    def record_key(records):
        # MergeRecord carries NaN (cross_within_ratio on residual merges),
        # and NaN != NaN under dataclass equality; repr round-trips floats.
        return [repr(r) for r in records]

    store2, rc2, cand_ids2 = run()
    assert cand_ids2 == cand_ids
    assert record_key(rc2.sweeper.merge_log) == record_key(log)
    assert rc2.run_log == rc.run_log
    assert rc2.pool_ids == rc.pool_ids
    for a, b in zip(sorted(store.stm, key=lambda c: c.concept_id),
                    sorted(store2.stm, key=lambda c: c.concept_id)):
        assert a.concept_id == b.concept_id
        assert a.ref_set.tobytes() == b.ref_set.tobytes()
        assert a.centroid.tobytes() == b.centroid.tobytes()
        assert (a.tau, a.tau_vmf, a.kappa) == (b.tau, b.tau_vmf, b.kappa)


# ------------------------------------------------------------ real data ([I])


@pytest.mark.slow
@pytest.mark.skipif(not AVAILABLE, reason=REASON)
def test_residual_consolidation_real():
    """TASKS T10 [I]: under-segmentation scenario from real embeddings —
    6 immature candidates from split halves of 3 near-OOD classes; the real
    UMAP+HDBSCAN consolidation reduces them to 3 concepts with correct
    pairings (ground truth used to verify pairing only)."""
    near = load_pool(POOL_SPECS[3])  # near_ood: 500 x 1024, 6 classes
    classes = list(np.unique(near.subclass_names)[:3])
    config = FPCMCConfig(w_residual=100, T_cluster=100, n_mature=5, seed=SEED)
    prior = GlobalPrior(tau=0.30, tau_vmf=0.0)
    store = ConceptStore(config, prior)
    rc = ResidualClusterer(config, MergeSweeper(config, prior))

    half_ids: dict[str, list[str]] = {}
    owner_class: dict[str, str] = {}
    for cls in classes:
        rows = near.x[near.subclass_names == cls]
        half_ids[cls] = []
        for half in (rows[:20], rows[20:40]):
            cid = store.new_concept_id("stm")
            centroid = half.mean(axis=0)
            centroid /= np.linalg.norm(centroid)
            store.register(Concept(
                concept_id=cid,
                centroid=centroid,
                ref_set=half.copy(),
                tau=0.30,
                kappa=estimate_kappa(half),
                status="STM",
                provenance="seeded",
                match_count=1,  # immature
                created_at=0,
                last_matched_at=0,
                rng=make_rng(SEED, f"t10/reservoir/{cid}"),
            ))
            half_ids[cls].append(cid)
            owner_class[cid] = cls
        for z in rows[40:50]:  # 10 residual singletons per class = pool of 30
            cid = store.new_concept_id("stm")
            store.register(Concept.seed(
                z, 0, prior.tau, prior.tau_vmf,
                concept_id=cid, rng=make_rng(SEED, f"t10/reservoir/{cid}"),
            ))
            rc.note_seed(cid, z, 0)
            owner_class[cid] = cls

    rc.hook(store, 100)  # singletons age into the pool at 100; trigger fires

    # Reduced to 3 concepts, one per class, with correct pairings.
    assert len(store.concepts) == 3
    for cls in classes:
        first, second = half_ids[cls]
        assert first in store, f"{cls}: expected surviving half {first}"
        survivor = store.get(first)
        assert second in survivor.merged_from, f"{cls}: halves not paired"
        absorbed_classes = {owner_class[a] for a in survivor.merged_from}
        assert absorbed_classes == {cls}, (
            f"{cls}: survivor absorbed foreign-class candidates: {absorbed_classes}"
        )
    assert all(r.kind == "residual" for r in rc.sweeper.merge_log)
