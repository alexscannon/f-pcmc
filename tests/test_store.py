"""T5 [U] tests — ConceptStore and routing core (TASKS Task 5; PRD FR-9 loop
body + FR-3.3 routing order).

All tests are synthetic (crafted axis-aligned clusters in D=8 or the vMF
fixture world); no real data; all randomness through fpcmc.rng.make_rng.

Owner-approved T5 semantics (Q&A 2026-07-11, recorded in docs/CHANGES.md T5):
  - RoutingResult.score is the winning concept's ScoreDetail.score (under
    knn_vmf: the composed scalar = knn_ref sub-score, per the T2 decision);
    RoutingResult additionally carries via/fallback for the event log and A5.
    At tier 3 nothing accepted: score = margin = NaN, via = None.
  - The FR-5.1 lazy check (fpcmc.thresholds.maybe_recompute) runs after every
    assignment's add_observation, on the matched concept only, tiers 1 and 2
    alike. Tier-3 seeds skip it (fresh counter).
  - Every seeded singleton gets tau = prior.tau AND tau_vmf = prior.tau_vmf
    (GlobalPrior is always a full pair) — NaN never enters a routed concept.
  - Concept ids are PRD-literal zero-padded: ltm_{:03d} / stm_{:04d},
    store-owned counters, never reused; overflow raises.
  - The batch scoring path computes per-concept `ref_set @ z` (the exact
    frozen-scorer op) and vectorizes composition/selection over concept
    arrays — a stacked single GEMV is measurably not bitwise-equal to the
    per-concept op, and test_vectorized_matches_loop's identity guard is
    the binding clause.
"""

import math

import numpy as np
import pytest

from fpcmc.concepts import Concept, ConceptStore, RoutingResult
from fpcmc.config import FPCMCConfig
from fpcmc.rng import make_rng
from fpcmc.scorers import estimate_kappa, make_scorer
from fpcmc.thresholds import (
    GlobalPrior,
    compute_global_prior,
    recompute_on_promotion,
    recompute_thresholds,
)
from tests.fixtures.vmf_world import VMFWorld

SEED = 501


# ------------------------------------------------------------------- helpers


def _concept_from(ref_set: np.ndarray, status: str, **kw) -> Concept:
    """A consistent Concept over an owned copy of ref_set (kappa cached)."""
    ref = np.array(ref_set, dtype=np.float64)
    centroid = ref.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    return Concept(
        concept_id=kw.pop("concept_id"),
        centroid=centroid,
        ref_set=ref,
        tau=kw.pop("tau", 0.5),
        kappa=estimate_kappa(ref),
        status=status,
        **kw,
    )


def _cluster(axis: int, n: int, stream: str, d: int = 8, spread: float = 0.05) -> np.ndarray:
    """(n, d) unit rows in a tight deterministic cone around basis axis `axis`."""
    rng = make_rng(SEED, f"t5/cluster/{stream}")
    e = np.zeros(d)
    e[axis] = 1.0
    x = e[None, :] + spread * rng.standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _assert_results_equal(a: RoutingResult, b: RoutingResult) -> None:
    assert a.prediction == b.prediction
    assert a.concept_id == b.concept_id
    assert a.tier == b.tier
    assert a.via == b.via
    assert a.fallback == b.fallback
    for fa, fb in ((a.score, b.score), (a.margin, b.margin), (a.novelty, b.novelty)):
        assert (math.isnan(fa) and math.isnan(fb)) or fa == fb


def _assert_stores_equal(s1: ConceptStore, s2: ConceptStore) -> None:
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


# The fixture world shared by the vectorized/determinism tests. World sampling
# is a pure function of (seed, stream, class, n), so two identically built
# stores draw bit-identical material.
_WORLD = VMFWorld(seed=23, k_known=3, k_novel=2)


def _fixture_store(config: FPCMCConfig, vectorized: bool) -> ConceptStore:
    """LTM store over the fixture world's known classes, full FR-5 init."""
    pool = _WORLD.t0_pool(n_per_class=20)
    concepts = []
    for i, name in enumerate(_WORLD.known_names):
        cid = f"ltm_{i:03d}"
        concepts.append(
            _concept_from(
                pool.x[pool.labels == name],
                "LTM",
                provenance="initial",
                concept_id=cid,
                rng=make_rng(SEED, f"t5/store/reservoir/{cid}"),
                k_max=config.K_max_refset,
                window_W=config.window_W,
                alpha_ema=config.alpha_stm_ema,
            )
        )
    prior = compute_global_prior(concepts, config)
    for c in concepts:
        recompute_thresholds(c, config, prior)
    return ConceptStore(config, prior, concepts, vectorized=vectorized)


def _mixed_queries(n_known: int, n_novel: int, n_distract: int, stream: str) -> np.ndarray:
    """Deterministic shuffled mix of known/novel/distractor embeddings."""
    parts = [_WORLD.sample_class(n, n_known, stream=stream) for n in _WORLD.known_names]
    parts += [_WORLD.sample_class(n, n_novel, stream=stream) for n in _WORLD.novel_names]
    parts += [_WORLD.distractor_point(i)[None, :] for i in range(n_distract)]
    q = np.vstack(parts)
    perm = make_rng(SEED, f"{stream}/perm").permutation(len(q))
    return q[perm]


# ------------------------------------------------------- tier order (FR-3.3)


def test_routing_tier_order():
    """A z accepted by both a mature-STM and a better-margin immature-STM
    concept must go to the mature one: immature candidates cannot claim
    traffic away from tier 1 (FR-3.3)."""
    config = FPCMCConfig()  # n_mature=5
    prior = GlobalPrior(tau=0.5, tau_vmf=0.0)

    # Both concepts sit on the same axis; the immature one has a far looser
    # tau, hence the better normalized margin for any nearby query.
    mature = _concept_from(
        _cluster(0, 5, "tier_order/mature"), "STM",
        match_count=config.n_mature, tau=0.30, concept_id="stm_0000",
    )
    immature = _concept_from(
        _cluster(0, 5, "tier_order/immature"), "STM",
        match_count=0, tau=1.50, concept_id="stm_0001",
    )
    z = _cluster(0, 1, "tier_order/query")[0]

    # Premise: both accept, and the immature concept's margin is strictly better.
    scorer = make_scorer(config)
    assert scorer.accepts(z, mature) and scorer.accepts(z, immature)
    assert scorer.margin(z, immature) > scorer.margin(z, mature)

    store = ConceptStore(config, prior, [mature, immature])
    r = store.route(z, step=7)

    assert r.tier == 1
    assert r.concept_id == "stm_0000"
    assert r.prediction == "stm_0000"
    assert mature.match_count == config.n_mature + 1
    assert immature.match_count == 0, "immature STM claimed traffic from tier 1"


# ------------------------------------------------------------ tier 2 (FR-9)


def test_routing_tier2():
    """z rejected by all tier-1 concepts but accepted by one immature STM
    candidate: assigned there, prediction 'unknown'."""
    config = FPCMCConfig()
    prior = GlobalPrior(tau=0.5, tau_vmf=0.0)

    ltm = _concept_from(
        _cluster(0, 8, "tier2/ltm"), "LTM",
        provenance="initial", tau=0.30, concept_id="ltm_000",
    )
    immature = _concept_from(
        _cluster(1, 5, "tier2/immature"), "STM",
        match_count=0, tau=0.30, concept_id="stm_0000",
    )
    z = _cluster(1, 1, "tier2/query")[0]

    store = ConceptStore(config, prior, [ltm, immature])
    r = store.route(z, step=12)

    assert r.tier == 2
    assert r.prediction == "unknown"
    assert r.concept_id == "stm_0000"
    assert immature.match_count == 1
    assert immature.last_matched_at == 12
    assert ltm.match_count == 0
    assert len(store.concepts) == 2, "an accepted embedding must not seed"


# ------------------------------------------------------------ seeding (FR-3.2)


def test_routing_seeds():
    """z rejected everywhere: a new STM singleton exists with ref_set=[z] and
    both taus bootstrapped from the global prior pair."""
    config = FPCMCConfig()
    prior = GlobalPrior(tau=0.42, tau_vmf=-3.5)
    ltm = _concept_from(
        _cluster(0, 8, "seeds/ltm"), "LTM",
        provenance="initial", tau=0.30, concept_id="ltm_000",
    )
    store = ConceptStore(config, prior, [ltm])

    z = _cluster(1, 1, "seeds/query")[0]
    r = store.route(z, step=33)

    assert r.tier == 3
    assert r.prediction == "unknown"
    assert r.concept_id == "stm_0000"
    assert math.isnan(r.score) and math.isnan(r.margin)
    assert r.via is None and r.fallback is False

    (seeded,) = store.stm
    assert seeded.concept_id == "stm_0000"
    assert seeded.status == "STM"
    assert seeded.provenance == "seeded"
    assert seeded.ref_set.shape == (1, z.shape[0])
    assert np.array_equal(seeded.ref_set[0], z)
    assert np.array_equal(seeded.centroid, z)
    assert seeded.tau == prior.tau
    assert seeded.tau_vmf == prior.tau_vmf, "seed must get both priors (owner decision)"
    assert seeded.match_count == 0
    assert seeded.created_at == 33 and seeded.last_matched_at == 33
    # Concept plumbing comes from the config (approved T3 decision 3).
    assert seeded.k_max == config.K_max_refset
    assert seeded.window_W == config.window_W
    assert seeded.alpha_ema == config.alpha_stm_ema

    # A second novel direction seeds the next id (store-owned allocation).
    z2 = _cluster(2, 1, "seeds/query2")[0]
    r2 = store.route(z2, step=34)
    assert r2.tier == 3
    assert r2.concept_id == "stm_0001"


# -------------------------------------- promotion-aware routing (FR-5.4/FR-7.1)


def test_promoted_participates_immediately():
    """Flip a concept to LTM via the T4 promotion hook + manual status flip:
    the very next route call must accept a near sample at tier 1. This is the
    promotion-aware-routing invariant, tested before the full loop exists."""
    config = FPCMCConfig()
    world = VMFWorld(seed=19, k_known=1, k_novel=1)

    t0 = world.t0_pool(n_per_class=32)
    ltm = _concept_from(t0.x, "LTM", provenance="initial", concept_id="ltm_000")
    prior = compute_global_prior([ltm], config)
    recompute_thresholds(ltm, config, prior)

    # A candidate around the novel class, big enough for the native vmf branch
    # (16 >= n_vmf_min) and for the promotion recompute; immature (0 matches).
    cand = _concept_from(
        world.sample_class("novel_00", 16, stream="t5/promote/refs"),
        "STM", provenance="seeded", concept_id="stm_0000",
    )
    recompute_thresholds(cand, config, prior)  # FR-5.2 shrunk STM taus

    store = ConceptStore(config, prior, [ltm, cand])
    z0, z1 = world.sample_class("novel_00", 2, stream="t5/promote/queries")

    # Pre-promotion: the immature candidate can only match at tier 2.
    r0 = store.route(z0, step=0)
    assert r0.tier == 2
    assert r0.concept_id == "stm_0000"
    assert r0.prediction == "unknown"

    # Promotion = FR-5.4 recompute + status flip (T8 wires this atomically).
    recompute_on_promotion(cand, config)
    cand.status = "LTM"

    r1 = store.route(z1, step=1)
    assert r1.tier == 1, "promoted concept must participate in tier 1 immediately"
    assert r1.concept_id == "stm_0000"
    assert r1.prediction == "stm_0000"


# ------------------------------------------------------- bookkeeping (FR-1/FR-9)


def test_route_updates_bookkeeping():
    """An assignment updates match_count, last_matched_at, reservoir and
    windows on exactly one concept; everything else is untouched."""
    config = FPCMCConfig()  # window_W=250
    prior = GlobalPrior(tau=0.5, tau_vmf=0.0)

    a = _concept_from(
        _cluster(0, 8, "bookkeeping/a"), "LTM",
        provenance="initial", tau=0.30, concept_id="ltm_000",
    )
    b = _concept_from(
        _cluster(1, 7, "bookkeeping/b"), "STM",
        match_count=7, match_windows={0}, last_matched_at=3,
        tau=0.30, concept_id="stm_0000",
    )
    store = ConceptStore(config, prior, [a, b])

    b_ref_before = b.ref_set.copy()
    b_centroid_before = b.centroid.copy()
    a_centroid_before = a.centroid.copy()

    z = _cluster(0, 1, "bookkeeping/query")[0]
    r = store.route(z, step=260)  # window 260 // 250 = 1

    assert r.tier == 1 and r.concept_id == "ltm_000"

    # Exactly one concept absorbed the observation.
    assert a.match_count == 1
    assert a.ref_count_seen == 9
    assert a.last_matched_at == 260
    assert a.match_windows == {1}
    assert a.ref_set.shape[0] == 9, "below k_max the reservoir must append"
    assert np.array_equal(a.ref_set[-1], z)
    assert np.array_equal(a.centroid, a_centroid_before), "LTM centroid is frozen"

    # The other concept is bitwise untouched.
    assert b.match_count == 7
    assert b.ref_count_seen == 7
    assert b.last_matched_at == 3
    assert b.match_windows == {0}
    assert np.array_equal(b.ref_set, b_ref_before)
    assert np.array_equal(b.centroid, b_centroid_before)

    assert len(store.concepts) == 2, "an accepted embedding must not seed"


# --------------------------------------------------- batch-path guard (NFR-1)


@pytest.mark.parametrize("scorer_name", ["knn_ref", "vmf", "knn_vmf"])
def test_vectorized_matches_loop(scorer_name):
    """The batch scoring path is identical — RoutingResult by RoutingResult
    and in final store state — to the naive per-concept Scorer.select loop,
    over 200 mixed queries that exercise all three tiers."""
    config = FPCMCConfig(scorer=scorer_name)
    fast = _fixture_store(config, vectorized=True)
    slow = _fixture_store(config, vectorized=False)

    queries = _mixed_queries(n_known=40, n_novel=30, n_distract=20, stream="t5/vec")
    assert len(queries) == 200

    tiers = set()
    for step, z in enumerate(queries):
        r_fast = fast.route(z, step)
        r_slow = slow.route(z, step)
        _assert_results_equal(r_fast, r_slow)
        tiers.add(r_fast.tier)

    assert tiers == {1, 2, 3}, f"query mix must exercise every tier, saw {tiers}"
    _assert_stores_equal(fast, slow)

    # Exact-tie semantics: twin concepts over the same ref_set produce
    # bitwise-identical margins under every scorer, so the batch path must
    # reproduce Scorer.select's tie-break to the lexicographically smallest
    # concept_id (random queries never hit an exact cross-concept tie).
    twin_ref = _WORLD.sample_class("known_00", 12, stream="t5/vec/tie")

    def _tie_store(vectorized: bool) -> ConceptStore:
        twins = [
            _concept_from(
                twin_ref, "STM", match_count=9, tau=0.6, tau_vmf=1e6, concept_id=cid
            )
            for cid in ("stm_0400", "stm_0007")  # registration order != id order
        ]
        return ConceptStore(
            config, GlobalPrior(tau=0.6, tau_vmf=1e6), twins, vectorized=vectorized
        )

    zq = _WORLD.sample_class("known_00", 1, stream="t5/vec/tie/query")[0]
    r_fast = _tie_store(True).route(zq, step=0)
    r_slow = _tie_store(False).route(zq, step=0)
    _assert_results_equal(r_fast, r_slow)
    assert r_fast.tier == 1
    assert r_fast.concept_id == "stm_0007", "exact tie must break lexicographically"


# ----------------------------------------------------- determinism (FR-9.2)


def test_routing_determinism():
    """Two full replays of 500 fixture queries through identically built
    stores produce identical RoutingResult sequences and final state."""
    config = FPCMCConfig()

    def _run() -> tuple[list[RoutingResult], ConceptStore]:
        store = _fixture_store(config, vectorized=True)
        queries = _mixed_queries(n_known=105, n_novel=80, n_distract=25, stream="t5/det")
        assert len(queries) == 500
        return [store.route(z, step) for step, z in enumerate(queries)], store

    results_1, store_1 = _run()
    results_2, store_2 = _run()

    for r1, r2 in zip(results_1, results_2, strict=True):
        _assert_results_equal(r1, r2)
    _assert_stores_equal(store_1, store_2)
