"""T8 [U] tests — promotion (TASKS Task 8; PRD FR-7.1–7.2, FR-5.4).

Owner-approved T8 semantics (Q&A 2026-07-11, recorded in docs/CHANGES.md T8):
  - Cadence: per-assignment — the runner calls `check(concept, store, step)`
    after every tier-1/2 assignment (FR-7.1 "immediate"; no new config key);
    `evaluate(store, step)` is the full sweep for hooks and tests.
  - Separation (criterion 3): the candidate's centroid must be rejected by
    every LTM concept under the configured scorer with sep_factor-scaled
    thresholds — under knn_vmf BOTH sub-scorers must reject.
  - Sweeps are live, ascending concept_id: a promotion takes effect for the
    separation checks of candidates evaluated later in the same pass.
  - Decisions are returned (all four criteria always evaluated, failures
    named); only successful promotions persist, on evaluator.promotion_log;
    the FR-7.2 eval-only fields ride as placeholders (gt_majority_label from
    the concept — None at runtime; purity None, filled by T13).

The unit-test world: 4 known + 3 novel + 1 burst class, exactly orthogonal
(separation_deg=90) at kappa 150/150/500 in D=32. Single-class candidates cohere
at ~0.81 mean pairwise cosine similarity; an m-lobe blob of orthogonal classes
coheres at ~0.81/m. The FR-7 cohesion bar is RELATIVE since 2026-07-13
(min_cohesion_ratio 0.35 x the median cohesion of the T0 LTM concepts, ~0.81
here => ~0.284), so the four-lobe blocking candidate (~0.20) fails it with room
to spare. Novel-class centroids are rejected by every LTM concept, while a
known-class clone's centroid is accepted by its LTM twin.

test_outlier_burst_never_promotes runs the frozen golden stream (T1) through
2,000 steps with stm_capacity=10 (the golden-run config must use <= ~25 so
the 25 planted distractors create real LRU pressure — CLAUDE.md).
"""

import numpy as np
import pytest

from fpcmc.concepts import Concept, ConceptStore
from fpcmc.config import FPCMCConfig
from fpcmc.init import initialize_ltm
from fpcmc.memory import PromotionEvaluator, PromotionRecord, cohesion
from fpcmc.rng import make_rng
from fpcmc.scorers import estimate_kappa
from fpcmc.thresholds import (
    GlobalPrior,
    compute_global_prior,
    recompute_on_promotion,
    recompute_thresholds,
)
from tests.fixtures.golden_stream import load_golden
from tests.fixtures.vmf_world import VMFWorld

SEED = 801


# ------------------------------------------------------------------- helpers


def _ltm_world(config: FPCMCConfig) -> tuple[VMFWorld, ConceptStore, GlobalPrior]:
    """A 4-known-class LTM store via the production T6 init, plus the world
    and the (identical) frozen prior for hand-building candidates."""
    world = VMFWorld(seed=SEED, k_known=4, k_novel=3, n_burst=1, separation_deg=90.0)
    pool = world.t0_pool(n_per_class=100)
    store = initialize_ltm(pool.x, pool.labels, config)
    # Pure function of the untouched T0 concepts => equals the store's prior.
    prior = compute_global_prior(store.ltm, config)
    return world, store, prior


def _candidate(
    x: np.ndarray,
    concept_id: str,
    config: FPCMCConfig,
    prior: GlobalPrior,
    *,
    match_count: int,
    windows: set[int],
) -> Concept:
    """A hand-built mature STM candidate with crafted FR-7 statistics and
    realistic shrunk STM thresholds (FR-5.2 via recompute_thresholds)."""
    centroid = x.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    c = Concept(
        concept_id=concept_id,
        centroid=centroid,
        ref_set=np.array(x, dtype=np.float64),
        tau=float("nan"),
        kappa=estimate_kappa(x),
        status="STM",
        match_count=match_count,
        match_windows=set(windows),
        created_at=0,
        last_matched_at=0,
        provenance="seeded",
        window_W=config.window_W,
        k_max=config.K_max_refset,
        alpha_ema=config.alpha_stm_ema,
        rng=make_rng(SEED, f"t8/reservoir/{concept_id}"),
    )
    recompute_thresholds(c, config, prior)
    return c


# --------------------------------------------- each criterion blocks (FR-7)


@pytest.mark.parametrize("criterion", ["size", "cohesion", "separation", "recurrence"])
def test_each_criterion_blocks(criterion):
    """A candidate passing exactly three criteria and failing one is not
    promoted, and the decision names exactly the failing criterion."""
    config = FPCMCConfig()
    world, store, prior = _ltm_world(config)

    if criterion == "size":
        # theta - 1 = 29 matches; everything else passing.
        x = world.sample_class("novel_00", 30, stream="t8/blocks/size")
        cand = _candidate(
            x, "stm_0000", config, prior,
            match_count=config.theta_promote - 1, windows={0, 1, 2},
        )
    elif criterion == "cohesion":
        # Four-lobe candidate: quarters from four mutually orthogonal non-known
        # classes. An m-lobe blob of orthogonal classes coheres at ~within/m, so
        # this lands at ~0.20 against a bar of ~0.284 — a ~29% margin.
        #
        # Was a TWO-lobe candidate under the retired absolute `min_cohesion=0.55`
        # (2026-07-13: FR-7 criterion 2 became RELATIVE — min_cohesion_ratio 0.35
        # x the median cohesion of the T0 LTM concepts, ~0.81 here). Two lobes
        # cohere at ~0.40 and now CLEAR the bar, so the fixture — not the
        # assertion — had to move. Three lobes (~0.26) would fail by only 7%,
        # too thin to trust. Four lobes is both robust and the truer shape of what
        # this criterion exists to catch: the real blob is MANY classes with few
        # members each. A clean 2-class blob can only arise from a cross-class
        # merge, which the FR-6/FR-8.2 merge guards now prevent outright.
        #
        # Every other criterion still passes (size 40 >= 30; 3 windows; a centroid
        # averaged over four orthogonal non-known directions is rejected by every
        # LTM), so a promotion here would be a cohesion failure and nothing else.
        x = np.vstack([
            world.sample_class("novel_00", 10, stream="t8/blocks/cohesion"),
            world.sample_class("novel_01", 10, stream="t8/blocks/cohesion"),
            world.sample_class("novel_02", 10, stream="t8/blocks/cohesion"),
            world.sample_class("burst_00", 10, stream="t8/blocks/cohesion"),
        ])
        cand = _candidate(x, "stm_0000", config, prior, match_count=40, windows={0, 1, 2})
    elif criterion == "separation":
        # Seeded inside a known LTM class: its centroid is accepted by that
        # LTM concept's taus, so it is not separated.
        x = world.sample_class("known_00", 40, stream="t8/blocks/separation")
        cand = _candidate(x, "stm_0000", config, prior, match_count=40, windows={0, 1, 2})
    else:  # recurrence
        # 40 matches all within one window (the outlier-burst shape).
        x = world.sample_class("burst_00", 40, stream="t8/blocks/recurrence")
        cand = _candidate(x, "stm_0000", config, prior, match_count=40, windows={2})

    store.register(cand)
    evaluator = PromotionEvaluator(config)
    decisions = evaluator.evaluate(store, step=1_000)

    assert len(decisions) == 1
    d = decisions[0]
    assert d.concept_id == "stm_0000"
    assert not d.promoted
    assert d.failed == (criterion,), (
        f"exactly {criterion!r} must fail (the other three pass), got {d.failed}"
    )
    assert evaluator.promotion_log == []
    assert cand.status == "STM"
    assert cand.provenance == "seeded"


# ----------------------------------------------------- happy path (FR-7.1/7.2)


def test_promotion_happy_path():
    """A recurring novel-class candidate is promoted atomically in a single
    hook call: status LTM, provenance flip, FR-5.4 tau recompute, STM slot
    released, centroid frozen thereafter, complete FR-7.2 log record."""
    config = FPCMCConfig()
    world, store, prior = _ltm_world(config)

    x = world.sample_class("novel_00", 36, stream="t8/happy")
    cand = _candidate(x, "stm_0000", config, prior, match_count=35, windows={1, 2, 3})
    store.register(cand)
    tau_pre, tau_vmf_pre = cand.tau, cand.tau_vmf
    # Expected FR-5.4 taus: an identical unregistered twin through the T4 hook.
    twin = _candidate(x, "stm_0999", config, prior, match_count=35, windows={1, 2, 3})
    recompute_on_promotion(twin, config)

    n_stm, n_ltm = len(store.stm), len(store.ltm)
    evaluator = PromotionEvaluator(config)
    decisions = evaluator.evaluate(store, step=990)  # the single hook call

    assert [d.promoted for d in decisions] == [True]
    assert decisions[0].failed == ()
    assert cand.status == "LTM"
    assert cand.provenance == "promoted"
    # FR-5.4: both taus equal the pure FR-5.1 recompute — not the shrunk
    # STM values they had before promotion.
    assert cand.tau == twin.tau
    assert cand.tau_vmf == twin.tau_vmf
    assert cand.tau != tau_pre
    assert cand.tau_vmf != tau_vmf_pre
    assert cand.refset_changes_since_tau == 0
    # STM capacity accounting released, LTM grown (live views).
    assert len(store.stm) == n_stm - 1
    assert len(store.ltm) == n_ltm + 1

    # FR-7.2 log record, complete (asserted before mutating the concept).
    assert len(evaluator.promotion_log) == 1
    rec = evaluator.promotion_log[0]
    assert isinstance(rec, PromotionRecord)
    assert rec.step == 990
    assert rec.concept_id == "stm_0000"
    assert rec.size == 35 == decisions[0].size
    assert rec.cohesion == decisions[0].cohesion == cohesion(cand.ref_set)
    # FR-7 criterion 2 is RELATIVE (amended 2026-07-13): the bar is
    # min_cohesion_ratio x median cohesion of the T0 LTM concepts, not a constant.
    assert rec.cohesion >= evaluator.cohesion_bar(store)
    assert rec.separation_margin == decisions[0].separation_margin
    assert rec.separation_margin > 0.0
    assert rec.window_count == 3 == decisions[0].window_count
    assert rec.gt_majority_label is None
    assert rec.purity is None

    # Centroid frozen from the flip onward (FR-1.3 via status).
    frozen = cand.centroid.tobytes()
    cand.add_observation(world.sample_class("novel_00", 1, stream="t8/happy/post")[0], step=995)
    assert cand.centroid.tobytes() == frozen


# ------------------------------------------- separation reads per-concept tau


def test_separation_uses_per_concept_tau():
    """Criterion 3 reads the individual LTM concepts' taus, not a global one:
    the same candidate flips between blocked and promoted when only the
    nearest LTM concept's thresholds are tightened/loosened."""
    config = FPCMCConfig()

    # (a) A known-class clone blocks under the nearest concept's own taus;
    # tightening ONLY that concept's taus makes the same candidate promote.
    world, store, prior = _ltm_world(config)
    x = world.sample_class("known_00", 40, stream="t8/pertau/dup")
    cand = _candidate(x, "stm_0000", config, prior, match_count=40, windows={0, 1, 2})
    store.register(cand)
    evaluator = PromotionEvaluator(config)
    d = evaluator.evaluate(store, step=900)[0]
    assert not d.promoted
    assert d.failed == ("separation",)

    nearest = max(store.ltm, key=lambda c: float(c.centroid @ cand.centroid))
    nearest.tau = -1.0        # knn scores are >= 0: rejects everything
    nearest.tau_vmf = -1e12   # vmf scores are finite: rejects everything
    d2 = evaluator.evaluate(store, step=901)[0]
    assert d2.promoted
    assert d2.failed == ()

    # (b) A separated novel candidate blocks when ONLY one LTM concept's knn
    # tau is loosened to accept everything, and promotes once it is restored.
    world2, store2, prior2 = _ltm_world(config)
    y = world2.sample_class("novel_00", 40, stream="t8/pertau/novel")
    cand2 = _candidate(y, "stm_0000", config, prior2, match_count=40, windows={0, 1, 2})
    store2.register(cand2)
    evaluator2 = PromotionEvaluator(config)

    loose = max(store2.ltm, key=lambda c: float(c.centroid @ cand2.centroid))
    tau_orig = loose.tau
    loose.tau = 2.0  # max cosine distance is 2: accepts every unit vector
    d3 = evaluator2.evaluate(store2, step=900)[0]
    assert not d3.promoted
    assert d3.failed == ("separation",)

    loose.tau = tau_orig
    d4 = evaluator2.evaluate(store2, step=901)[0]
    assert d4.promoted
    assert d4.failed == ()


# ------------------------------------------------- burst discrimination (G)


def test_outlier_burst_never_promotes():
    """The golden-world burst class through 2,000 steps: never promoted,
    eventually LRU-evicted. This is the recurring-novelty-vs-outlier
    discrimination test — recurring novel classes DO promote on the same
    stream, the one-shot burst does not (fails size and recurrence) and is
    forgotten under the distractor-driven LRU pressure."""
    g = load_golden()
    config = FPCMCConfig(stm_capacity=10)  # golden-run config: <= ~25 (CLAUDE.md)
    store = initialize_ltm(g["t0_x"], g["t0_labels"], config)
    evaluator = PromotionEvaluator(config)

    burst_ids: set[str] = set()
    for step, (z, label) in enumerate(zip(g["stream_x"], g["stream_labels"])):
        r = store.route(z, step)
        if label == "burst_00":
            burst_ids.add(r.concept_id)
        if r.tier in (1, 2):  # owner cadence decision: check after every assignment
            evaluator.check(store.get(r.concept_id), store, step)

    assert burst_ids, "burst samples must have routed somewhere"
    assert not any(cid.startswith("ltm_") for cid in burst_ids), (
        "burst samples must never land in a T0 LTM concept"
    )

    promoted_ids = {rec.concept_id for rec in evaluator.promotion_log}
    assert not (burst_ids & promoted_ids), (
        f"the burst class must never promote; burst={sorted(burst_ids)} "
        f"promoted={sorted(promoted_ids)}"
    )
    evicted_ids = {rec.concept_id for rec in store.eviction_log}
    assert burst_ids & evicted_ids, (
        f"the burst candidate must eventually be LRU-evicted; "
        f"burst={sorted(burst_ids)} evicted={sorted(evicted_ids)}"
    )
    for cid in burst_ids & evicted_ids:
        assert cid not in store, "evicted burst candidates must be gone from the registry"

    # The discrimination has to be real: recurring novelty promoted on the
    # very same stream and cadence.
    assert promoted_ids, "no promotion at all — the burst assertion would be vacuous"


# --------------------------------------------------------------- idempotence


def test_promotion_idempotent():
    """The evaluator is a no-op on an already-promoted concept: neither the
    sweep nor the per-assignment check touches it again."""
    config = FPCMCConfig()
    world, store, prior = _ltm_world(config)
    x = world.sample_class("novel_01", 36, stream="t8/idem")
    cand = _candidate(x, "stm_0000", config, prior, match_count=40, windows={0, 1, 2})
    store.register(cand)
    evaluator = PromotionEvaluator(config)
    assert [d.promoted for d in evaluator.evaluate(store, step=500)] == [True]

    snap = (cand.tau, cand.tau_vmf, cand.centroid.tobytes(), cand.status, cand.provenance)
    assert evaluator.evaluate(store, step=501) == [], (
        "a promoted concept is no longer a candidate for the sweep"
    )
    assert evaluator.check(cand, store, step=501) is None, (
        "the per-assignment check must no-op on a promoted concept"
    )
    assert len(evaluator.promotion_log) == 1
    assert (cand.tau, cand.tau_vmf, cand.centroid.tobytes(), cand.status, cand.provenance) == snap


# --------------------------------------------- live same-sweep LTM visibility


def test_live_sweep_blocks_same_class_fragment():
    """Owner decision (live, ascending id): with two promotable fragments of
    the same novel class in one sweep, the smaller id promotes and the other
    is blocked by separation against the freshly promoted concept."""
    config = FPCMCConfig()
    world, store, prior = _ltm_world(config)
    a = world.sample_class("novel_02", 36, stream="t8/frag/a")
    b = world.sample_class("novel_02", 36, stream="t8/frag/b")
    frag_a = _candidate(a, "stm_0000", config, prior, match_count=35, windows={0, 1, 2})
    frag_b = _candidate(b, "stm_0001", config, prior, match_count=34, windows={0, 1, 3})

    # Premise: fragment b alone (no fragment a anywhere) would promote.
    world_p, store_p, prior_p = _ltm_world(config)
    solo_b = _candidate(b, "stm_0001", config, prior_p, match_count=34, windows={0, 1, 3})
    store_p.register(solo_b)
    assert [d.promoted for d in PromotionEvaluator(config).evaluate(store_p, step=980)] == [True]

    store.register(frag_a)
    store.register(frag_b)
    evaluator = PromotionEvaluator(config)
    decisions = evaluator.evaluate(store, step=980)

    assert [d.concept_id for d in decisions] == ["stm_0000", "stm_0001"]
    assert decisions[0].promoted
    assert not decisions[1].promoted
    assert decisions[1].failed == ("separation",), (
        "the second fragment must be blocked by the newly promoted first"
    )
    assert frag_a.status == "LTM"
    assert frag_b.status == "STM"
    assert [rec.concept_id for rec in evaluator.promotion_log] == ["stm_0000"]
