"""T11 tests — stream runner, event log, periodic hooks, golden gate
(TASKS Task 11; PRD FR-9, NFR-1, NFR-3).

T11 decisions (2026-07-11, agent recommendations under the owner's
proceed-on-recommendations instruction; recorded in docs/CHANGES.md T11):

  25. Runner surface: StreamRunner(config, store, prior, log_path=,
      checkpoint_steps=()) — the prior rides alongside the store (the T9
      plumbing note; invariant 5 keeps it out of the store's public surface);
      promoter/sweeper/residual are public attributes sharing ONE MergeSweeper.
  26. Wake-loop order per step: route -> drain eviction records (they precede
      the seed's registration inside _seed, so the log stays in mutation
      order) -> assign/seed record (+ note_seed at tier 3) -> per-assignment
      promotion check (T8 decision 12; promote record + sweeper.on_promotion
      on success) -> residual.hook (self-scheduled on T_cluster) -> merge
      sweep at step % T_merge == 0 (step > 0) -> checkpoint record if
      step in checkpoint_steps. Merge records drain from the shared
      merge_log after every mutating sub-step.
  27. Log format: strict JSON lines (sort_keys, compact separators,
      allow_nan=False; non-finite floats -> null), exactly the seven TASKS
      record types; residual-driven merges appear as kind="residual" merge
      records. config_header embeds the resolved config (NFR-2), n_steps,
      checkpoint_steps and a schema version.
  28. Replay = event application, never re-scoring: the log is authoritative
      for every decision; mutations re-apply with the same deterministic
      substreams (per-concept reservoir, per-merge bounded union), and a
      replayed merge whose recomputed survivor disagrees with the logged
      survivor raises ReplayError. MergeSweeper gains the additive public
      fold_pair seam so stm_ltm records replay through T9 code.

The [G] golden gate consumes the agent-proposed frozen configs/golden_run.yaml
(stm_capacity=16; T_merge/T_cluster/w_residual=250 — one hook opportunity per
window; see that file's provenance header) — chosen before the gate first ran,
never tuned against gate results. Ground truth (fixture labels) is used
eval-side in this test only, per PRD §7.2; distractor examples enter no metric
denominators (golden-stream freeze note).
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from fpcmc.concepts import ConceptStore
from fpcmc.config import FPCMCConfig, UmapConfig
from fpcmc.data import embeddings_available, load_all_pools
from fpcmc.init import initialize_ltm
from fpcmc.memory import MergeSweeper, PromotionEvaluator
from fpcmc.residual import ResidualClusterer
from fpcmc.replay import ReplayError, read_log, replay
from fpcmc.rng import make_rng
from fpcmc.stream import LOG_SCHEMA_VERSION, StreamRunner
from fpcmc.thresholds import compute_global_prior
from tests.fixtures.golden_stream import load_golden
from tests.fixtures.vmf_world import Segment, VMFWorld

AVAILABLE, REASON = embeddings_available()

SEED = 1101


# ------------------------------------------------------------------- fixtures


def _runner_config(**kw) -> FPCMCConfig:
    """[U] runner config: small capacities/cadences sized to the 600-step
    fixture stream; umap.dim=200 keeps every consolidation on the
    HDBSCAN-only branch (input N <= 200), so no [U] test pays for UMAP."""
    base = dict(
        stm_capacity=8,
        n_mature=3,
        theta_promote=10,
        m_windows=2,
        window_W=100,
        T_merge=200,
        T_cluster=200,
        w_residual=100,
        umap=UmapConfig(dim=200),
        seed=42,
    )
    base.update(kw)
    return FPCMCConfig(**base)


# novel_00 tight (kappa=600: its first singleton captures the class, so it
# promotes and exercises promote/on-promotion records); novel_01/novel_02
# loose (kappa=150: at D=32 the prior tau is tighter than typical
# intra-class distance, so they fragment — feeding evictions, the residual
# pool, and residual-driven merges). Two loose classes matter: HDBSCAN with
# allow_single_cluster=False (the lib default) cannot return a lone blob,
# so a single-class pool clusters to nothing.
_WORLD = VMFWorld(
    seed=SEED,
    k_known=4,
    k_novel=3,
    separation_deg=75.0,
    kappa_novel=(600.0, 150.0, 150.0),
)


def _fixture_stream():
    """600 steps / six 100-step windows: knowns throughout, three recurring
    novel classes, and 35 one-off distractors (30 in window 3 for LRU/pool
    pressure at capacity 8, 5 later)."""
    known = _WORLD.known_names

    def kn(n):
        base, rem = divmod(n, len(known))
        return {name: base + (1 if i < rem else 0) for i, name in enumerate(known)}

    schedule = [
        Segment(counts=kn(100)),
        Segment(counts={**kn(70), "novel_00": 30}),
        Segment(counts={**kn(50), "novel_00": 25, "novel_01": 25}),
        Segment(
            counts={**kn(25), "novel_01": 25, "novel_02": 20},
            distractors=tuple(range(30)),
        ),
        Segment(
            counts={**kn(45), "novel_00": 25, "novel_02": 25},
            distractors=tuple(range(30, 35)),
        ),
        Segment(counts={**kn(50), "novel_01": 25, "novel_02": 25}),
    ]
    stream = _WORLD.make_stream(schedule)
    assert stream.x.shape[0] == 600
    return stream


def _fixture_run(config: FPCMCConfig, log_path: Path, checkpoint_steps=(299, 599)):
    stream = _fixture_stream()
    pool = _WORLD.t0_pool(n_per_class=100)
    store = initialize_ltm(pool.x, pool.labels, config)
    prior = compute_global_prior(store.ltm, config)
    runner = StreamRunner(
        config, store, prior, log_path=log_path, checkpoint_steps=checkpoint_steps
    )
    runner.run(stream.x)
    return stream, runner


# -------------------------------------------------------------- hook schedule


def test_hook_schedule(mocker):
    """TASKS T11: spies confirm each hook fires at exactly the configured
    steps over a 2,000-step run (the frozen golden stream)."""
    g = load_golden()
    config = FPCMCConfig.from_yaml("configs/golden_run.yaml")
    store = initialize_ltm(g["t0_x"], g["t0_labels"], config)
    prior = compute_global_prior(store.ltm, config)

    residual_hook = mocker.spy(ResidualClusterer, "hook")
    sweep = mocker.spy(MergeSweeper, "sweep")
    check = mocker.spy(PromotionEvaluator, "check")
    # Schedule test, not clustering correctness: skip the UMAP cost.
    mocker.patch.object(ResidualClusterer, "_cluster", return_value=[])

    checkpoints = tuple(range(249, 2000, 250))
    log_path = Path("/tmp/does-not-matter")  # replaced below with tmp handling

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "golden.jsonl"
        runner = StreamRunner(
            config, store, prior, log_path=log_path, checkpoint_steps=checkpoints
        )
        runner.run(g["stream_x"])
        records = read_log(log_path)

    n_steps = g["stream_x"].shape[0]
    assert [c.args[2] for c in residual_hook.call_args_list] == list(range(n_steps)), (
        "residual hook must run at every step (it self-schedules on T_cluster)"
    )
    assert [c.args[2] for c in sweep.call_args_list] == [
        s for s in range(1, n_steps) if s % config.T_merge == 0
    ], "merge sweep fires exactly at positive T_merge multiples"
    n_assigns = sum(1 for r in records if r["type"] == "assign")
    assert check.call_count == n_assigns, (
        "promotion check runs after every tier-1/2 assignment (T8 decision 12)"
    )
    assert [r["step"] for r in records if r["type"] == "checkpoint"] == list(checkpoints)


# ---------------------------------------------------------------- log schema


# Schema v2 (T13, owner-approved 2026-07-13): assign/seed gained "novelty"
# (the min tier-1 scorer scalar — §7.3 streaming detection statistic) and
# checkpoint gained "taus" (per-concept {status, tau, tau_vmf} snapshot for
# the τ-distribution/threshold-health metrics). Additive only; no field was
# removed or weakened.
_REQUIRED_KEYS = {
    "config_header": {"type", "schema", "config", "n_steps", "checkpoint_steps"},
    "assign": {"type", "step", "concept_id", "prediction", "tier", "score",
               "margin", "via", "fallback", "novelty"},
    "seed": {"type", "step", "concept_id", "novelty"},
    "evict": {"type", "step", "concept_id", "size", "age", "created_at",
              "last_matched_at", "ref_count_seen"},
    "promote": {"type", "step", "concept_id", "size", "cohesion",
                "separation_margin", "window_count", "gt_majority_label", "purity"},
    "merge": {"type", "step", "kind", "survivor_id", "absorbed_id",
              "centroid_sim", "cross_within_ratio", "survivor_match_count",
              "absorbed_match_count"},
    "checkpoint": {"type", "step", "n_ltm", "n_stm", "n_concepts", "n_evictions",
                   "n_promotions", "n_merges", "residual_pool_size", "taus"},
}


def test_log_schema_complete(tmp_path, mocker):
    """TASKS T11: every record validates against the schema; every store
    mutation during the run has a corresponding record (spies on the
    mutating methods reconcile against record counts)."""
    config = _runner_config()
    assigned = mocker.spy(ConceptStore, "_assign")
    seeded = mocker.spy(ConceptStore, "_seed")
    evicted = mocker.spy(ConceptStore, "_evict")
    promoted = mocker.spy(PromotionEvaluator, "_promote")
    merged = mocker.spy(MergeSweeper, "merge_pair")
    folded = mocker.spy(MergeSweeper, "_fold")

    log_path = tmp_path / "run.jsonl"
    _, runner = _fixture_run(config, log_path)

    # Strict JSON: NaN/Infinity literals must not appear.
    def reject_constants(name):  # pragma: no cover - triggers only on failure
        raise AssertionError(f"non-strict JSON constant in log: {name}")

    lines = log_path.read_text().splitlines()
    records = [json.loads(line, parse_constant=reject_constants) for line in lines]

    assert records[0]["type"] == "config_header"
    assert records[0]["schema"] == LOG_SCHEMA_VERSION
    assert records[0]["config"] == yaml.safe_load(config.to_yaml()), (
        "the header must embed the resolved config (NFR-2)"
    )
    counts: dict[str, int] = {}
    for rec in records:
        rtype = rec["type"]
        counts[rtype] = counts.get(rtype, 0) + 1
        assert rtype in _REQUIRED_KEYS, f"unknown record type {rtype!r}"
        assert set(rec) == _REQUIRED_KEYS[rtype], (
            f"{rtype} keys {sorted(rec)} != {sorted(_REQUIRED_KEYS[rtype])}"
        )
        if rtype != "config_header":
            assert isinstance(rec["step"], int)

    # The fixture must exercise every record type, or this test proves little.
    assert set(counts) == set(_REQUIRED_KEYS), f"missing record types: {counts}"

    # Mutation <-> record reconciliation.
    assert counts["assign"] == assigned.call_count
    assert counts["seed"] == seeded.call_count
    assert counts["evict"] == evicted.call_count == len(runner.store.eviction_log)
    assert counts["promote"] == promoted.call_count == len(runner.promoter.promotion_log)
    assert counts["merge"] == merged.call_count + folded.call_count
    assert counts["merge"] == len(runner.sweeper.merge_log)
    assert counts["assign"] + counts["seed"] == 600
    assert counts["checkpoint"] == 2


# ------------------------------------------------------------ byte determinism


def test_byte_determinism(tmp_path):
    """TASKS T11 / FR-9.2 / NFR-3: same config+seed => byte-identical JSONL;
    seed+1 => different bytes."""
    config = _runner_config(seed=42)
    _fixture_run(config, tmp_path / "a.jsonl")
    _fixture_run(config, tmp_path / "b.jsonl")
    a = (tmp_path / "a.jsonl").read_bytes()
    assert a == (tmp_path / "b.jsonl").read_bytes()

    _fixture_run(_runner_config(seed=43), tmp_path / "c.jsonl")
    assert a != (tmp_path / "c.jsonl").read_bytes()


# --------------------------------------------------------------------- replay


def test_replay_reconstruction(tmp_path):
    """TASKS T11: replayed final state equals live final state (concept ids,
    statuses, match_counts, lineage; centroids atol 1e-9 — asserted exact,
    which is stronger and holds because replay re-applies the same
    deterministic mutations)."""
    config = _runner_config()
    log_path = tmp_path / "run.jsonl"
    stream, runner = _fixture_run(config, log_path)
    live = runner.store

    pool = _WORLD.t0_pool(n_per_class=100)
    fresh = initialize_ltm(pool.x, pool.labels, config)
    prior = compute_global_prior(fresh.ltm, config)
    replayed = replay(log_path, stream.x, fresh, prior)

    live_by_id = {c.concept_id: c for c in live.concepts}
    replay_by_id = {c.concept_id: c for c in replayed.concepts}
    assert set(live_by_id) == set(replay_by_id)
    for cid, c in live_by_id.items():
        r = replay_by_id[cid]
        assert (c.status, c.provenance) == (r.status, r.provenance), cid
        assert c.match_count == r.match_count, cid
        assert c.match_windows == r.match_windows, cid
        assert c.merged_from == r.merged_from, cid
        assert c.ref_count_seen == r.ref_count_seen, cid
        assert c.last_matched_at == r.last_matched_at, cid
        np.testing.assert_allclose(c.centroid, r.centroid, atol=1e-9)
        assert c.centroid.tobytes() == r.centroid.tobytes(), cid
        assert c.ref_set.tobytes() == r.ref_set.tobytes(), cid
        assert (c.tau, c.kappa) == (r.tau, r.kappa), cid
        assert (c.tau_vmf == r.tau_vmf) or (
            math.isnan(c.tau_vmf) and math.isnan(r.tau_vmf)
        ), cid

    # Replay is validating, not trusting: a log whose merge record names the
    # wrong survivor must raise.
    records = read_log(log_path)
    merge_idx = next(i for i, r in enumerate(records) if r["type"] == "merge")
    bad = dict(records[merge_idx])
    bad["survivor_id"], bad["absorbed_id"] = bad["absorbed_id"], bad["survivor_id"]
    records[merge_idx] = bad
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n")
    fresh2 = initialize_ltm(pool.x, pool.labels, config)
    with pytest.raises((ReplayError, KeyError)):
        replay(bad_path, stream.x, fresh2, compute_global_prior(fresh2.ltm, config))


# ------------------------------------------------------------ golden gate [G]


def _lineage_resolver(records) -> dict[str, str]:
    """Map every concept id to its end-of-stream lineage root (survivor
    chains followed transitively; unmerged ids map to themselves)."""
    parent: dict[str, str] = {}
    for rec in records:
        if rec["type"] == "merge":
            parent[rec["absorbed_id"]] = rec["survivor_id"]

    def resolve(cid: str) -> str:
        while cid in parent:
            cid = parent[cid]
        return cid

    return {cid: resolve(cid) for cid in set(parent)}


@pytest.mark.slow
def test_golden_stream_end_to_end(tmp_path):
    """TASKS T11 [G] — the executable specification of the whole system, on
    the frozen golden stream under the frozen configs/golden_run.yaml.

    Ground truth is consumed eval-side only (PRD §7.2); the 25 planted
    distractors are one-off outliers, not novel classes — they enter none of
    the promotion/purity/coverage/unknown-residual denominators (golden
    freeze note, 2026-07-10)."""
    g = load_golden()
    config = FPCMCConfig.from_yaml("configs/golden_run.yaml")
    store = initialize_ltm(g["t0_x"], g["t0_labels"], config)
    prior = compute_global_prior(store.ltm, config)
    log_path = tmp_path / "golden.jsonl"
    runner = StreamRunner(
        config, store, prior,
        log_path=log_path,
        checkpoint_steps=tuple(range(249, 2000, 250)),
    )
    runner.run(g["stream_x"])

    records = read_log(log_path)
    labels = g["stream_labels"]
    novel_classes = ("novel_00", "novel_01", "novel_02")
    known_classes = sorted({l for l in labels if l.startswith("known_")})

    # --- eval-side bookkeeping (GT used for verification only) --------------
    step_concept: dict[int, str] = {}   # arrival-time container per step
    step_tier: dict[int, int] = {}
    for rec in records:
        if rec["type"] in ("assign", "seed"):
            step_concept[rec["step"]] = rec["concept_id"]
            step_tier[rec["step"]] = rec["tier"] if rec["type"] == "assign" else 3
    resolve = _lineage_resolver(records)

    def root(cid: str) -> str:
        return resolve.get(cid, cid)

    # Majority GT label per end-of-stream concept, over lineage-resolved
    # arrivals (distractor examples excluded everywhere).
    members: dict[str, list[str]] = {}
    for step, cid in step_concept.items():
        if labels[step].startswith("distractor"):
            continue
        members.setdefault(root(cid), []).append(labels[step])
    majority = {
        cid: max(set(ls), key=ls.count) for cid, ls in members.items()
    }

    final_ltm = {c.concept_id for c in runner.store.ltm}
    promoted_final = {
        cid for cid in final_ltm if runner.store.get(cid).provenance == "promoted"
    }

    # 1. All 3 recurring novel classes promoted, each exactly once
    #    (fragmentation index = 1.0 after LTM<->LTM merging).
    for cls in novel_classes:
        owners = {cid for cid in promoted_final if majority.get(cid) == cls}
        assert len(owners) == 1, (
            f"{cls}: expected exactly one promoted concept, got {sorted(owners)}"
        )

    # 2. Burst class: zero promotions, >= 1 eviction record.
    burst_ids = {step_concept[s] for s in range(len(labels)) if labels[s] == "burst_00"}
    promoted_ever = {r["concept_id"] for r in records if r["type"] == "promote"}
    assert not (burst_ids & promoted_ever), "burst concepts must never promote"
    assert not any(root(cid) in promoted_final for cid in burst_ids), (
        "burst traffic must not be merged into a promoted concept"
    )
    evicted_ids = {r["concept_id"] for r in records if r["type"] == "evict"}
    assert burst_ids & evicted_ids, "the burst candidate must be LRU-evicted"

    # 3. End-of-stream purity of each promoted concept >= 0.95.
    for cid in promoted_final:
        ls = members[cid]
        purity = ls.count(majority[cid]) / len(ls)
        assert purity >= 0.95, f"{cid} ({majority[cid]}): purity {purity:.3f} < 0.95"

    # 4. Promotion-aware routing: >= 0.85 of each promoted class's
    #    post-promotion arrivals route at tier 1 (TASKS T11, owner amendment
    #    2026-07-11 from the original 0.90). The rate is structurally ceilinged
    #    at tau_percentile_q/100 = 0.95 -- FR-5.1 calibrates tau at the q-th
    #    percentile of a concept's own LOO scores, so ~5% of its own arrivals
    #    fall beyond its own tau by construction. 0.85 is that ceiling less a
    #    3-sigma binomial band at the smallest post-promotion n (~49).
    #    Q-LINKED: re-derive this floor if tau_percentile_q changes.
    promote_step = {
        r["concept_id"]: r["step"] for r in records if r["type"] == "promote"
    }
    for cls in novel_classes:
        owner = next(cid for cid in promoted_final if majority.get(cid) == cls)
        p_step = min(
            step for cid, step in promote_step.items() if root(cid) == owner
        )
        post = [s for s in range(len(labels)) if labels[s] == cls and s > p_step]
        assert post, f"{cls}: no post-promotion arrivals (fixture guarantees some)"
        tier1 = sum(1 for s in post if step_tier[s] == 1)
        assert tier1 / len(post) >= 0.85, (
            f"{cls}: only {tier1}/{len(post)} post-promotion arrivals at tier 1"
        )

    # 5. Known-class expanding accuracy at every window end (TASKS T11, owner
    #    amendment 2026-07-11 from the flat >= 0.95). A known arrival is
    #    correct only if its OWN LTM accepts it at tier 1, and FR-5.1 puts that
    #    concept's tau at the q-th percentile of its own LOO scores -- so
    #    ~(1 - q/100) of its own arrivals fall beyond its own tau by
    #    construction. Accuracy is thus ceilinged at tau_percentile_q/100, and
    #    the floor is that ceiling less a 3-sigma binomial band at the n known
    #    arrivals seen so far (n grows across window ends, so the band does
    #    too). Q-LINKED: this is derived from config, not hardcoded, so it
    #    tracks tau_percentile_q automatically.
    #    T0 concepts are ltm_{i:03d} in sorted class order (T6 decision 2).
    ceiling = config.tau_percentile_q / 100.0
    known_to_ltm = {cls: f"ltm_{i:03d}" for i, cls in enumerate(known_classes)}
    for end in range(249, 2000, 250):
        seen = [s for s in range(end + 1) if labels[s].startswith("known_")]
        correct = sum(
            1 for s in seen if step_concept[s] == known_to_ltm[labels[s]] and step_tier[s] == 1
        )
        acc = correct / len(seen)
        floor = ceiling - 3.0 * math.sqrt(ceiling * (1.0 - ceiling) / len(seen))
        assert acc >= floor, (
            f"expanding accuracy {acc:.4f} < {floor:.4f} at step {end} "
            f"(n={len(seen)}, ceiling={ceiling})"
        )

    # 6. Residual "unknown" rate for PROMOTED classes (TASKS T11, owner
    #    amendment 2026-07-11). Population: the POST-promotion arrivals of each
    #    promoted novel class -- PRD §7 states the metric as "residual unknowns
    #    for promoted classes", and PRD §7.3 makes "unknown" a CORRECT
    #    prediction for a class not yet promoted, so pre-promotion arrivals
    #    cannot count against it. Bound: the same q-linked tau tail as clauses
    #    4 and 5, from the other side -- a promoted concept's tau rejects
    #    ~(1 - q/100) of its own class by construction, and those arrivals fall
    #    to tier 2/3, where FR-9.1 emits "unknown". So the unknown rate has a
    #    structural FLOOR of 1 - q/100, and the bound is that floor plus a
    #    3-sigma binomial band. (A post-promotion arrival is "unknown" exactly
    #    when it did not route at tier 1, so this is the complement of clause 4
    #    -- the two must stay consistent.)
    #    Q-LINKED: derived from config, so it tracks tau_percentile_q.
    unknown_floor = 1.0 - ceiling
    post_all: list[int] = []
    n_unknown = 0
    for cls in novel_classes:
        owner = next(cid for cid in promoted_final if majority.get(cid) == cls)
        p_step = min(
            step for cid, step in promote_step.items() if root(cid) == owner
        )
        post = [s for s in range(len(labels)) if labels[s] == cls and s > p_step]
        post_all += post
        n_unknown += sum(1 for s in post if step_tier[s] != 1)
    bound = unknown_floor + 3.0 * math.sqrt(
        unknown_floor * (1.0 - unknown_floor) / len(post_all)
    )
    assert n_unknown / len(post_all) < bound, (
        f"residual unknown rate {n_unknown}/{len(post_all)} = "
        f"{n_unknown / len(post_all):.4f} >= {bound:.4f} "
        f"(structural floor {unknown_floor}, n={len(post_all)})"
    )


# --------------------------------------------------------- runtime budget [I]


@pytest.mark.slow
@pytest.mark.skipif(not AVAILABLE, reason=REASON)
def test_runtime_budget(tmp_path):
    """TASKS T11 [I]: a P1-sized run (13,326 real embeddings) completes
    within the NFR-1 budget (< 30 min including UMAP; the protocol-exact P1
    builder is T12 — this uses the same composition with an ad-hoc seeded
    shuffle). Wall-time per 1k steps goes in the session report."""
    pools = load_all_pools()
    config = FPCMCConfig()  # PRD §8 defaults, seed 42
    store = initialize_ltm(
        pools["ind_reference"].x, pools["ind_reference"].subclass_names, config
    )
    prior = compute_global_prior(store.ltm, config)

    ind = pools["ind_test"].x
    rest = np.vstack([ind[1000:], pools["synthetic_ind"].x,
                      pools["near_ood"].x, pools["far_ood"].x])
    perm = make_rng(config.seed, "t11/p1-sized-shuffle").permutation(rest.shape[0])
    stream_x = np.vstack([ind[:1000], rest[perm]])
    assert stream_x.shape == (13_326, 1024)

    runner = StreamRunner(config, store, prior, log_path=tmp_path / "p1.jsonl")
    t0 = time.perf_counter()
    runner.run(stream_x)
    elapsed = time.perf_counter() - t0

    per_1k = elapsed / (stream_x.shape[0] / 1000.0)
    print(f"\nP1-sized run: {elapsed:.1f}s total, {per_1k:.1f}s per 1k steps, "
          f"{len(runner.sweeper.merge_log)} merges, "
          f"{len(runner.promoter.promotion_log)} promotions, "
          f"{len(runner.store.eviction_log)} evictions, "
          f"{len(runner.residual.run_log)} residual passes")
    assert elapsed < 30 * 60, f"NFR-1: {elapsed:.0f}s exceeds the 30 min budget"
