"""T13 tests — evaluation harness (TASKS Task 13; PRD §7.2–7.3).

T13 decisions (owner-approved 2026-07-13, this session; docs/CHANGES.md T13):

  - Log schema v2: assign/seed records carry ``novelty`` (min tier-1 scorer
    scalar — the continuous streaming-detection statistic) and checkpoint
    records carry ``taus`` (per-concept threshold snapshot). Both additive.
  - Mapping timing: checkpoint metrics use the PREFIX mapping/lineage/
    promotions (records up to that step); end-of-stream metrics and the
    golden reproduction use the full-log mapping (T11 parity).
  - "unknown" scoring (PRD §7.2): correct in both variants before the class
    is introduced; lenient-only once introduced but not yet promoted (an
    arrival AT the promotion step is pre-promotion); wrong in both after.

The microcases below are hand-computed over a 10-example crafted log —
every expected value derived in the comments, asserted exactly.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from eval.gt import (
    StreamGroundTruth,
    arrivals_from_records,
    class_promotion_steps,
    prediction_is_correct,
    snapshot,
    unknown_is_correct,
)
from eval.harness import evaluate_run
from eval.metrics import (
    coverage,
    end_of_stream_purity,
    expanding_accuracy,
    fragmentation_index,
    ood_metrics,
    promotion_purity,
    residual_unknown_promoted,
    stm_occupancy,
    tier1_post_promotion_rates,
)
from fpcmc.config import FPCMCConfig, UmapConfig
from fpcmc.data import embeddings_available
from fpcmc.init import initialize_ltm
from fpcmc.replay import read_log
from fpcmc.rng import make_rng
from fpcmc.stream import StreamRunner
from fpcmc.thresholds import compute_global_prior
from tests.fixtures.golden_stream import load_golden
from tests.fixtures.vmf_world import Segment, VMFWorld

AVAILABLE, REASON = embeddings_available()


# ------------------------------------------------------- microcase fixture
#
# 10-example crafted stream. Classes: known k_a/k_b (T0), novel n_1/n_2, and
# n_3 which never arrives (coverage denominator). Step-by-step design:
#
#   step  class  record                              correctness (strict/lenient)
#   0     k_a    assign ltm_000 tier 1               Y / Y
#   1     k_b    assign ltm_001 tier 1               Y / Y
#   2     n_1    seed stm_0000                       N / Y  (introduced@2, unpromoted)
#   3     n_1    seed stm_0001 (fragment)            N / Y
#   4     n_1    assign stm_0000 tier 2, "unknown"   N / Y
#   5     k_a    assign ltm_000 tier 1               Y / Y
#   6     n_2    seed stm_0002                       N / Y  (introduced@6)
#   7     n_2    assign stm_0002 tier 2, "unknown"   N / Y  (promotion@7 is post-route)
#                promote stm_0002 @7
#   8     n_1    assign stm_0000 tier 1              Y / Y  (majority(stm_0000)=n_1)
#                promote stm_0000 @8, promote stm_0001 @8,
#                merge ltm_ltm stm_0000 <- stm_0001 @8
#   9     k_b    assign stm_0002 tier 1 (WRONG)      N / N  (majority(stm_0002)=n_2)
#
# End majorities: ltm_000 [k_a,k_a] -> k_a; ltm_001 [k_b] -> k_b;
# stm_0000 (+absorbed stm_0001) [n_1 x4] -> n_1; stm_0002 [n_2,n_2,k_b] -> n_2.
# Class promotion steps: n_2 -> 7, n_1 -> 8 (lineage-min over its fragments).

_LABELS = ["k_a", "k_b", "n_1", "n_1", "n_1", "k_a", "n_2", "n_2", "n_1", "k_b"]


def _assign(step, cid, tier, prediction=None, novelty=0.5):
    return {"type": "assign", "step": step, "concept_id": cid, "tier": tier,
            "prediction": cid if prediction is None else prediction,
            "score": 0.1, "margin": 0.5, "via": "knn_ref", "fallback": False,
            "novelty": novelty}


def _seed(step, cid, novelty=0.9):
    return {"type": "seed", "step": step, "concept_id": cid, "novelty": novelty}


def _promote(step, cid):
    return {"type": "promote", "step": step, "concept_id": cid, "size": 30,
            "cohesion": 0.8, "separation_margin": 0.1, "window_count": 3,
            "gt_majority_label": None, "purity": None}


def _merge(step, survivor, absorbed, kind):
    return {"type": "merge", "step": step, "kind": kind, "survivor_id": survivor,
            "absorbed_id": absorbed, "centroid_sim": 0.9,
            "cross_within_ratio": 1.0, "survivor_match_count": 10,
            "absorbed_match_count": 2}


def _micro_records(with_merge=True):
    records = [
        _assign(0, "ltm_000", 1, novelty=0.10),
        _assign(1, "ltm_001", 1, novelty=0.12),
        _seed(2, "stm_0000", novelty=0.90),
        _seed(3, "stm_0001", novelty=0.80),
        _assign(4, "stm_0000", 2, prediction="unknown", novelty=0.85),
        _assign(5, "ltm_000", 1, novelty=0.11),
        _seed(6, "stm_0002", novelty=0.70),
        _assign(7, "stm_0002", 2, prediction="unknown", novelty=0.75),
        _promote(7, "stm_0002"),
        _assign(8, "stm_0000", 1, novelty=0.60),
        _promote(8, "stm_0000"),
        _promote(8, "stm_0001"),
        _assign(9, "stm_0002", 1, novelty=0.20),
    ]
    if with_merge:
        records.insert(-1, _merge(8, "stm_0000", "stm_0001", "ltm_ltm"))
    return records


def _micro_gt():
    return StreamGroundTruth.from_labels(
        _LABELS,
        known_classes=("k_a", "k_b"),
        novel_classes=("n_1", "n_2", "n_3"),
    )


# --------------------------------------------------------------- microcases


def test_metric_microcases():
    """TASKS T13: hand-computed 10-example cases — expanding accuracy,
    fragmentation index (lineage-merged fragments count as one), promotion
    vs end purity divergence, coverage. Exact equality throughout."""
    records = _micro_records()
    gt = _micro_gt()
    end = snapshot(records, gt)

    # Mapping sanity (every expected value below hangs off these).
    assert end.majority == {"ltm_000": "k_a", "ltm_001": "k_b",
                            "stm_0000": "n_1", "stm_0002": "n_2"}
    assert end.promoted == {"stm_0002": 7, "stm_0000": 8}
    assert class_promotion_steps(records, gt) == {
        "k_a": -1, "k_b": -1, "n_2": 7, "n_1": 8
    }

    # Expanding accuracy at the end (design table above):
    # strict 4/10; lenient 9/10; initial 3/4 both; novel 1/6 strict, 6/6
    # lenient; promoted == novel here (both novel classes promoted).
    acc = expanding_accuracy(records, gt, 9, snap=end)
    assert acc["strict"]["overall"] == {"accuracy": 0.4, "n_correct": 4, "n": 10}
    assert acc["lenient"]["overall"] == {"accuracy": 0.9, "n_correct": 9, "n": 10}
    for variant in ("strict", "lenient"):
        assert acc[variant]["initial"] == {"accuracy": 0.75, "n_correct": 3, "n": 4}
    assert acc["strict"]["novel"] == {"accuracy": 1 / 6, "n_correct": 1, "n": 6}
    assert acc["lenient"]["novel"] == {"accuracy": 1.0, "n_correct": 6, "n": 6}
    assert acc["strict"]["promoted"] == acc["strict"]["novel"]

    # PREFIX mapping at step 4 (owner timing decision): nothing promoted yet,
    # so every "unknown" on n_1 is lenient-correct; strict 2/5, lenient 5/5.
    pre = expanding_accuracy(records, gt, 4, snap=snapshot(records, gt, up_to=4))
    assert pre["strict"]["overall"] == {"accuracy": 0.4, "n_correct": 2, "n": 5}
    assert pre["lenient"]["overall"] == {"accuracy": 1.0, "n_correct": 5, "n": 5}

    # Fragmentation: 2 promoted lineage roots / 2 discovered classes = 1.0 —
    # the merged n_1 fragments count as ONE. Without the merge record the
    # same promotions give 3 roots / 2 classes = 1.5.
    assert fragmentation_index(records, gt, snap=end) == 1.0
    assert fragmentation_index(_micro_records(with_merge=False), gt) == 1.5

    # Coverage: {n_1, n_2} discovered of {n_1, n_2, n_3} -> 2/3.
    assert coverage(records, gt, snap=end) == 2 / 3

    # Purity divergence: stm_0002 promoted pure (members [n_2, n_2] @7) and
    # diluted afterwards by the step-9 k_b capture -> end purity 2/3.
    # stm_0000 (with its absorbed fragment) stays pure at both times.
    assert end_of_stream_purity(records, gt, snap=end) == {
        "stm_0000": 1.0, "stm_0002": 2 / 3
    }
    rows = {r["concept_id"]: r for r in promotion_purity(records, gt)}
    assert rows["stm_0002"]["purity_at_promotion"] == 1.0
    assert rows["stm_0002"]["purity_at_end"] == 2 / 3
    assert rows["stm_0000"]["purity_at_promotion"] == 1.0
    assert rows["stm_0000"]["purity_at_end"] == 1.0
    assert rows["stm_0001"]["end_root"] == "stm_0000"

    # STM occupancy from the log alone: seeds at 2/3/6, promotions at 7/8/8;
    # the ltm_ltm merge releases nothing (absorbed side already LTM).
    assert stm_occupancy(records, 10).tolist() == [0, 0, 1, 2, 2, 2, 3, 2, 0, 0]

    # No post-promotion arrivals exist for either class in this fixture.
    assert residual_unknown_promoted(records, gt, snap=end)["n_post"] == 0
    assert tier1_post_promotion_rates(records, gt, snap=end)["n_1"]["n_post"] == 0


def test_unknown_variants():
    """TASKS T13: class introduced at step 100, promoted at step 300 —
    "unknown" at step 50 correct in both variants; at step 200 correct only
    in lenient; at step 400 wrong in both. Both scorers asserted, plus the
    boundary (an arrival AT the promotion step is still pre-promotion) and
    the known-class rule (always wrong)."""
    gt = StreamGroundTruth(
        true_class=np.array(["c"] * 500, dtype=str),
        ood_kind=np.array(["ood"] * 500, dtype=str),
        excluded=np.zeros(500, dtype=bool),
        phase=None,
        known_classes=frozenset({"k"}),
        novel_classes=frozenset({"c"}),
        introduced_at={"k": 0, "c": 100},
    )
    promoted_at = {"k": -1, "c": 300}

    expected = {50: (True, True), 200: (False, True), 300: (False, True),
                400: (False, False)}
    for step, (strict_ok, lenient_ok) in expected.items():
        assert unknown_is_correct("c", step, gt, promoted_at, "strict") is strict_ok
        assert unknown_is_correct("c", step, gt, promoted_at, "lenient") is lenient_ok
        # Same law through the full prediction scorer.
        for variant, ok in (("strict", strict_ok), ("lenient", lenient_ok)):
            assert prediction_is_correct(
                "unknown", "c", step, gt, {}, {}, promoted_at, variant
            ) is ok

    for variant in ("strict", "lenient"):
        assert unknown_is_correct("k", 50, gt, promoted_at, variant) is False
    with pytest.raises(ValueError):
        unknown_is_correct("c", 50, gt, promoted_at, "bogus")


def test_auroc_against_sklearn():
    """TASKS T13: streaming AUROC on synthetic scores equals
    sklearn.metrics.roc_auc_score (atol 1e-9); FPR@95 orientation checked on
    a separable case."""
    rng = make_rng(42, "t13/auroc")
    ind = rng.normal(0.0, 1.0, 500)
    ood = rng.normal(1.2, 1.0, 300)

    out = ood_metrics(ind, ood)
    labels = np.concatenate([np.zeros(500), np.ones(300)])
    scores = np.concatenate([ind, ood])
    assert out["auroc"] == pytest.approx(roc_auc_score(labels, scores), abs=1e-9)
    assert out["n_ind"] == 500 and out["n_ood"] == 300

    # Orientation: perfectly separable scores -> AUROC 1.0, FPR@95 = 0.
    sep = ood_metrics(np.zeros(50), np.ones(50))
    assert sep["auroc"] == 1.0
    assert sep["fpr_at_95_tpr"] == 0.0


# ------------------------------------------------ fixture-run integration [U]


def _fixture_config(**kw) -> FPCMCConfig:
    base = dict(
        stm_capacity=8,
        n_mature=3,
        theta_promote=10,
        m_windows=2,
        window_W=100,
        T_merge=200,
        T_cluster=200,
        w_residual=100,
        umap=UmapConfig(dim=200),  # N <= 200 keeps consolidation UMAP-free
        seed=42,
    )
    base.update(kw)
    return FPCMCConfig(**base)


_WORLD = VMFWorld(seed=1301, k_known=4, k_novel=2, separation_deg=75.0,
                  kappa_novel=(600.0, 600.0))


def _fixture_run(tmp_path: Path):
    """400-step fixture run exercising every record type; returns
    (log_path, gt, checkpoint_steps)."""
    known = _WORLD.known_names

    def kn(n):
        base, rem = divmod(n, len(known))
        return {name: base + (1 if i < rem else 0) for i, name in enumerate(known)}

    schedule = [
        Segment(counts=kn(100)),
        Segment(counts={**kn(60), "novel_00": 25}, distractors=tuple(range(15))),
        Segment(counts={**kn(50), "novel_00": 25, "novel_01": 25}),
        Segment(counts={**kn(50), "novel_00": 25, "novel_01": 25}),
    ]
    stream = _WORLD.make_stream(schedule)
    assert stream.x.shape[0] == 400

    config = _fixture_config()
    pool = _WORLD.t0_pool(n_per_class=100)
    store = initialize_ltm(pool.x, pool.labels, config)
    prior = compute_global_prior(store.ltm, config)
    checkpoints = (99, 199, 299, 399)
    log_path = tmp_path / "run.jsonl"
    StreamRunner(config, store, prior, log_path=log_path,
                 checkpoint_steps=checkpoints).run(stream.x)

    labels = stream.labels.tolist()
    gt = StreamGroundTruth.from_labels(
        labels,
        known_classes=known,
        novel_classes=("novel_00", "novel_01"),
        excluded_classes={l for l in labels if l.startswith("distractor")},
    )
    return log_path, gt, checkpoints


def test_figures_from_log_only(tmp_path):
    """TASKS T13: figure generation succeeds given only the JSONL file and
    ground truth built from labels/manifest (no live pipeline objects),
    producing the §7.3 figure set without exceptions."""
    from eval.figures import generate_figures  # deferred: pulls in matplotlib

    log_path, gt, checkpoints = _fixture_run(tmp_path)
    out_dir = tmp_path / "figures"
    paths = generate_figures(log_path, gt, out_dir)

    expected = {"detection.png", "expanding_accuracy.png", "purity_drift.png",
                "memory_dynamics.png", "threshold_health.png", "summary.json"}
    assert {p.name for p in paths} == expected
    for p in paths:
        assert p.exists() and p.stat().st_size > 0, p

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["n_steps"] == 400
    assert 0.0 <= summary["auroc_all_ood"] <= 1.0

    # The report behind the figures is checkpoint-complete and finite where
    # the fixture guarantees data.
    report = evaluate_run(log_path, gt)
    assert [c["step"] for c in report["checkpoints"]] == list(checkpoints)
    end_acc = report["end_of_stream"]["expanding_accuracy"]
    assert end_acc["strict"]["initial"]["n"] > 0
    assert report["end_of_stream"]["tau"]["final"] is not None


# ------------------------------------------------------------ golden gate [G]


def _lineage_resolver(records):
    """T11's resolver, copied verbatim as the parity reference."""
    parent = {}
    for rec in records:
        if rec["type"] == "merge":
            parent[rec["absorbed_id"]] = rec["survivor_id"]

    def resolve(cid):
        while cid in parent:
            cid = parent[cid]
        return cid

    return {cid: resolve(cid) for cid in set(parent)}


@pytest.mark.slow
def test_eval_on_golden(tmp_path):
    """TASKS T13 [G]: the harness on the golden run reproduces the exact
    numbers asserted in T11's golden test — single source of truth for the
    metric definitions. The reference values below are computed with T11's
    own eval-side bookkeeping (copied from test_golden_stream_end_to_end),
    then asserted equal to the harness's outputs."""
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

    # ---- T11 reference bookkeeping (copied) --------------------------------
    step_concept, step_tier = {}, {}
    for rec in records:
        if rec["type"] in ("assign", "seed"):
            step_concept[rec["step"]] = rec["concept_id"]
            step_tier[rec["step"]] = rec["tier"] if rec["type"] == "assign" else 3
    resolve = _lineage_resolver(records)

    def root(cid):
        return resolve.get(cid, cid)

    members = {}
    for step, cid in step_concept.items():
        if labels[step].startswith("distractor"):
            continue
        members.setdefault(root(cid), []).append(labels[step])
    majority = {cid: max(set(ls), key=ls.count) for cid, ls in members.items()}

    final_ltm = {c.concept_id for c in runner.store.ltm}
    promoted_final = {
        cid for cid in final_ltm if runner.store.get(cid).provenance == "promoted"
    }
    promote_step = {
        r["concept_id"]: r["step"] for r in records if r["type"] == "promote"
    }

    ref_purity = {}
    for cid in promoted_final:
        ls = members[cid]
        ref_purity[cid] = ls.count(majority[cid]) / len(ls)

    ref_tier1, ref_unknown, ref_p_step = {}, {}, {}
    for cls in novel_classes:
        owner = next(cid for cid in promoted_final if majority.get(cid) == cls)
        p_step = min(step for cid, step in promote_step.items() if root(cid) == owner)
        post = [s for s in range(len(labels)) if labels[s] == cls and s > p_step]
        ref_p_step[cls] = p_step
        ref_tier1[cls] = (sum(1 for s in post if step_tier[s] == 1), len(post))
        ref_unknown[cls] = (sum(1 for s in post if step_tier[s] != 1), len(post))

    known_to_ltm = {cls: f"ltm_{i:03d}" for i, cls in enumerate(known_classes)}
    ref_known_acc = {}
    for end in range(249, 2000, 250):
        seen = [s for s in range(end + 1) if labels[s].startswith("known_")]
        correct = sum(
            1 for s in seen
            if step_concept[s] == known_to_ltm[labels[s]] and step_tier[s] == 1
        )
        ref_known_acc[end] = (correct, len(seen))

    # ---- harness ------------------------------------------------------------
    gt = StreamGroundTruth.from_labels(
        labels.tolist(),
        known_classes=known_classes,
        novel_classes=novel_classes,
        excluded_classes={l for l in labels if l.startswith("distractor")},
    )
    report = evaluate_run(log_path, gt)
    end_block = report["end_of_stream"]

    # 1+3. Exactly one promoted root per novel class; end purity identical to
    # T11's, value by value (and above the gate's 0.95 bar).
    assert end_block["fragmentation_index"] == 1.0
    assert end_block["coverage"] == 1.0
    assert end_block["purity"]["end_by_root"] == ref_purity
    assert all(p >= 0.95 for p in ref_purity.values())

    # 4. Tier-1 post-promotion rates: same promotion step, numerator and
    # denominator as T11's clause 4.
    harness_tier1 = end_block["tier1_post_promotion"]
    for cls in novel_classes:
        t1, n = ref_tier1[cls]
        assert harness_tier1[cls]["promotion_step"] == ref_p_step[cls]
        assert (harness_tier1[cls]["n_tier1"], harness_tier1[cls]["n_post"]) == (t1, n)
        assert harness_tier1[cls]["rate"] == t1 / n

    # 5. Known-class expanding accuracy at every window end == T11's clause 5
    # instrument (own-LTM-at-tier-1), through the harness's §7.2 mapping.
    by_step = {c["step"]: c for c in report["checkpoints"]}
    assert sorted(by_step) == sorted(ref_known_acc)
    for end, (correct, n) in ref_known_acc.items():
        bucket = by_step[end]["expanding_accuracy"]["strict"]["initial"]
        assert (bucket["n_correct"], bucket["n"]) == (correct, n), (
            f"window end {end}: harness {bucket} != T11 reference {(correct, n)}"
        )

    # 6. Residual unknown rate for promoted classes: same population and
    # count as T11's clause 6 (the complement of clause 4).
    res = end_block["residual_unknown_promoted"]
    total_u = sum(u for u, _ in ref_unknown.values())
    total_n = sum(n for _, n in ref_unknown.values())
    assert (res["n_unknown"], res["n_post"]) == (total_u, total_n)
    assert res["rate"] == total_u / total_n
    for cls in novel_classes:
        u, n = ref_unknown[cls]
        assert (res["per_class"][cls]["n_unknown"], res["per_class"][cls]["n_post"]) == (u, n)

    # Detection sanity on the schema-v2 novelty statistic. The whole-stream
    # AUROC is deliberately NOT bounded high: the streaming detector is
    # growth-coupled — once a novel class promotes, its arrivals correctly
    # score low novelty while ground truth still marks them OOD — the same
    # ~0.08-AUROC depression the T6 re-pin investigation measured on the
    # source project (docs/CHANGES.md, `source_streaming_knn_vmf_run`).
    # Measured here: ~0.68 whole-stream. The calibrated check is the
    # PRE-promotion prefix, where tier 1 is still the static T0 detector.
    det = report["detection"]
    assert det["n_unscored"] == 0, "every golden step has 8 T0 LTM concepts at tier 1"
    assert det["all_ood"]["auroc"] > 0.5
    assert det["all_ood"]["n_ind"] + det["all_ood"]["n_ood"] == len(labels)

    first_promotion = min(promote_step.values())
    from eval.metrics import detection_metrics
    pre_mask = np.arange(len(labels)) < first_promotion
    pre_det = detection_metrics(records, gt, include_mask=pre_mask)
    assert pre_det["all_ood"]["auroc"] > 0.85, (
        "pre-promotion detection should separate strongly (mature STM "
        "candidates already blunt it before any promotion — measured 0.88 "
        f"vs 0.68 whole-stream); got {pre_det['all_ood']['auroc']:.4f}"
    )
    assert pre_det["all_ood"]["auroc"] > det["all_ood"]["auroc"], (
        "discovery must depress later detection (growth coupling), so the "
        "pre-promotion prefix outscores the whole stream"
    )

    # The eviction composition must classify the planted one-off outliers as
    # outliers ("are evictions actually outliers?" — PRD §7.3.4).
    comp = end_block["memory"]["eviction_composition"]
    assert comp["n_evictions"] == len(runner.store.eviction_log)
    assert comp["by_category"]["outlier"] > 0
