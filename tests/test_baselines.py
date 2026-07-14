"""T14 tests — baselines: v1 port, batch wrapper, oracle (TASKS Task 14;
PRD §7.4 B1–B3).

Owner-approved T14 decisions (Q&A 2026-07-13, recorded in docs/CHANGES.md):

  - The v1 pipeline is vendored byte-identical under ``baselines/v1/`` and
    run untouched as a subprocess; ``baselines/v1_stream.py`` is the only
    shim. The vendored blob hashes are recorded in ``lib/PROVENANCE.md``
    (the TASKS ``test_v1_untouched`` literal) — asserted here.
  - The v1 -> schema-v2 adapter is outputs-only (per_step.csv +
    clustering_events.json + final_cluster_assignments.csv); promotion-time
    membership is not persisted by v1, so promotion-time purity stays
    sourced from results_summary.json where the pin reads it.
  - The B3 oracle is a pure ground-truth router emitting schema-v2 JSONL
    (embedding-free); the T13 harness scores it like any run.

Gate discipline: ``test_v1_regression_pin`` is a HARD GATE — a red pin means
the port changed behavior; fix the port, never the pin.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from baselines.batch_knn_vmf import evaluate_batch_checkpoints
from baselines.oracle import run_oracle
from baselines.v1_stream import (
    V1_DIR,
    end_of_stream_median_purity,
    load_results,
    run_v1,
    v1_ground_truth,
    v1_run_to_jsonl,
)
from eval.gt import StreamGroundTruth
from eval.harness import evaluate_run
from fpcmc.config import FPCMCConfig
from fpcmc.data import embeddings_available, load_all_pools, read_roots_env
from fpcmc.init import initialize_ltm
from fpcmc.protocols import build_p1
from fpcmc.stream import StreamRunner
from fpcmc.thresholds import compute_global_prior
from tests.fixtures.golden_stream import load_golden

AVAILABLE, REASON = embeddings_available()

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = yaml.safe_load((REPO_ROOT / "tests" / "reference_numbers.yaml").read_text())

# Stable archive location for the ~11-minute v1 P1 run (v1's own resume
# semantics: an existing results_summary.json is reused, --force reruns).
_V1_OUT_SUBDIR = ("evaluation", "f_pcmc_baselines", "v1_p1_mahalanobis_seed42")


# ---------------------------------------------------------- test_v1_untouched


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def _provenance_hashes() -> dict[str, str]:
    """The baselines/v1 table from lib/PROVENANCE.md: rel path -> blob hash."""
    text = (REPO_ROOT / "lib" / "PROVENANCE.md").read_text()
    marker = "## `baselines/v1/`"
    assert marker in text, "lib/PROVENANCE.md lost its baselines/v1 section"
    section = text[text.index(marker):]
    rows = re.findall(r"\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{40})`\s*\|", section)
    assert rows, "no hash rows found in the baselines/v1 provenance table"
    return dict(rows)


def test_v1_untouched():
    """TASKS T14: checksum of every vendored v1 file matches the hash recorded
    in lib/PROVENANCE.md (only the shim file, baselines/v1_stream.py, may
    differ from source). Also asserts the vendored set is exactly the recorded
    set — nothing added, nothing dropped."""
    recorded = _provenance_hashes()

    on_disk = {
        str(p.relative_to(V1_DIR)): p
        for p in sorted(V1_DIR.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    assert set(on_disk) == set(recorded), (
        "baselines/v1/ file set diverged from lib/PROVENANCE.md: "
        f"unrecorded={sorted(set(on_disk) - set(recorded))}, "
        f"missing={sorted(set(recorded) - set(on_disk))}"
    )

    mismatched = {
        rel: (got, recorded[rel])
        for rel, path in on_disk.items()
        if (got := _git_blob_sha1(path)) != recorded[rel]
    }
    assert not mismatched, (
        f"vendored v1 files modified (fix the port, never the pin): {mismatched}"
    )

    # Cross-check against the reference-numbers pins where one exists: the
    # provenance table must never drift from what the pinned run actually ran.
    pin_hashes = {
        **REFERENCE["t14_v1_regression_pin"]["source"]["git_blob_hashes"],
        **REFERENCE["t6_m1_gate"]["source"]["git_blob_hashes"],
    }
    for src_path, blob in pin_hashes.items():
        rel = src_path.removeprefix("evaluation/continual/")
        if rel in recorded:
            assert recorded[rel] == blob, (
                f"{rel}: provenance table hash != reference_numbers.yaml pin"
            )


# --------------------------------------------------- test_oracle_upper_bounds


def _golden_reports(tmp_path: Path) -> tuple[dict, dict]:
    """(F-PCMC report, oracle report) for the frozen golden stream, both
    scored by the T13 harness with identical ground truth and checkpoints."""
    g = load_golden()
    config = FPCMCConfig.from_yaml("configs/golden_run.yaml")
    checkpoints = tuple(range(249, 2000, 250))

    store = initialize_ltm(g["t0_x"], g["t0_labels"], config)
    prior = compute_global_prior(store.ltm, config)
    fp_log = tmp_path / "golden_fpcmc.jsonl"
    StreamRunner(
        config, store, prior, log_path=fp_log, checkpoint_steps=checkpoints
    ).run(g["stream_x"])

    labels = g["stream_labels"]
    gt = StreamGroundTruth.from_labels(
        labels.tolist(),
        known_classes=sorted({l for l in labels if l.startswith("known_")}),
        novel_classes=("novel_00", "novel_01", "novel_02"),
        excluded_classes={l for l in labels if l.startswith("distractor")},
    )
    oracle_log = run_oracle(
        gt, config, tmp_path / "golden_oracle.jsonl", checkpoint_steps=checkpoints
    )
    return evaluate_run(fp_log, gt), evaluate_run(oracle_log, gt)


def test_oracle_upper_bounds(tmp_path):
    """TASKS T14: on the fixture world (the frozen golden stream), oracle
    accuracy >= every F-PCMC golden-run accuracy metric. A ceiling that isn't
    a ceiling indicates a scoring bug in the harness or the oracle."""
    fp, orc = _golden_reports(tmp_path)

    # The oracle's structural guarantees (one concept per class, GT routing).
    orc_end = orc["end_of_stream"]
    assert orc_end["fragmentation_index"] == 1.0
    assert orc_end["coverage"] == 1.0
    assert all(p == 1.0 for p in orc_end["purity"]["end_by_root"].values())
    orc_acc = orc_end["expanding_accuracy"]
    assert orc_acc["lenient"]["overall"]["accuracy"] == 1.0
    assert orc_acc["strict"]["initial"]["accuracy"] == 1.0

    fp_end = fp["end_of_stream"]
    fp_acc = fp_end["expanding_accuracy"]

    # Accuracy-family ceilings, both variants, end of stream.
    for variant in ("strict", "lenient"):
        for bucket in ("overall", "initial", "promoted"):
            o = orc_acc[variant][bucket]["accuracy"]
            s = fp_acc[variant][bucket]["accuracy"]
            assert s is not None and o is not None
            assert o >= s, (variant, bucket, o, s)

    # ... and at every checkpoint (same prefix-snapshot definitions).
    orc_cps = {c["step"]: c for c in orc["checkpoints"]}
    for cp in fp["checkpoints"]:
        o = orc_cps[cp["step"]]["expanding_accuracy"]
        s = cp["expanding_accuracy"]
        for variant in ("strict", "lenient"):
            assert o[variant]["overall"]["accuracy"] >= s[variant]["overall"]["accuracy"], (
                cp["step"], variant
            )

    # Discovery-quality ceilings: purity / coverage at least as good,
    # fragmentation and residual-unknown at most as high, tier-1
    # post-promotion rate at least as high per promoted class.
    assert orc_end["purity"]["median_at_end"] >= fp_end["purity"]["median_at_end"]
    assert orc_end["coverage"] >= fp_end["coverage"]
    assert orc_end["fragmentation_index"] <= fp_end["fragmentation_index"]
    assert orc_end["residual_unknown_promoted"]["rate"] <= (
        fp_end["residual_unknown_promoted"]["rate"]
    )
    for cls, row in fp_end["tier1_post_promotion"].items():
        orc_row = orc_end["tier1_post_promotion"][cls]
        assert orc_row["rate"] >= row["rate"], cls
        # The oracle promotes at theta matches sharp — never later than a
        # mechanism that also needs cohesion/separation/recurrence.
        assert orc_row["promotion_step"] <= row["promotion_step"], cls


# ------------------------------------------- test_batch_wrapper_matches_existing


@pytest.mark.slow
@pytest.mark.skipif(not AVAILABLE, reason=REASON)
def test_batch_wrapper_matches_existing():
    """TASKS T14: the B2 wrapper at the end-of-stream checkpoint reproduces
    the stored batch knn_vmf metrics (the t6_m1_gate static-batch pins,
    tests/reference_numbers.yaml) within ±0.005."""
    pools = load_all_pools()
    protocol = build_p1(FPCMCConfig(), seed=42, pools=pools)
    report = evaluate_batch_checkpoints(protocol, pools)

    # P1 declares no checkpoints -> exactly the end-of-stream block.
    assert report["checkpoint_steps"] == [13_325]
    assert report["gallery_size"] == 50_000
    end = report["checkpoints"][-1]["detection"]

    # Population identity with the pinned run: 10,250 IND vs 500/2,576 OOD.
    src = REFERENCE["t6_m1_gate"]["source"]
    assert end["all_ood"]["n_ind"] == src["n_ind"]
    assert end["near_ood"]["n_ood"] == src["n_near_ood"]
    assert end["far_ood"]["n_ood"] == src["n_far_ood"]

    pins = REFERENCE["t6_m1_gate"]["metrics"]
    for block, pin_key in (
        ("all_ood", "auroc_all_ood"),
        ("near_ood", "auroc_near_ood"),
        ("far_ood", "auroc_far_ood"),
    ):
        got = end[block]["auroc"]
        assert abs(got - pins[pin_key]) <= 0.005, (
            f"{block}: wrapper AUROC {got:.6f} vs stored batch {pins[pin_key]:.6f}"
        )


# ------------------------------------------------------ test_v1_regression_pin


@pytest.mark.slow
@pytest.mark.skipif(not AVAILABLE, reason=REASON)
def test_v1_regression_pin(tmp_path):
    """TASKS T14 HARD GATE: v1 (vendored untouched) on P1, seed 42, reproduces
    its original headline numbers within the tolerances pinned in
    tests/reference_numbers.yaml. A red pin means the port changed behavior —
    fix the port, never the pin.

    Also proves T13-harness comparability: the outputs-only adapter's JSONL,
    scored by eval.harness.evaluate_run, reproduces v1's own detection AUROC
    (same score column, same non-drain population) to within the per_step.csv
    6-decimal rounding.
    """
    roots = read_roots_env()
    out_dir = Path(roots["DATA_ROOT"]).joinpath(*_V1_OUT_SUBDIR)
    run_v1(out_dir, paradigm="mahalanobis_hdbscan", seed=42)
    results = load_results(out_dir)

    pin = REFERENCE["t14_v1_regression_pin"]
    metrics, tol = pin["metrics"], pin["tolerance"]

    got_auroc = results["detection_metrics"]["all_ood"]["auroc"]
    assert abs(got_auroc - metrics["detection_auroc_all_ood"]) <= tol[
        "detection_auroc_all_ood"
    ], f"detection AUROC {got_auroc} vs pin {metrics['detection_auroc_all_ood']}"

    got_acc = results["classification_accuracy"]["accuracy"]
    assert abs(got_acc - metrics["overall_accuracy"]) <= tol["overall_accuracy"], (
        f"overall accuracy {got_acc} vs pin {metrics['overall_accuracy']}"
    )

    got_promoted = results["cluster_quality"]["n_promoted_total"]
    assert got_promoted == metrics["n_promoted_clusters"], (
        f"promoted clusters {got_promoted} vs pin {metrics['n_promoted_clusters']}"
    )

    got_median = end_of_stream_median_purity(results)
    assert abs(got_median - metrics["end_of_stream_median_purity"]) <= tol[
        "end_of_stream_median_purity"
    ], f"median purity {got_median} vs pin {metrics['end_of_stream_median_purity']}"

    got_buffer = results["metadata"]["final_ood_buffer_size"]
    assert got_buffer == metrics["residual_ood_buffer_size"], (
        f"residual buffer {got_buffer} vs pin {metrics['residual_ood_buffer_size']}"
    )

    # ---- adapter comparability (T13 harness scores the v1 run) -------------
    jsonl = v1_run_to_jsonl(out_dir, tmp_path / "v1_p1.jsonl")
    gt = v1_ground_truth(out_dir)
    assert len(gt) == pin["source"]["stream_length"]
    report = evaluate_run(jsonl, gt)

    det = report["detection"]
    v1_det = results["detection_metrics"]
    for block in ("all_ood", "near_ood", "far_ood"):
        assert det[block]["n_ind"] == v1_det[block]["n_ind"]
        assert det[block]["n_ood"] == v1_det[block]["n_ood"]
        # per_step.csv rounds scores to 6 decimals; that is the only
        # difference between the two computations.
        assert abs(det[block]["auroc"] - v1_det[block]["auroc"]) <= 5e-4, block
    assert abs(det["all_ood"]["auroc"] - metrics["detection_auroc_all_ood"]) <= tol[
        "detection_auroc_all_ood"
    ]
