"""Figure/table generators reading only the JSONL log + ground truth (T13;
PRD §7.3, NFR-3).

``generate_figures(log_path, gt, out_dir)`` produces one figure per §7.3
metric group plus a summary table, from the log file and a
``StreamGroundTruth`` (built from the T12 manifest or raw fixture labels) —
no live pipeline objects, per ``test_figures_from_log_only``. Styling
follows the vendored source module lib/evaluation/continual/evaluation.py
(blob 2d0d7e5e): Agg backend, 150 dpi PNGs under ``out_dir``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.gt import StreamGroundTruth, arrivals_from_records
from eval.harness import evaluate_run
from fpcmc.replay import read_log

_KIND_STYLE = {
    "ind": ("IND", "#2196F3"),
    "near": ("Near-OOD", "#FF9800"),
    "far": ("Far-OOD", "#F44336"),
    "ood": ("OOD", "#F44336"),
}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _fig_detection(records, gt: StreamGroundTruth, report: dict, out: Path) -> Path:
    """§7.3.1 — novelty-score distributions by stratum + stratified AUROC."""
    arr = arrivals_from_records(records)
    keep = np.isfinite(arr.novelty)
    kind = gt.ood_kind[arr.step]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for k, (label, color) in _KIND_STYLE.items():
        scores = arr.novelty[keep & (kind == k)]
        if len(scores):
            ax1.hist(scores, bins=60, alpha=0.5, label=label, color=color, density=True)
    ax1.set_xlabel("novelty (min tier-1 score)")
    ax1.set_ylabel("density")
    ax1.set_title("Novelty-score distributions")
    ax1.legend()

    det = report["detection"]
    strata = [k for k in ("all_ood", "near_ood", "far_ood") if k in det]
    if strata:
        aurocs = [det[k]["auroc"] for k in strata]
        ax2.bar(strata, aurocs, color="#1976D2")
        for i, v in enumerate(aurocs):
            ax2.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Streaming detection AUROC")
    return _save(fig, out / "detection.png")


def _fig_accuracy(report: dict, out: Path) -> Path:
    """§7.3.2 — expanding accuracy + forgetting curve over checkpoints."""
    rows = [c["expanding_accuracy"] for c in report["checkpoints"]]
    if not rows:  # P1 has no checkpoints: plot the end-of-stream point alone
        rows = [report["end_of_stream"]["expanding_accuracy"]]
    steps = [r["step"] for r in rows]

    def _series(variant: str, bucket: str):
        return [
            (r[variant][bucket]["accuracy"] if r[variant][bucket]["n"] else np.nan)
            for r in rows
        ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, _series("strict", "overall"), "o-", color="#1976D2",
            label="overall (strict)")
    ax.plot(steps, _series("lenient", "overall"), "o--", color="#1976D2",
            alpha=0.6, label="overall (lenient)")
    ax.plot(steps, _series("strict", "initial"), "s-", color="#6A1B9A",
            label="initial classes (forgetting curve)")
    ax.plot(steps, _series("strict", "promoted"), "^-", color="#E65100",
            label="promoted classes (strict)")
    ax.set_xlabel("step")
    ax.set_ylabel("expanding accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Expanding classification accuracy at checkpoints")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, out / "expanding_accuracy.png")


def _fig_purity(report: dict, out: Path) -> Path:
    """§7.3.3 — promotion-time vs end-of-stream purity per promoted concept."""
    rows = report["end_of_stream"]["purity"]["promotions"]
    fig, ax = plt.subplots(figsize=(6, 6))
    xs = [r["purity_at_promotion"] for r in rows]
    ys = [r["purity_at_end"] for r in rows]
    ok = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if ok:
        ax.scatter(*zip(*ok), color="#1976D2")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("purity at promotion")
    ax.set_ylabel("purity at end of stream")
    ax.set_title("Promoted-concept purity drift (v1's 1.0 → 0.61 metric)")
    return _save(fig, out / "purity_drift.png")


def _fig_memory(records, gt: StreamGroundTruth, report: dict, out: Path) -> Path:
    """§7.3.4 — STM occupancy, cumulative evictions, unknown rate."""
    from eval.metrics import stm_occupancy

    n_steps = report["n_steps"]
    occupancy = stm_occupancy(records, n_steps)
    evict_steps = sorted(r["step"] for r in records if r["type"] == "evict")

    arr = arrivals_from_records(records)
    keep = ~gt.excluded[arr.step]
    unknown = ((arr.prediction == "unknown") & keep).astype(np.float64)
    cum_unknown = np.cumsum(unknown) / np.maximum(np.cumsum(keep.astype(np.float64)), 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(np.arange(n_steps), occupancy, color="#1976D2", label="|STM|")
    if evict_steps:
        ax1.plot(evict_steps, np.arange(1, len(evict_steps) + 1), color="#F44336",
                 label="cumulative evictions")
    ax1.set_ylabel("count")
    ax1.set_title("STM occupancy and evictions")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(arr.step, cum_unknown, color="#6A1B9A")
    ax2.set_xlabel("step")
    ax2.set_ylabel('cumulative "unknown" rate')
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)
    return _save(fig, out / "memory_dynamics.png")


def _fig_threshold_health(report: dict, out: Path) -> Path:
    """§7.3.5 — τ distribution + per-concept FPR/FNR scatter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    tau = report["end_of_stream"]["tau"]["final"]
    if tau is not None:
        for status, color in (("LTM", "#1976D2"), ("STM", "#FF9800")):
            values = tau["by_status"].get(status, [])
            if values:
                ax1.hist(values, bins=30, alpha=0.6, label=f"{status} (n={len(values)})",
                         color=color)
        ax1.legend()
    ax1.set_xlabel("tau (knn_ref scale)")
    ax1.set_ylabel("concepts")
    ax1.set_title("Per-concept τ distribution (final checkpoint)")

    health = report["end_of_stream"]["threshold_health"]
    pts = [(h["fpr"], h["fnr"]) for h in health.values()
           if h["fpr"] is not None and h["fnr"] is not None]
    if pts:
        ax2.scatter(*zip(*pts), alpha=0.6, color="#6A1B9A")
    ax2.set_xlabel("post-hoc FPR")
    ax2.set_ylabel("post-hoc FNR")
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Per-concept threshold health")
    return _save(fig, out / "threshold_health.png")


def _summary_table(report: dict, out: Path) -> Path:
    """Machine-readable summary (JSON) of the headline §7.3 numbers."""
    end = report["end_of_stream"]
    det = report["detection"]
    summary = {
        "n_steps": report["n_steps"],
        "auroc_all_ood": det.get("all_ood", {}).get("auroc"),
        "fpr_at_95_all_ood": det.get("all_ood", {}).get("fpr_at_95_tpr"),
        "expanding_accuracy_strict": end["expanding_accuracy"]["strict"]["overall"]["accuracy"],
        "expanding_accuracy_lenient": end["expanding_accuracy"]["lenient"]["overall"]["accuracy"],
        "initial_class_accuracy": end["expanding_accuracy"]["strict"]["initial"]["accuracy"],
        "median_purity_at_promotion": end["purity"]["median_at_promotion"],
        "median_purity_at_end": end["purity"]["median_at_end"],
        "fragmentation_index": end["fragmentation_index"],
        "coverage": end["coverage"],
        "residual_unknown_promoted_rate": end["residual_unknown_promoted"]["rate"],
        "n_evictions": end["memory"]["eviction_composition"]["n_evictions"],
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def generate_figures(
    log_path: str | Path,
    gt: StreamGroundTruth,
    out_dir: str | Path,
    *,
    report: Optional[dict] = None,
) -> list[Path]:
    """Generate every §7.3 figure + the summary table from the log alone."""
    out = Path(out_dir)
    records = read_log(log_path)
    if report is None:
        report = evaluate_run(log_path, gt)
    return [
        _fig_detection(records, gt, report, out),
        _fig_accuracy(report, out),
        _fig_purity(report, out),
        _fig_memory(records, gt, report, out),
        _fig_threshold_health(report, out),
        _summary_table(report, out),
    ]
