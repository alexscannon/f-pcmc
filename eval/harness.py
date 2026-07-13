"""Evaluation-run orchestrator (T13; PRD §7.3 "per checkpoint and
end-of-stream").

``evaluate_run(log_path, gt)`` reads one JSONL event log (fpcmc runs and
T14's schema-compatible baseline adapters alike) and produces the full §7.3
report: streaming detection (stratified), per-checkpoint expanding accuracy
in both §7.2 variants (prefix mapping — owner-approved timing decision,
2026-07-13), discovery quality, memory dynamics, and threshold health.
Checkpoint steps come from the log's own checkpoint records; a run without
checkpoints (P1) still gets every end-of-stream block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from fpcmc.replay import read_log

from eval.gt import StreamGroundTruth, arrivals_from_records, snapshot
from eval.metrics import (
    coverage,
    detection_metrics,
    discovery_counts,
    end_of_stream_purity,
    eviction_composition,
    expanding_accuracy,
    fragmentation_index,
    promotion_purity,
    residual_unknown_promoted,
    stm_occupancy,
    tau_distribution,
    threshold_health,
    tier1_post_promotion_rates,
    unknown_rate_series,
)


def evaluate_run(
    log_path: str | Path,
    gt: StreamGroundTruth,
    *,
    detection_include_mask: Optional[np.ndarray] = None,
) -> dict:
    """Compute every §7.3 metric for one logged run.

    ``detection_include_mask`` optionally restricts the detection population
    (e.g. dropping P1's warmup prefix for v1 comparability at T14); all other
    metrics use the gt's own exclusion mask.
    """
    records = read_log(log_path)
    if not records or records[0].get("type") != "config_header":
        raise ValueError(f"{log_path}: log must start with a config_header record")
    header = records[0]
    n_steps = int(header["n_steps"])
    if n_steps != len(gt):
        raise ValueError(
            f"{log_path}: log has {n_steps} steps but ground truth covers {len(gt)}"
        )

    checkpoint_steps: Sequence[int] = [
        r["step"] for r in records if r["type"] == "checkpoint"
    ]
    arrivals = arrivals_from_records(records)
    end_snap = snapshot(records, gt)

    checkpoints = []
    for s in checkpoint_steps:
        snap = snapshot(records, gt, up_to=s)
        checkpoints.append({
            "step": int(s),
            "expanding_accuracy": expanding_accuracy(
                records, gt, s, snap=snap, arrivals=arrivals
            ),
        })

    last_step = n_steps - 1
    occupancy = stm_occupancy(records, n_steps)
    purity_rows = promotion_purity(records, gt)
    end_purity = end_of_stream_purity(records, gt, snap=end_snap)

    def _median(values: list[float]) -> Optional[float]:
        return float(np.median(values)) if values else None

    report = {
        "n_steps": n_steps,
        "n_excluded": int(np.count_nonzero(gt.excluded)),
        "checkpoint_steps": [int(s) for s in checkpoint_steps],
        "config": header["config"],
        "detection": detection_metrics(
            records, gt, include_mask=detection_include_mask
        ),
        "checkpoints": checkpoints,
        "end_of_stream": {
            "expanding_accuracy": expanding_accuracy(
                records, gt, last_step, snap=end_snap, arrivals=arrivals
            ),
            "purity": {
                "promotions": purity_rows,
                "end_by_root": end_purity,
                "median_at_promotion": _median(
                    [p["purity_at_promotion"] for p in purity_rows
                     if p["purity_at_promotion"] is not None]
                ),
                "median_at_end": _median(list(end_purity.values())),
            },
            "fragmentation_index": fragmentation_index(records, gt, snap=end_snap),
            "coverage": coverage(records, gt, snap=end_snap),
            "discovery": discovery_counts(records, gt, snap=end_snap),
            "tier1_post_promotion": tier1_post_promotion_rates(
                records, gt, snap=end_snap
            ),
            "residual_unknown_promoted": residual_unknown_promoted(
                records, gt, snap=end_snap
            ),
            "memory": {
                "stm_occupancy_final": int(occupancy[-1]) if n_steps else 0,
                "stm_occupancy_max": int(occupancy.max()) if n_steps else 0,
                "eviction_composition": eviction_composition(records, gt),
                "unknown_rate": unknown_rate_series(
                    records, gt, list(checkpoint_steps) + [last_step]
                ),
            },
            "threshold_health": threshold_health(records, gt, snap=end_snap),
            "tau": tau_distribution(records),
        },
    }
    return report
