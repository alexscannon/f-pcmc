"""Evaluation harness (T13): metrics, checkpointing, ground-truth mapping, figures.

Ground truth lives here and only here — nothing under fpcmc/ may import from
this package (label-leakage invariant, TASKS.md T13 test_gt_map_isolation).

Public surface: ``eval.gt.StreamGroundTruth`` (build from a T12 manifest or
raw labels), ``eval.harness.evaluate_run`` (full §7.3 report from one JSONL
log), ``eval.figures.generate_figures`` (figures/tables from the log alone —
imported lazily; it pulls in matplotlib).
"""

from eval.gt import StreamGroundTruth
from eval.harness import evaluate_run

__all__ = ["StreamGroundTruth", "evaluate_run"]
