"""Evaluation harness (T13): metrics, checkpointing, ground-truth mapping, figures.

Ground truth lives here and only here — nothing under fpcmc/ may import from
this package (label-leakage invariant, TASKS.md T13 test_gt_map_isolation).
"""
