"""Ground-truth mapping for evaluation (T13; PRD §7.2).

Ground truth lives HERE and only here — nothing under ``fpcmc/`` may import
this module (label-leakage invariant 2; ``test_gt_map_isolation`` enforces it
with an AST scan). Everything in this module is a pure function of the JSONL
event-log records (``fpcmc.replay.read_log``) plus a ``StreamGroundTruth``,
so every metric and figure is reconstructable from the log alone (NFR-3).

Semantics pinned by the T11 golden gate (``test_eval_on_golden`` asserts the
harness reproduces its numbers exactly):

  - **Lineage resolution is transitive**: an absorbed concept's arrivals
    count toward its end-of-stream survivor (T11 ``_lineage_resolver``).
  - **Majority mapping** (PRD §7.2): each lineage-resolved concept maps to
    the majority ground-truth class over its assigned arrivals; excluded
    steps (golden distractors) enter no denominator. Ties break to the
    lexicographically smallest class name (deterministic; T11's
    ``max(set, key=count)`` is tie-ambiguous but tie-free in practice).
  - **A class's promotion step** is the earliest promote-record step among
    concepts that resolve to the class's owning promoted root.
  - **"unknown" scoring** (PRD §7.2/§7.3): correct in BOTH variants before
    the class is introduced; correct only in the LENIENT variant once
    introduced but not yet promoted (an arrival AT the promotion step is
    still pre-promotion — the FR-9 route happens before the promotion hook);
    wrong in both once promoted. Known (T0) classes are introduced and
    promoted from initialization, so "unknown" on them is always wrong.

Mapping timing (owner-approved 2026-07-13): metrics evaluated AT a checkpoint
use the PREFIX mapping/lineage/promotions (records up to that step only — no
future information in a mid-stream number); end-of-stream metrics and the
golden reproduction use the full-log mapping, matching T11 exactly. Every
function below takes ``up_to`` accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

# Pool name -> detection stratum for manifest-built ground truth (PRD §7.3
# metric 1: stratified near/far OOD; every IND pool is the negative class).
_POOL_OOD_KIND = {
    "ind_reference": "ind",
    "ind_test": "ind",
    "synthetic_ind": "ind",
    "near_ood": "near",
    "far_ood": "far",
}

STRICT = "strict"
LENIENT = "lenient"
VARIANTS = (STRICT, LENIENT)


@dataclass(frozen=True)
class StreamGroundTruth:
    """Per-step ground truth for one stream run (eval-side only).

    true_class:     (N,) class name per stream step.
    ood_kind:       (N,) detection stratum per step: "ind" | "near" | "far"
                    | "ood" (novel but unstratified, e.g. golden-world
                    classes and distractors).
    excluded:       (N,) bool — steps excluded from every classification /
                    discovery / mapping denominator (golden distractors, per
                    the golden-stream freeze note). Detection metrics keep
                    them: a one-off outlier is still genuinely OOD.
    phase:          (N,) phase label per step, or None.
    known_classes:  the T0 classes (their LTM concepts exist from step 0).
    novel_classes:  the discoverable classes — the coverage / fragmentation /
                    purity / residual-unknown population. Deliberately need
                    not cover every non-known class (the golden burst class
                    and distractors are outliers, not discoverable classes).
    introduced_at:  class -> formal introduction step. Known classes are
                    introduced at 0; defaults to first arrival for others.
    """

    true_class: np.ndarray
    ood_kind: np.ndarray
    excluded: np.ndarray
    phase: Optional[np.ndarray]
    known_classes: frozenset[str]
    novel_classes: frozenset[str]
    introduced_at: Mapping[str, int]

    def __len__(self) -> int:
        return int(self.true_class.shape[0])

    # ------------------------------------------------------------ builders

    @classmethod
    def from_manifest(cls, protocol_stream) -> "StreamGroundTruth":
        """Build from a T12 ``ProtocolStream`` (manifest + phases)."""
        manifest = protocol_stream.manifest
        true_class = np.asarray(manifest.true_class, dtype=str)
        ood_kind = np.array(
            [_POOL_OOD_KIND[p] for p in manifest.pool.tolist()], dtype=str
        )
        known = frozenset(protocol_stream.t0_classes)
        introduced: dict[str, int] = {c: 0 for c in known}
        novel: set[str] = set()
        for phase in protocol_stream.phases:
            for c in phase.introduced_classes:
                novel.add(c)
                introduced.setdefault(c, phase.start)
        # P1 introduces nothing via phases: every non-T0 class is novel from
        # its first arrival.
        for step, c in enumerate(true_class.tolist()):
            if c not in introduced:
                novel.add(c)
                introduced[c] = step
        return cls(
            true_class=true_class,
            ood_kind=ood_kind,
            excluded=np.zeros(len(true_class), dtype=bool),
            phase=np.asarray(manifest.phase, dtype=str),
            known_classes=known,
            novel_classes=frozenset(novel),
            introduced_at=introduced,
        )

    @classmethod
    def from_labels(
        cls,
        labels: Sequence[str],
        *,
        known_classes: Iterable[str],
        novel_classes: Iterable[str],
        excluded_classes: Iterable[str] = (),
        phase: Optional[Sequence[str]] = None,
    ) -> "StreamGroundTruth":
        """Build from a raw per-step label array (fixture / golden streams).

        ``ood_kind`` is "ind" for known classes and "ood" otherwise (fixture
        worlds have no near/far stratification); ``excluded_classes`` marks
        classes (e.g. golden distractors) whose steps leave every
        classification/discovery denominator while remaining OOD for
        detection.
        """
        true_class = np.asarray(list(labels), dtype=str)
        known = frozenset(known_classes)
        excluded_set = frozenset(excluded_classes)
        ood_kind = np.array(
            ["ind" if c in known else "ood" for c in true_class.tolist()], dtype=str
        )
        excluded = np.array([c in excluded_set for c in true_class.tolist()], dtype=bool)
        introduced: dict[str, int] = {c: 0 for c in known}
        for step, c in enumerate(true_class.tolist()):
            introduced.setdefault(c, step)
        return cls(
            true_class=true_class,
            ood_kind=ood_kind,
            excluded=excluded,
            phase=None if phase is None else np.asarray(list(phase), dtype=str),
            known_classes=known,
            novel_classes=frozenset(novel_classes),
            introduced_at=introduced,
        )


# ------------------------------------------------------------------ arrivals


@dataclass(frozen=True)
class Arrivals:
    """Per-step routing outcome extracted from assign/seed records.

    Parallel arrays over the steps that carry an assign or seed record
    (exactly one per stream step — invariant 1). ``prediction`` is the
    emitted prediction (concept_id at tier 1, "unknown" at tiers 2/3);
    ``concept_id`` is the arrival-time container (assigned or seeded);
    ``novelty`` is the schema-v2 min tier-1 scalar (NaN where null).
    """

    step: np.ndarray
    concept_id: np.ndarray
    prediction: np.ndarray
    tier: np.ndarray
    novelty: np.ndarray


def arrivals_from_records(records: Sequence[dict]) -> Arrivals:
    """Extract the per-step arrival table from log records (step-ordered)."""
    rows = [r for r in records if r["type"] in ("assign", "seed")]
    rows.sort(key=lambda r: r["step"])
    n = len(rows)
    step = np.empty(n, dtype=np.int64)
    tier = np.empty(n, dtype=np.int64)
    novelty = np.empty(n, dtype=np.float64)
    concept_id = np.empty(n, dtype=object)
    prediction = np.empty(n, dtype=object)
    for i, r in enumerate(rows):
        step[i] = r["step"]
        concept_id[i] = r["concept_id"]
        if r["type"] == "seed":
            tier[i] = 3
            prediction[i] = "unknown"
        else:
            tier[i] = r["tier"]
            prediction[i] = r["prediction"]
        nv = r.get("novelty")
        novelty[i] = np.nan if nv is None else float(nv)
    return Arrivals(
        step=step,
        concept_id=concept_id.astype(str),
        prediction=prediction.astype(str),
        tier=tier,
        novelty=novelty,
    )


# ------------------------------------------------------------------- lineage


def lineage_parent(records: Sequence[dict], up_to: Optional[int] = None) -> dict[str, str]:
    """absorbed_id -> survivor_id over merge records (optionally prefix)."""
    return {
        r["absorbed_id"]: r["survivor_id"]
        for r in records
        if r["type"] == "merge" and (up_to is None or r["step"] <= up_to)
    }


def resolve_root(parent: Mapping[str, str], concept_id: str) -> str:
    """Follow survivor chains transitively; unmerged ids map to themselves."""
    while concept_id in parent:
        concept_id = parent[concept_id]
    return concept_id


# ------------------------------------------------------- mapping & promotion


def majority_map(
    records: Sequence[dict],
    gt: StreamGroundTruth,
    up_to: Optional[int] = None,
) -> dict[str, str]:
    """PRD §7.2: lineage-resolved concept -> majority ground-truth class.

    Majorities are computed over non-excluded arrivals (assign + seed alike —
    the seed embedding is a member) up to ``up_to`` (None = full log), with
    lineage resolved over the same prefix. Ties break to the largest count
    then the lexicographically smallest class name (deterministic).
    """
    parent = lineage_parent(records, up_to)
    members: dict[str, dict[str, int]] = {}
    for r in records:
        if r["type"] not in ("assign", "seed"):
            continue
        step = r["step"]
        if up_to is not None and step > up_to:
            continue
        if gt.excluded[step]:
            continue
        root = resolve_root(parent, r["concept_id"])
        counts = members.setdefault(root, {})
        cls = str(gt.true_class[step])
        counts[cls] = counts.get(cls, 0) + 1
    return {
        root: min(counts, key=lambda c: (-counts[c], c))
        for root, counts in members.items()
    }


def promoted_roots(
    records: Sequence[dict], up_to: Optional[int] = None
) -> dict[str, int]:
    """Lineage-resolved promoted concepts alive in the (prefix) end state.

    Returns root -> earliest promote-record step among the concepts that
    resolve to it (T11's ``min`` over the merged chain). A promoted concept
    later absorbed by another root is represented by its survivor.
    """
    parent = lineage_parent(records, up_to)
    out: dict[str, int] = {}
    for r in records:
        if r["type"] != "promote":
            continue
        if up_to is not None and r["step"] > up_to:
            continue
        root = resolve_root(parent, r["concept_id"])
        out[root] = min(out.get(root, r["step"]), r["step"])
    return out


def class_promotion_steps(
    records: Sequence[dict],
    gt: StreamGroundTruth,
    up_to: Optional[int] = None,
    *,
    majority: Optional[Mapping[str, str]] = None,
) -> dict[str, int]:
    """Class -> promotion step, for classes owned by a promoted root.

    A class is "promoted" once some promoted root's majority is that class;
    its promotion step is that root's earliest promote step (T11 clause 4/6
    semantics). Known (T0) classes are promoted from initialization (step -1,
    i.e. before any stream step).
    """
    if majority is None:
        majority = majority_map(records, gt, up_to)
    out: dict[str, int] = {c: -1 for c in gt.known_classes}
    for root, p_step in promoted_roots(records, up_to).items():
        cls = majority.get(root)
        if cls is None or cls in gt.known_classes:
            continue
        out[cls] = min(out.get(cls, p_step), p_step)
    return out


# ------------------------------------------------------------ "unknown" law


def unknown_is_correct(
    cls: str,
    step: int,
    gt: StreamGroundTruth,
    promoted_at: Mapping[str, int],
    variant: str,
) -> bool:
    """PRD §7.2: is an "unknown" prediction correct for class ``cls`` at ``step``?

    Correct in both variants before the class is introduced; once introduced
    but not yet promoted, correct only in the lenient variant; wrong in both
    once promoted (an arrival AT the promotion step is still pre-promotion).
    Known classes are introduced and promoted from initialization.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if cls in gt.known_classes:
        return False
    intro = gt.introduced_at.get(cls)
    if intro is None or step < intro:
        return True
    p_step = promoted_at.get(cls)
    if p_step is None or step <= p_step:
        return variant == LENIENT
    return False


def prediction_is_correct(
    prediction: str,
    cls: str,
    step: int,
    gt: StreamGroundTruth,
    majority: Mapping[str, str],
    parent: Mapping[str, str],
    promoted_at: Mapping[str, int],
    variant: str,
) -> bool:
    """Score one prediction (concept_id or "unknown") against ground truth."""
    if prediction == "unknown":
        return unknown_is_correct(cls, step, gt, promoted_at, variant)
    return majority.get(resolve_root(parent, prediction)) == cls


# ------------------------------------------------------------- eval snapshot


@dataclass(frozen=True)
class EvalSnapshot:
    """Everything mapping-dependent, computed once for one evaluation point.

    ``up_to=None`` is the end-of-stream snapshot (T11 golden semantics);
    an integer gives the prefix snapshot for checkpoint-time metrics
    (owner-approved mapping-timing decision, 2026-07-13).
    """

    up_to: Optional[int]
    parent: Mapping[str, str]
    majority: Mapping[str, str]
    promoted: Mapping[str, int]      # promoted root -> earliest promote step
    promoted_at: Mapping[str, int]   # class -> promotion step (known: -1)


def snapshot(
    records: Sequence[dict],
    gt: StreamGroundTruth,
    up_to: Optional[int] = None,
) -> EvalSnapshot:
    parent = lineage_parent(records, up_to)
    majority = majority_map(records, gt, up_to)
    return EvalSnapshot(
        up_to=up_to,
        parent=parent,
        majority=majority,
        promoted=promoted_roots(records, up_to),
        promoted_at=class_promotion_steps(records, gt, up_to, majority=majority),
    )
