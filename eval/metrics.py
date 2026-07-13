"""§7.3 metrics over the JSONL event log (T13; PRD §7.2–7.3, NFR-3).

Every function here is a pure function of (log records, StreamGroundTruth)
— no live pipeline objects — so any number is reconstructable from the log
alone. Ground truth flows exclusively through ``eval.gt`` (invariant 2).

Detection math is ported with citation from the vendored source module
``lib/evaluation/continual/evaluation.py::_ood_metrics`` (blob 2d0d7e5e,
read-only): sklearn ``roc_auc_score`` / ``average_precision_score`` /
``roc_curve``, FPR at the first threshold reaching TPR >= 0.95. The novelty
statistic is the schema-v2 per-step ``novelty`` field (min tier-1 scorer
scalar — the same min-over-concepts statistic as the T6 M1 gate), so the
streaming AUROC is directly comparable to the pinned batch/v1 numbers.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from eval.gt import (
    Arrivals,
    EvalSnapshot,
    StreamGroundTruth,
    VARIANTS,
    arrivals_from_records,
    prediction_is_correct,
    resolve_root,
    snapshot,
)

# ------------------------------------------------------------- detection (1)


def ood_metrics(ind_scores: np.ndarray, ood_scores: np.ndarray) -> dict:
    """AUROC / AUPR / FPR@95TPR from IND and OOD score arrays.

    Ported verbatim in behavior from lib/evaluation/continual/evaluation.py::
    _ood_metrics (blob 2d0d7e5e): labels 0=IND / 1=OOD, higher score = more
    novel, FPR at the first ROC point with TPR >= 0.95 (1.0 if unreached).
    """
    ind_scores = np.asarray(ind_scores, dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)
    labels = np.concatenate([
        np.zeros(len(ind_scores), dtype=np.int32),
        np.ones(len(ood_scores), dtype=np.int32),
    ])
    all_scores = np.concatenate([ind_scores, ood_scores])

    auroc = float(roc_auc_score(labels, all_scores))
    aupr_ood = float(average_precision_score(labels, all_scores))
    aupr_ind = float(average_precision_score(1 - labels, -all_scores))

    fpr, tpr, _ = roc_curve(labels, all_scores)
    idx = np.where(tpr >= 0.95)[0]
    fpr_at_95 = float(fpr[idx[0]]) if len(idx) > 0 else 1.0

    return {
        "auroc": auroc,
        "aupr_ood_positive": aupr_ood,
        "aupr_ind_positive": aupr_ind,
        "fpr_at_95_tpr": fpr_at_95,
        "n_ind": int(len(ind_scores)),
        "n_ood": int(len(ood_scores)),
    }


def detection_metrics(
    records: Sequence[dict],
    gt: StreamGroundTruth,
    *,
    include_mask: Optional[np.ndarray] = None,
) -> dict:
    """Streaming detection metrics, stratified all/near/far (PRD §7.3.1).

    The per-step score is the schema-v2 ``novelty`` field. Steps whose
    novelty is null (no tier-1 concept existed) are dropped and counted in
    ``n_unscored``. ``gt.excluded`` steps stay IN — a planted distractor is
    genuinely OOD for detection (it leaves only classification/discovery
    denominators). ``include_mask`` optionally restricts the evaluated steps
    (e.g. T14 excluding P1's warmup for v1 comparability).
    """
    arr = arrivals_from_records(records)
    kind = gt.ood_kind[arr.step]
    keep = np.isfinite(arr.novelty)
    if include_mask is not None:
        keep &= np.asarray(include_mask, dtype=bool)[arr.step]
    n_unscored = int(np.count_nonzero(~np.isfinite(arr.novelty)))

    scores = arr.novelty[keep]
    kind = kind[keep]
    ind_scores = scores[kind == "ind"]
    out: dict = {"n_unscored": n_unscored}
    ood_all = scores[kind != "ind"]
    if len(ind_scores) and len(ood_all):
        out["all_ood"] = ood_metrics(ind_scores, ood_all)
        for stratum in ("near", "far"):
            sel = scores[kind == stratum]
            if len(sel):
                out[f"{stratum}_ood"] = ood_metrics(ind_scores, sel)
    return out


# --------------------------------------------- expanding accuracy (2)


def expanding_accuracy(
    records: Sequence[dict],
    gt: StreamGroundTruth,
    at_step: int,
    *,
    snap: Optional[EvalSnapshot] = None,
    arrivals: Optional[Arrivals] = None,
) -> dict:
    """Expanding classification accuracy at ``at_step`` (PRD §7.3.2).

    Accuracy over every non-excluded arrival with step <= at_step (their
    classes are by definition "seen so far"), in both §7.2 variants, plus
    the decomposition: initial-class accuracy (the forgetting-curve series;
    strict == lenient there because "unknown" on a known class is wrong in
    both) and promoted-class accuracy (arrivals of novel classes promoted by
    the snapshot's evaluation point).

    Pass ``snap=snapshot(records, gt, up_to=at_step)`` for checkpoint-time
    numbers (prefix mapping — the owner-approved timing decision) or an
    end-of-stream snapshot for final numbers / T11 golden parity.
    """
    if snap is None:
        snap = snapshot(records, gt, up_to=at_step)
    if arrivals is None:
        arrivals = arrivals_from_records(records)

    counts = {v: {"overall": [0, 0], "initial": [0, 0], "promoted": [0, 0], "novel": [0, 0]}
              for v in VARIANTS}
    for i in range(len(arrivals.step)):
        step = int(arrivals.step[i])
        if step > at_step or gt.excluded[step]:
            continue
        cls = str(gt.true_class[step])
        known = cls in gt.known_classes
        promoted_cls = not known and snap.promoted_at.get(cls) is not None
        for variant in VARIANTS:
            ok = prediction_is_correct(
                str(arrivals.prediction[i]), cls, step, gt,
                snap.majority, snap.parent, snap.promoted_at, variant,
            )
            c = counts[variant]
            for bucket, hit in (
                ("overall", True),
                ("initial", known),
                ("novel", not known),
                ("promoted", promoted_cls),
            ):
                if hit:
                    c[bucket][0] += int(ok)
                    c[bucket][1] += 1

    def _acc(pair: list[int]) -> Optional[float]:
        return pair[0] / pair[1] if pair[1] else None

    return {
        "step": int(at_step),
        **{
            variant: {
                bucket: {"accuracy": _acc(pair), "n_correct": pair[0], "n": pair[1]}
                for bucket, pair in buckets.items()
            }
            for variant, buckets in counts.items()
        },
    }


# ----------------------------------------------------- discovery quality (3)


def _root_members(
    records: Sequence[dict],
    gt: StreamGroundTruth,
    snap: EvalSnapshot,
    up_to: Optional[int] = None,
) -> dict[str, list[str]]:
    """Non-excluded arrival classes per lineage root (snapshot's lineage)."""
    members: dict[str, list[str]] = {}
    for r in records:
        if r["type"] not in ("assign", "seed"):
            continue
        step = r["step"]
        if up_to is not None and step > up_to:
            continue
        if gt.excluded[step]:
            continue
        root = resolve_root(snap.parent, r["concept_id"])
        members.setdefault(root, []).append(str(gt.true_class[step]))
    return members


def _purity(classes: list[str], majority: str) -> float:
    return classes.count(majority) / len(classes)


def promotion_purity(
    records: Sequence[dict],
    gt: StreamGroundTruth,
) -> list[dict]:
    """Per promotion event: purity at promotion time vs at end of stream.

    Promotion-time purity uses the PREFIX snapshot at the promotion step
    (members and majority as they stood when the concept was promoted — the
    v1 drift metric's "at promotion" side); end-of-stream purity uses the
    full-log snapshot and the promoted concept's end lineage root (T11
    clause-3 semantics). One row per promote record.
    """
    end_snap = snapshot(records, gt)
    end_members = _root_members(records, gt, end_snap)
    out: list[dict] = []
    for r in records:
        if r["type"] != "promote":
            continue
        cid, p_step = r["concept_id"], r["step"]
        pre_snap = snapshot(records, gt, up_to=p_step)
        pre_members = _root_members(records, gt, pre_snap, up_to=p_step)
        pre = pre_members.get(resolve_root(pre_snap.parent, cid), [])
        end_root = resolve_root(end_snap.parent, cid)
        end = end_members.get(end_root, [])
        out.append({
            "concept_id": cid,
            "step": int(p_step),
            "end_root": end_root,
            "majority_at_promotion": (
                pre_snap.majority.get(resolve_root(pre_snap.parent, cid))
            ),
            "majority_at_end": end_snap.majority.get(end_root),
            "purity_at_promotion": _purity(pre, pre_snap.majority[
                resolve_root(pre_snap.parent, cid)
            ]) if pre else None,
            "purity_at_end": _purity(end, end_snap.majority[end_root]) if end else None,
            "n_members_at_promotion": len(pre),
            "n_members_at_end": len(end),
        })
    return out


def end_of_stream_purity(
    records: Sequence[dict],
    gt: StreamGroundTruth,
    *,
    snap: Optional[EvalSnapshot] = None,
) -> dict[str, float]:
    """End-of-stream purity per promoted lineage root (T11 clause 3)."""
    if snap is None:
        snap = snapshot(records, gt)
    members = _root_members(records, gt, snap)
    return {
        root: _purity(members[root], snap.majority[root])
        for root in snap.promoted
        if root in members
    }


def fragmentation_index(
    records: Sequence[dict],
    gt: StreamGroundTruth,
    *,
    snap: Optional[EvalSnapshot] = None,
) -> Optional[float]:
    """Promoted concepts per unique discovered novel class (PRD §7.3.3).

    Lineage-merged fragments count as one (the metric is stated "post
    LTM<->LTM merge": promoted roots, not promote events). Roots whose
    majority is not a novel class (contaminants) are excluded here and
    surface in ``discovery_counts``. None when nothing novel was discovered.
    """
    if snap is None:
        snap = snapshot(records, gt)
    novel_roots = [
        root for root in snap.promoted
        if snap.majority.get(root) in gt.novel_classes
    ]
    discovered = {snap.majority[root] for root in novel_roots}
    if not discovered:
        return None
    return len(novel_roots) / len(discovered)


def coverage(
    records: Sequence[dict],
    gt: StreamGroundTruth,
    *,
    snap: Optional[EvalSnapshot] = None,
) -> float:
    """Fraction of novel classes with >= 1 promoted concept (PRD §7.3.3)."""
    if snap is None:
        snap = snapshot(records, gt)
    if not gt.novel_classes:
        return 0.0
    discovered = {
        snap.majority.get(root) for root in snap.promoted
    } & gt.novel_classes
    return len(discovered) / len(gt.novel_classes)


def discovery_counts(
    records: Sequence[dict],
    gt: StreamGroundTruth,
    *,
    snap: Optional[EvalSnapshot] = None,
) -> dict:
    """Discovered-vs-true class counts + contaminant roots (PRD §7.3.3)."""
    if snap is None:
        snap = snapshot(records, gt)
    majorities = [snap.majority.get(root) for root in snap.promoted]
    novel = [m for m in majorities if m in gt.novel_classes]
    return {
        "n_promote_events": sum(1 for r in records if r["type"] == "promote"),
        "n_promoted_roots": len(snap.promoted),
        "n_novel_majority_roots": len(novel),
        "n_known_majority_roots": sum(1 for m in majorities if m in gt.known_classes),
        "n_true_novel_classes": len(gt.novel_classes),
        "n_discovered_novel_classes": len(set(novel)),
    }


# ------------------------------------------------------- memory dynamics (4)


def stm_occupancy(records: Sequence[dict], n_steps: int) -> np.ndarray:
    """|STM| after each step, reconstructed from the event log.

    Seeds grow STM by one; evictions and promotions release a slot; merges
    absorb the (STM) absorbed concept except LTM<->LTM merges, whose absorbed
    side is already LTM. Matches the live ``len(store.stm)`` because those
    four record types are exactly the STM-cardinality mutations.
    """
    delta = np.zeros(n_steps, dtype=np.int64)
    for r in records:
        t = r.get("step")
        if r["type"] == "seed":
            delta[t] += 1
        elif r["type"] == "evict" or r["type"] == "promote":
            delta[t] -= 1
        elif r["type"] == "merge" and r["kind"] != "ltm_ltm":
            delta[t] -= 1
    return np.cumsum(delta)


def eviction_composition(records: Sequence[dict], gt: StreamGroundTruth) -> dict:
    """Ground-truth composition of evicted concepts (PRD §7.3.4).

    "Are evictions actually outliers?" — per evicted concept, its arrival
    members (assign + seed, raw concept id: an evicted candidate was never
    merged away) are classified: all-excluded steps (planted outliers /
    distractors), known-class majority (tau-tail rejects of known classes),
    novel-class majority, or other/mixed.
    """
    members: dict[str, list[int]] = {}
    for r in records:
        if r["type"] in ("assign", "seed"):
            members.setdefault(r["concept_id"], []).append(r["step"])

    per_concept: list[dict] = []
    tally = {"outlier": 0, "known": 0, "novel": 0, "other": 0}
    for r in records:
        if r["type"] != "evict":
            continue
        steps = [s for s in members.get(r["concept_id"], []) if s <= r["step"]]
        classes = [str(gt.true_class[s]) for s in steps if not gt.excluded[s]]
        if not classes:
            category, major = "outlier", None
        else:
            major = min(set(classes), key=lambda c: (-classes.count(c), c))
            if major in gt.known_classes:
                category = "known"
            elif major in gt.novel_classes:
                category = "novel"
            else:
                category = "other"
        tally[category] += 1
        per_concept.append({
            "concept_id": r["concept_id"],
            "step": r["step"],
            "size": r["size"],
            "age": r["age"],
            "category": category,
            "majority": major,
            "n_members": len(steps),
        })
    return {"n_evictions": len(per_concept), "by_category": tally,
            "evictions": per_concept}


def unknown_rate_series(
    records: Sequence[dict],
    gt: StreamGroundTruth,
    at_steps: Sequence[int],
) -> list[dict]:
    """Cumulative "unknown" rate over non-excluded arrivals at each step."""
    arr = arrivals_from_records(records)
    keep = ~gt.excluded[arr.step]
    unknown = (arr.prediction == "unknown") & keep
    out = []
    for s in at_steps:
        mask = arr.step <= s
        n = int(np.count_nonzero(keep & mask))
        u = int(np.count_nonzero(unknown & mask))
        out.append({"step": int(s), "n": n, "n_unknown": u,
                    "rate": u / n if n else None})
    return out


def residual_unknown_promoted(
    records: Sequence[dict],
    gt: StreamGroundTruth,
    *,
    snap: Optional[EvalSnapshot] = None,
) -> dict:
    """Residual "unknown" rate for promoted classes (PRD §7 / T11 clause 6).

    Population: the post-promotion arrivals of each promoted novel class
    (strictly after the class's promotion step — PRD §7.3 makes "unknown"
    CORRECT pre-promotion). An arrival is residual-unknown iff it did not
    route at tier 1 (the exact complement of the tier-1 clause).
    """
    if snap is None:
        snap = snapshot(records, gt)
    arr = arrivals_from_records(records)
    per_class: dict[str, dict] = {}
    n_total = 0
    n_unknown = 0
    for cls in sorted(gt.novel_classes):
        p_step = snap.promoted_at.get(cls)
        if p_step is None:
            continue
        mask = (gt.true_class[arr.step] == cls) & (arr.step > p_step)
        n = int(np.count_nonzero(mask))
        u = int(np.count_nonzero(mask & (arr.tier != 1)))
        per_class[cls] = {"promotion_step": int(p_step), "n_post": n,
                          "n_unknown": u, "rate": u / n if n else None}
        n_total += n
        n_unknown += u
    return {
        "per_class": per_class,
        "n_post": n_total,
        "n_unknown": n_unknown,
        "rate": n_unknown / n_total if n_total else None,
    }


def tier1_post_promotion_rates(
    records: Sequence[dict],
    gt: StreamGroundTruth,
    *,
    snap: Optional[EvalSnapshot] = None,
) -> dict[str, dict]:
    """Per promoted novel class: fraction of post-promotion arrivals routed
    at tier 1 (T11 clause 4 — promotion-aware routing)."""
    if snap is None:
        snap = snapshot(records, gt)
    arr = arrivals_from_records(records)
    out: dict[str, dict] = {}
    for cls in sorted(gt.novel_classes):
        p_step = snap.promoted_at.get(cls)
        if p_step is None:
            continue
        mask = (gt.true_class[arr.step] == cls) & (arr.step > p_step)
        n = int(np.count_nonzero(mask))
        t1 = int(np.count_nonzero(mask & (arr.tier == 1)))
        out[cls] = {"promotion_step": int(p_step), "n_post": n,
                    "n_tier1": t1, "rate": t1 / n if n else None}
    return out


# ----------------------------------------------------- threshold health (5)


def threshold_health(
    records: Sequence[dict],
    gt: StreamGroundTruth,
    *,
    snap: Optional[EvalSnapshot] = None,
) -> dict[str, dict]:
    """Post-hoc per-concept FPR/FNR estimates from ground truth (PRD §7.3.5).

    For each end-of-stream lineage root with a majority class: FPR = fraction
    of its non-excluded arrivals whose true class differs from its majority
    (1 - purity, the accept-side error); FNR = fraction of its majority
    class's non-excluded arrivals it did NOT capture at tier 1 (the
    reject-side error). Estimates, not exact rates: the log records only the
    winning concept, so rejected-by-this-concept-specifically is
    unobservable — documented limitation of any log-only estimate.
    """
    if snap is None:
        snap = snapshot(records, gt)
    arr = arrivals_from_records(records)
    keep = ~gt.excluded[arr.step]
    roots = np.array(
        [resolve_root(snap.parent, c) for c in arr.concept_id.tolist()], dtype=str
    )
    classes = gt.true_class[arr.step]

    out: dict[str, dict] = {}
    for root, majority in snap.majority.items():
        mine = keep & (roots == root)
        n_assigned = int(np.count_nonzero(mine))
        n_wrong = int(np.count_nonzero(mine & (classes != majority)))
        cls_mask = keep & (classes == majority)
        n_class = int(np.count_nonzero(cls_mask))
        n_missed = int(np.count_nonzero(cls_mask & ~((roots == root) & (arr.tier == 1))))
        out[root] = {
            "majority": majority,
            "fpr": n_wrong / n_assigned if n_assigned else None,
            "fnr": n_missed / n_class if n_class else None,
            "n_assigned": n_assigned,
            "n_class": n_class,
        }
    return out


def tau_distribution(records: Sequence[dict]) -> dict:
    """τ distribution across concepts, from checkpoint ``taus`` snapshots
    (schema v2). Returns the last checkpoint's per-status values plus a
    per-checkpoint quantile series (PRD §7.3.5)."""
    checkpoints = [r for r in records if r["type"] == "checkpoint" and "taus" in r]
    if not checkpoints:
        return {"final": None, "series": []}

    def _by_status(rec: dict) -> dict:
        vals: dict[str, list[float]] = {"LTM": [], "STM": []}
        for entry in rec["taus"].values():
            if entry["tau"] is not None:
                vals[entry["status"]].append(float(entry["tau"]))
        return vals

    series = []
    for rec in checkpoints:
        vals = _by_status(rec)
        row: dict = {"step": rec["step"]}
        for status, taus in vals.items():
            row[status.lower()] = (
                {
                    "n": len(taus),
                    "min": float(np.min(taus)),
                    "median": float(np.median(taus)),
                    "max": float(np.max(taus)),
                }
                if taus
                else {"n": 0, "min": None, "median": None, "max": None}
            )
        series.append(row)

    last = checkpoints[-1]
    return {
        "final": {
            "step": last["step"],
            "by_status": _by_status(last),
        },
        "series": series,
    }
