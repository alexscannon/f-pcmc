"""T16 results workbook: §7.3 aggregation, §7.5 scorecard, ablation
attribution (TASKS T16; PRD §7.3, §7.5, NFR-1..3).

Reads the archived matrix cells that ``run_matrix.py`` wrote under
``{out_root}/{system}/{protocol}_seed{S}/summary.json`` (plus the sweep cells
under ``{out_root}/sweep/``) and generates, under ``{out_root}/report/``:

  * ``workbook.json`` + ``workbook.md`` — every §7.3 headline metric as
    mean ± std across seeds per system × protocol, the runtime table
    (NFR-1 budgets), the sweep appendix, and the limitations list;
  * ``scorecard.json`` — the five §7.5 success criteria with numeric
    evidence and explicit pass/fail, plus the ablation attribution table
    (populated for every criterion, failed or not).

Owner-approved T16 decisions (Q&A 2026-07-14, recorded in docs/CHANGES.md):

  * Final numbers use the §8 DEFAULTS; the sweep is reported as an appendix
    and never re-bases the matrix (PRD §8 "defaults or the single best sweep
    configuration" — no selection criterion is defined, so defaults it is).
  * §7.5 "residual unknown pool" = the whole-stream count of ``"unknown"``
    predictions (the structural analog of v1's 1,962-example terminal
    buffer: arrivals never classified at stream time; the T14 adapter maps
    exactly those to unknown-prediction records, so the definition is
    symmetric across systems).
  * §7.5 "overall accuracy ≥ v1 + 8 points" compares the STRICT §7.2
    variant on both sides (harness-scored v1, not its native force-classify
    0.7403 — different scoring rules are not comparable; both v1 numbers are
    shown as evidence).
  * The six numeric clauses of §7.5 are presented as FIVE criteria (TASKS
    T16's literal): the four P1-vs-B1 clauses, plus P2 forgetting-bound +
    coverage as one two-clause criterion (pass iff both).

Pass/fail is evaluated on the across-seed mean; per-seed values are always
shown as evidence. A criterion whose metric is unavailable fails with a
note — this is a diagnostic scorecard, and per TASKS a failed research
criterion is a reportable finding, never a test failure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

#: §7.5 relative-to-B1 thresholds (PRD literals).
PURITY_BAR = 0.85
FRAGMENTATION_BAR = 1.3
RESIDUAL_POOL_FRACTION = 0.25
ACCURACY_DELTA = 0.08
FORGETTING_BOUND = 0.05
COVERAGE_BAR = 0.60

#: NFR-1: full stream run < 30 min including UMAP calls (the inclusive
#: budget — wall_time_seconds does not separate UMAP time).
NFR1_BUDGET_SECONDS = 30 * 60

#: §7.4 ablation rows (A5 is two config files, per the T15 inventory).
ABLATION_SYSTEMS = (
    "a1_global_tau", "a2_no_stm", "a3_no_recurrence", "a4_no_merge",
    "a5_knn_ref", "a5_vmf", "a6_resnet50",
)

MAIN_SYSTEM = "fpcmc_default"
B1 = "b1_v1"

#: The known matrix gap, reported (never silently absent) — owner ruling
#: 2026-07-13: the vendored v1 builds its own P1 stream and cannot consume
#: P2 without modifying the frozen port.
LIMITATIONS = [
    "B1 x P2 is unsupported by construction: the vendored v1 pipeline "
    "builds its own P1 stream internally; running it on P2 would require "
    "modifying the frozen port, whose regression pin covers P1/seed 42 "
    "only. All P2 rows therefore have no B1 reference; the §7.5 P2 "
    "criterion (C5) is absolute, not B1-relative, so no criterion loses "
    "its baseline.",
    "v1's native overall accuracy (0.7403, the T14 pin) force-classifies "
    "its terminal buffer; the §7.5 accuracy criterion compares the strict "
    "§7.2 variant on both sides instead (owner decision 2026-07-14). Both "
    "v1 numbers appear as evidence.",
]


# ------------------------------------------------------------------- loading


def load_matrix(out_root: str | Path) -> dict[tuple[str, str, int], dict]:
    """All archived matrix cells: {(system, protocol, seed): summary}."""
    root = Path(out_root)
    cells: dict[tuple[str, str, int], dict] = {}
    for summary_path in sorted(root.glob("*/*/summary.json")):
        if summary_path.parts[-3] in ("sweep", "report"):
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cell = summary.get("cell")
        if not cell:
            continue
        cells[(cell["system"], cell["protocol"], int(cell["seed"]))] = summary
    return cells


def load_sweep(out_root: str | Path) -> list[dict]:
    """Sweep cells ({out_root}/sweep/{param}={value}/p1_seed42/) in order."""
    rows = []
    for summary_path in sorted(Path(out_root).glob("sweep/*/*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cell = summary.get("cell", {})
        rows.append({
            "param": cell.get("param"),
            "value": cell.get("value"),
            **headline_metrics(summary),
        })
    return rows


# ------------------------------------------------------------- metric access


def _get(summary: dict, *path, default=None):
    node: Any = summary
    for key in path:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return default
    return node


def headline_metrics(summary: dict) -> dict[str, Optional[float]]:
    """The flat §7.3 headline view of one cell's summary.

    Works for every runner family: F-PCMC/ablations, B1 and B3 share the
    ``evaluate_run`` report shape; B2 is detection-only (its last checkpoint
    is end-of-stream by construction), so its other metrics are None.
    """
    if "checkpoints" in summary and "end_of_stream" not in summary:
        # B2: detection-only batch report.
        last = summary["checkpoints"][-1]["detection"] if summary["checkpoints"] else {}
        return {
            "detection_auroc_all": _get(last, "all_ood", "auroc"),
            "detection_auroc_near": _get(last, "near_ood", "auroc"),
            "detection_auroc_far": _get(last, "far_ood", "auroc"),
            "detection_fpr95_all": _get(last, "all_ood", "fpr_at_95_tpr"),
            "wall_time_seconds": summary.get("wall_time_seconds"),
        }

    eos = summary.get("end_of_stream", {})
    unknown_series = _get(eos, "memory", "unknown_rate", default=[]) or []
    last_unknown = unknown_series[-1] if unknown_series else {}
    return {
        "detection_auroc_all": _get(summary, "detection", "all_ood", "auroc"),
        "detection_auroc_near": _get(summary, "detection", "near_ood", "auroc"),
        "detection_auroc_far": _get(summary, "detection", "far_ood", "auroc"),
        "detection_fpr95_all": _get(summary, "detection", "all_ood", "fpr_at_95_tpr"),
        "accuracy_strict_overall": _get(eos, "expanding_accuracy", "strict", "overall", "accuracy"),
        "accuracy_lenient_overall": _get(eos, "expanding_accuracy", "lenient", "overall", "accuracy"),
        "accuracy_strict_initial": _get(eos, "expanding_accuracy", "strict", "initial", "accuracy"),
        "accuracy_strict_promoted": _get(eos, "expanding_accuracy", "strict", "promoted", "accuracy"),
        "purity_median_at_promotion": _get(eos, "purity", "median_at_promotion"),
        "purity_median_at_end": _get(eos, "purity", "median_at_end"),
        "fragmentation_index": _get(eos, "fragmentation_index"),
        "coverage": _get(eos, "coverage"),
        "n_promoted_roots": _get(eos, "discovery", "n_promoted_roots"),
        "residual_unknown_promoted_rate": _get(eos, "residual_unknown_promoted", "rate"),
        "unknown_count": last_unknown.get("n_unknown"),
        "unknown_rate": last_unknown.get("rate"),
        "stm_occupancy_final": _get(eos, "memory", "stm_occupancy_final"),
        "n_evictions": _get(eos, "memory", "eviction_composition", "n_evictions"),
        "wall_time_seconds": summary.get("wall_time_seconds"),
    }


def _mean_std(values: Sequence[Optional[float]]) -> dict:
    """Mean ± std (population, ddof=0) over the non-None values."""
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {"mean": None, "std": None, "n": 0, "values": list(values)}
    return {
        "mean": float(np.mean(xs)),
        "std": float(np.std(xs)),
        "n": len(xs),
        "values": list(values),
    }


def aggregate(cells: dict[tuple[str, str, int], dict]) -> dict:
    """{system: {protocol: {metric: mean/std/n/values}}}, seeds ascending."""
    out: dict[str, dict[str, dict]] = {}
    systems = sorted({k[0] for k in cells})
    for system in systems:
        out[system] = {}
        for protocol in ("p1", "p2"):
            seeds = sorted(s for (sys_, proto, s) in cells
                           if sys_ == system and proto == protocol)
            if not seeds:
                continue
            rows = [headline_metrics(cells[(system, protocol, s)]) for s in seeds]
            out[system][protocol] = {
                "seeds": seeds,
                "metrics": {
                    m: _mean_std([r.get(m) for r in rows]) for m in rows[0]
                },
            }
    return out


# ----------------------------------------------------------------- scorecard


def _seed_values(
    cells: dict, system: str, protocol: str, extract
) -> dict[int, Optional[float]]:
    return {
        seed: extract(summary)
        for (sys_, proto, seed), summary in sorted(cells.items())
        if sys_ == system and proto == protocol
    }


def _clause(name: str, values: dict[int, Optional[float]],
            comparator: str, threshold: Optional[float],
            evidence: Optional[dict] = None) -> dict:
    """One numeric clause: pass/fail on the across-seed mean."""
    stats = _mean_std(list(values.values()))
    mean = stats["mean"]
    ok: Optional[bool]
    if mean is None or threshold is None:
        ok = False
        note = "metric unavailable — fails diagnostically"
    else:
        ok = {"<": mean < threshold, "<=": mean <= threshold,
              ">=": mean >= threshold}[comparator]
        note = None
    clause = {
        "name": name,
        "values_by_seed": {str(k): v for k, v in values.items()},
        "mean": mean,
        "std": stats["std"],
        "comparator": comparator,
        "threshold": threshold,
        "pass": bool(ok),
    }
    if note:
        clause["note"] = note
    if evidence:
        clause["evidence"] = evidence
    return clause


def _initial_forgetting(summary: dict) -> Optional[float]:
    """P2 forgetting: first-to-last checkpoint drop in strict initial-class
    accuracy (strict == lenient on initial classes)."""
    checkpoints = summary.get("checkpoints") or []
    accs = [
        _get(c, "expanding_accuracy", "strict", "initial", "accuracy")
        for c in checkpoints
    ]
    accs = [a for a in accs if a is not None]
    if len(accs) < 2:
        return None
    return float(accs[0] - accs[-1])


def build_scorecard(cells: dict[tuple[str, str, int], dict]) -> dict:
    """The five §7.5 success criteria with numeric evidence and pass/fail."""
    h = lambda s, p, metric: _seed_values(  # noqa: E731
        cells, s, p, lambda summary: headline_metrics(summary).get(metric)
    )

    b1_evidence = {
        "b1_harness_purity_median_at_end": _mean_std(
            list(h(B1, "p1", "purity_median_at_end").values()))["mean"],
        "v1_native_purity_median": 0.61,
        "b1_harness_fragmentation": _mean_std(
            list(h(B1, "p1", "fragmentation_index").values()))["mean"],
        "v1_native_promoted_clusters_per_class": "14 concepts / 9 classes ~ 1.6 pre-merge",
    }

    c1 = _clause(
        "end-of-stream promoted purity (median) >= 0.85",
        h(MAIN_SYSTEM, "p1", "purity_median_at_end"), ">=", PURITY_BAR,
        evidence={k: b1_evidence[k] for k in
                  ("b1_harness_purity_median_at_end", "v1_native_purity_median")},
    )

    c2 = _clause(
        "fragmentation index <= 1.3",
        h(MAIN_SYSTEM, "p1", "fragmentation_index"), "<=", FRAGMENTATION_BAR,
        evidence={k: b1_evidence[k] for k in
                  ("b1_harness_fragmentation", "v1_native_promoted_clusters_per_class")},
    )

    # C3: residual unknown pool < 25% of v1's — whole-stream count of
    # "unknown" predictions on both sides (owner decision 2026-07-14).
    b1_pool = _mean_std(list(h(B1, "p1", "unknown_count").values()))["mean"]
    c3 = _clause(
        "residual unknown pool < 25% of v1's",
        h(MAIN_SYSTEM, "p1", "unknown_count"), "<",
        None if b1_pool is None else RESIDUAL_POOL_FRACTION * b1_pool,
        evidence={"b1_unknown_pool_mean": b1_pool,
                  "v1_native_terminal_buffer": 1962},
    )

    # C4: strict overall accuracy >= harness-scored v1 strict + 8 points
    # (owner decision 2026-07-14: same §7.2 variant on both sides).
    b1_acc = _mean_std(list(h(B1, "p1", "accuracy_strict_overall").values()))["mean"]
    c4 = _clause(
        "overall accuracy (strict) >= v1 (strict) + 8 points",
        h(MAIN_SYSTEM, "p1", "accuracy_strict_overall"), ">=",
        None if b1_acc is None else b1_acc + ACCURACY_DELTA,
        evidence={"b1_strict_overall_mean": b1_acc,
                  "b1_lenient_overall_mean": _mean_std(
                      list(h(B1, "p1", "accuracy_lenient_overall").values()))["mean"],
                  "v1_native_overall_accuracy": 0.7403},
    )

    # C5: P2, two clauses, pass iff both (owner-approved grouping).
    forgetting = _clause(
        "P2 initial-class accuracy degrades < 5 points (first -> last checkpoint)",
        _seed_values(cells, MAIN_SYSTEM, "p2", _initial_forgetting),
        "<", FORGETTING_BOUND,
    )
    cov = _clause(
        "P2 novel-class coverage >= 60%",
        h(MAIN_SYSTEM, "p2", "coverage"), ">=", COVERAGE_BAR,
    )
    c5 = {
        "name": "P2: forgetting bound AND novel-class coverage",
        "clauses": [forgetting, cov],
        "pass": bool(forgetting["pass"] and cov["pass"]),
    }

    criteria = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5}
    return {
        "protocol_basis": {"C1": "p1", "C2": "p1", "C3": "p1", "C4": "p1", "C5": "p2"},
        "criteria": criteria,
        "n_pass": sum(1 for c in criteria.values() if c["pass"]),
        "limitations": LIMITATIONS,
    }


# --------------------------------------------------------------- attribution


#: Criterion -> the headline metric its attribution row compares.
CRITERION_METRICS = {
    "C1": ("p1", "purity_median_at_end"),
    "C2": ("p1", "fragmentation_index"),
    "C3": ("p1", "unknown_count"),
    "C4": ("p1", "accuracy_strict_overall"),
    "C5": ("p2", "coverage"),
}


def build_attribution(cells: dict[tuple[str, str, int], dict]) -> dict:
    """Per-criterion metric across the ablations vs the main system.

    Populated for EVERY criterion (TASKS requires it for failed ones): each
    row is the criterion's metric for fpcmc_default and each §7.4 ablation
    on the criterion's protocol, mean across seeds, with the delta vs
    fpcmc_default — the mechanism whose removal moves the metric most is
    the localization TASKS asks for. C5 additionally carries the P2
    forgetting clause per system.
    """
    table: dict[str, dict] = {}
    for criterion, (protocol, metric) in CRITERION_METRICS.items():
        rows: dict[str, dict] = {}
        base = _mean_std([
            headline_metrics(s).get(metric)
            for (sys_, proto, _), s in sorted(cells.items())
            if sys_ == MAIN_SYSTEM and proto == protocol
        ])["mean"]
        for system in (MAIN_SYSTEM,) + ABLATION_SYSTEMS:
            stats = _mean_std([
                headline_metrics(s).get(metric)
                for (sys_, proto, _), s in sorted(cells.items())
                if sys_ == system and proto == protocol
            ])
            row = {"mean": stats["mean"], "std": stats["std"], "n": stats["n"]}
            if system != MAIN_SYSTEM and stats["mean"] is not None and base is not None:
                row["delta_vs_default"] = stats["mean"] - base
            if criterion == "C5":
                forget = _mean_std(list(_seed_values(
                    cells, system, "p2", _initial_forgetting).values()))
                row["forgetting_mean"] = forget["mean"]
            rows[system] = row
        table[criterion] = {"protocol": protocol, "metric": metric, "systems": rows}
    return table


# ------------------------------------------------------------------ runtimes


def build_runtime_table(cells: dict[tuple[str, str, int], dict]) -> dict:
    """Every cell's wall time vs the NFR-1 inclusive budget."""
    rows = []
    for (system, protocol, seed), summary in sorted(cells.items()):
        wall = summary.get("wall_time_seconds")
        rows.append({
            "system": system, "protocol": protocol, "seed": seed,
            "wall_time_seconds": wall,
            "within_budget": (wall is not None and wall < NFR1_BUDGET_SECONDS),
        })
    return {
        "budget_seconds": NFR1_BUDGET_SECONDS,
        "cells": rows,
        "all_within_budget": bool(rows) and all(r["within_budget"] for r in rows),
    }


# ----------------------------------------------------------------- rendering


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}"
    return str(v)


def _md_table(headers: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows]
    return lines


def render_markdown(workbook: dict) -> str:
    """The human-review view of the workbook (the T16 manual gate reads
    this): scorecard first, then aggregates, attribution, runtimes, sweep."""
    lines = ["# F-PCMC results workbook (T16)", ""]
    lines += [f"Generated from {workbook['n_cells']} archived matrix cells "
              f"under `{workbook['out_root']}`.", ""]

    sc = workbook["scorecard"]
    lines += ["## §7.5 success-criteria scorecard", ""]
    for cid, crit in sc["criteria"].items():
        if "clauses" in crit:
            lines += [f"### {cid} — {crit['name']}: **{_fmt(crit['pass'])}**", ""]
            rows = [[cl["name"], _fmt(cl["mean"]),
                     f"{cl['comparator']} {_fmt(cl['threshold'])}",
                     _fmt(cl["pass"])] for cl in crit["clauses"]]
            lines += _md_table(["clause", "mean", "bar", "verdict"], rows) + [""]
        else:
            lines += [
                f"### {cid} — {crit['name']}: **{_fmt(crit['pass'])}**", "",
                f"- mean {_fmt(crit['mean'])} (std {_fmt(crit['std'])}), "
                f"bar {crit['comparator']} {_fmt(crit['threshold'])}",
                f"- per-seed: {crit['values_by_seed']}",
            ]
            if crit.get("evidence"):
                lines += [f"- evidence: {crit['evidence']}"]
            lines += [""]
    lines += [f"**{sc['n_pass']}/5 criteria pass.**", ""]

    lines += ["### Reported limitations", ""]
    lines += [f"- {item}" for item in sc["limitations"]] + [""]

    lines += ["## Ablation attribution table", ""]
    for cid, block in workbook["attribution"].items():
        lines += [f"### {cid} ({block['metric']}, {block['protocol']})", ""]
        rows = [
            [system, _fmt(row["mean"]), _fmt(row["std"]),
             _fmt(row.get("delta_vs_default"))]
            for system, row in block["systems"].items()
        ]
        lines += _md_table(["system", "mean", "std", "delta vs default"], rows) + [""]

    lines += ["## §7.3 metrics, mean ± std across seeds", ""]
    for system, protocols in workbook["aggregates"].items():
        for protocol, block in protocols.items():
            lines += [f"### {system} × {protocol} (seeds {block['seeds']})", ""]
            rows = [
                [metric, _fmt(st["mean"]), _fmt(st["std"]), st["n"]]
                for metric, st in block["metrics"].items()
                if st["n"]
            ]
            lines += _md_table(["metric", "mean", "std", "n"], rows) + [""]

    rt = workbook["runtimes"]
    lines += ["## Runtime table (NFR-1)", "",
              f"Budget: {rt['budget_seconds']} s per cell (inclusive of UMAP).",
              ""]
    rows = [[f"{r['system']} × {r['protocol']} × seed{r['seed']}",
             _fmt(r["wall_time_seconds"]), _fmt(r["within_budget"])]
            for r in rt["cells"]]
    lines += _md_table(["cell", "wall (s)", "within budget"], rows) + [""]

    if workbook["sweep"]:
        lines += ["## Appendix: §8 sweep (P1, seed 42; never re-bases the matrix)", ""]
        rows = [[r["param"], r["value"], _fmt(r.get("accuracy_strict_overall")),
                 _fmt(r.get("purity_median_at_end")),
                 _fmt(r.get("fragmentation_index")), _fmt(r.get("coverage"))]
                for r in workbook["sweep"]]
        lines += _md_table(
            ["param", "value", "strict acc", "end purity", "fragmentation", "coverage"],
            rows) + [""]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- generation


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):  # before int: bool subclasses int
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def generate_workbook(out_root: str | Path, report_dir: str | Path | None = None) -> Path:
    """Build the full workbook from the archive; return the report directory."""
    out_root = Path(out_root)
    cells = load_matrix(out_root)
    if not cells:
        raise FileNotFoundError(f"no matrix cells found under {out_root}")

    workbook = {
        "out_root": str(out_root),
        "n_cells": len(cells),
        "scorecard": build_scorecard(cells),
        "attribution": build_attribution(cells),
        "aggregates": aggregate(cells),
        "runtimes": build_runtime_table(cells),
        "sweep": load_sweep(out_root),
    }

    report = Path(report_dir) if report_dir is not None else out_root / "report"
    report.mkdir(parents=True, exist_ok=True)
    (report / "workbook.json").write_text(
        json.dumps(_jsonable(workbook), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (report / "scorecard.json").write_text(
        json.dumps(_jsonable({
            "scorecard": workbook["scorecard"],
            "attribution": workbook["attribution"],
        }), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (report / "workbook.md").write_text(render_markdown(workbook), encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", help="matrix out_root "
                        "(default ${DATA_ROOT}/evaluation/f_pcmc_runs)")
    args = parser.parse_args(argv)
    if args.out:
        out_root = Path(args.out)
    else:
        from run_matrix import default_out_root
        out_root = default_out_root()
    report = generate_workbook(out_root)
    print(f"workbook -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
