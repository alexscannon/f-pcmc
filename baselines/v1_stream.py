"""B1 baseline: the v1 streaming pipeline (PRD §7.4 B1; TASKS T14).

The pipeline itself lives under ``baselines/v1/`` as a byte-identical vendored
copy of ``msproject_misc/evaluation/continual/`` (the working-tree state behind
``tests/reference_numbers.yaml``; every blob hash recorded in
``lib/PROVENANCE.md`` and asserted by ``test_v1_untouched``). This module is
the ONLY shim ("moved in unmodified except import-path shims"):

  - ``run_v1`` executes the vendored ``main.py`` as a subprocess with
    ``baselines/v1/`` as the script directory, so its flat top-level imports
    (``from config import ...``) resolve inside the vendored set and never
    collide with this repo's packages. The vendored ``config.yaml`` is used
    verbatim (its roots.env upward search finds this repo's ``roots.env``);
    only ``--output-dir`` / ``--seed`` / ``--paradigm`` CLI overrides — all
    pre-existing v1 flags — parameterize the run.
  - ``v1_run_to_jsonl`` + ``v1_ground_truth`` adapt a finished run's persisted
    outputs into a T13 schema-v2 JSONL + ``StreamGroundTruth`` so
    ``eval.harness.evaluate_run`` scores v1 under the same §7.3 definitions
    as F-PCMC (owner-approved "outputs-only adapter", 2026-07-13).

Adapter fidelity (owner-accepted limits of v1's persisted outputs):

  - ``per_step.csv`` logs the RESOLVED predicted label; the raw step-time
    identity for ``is_ood == False`` rows is recovered exactly from
    ``final_cluster_assignments.csv`` (main.py sets ``final_clusters[t]`` at
    step time for those rows and never overwrites them — promotion/drain
    rewrites touch only buffered, ``is_ood == True`` rows).
  - Buffered arrivals (``is_ood == True``) become singleton ``seed`` records
    (``v1_buffer_{t:05d}``): v1 has no per-arrival candidate concepts, and
    promotion-time cluster membership is not persisted, so promoted clusters
    appear in the harness with their POST-promotion arrivals only.
    Promotion-time purity is therefore not reconstructable from the adapter
    log — that metric stays sourced from v1's own ``results_summary.json``
    (where the T14 pin reads it).
  - Drain rows (``phase == "drain"``) are post-stream force-classifications
    with no real detection score; they are excluded, mirroring main.py's own
    ``stream_records`` filter for detection metrics.

Detection comparability is exact by construction: assign/seed records carry
``novelty`` = v1's Mahalanobis ``score`` for every non-drain step (warmup
included, matching the pinned run's 10,250-example IND side), so the
harness's stratified AUROC reproduces ``results_summary.json``'s to within
the CSV's 6-decimal rounding.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from eval.gt import StreamGroundTruth

V1_DIR = Path(__file__).resolve().parent / "v1"
V1_MAIN = V1_DIR / "main.py"

# v1 NoveltyType.value -> detection stratum (the source's ind_types split in
# main.py / evaluation.py, aligned with eval.gt's pool->kind convention).
_NOVELTY_KIND = {
    "ind_real": "ind",
    "ind_synthetic": "ind",
    "near_ood": "near",
    "far_ood": "far",
}

_STREAM_PHASES = ("warmup", "stream")


class V1RunError(RuntimeError):
    """The vendored v1 subprocess failed or produced no results."""


# ------------------------------------------------------------------ execution


def run_v1(
    output_dir: str | Path,
    *,
    paradigm: str = "mahalanobis_hdbscan",
    seed: int = 42,
    force: bool = False,
    limit: int | None = None,
    timeout: float = 3600.0,
) -> Path:
    """Run the vendored, untouched v1 pipeline; return its output directory.

    Reuses an existing ``results_summary.json`` unless ``force`` — the same
    resume semantics main.py itself implements (it exits early when results
    exist and ``--force`` is absent). The subprocess's stdout/stderr stream to
    ``<output_dir>/v1_run.log``.
    """
    output_dir = Path(output_dir)
    summary = output_dir / "results_summary.json"
    if summary.exists() and not force:
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(V1_MAIN),
        "--paradigm",
        paradigm,
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    if force:
        cmd.append("--force")
    if limit is not None:
        cmd += ["--limit", str(limit)]

    run_log = output_dir / "v1_run.log"
    with run_log.open("w", encoding="utf-8") as sink:
        proc = subprocess.run(
            cmd, cwd=V1_DIR, stdout=sink, stderr=sink, timeout=timeout
        )
    if proc.returncode != 0:
        raise V1RunError(
            f"v1 subprocess exited {proc.returncode}; see {run_log}"
        )
    if not summary.exists():
        raise V1RunError(f"v1 run finished but {summary} was not written")
    return output_dir


def load_results(output_dir: str | Path) -> dict:
    """Load a finished run's ``results_summary.json``."""
    return json.loads((Path(output_dir) / "results_summary.json").read_text())


def end_of_stream_median_purity(results: dict) -> float:
    """Median end-of-stream promoted-cluster purity (the pinned v1 metric)."""
    purities = sorted(
        s["end_of_stream"]["purity"] for s in results["cluster_snapshots"]
    )
    return float(np.median(purities))


# ------------------------------------------------------------- output parsing


def _read_per_step(output_dir: Path) -> list[dict]:
    """Non-drain rows of ``per_step.csv``, typed, in stream order."""
    rows: list[dict] = []
    with (output_dir / "per_step.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row["phase"] not in _STREAM_PHASES:
                continue  # drain rows are post-stream (see module docstring)
            rows.append(
                {
                    "t": int(row["t"]),
                    "score": float(row["score"]),
                    "is_ood": bool(int(row["is_ood"])),
                    "predicted_class": row["predicted_class"],
                    "true_class": row["true_class"],
                    "novelty_type": row["novelty_type"],
                    "phase": row["phase"],
                }
            )
    rows.sort(key=lambda r: r["t"])
    if [r["t"] for r in rows] != list(range(len(rows))):
        raise V1RunError("per_step.csv stream rows are not a dense 0..N-1 range")
    return rows


def _read_final_assignments(output_dir: Path) -> dict[int, tuple[str, str]]:
    """stream_idx -> (raw final_cluster, resolved_label)."""
    out: dict[int, tuple[str, str]] = {}
    with (output_dir / "final_cluster_assignments.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["stream_idx"])] = (row["final_cluster"], row["resolved_label"])
    return out


def v1_ground_truth(output_dir: str | Path) -> StreamGroundTruth:
    """Per-step ground truth for a v1 run, from its own ``per_step.csv``.

    Built directly from the run's logged stream (not re-derived via
    ``build_p1``) so the ground truth is exactly aligned with the arrivals the
    adapter emits. ``ood_kind`` carries the near/far stratification the §7.3
    detection metrics need; known classes are the IND (real + synthetic)
    classes, novel classes the near/far-OOD ones.
    """
    rows = _read_per_step(Path(output_dir))
    true_class = np.array([r["true_class"] for r in rows], dtype=str)
    ood_kind = np.array([_NOVELTY_KIND[r["novelty_type"]] for r in rows], dtype=str)
    phase = np.array([r["phase"] for r in rows], dtype=str)
    known = {r["true_class"] for r in rows if _NOVELTY_KIND[r["novelty_type"]] == "ind"}
    novel = {r["true_class"] for r in rows if _NOVELTY_KIND[r["novelty_type"]] != "ind"}
    introduced: dict[str, int] = {c: 0 for c in known}
    for step, cls in enumerate(true_class.tolist()):
        introduced.setdefault(cls, step)
    return StreamGroundTruth(
        true_class=true_class,
        ood_kind=ood_kind,
        excluded=np.zeros(len(rows), dtype=bool),
        phase=phase,
        known_classes=frozenset(known),
        novel_classes=frozenset(novel),
        introduced_at=introduced,
    )


# ------------------------------------------------------------------- adapter


def v1_run_to_jsonl(output_dir: str | Path, jsonl_path: str | Path) -> Path:
    """Adapt a finished v1 run into a T13 schema-v2 JSONL event log.

    Record mapping (module docstring has the fidelity contract):

      - ``is_ood == False`` step -> ``assign`` (tier 1, concept_id = the raw
        step-time identity from final_cluster_assignments.csv, novelty = the
        Mahalanobis score).
      - ``is_ood == True`` step  -> ``seed`` (``v1_buffer_{t:05d}``,
        novelty = score) — v1's buffer has no candidate-concept identities.
      - clustering_events.json promotions -> ``promote`` records at their
        event step (after that step's arrival record, matching main.py's
        route-then-cluster order).
      - no evict / merge / checkpoint records: v1 has no such mechanisms.
    """
    output_dir = Path(output_dir)
    jsonl_path = Path(jsonl_path)
    rows = _read_per_step(output_dir)
    finals = _read_final_assignments(output_dir)
    results = load_results(output_dir)
    events = json.loads((output_dir / "clustering_events.json").read_text())

    promotes: dict[int, list[dict]] = {}
    for event in events:
        for cluster in event["promoted"]:
            promotes.setdefault(int(event["step"]), []).append(
                {
                    "type": "promote",
                    "step": int(event["step"]),
                    "concept_id": cluster["id"],
                    "size": int(cluster["size"]),
                    # v1's intra_sim IS mean pairwise cosine similarity — the
                    # same statistic family as FR-7 cohesion.
                    "cohesion": float(cluster["intra_sim"]),
                    "separation_margin": None,
                    "window_count": None,
                    "gt_majority_label": None,
                    "purity": None,
                }
            )

    meta = results["metadata"]
    header = {
        "type": "config_header",
        "schema": 2,
        "config": {
            "baseline": "v1_stream",
            "paradigm": meta.get("paradigm", "mahalanobis_hdbscan"),
            "random_seed": meta["random_seed"],
            "threshold_percentile": meta["threshold_percentile"],
            "threshold_tau": meta["threshold_tau"],
            "source_output_dir": str(output_dir),
        },
        "n_steps": len(rows),
        "checkpoint_steps": [],
    }

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(header) + "\n")
        for row in rows:
            t = row["t"]
            if row["is_ood"]:
                record = {
                    "type": "seed",
                    "step": t,
                    "concept_id": f"v1_buffer_{t:05d}",
                    "novelty": row["score"],
                }
            else:
                raw, resolved = finals[t]
                if resolved != row["predicted_class"]:
                    raise V1RunError(
                        f"step {t}: final_cluster_assignments resolved label "
                        f"{resolved!r} != per_step predicted_class "
                        f"{row['predicted_class']!r} — raw-identity join broken"
                    )
                record = {
                    "type": "assign",
                    "step": t,
                    "concept_id": raw,
                    "prediction": raw,
                    "tier": 1,
                    "score": row["score"],
                    "margin": None,
                    "via": "v1",
                    "fallback": False,
                    "novelty": row["score"],
                }
            f.write(json.dumps(record) + "\n")
            for promo in promotes.get(t, ()):
                f.write(json.dumps(promo) + "\n")
    return jsonl_path


__all__ = [
    "V1_DIR",
    "V1_MAIN",
    "V1RunError",
    "end_of_stream_median_purity",
    "load_results",
    "run_v1",
    "v1_ground_truth",
    "v1_run_to_jsonl",
]
