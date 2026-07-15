"""T17 Phase 4: batch runner over the 12 production PCMC cells.

Runs in the repo's pinned CPU env; each cell shells out (via ``launch.run_cell``)
to the GPU env's interpreter for ``driver.py``. The 12 cells are
``{resnet18, resnet50} x {sleep, nosleep} x seeds {42, 43, 44}``, landing at
``${DATA_ROOT}/evaluation/f_pcmc_runs/pcmc_sleep/pcmc_{arch}_{variant}/p2_seed{N}/``.

Owner decisions (Phase 4 Q&A 2026-07-14, PLAN.md — verbatim):

  * Q11 pretrain sharing: the sleep and no-sleep cells of each (arch, seed)
    branch from the byte-identical T0 encoder via ``--pretrain-cache`` — 6
    pretrains, not 12. Execution order runs the SLEEP variant of each
    (arch, seed) FIRST so it populates the cache the no-sleep variant reuses.
    The cache key ``{arch}_seed{seed}_ep{epochs}_img{size}.pkl`` (driver.py)
    scopes sharing to exactly (arch, seed); sleep/no-sleep collide by design.
  * Q12 scheduling: one card => serial; ``pretrain_bs`` stays 256 (the driver
    config) unless a real RN50 OOM forces a recorded drop.
  * Q13 failure policy: STOP-AND-REPORT — on the first cell failure the runner
    records it to the manifest, writes the manifest, prints a report, and
    halts the matrix (returns non-zero); resumability makes the restart cheap.
  * Q14 snapshots: unchanged (driver's phase-end cadence, both variants).

The batch runner adds no fidelity surface: cell execution, resumability, and
all PCMC behavior live in ``driver.py`` / the vendored code. This file only
orders the cells, threads the shared cache, and records the run manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from baselines.pcmc_sleep import launch
from baselines.pcmc_sleep.run_config import ARCHS, IMG_SIZE, INIT_EPOCHS, SEEDS

#: Sleep BEFORE nosleep: the sleep cell of each (arch, seed) populates the
#: shared T0 pretrain cache that the nosleep cell then reuses (owner Q11).
VARIANTS = ("sleep", "nosleep")

#: §5 cost-projection basis (GPU-h), used only to flag cells that blew the
#: budget in the manifest (handoff §4.4). The pretrain figure is the RN18
#: 500-epoch projection from the Phase 0.3 spike; the wake+eval figure is the
#: measured Phase 3 cost (21,538 wake steps + 44 evals at CLUST_SIZE=25).
#: Refined in the manifest by the recorded actuals — this is a flag threshold,
#: not a claim.
PRETRAIN_PROJECTION_H = 30.0
WAKE_EVAL_PROJECTION_H = 0.2
BUDGET_TOLERANCE = 1.5


@dataclass(frozen=True)
class Cell:
    arch: str
    seed: int
    variant: str  # "sleep" | "nosleep"

    @property
    def sleep_on(self) -> bool:
        return self.variant == "sleep"

    @property
    def key(self) -> str:
        """Manifest key == the on-disk cell path suffix (launch.cell_dir)."""
        return f"pcmc_{self.arch}_{self.variant}/p2_seed{self.seed}"

    @property
    def selector(self) -> str:
        """Compact CLI selector for ``--only`` (e.g. resnet18_sleep_42)."""
        return f"{self.arch}_{self.variant}_{self.seed}"

    def cell_dir(self, out_root: Path) -> Path:
        return launch.cell_dir(out_root, self.arch, self.sleep_on, self.seed)


def enumerate_cells() -> list[Cell]:
    """The 12 production cells in execution order: arch outer, then seed, then
    sleep BEFORE nosleep so the shared-T0 cache is warm for the nosleep cell
    of each (arch, seed) (owner Q11)."""
    return [
        Cell(arch, seed, variant)
        for arch in ARCHS
        for seed in SEEDS
        for variant in VARIANTS
    ]


def default_cache_dir(out_root: Path) -> Path:
    """Shared T0 encoder cache, a sibling of the cell dirs and paper_protocol/
    under the pcmc_sleep root (underscore-prefixed so it never looks like a
    cell)."""
    return Path(out_root) / "_pretrain_cache"


def pretrain_cache_file(cache_dir: Path, arch: str, seed: int) -> Path:
    """The exact cache path driver.py reads/writes for the production
    (non-smoke) path — must stay byte-identical to driver.py's ``cache_key``."""
    return Path(cache_dir) / f"{arch}_seed{seed}_ep{INIT_EPOCHS}_img{IMG_SIZE}.pkl"


def projected_hours(from_cache: bool) -> float:
    """Per-cell wall projection: a cache-miss pays T0 pretrain, a cache-hit
    reuses it. Sleep-cycle cost folds into the tolerance headroom."""
    return (0.0 if from_cache else PRETRAIN_PROJECTION_H) + WAKE_EVAL_PROJECTION_H


# ---------------------------------------------------------------- manifest I/O


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolved_config_sha256(cell_dir: Path) -> str | None:
    p = cell_dir / "resolved_config.yaml"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def _load_manifest(path: Path) -> dict:
    if path.is_file():
        m = json.loads(path.read_text())
        m.setdefault("cells", {})
        return m
    return {"schema": "pcmc_matrix_manifest_v1", "cells": {}}


def _write_manifest(path: Path, manifest: dict) -> None:
    manifest["updated_at"] = _now()
    manifest["projection_basis_hours"] = {
        "pretrain": PRETRAIN_PROJECTION_H,
        "wake_eval": WAKE_EVAL_PROJECTION_H,
        "tolerance": BUDGET_TOLERANCE,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _ok_entry(cell: Cell, cell_dir: Path, from_cache: bool,
              runner_seconds: float) -> dict:
    summary = json.loads((cell_dir / "summary.json").read_text())
    wall_s = float(summary["wall_time_seconds"])
    actual_h = wall_s / 3600.0
    projected = projected_hours(from_cache)
    return {
        "arch": cell.arch,
        "seed": cell.seed,
        "variant": cell.variant,
        "path": str(cell_dir),
        "status": "ok",
        "resolved_config_sha256": _resolved_config_sha256(cell_dir),
        "pretrain_from_cache": from_cache,
        "wall_time_seconds": wall_s,
        "wall_time_hours": round(actual_h, 3),
        "runner_observed_seconds": round(runner_seconds, 1),
        "sleep_steps_executed": summary.get("sleep_steps_executed"),
        "final_ltm_size": summary.get("final_ltm_size"),
        "final_class_acc": summary.get("final_class_acc"),
        "final_clust_acc": summary.get("final_clust_acc"),
        "projected_hours": round(projected, 3),
        "over_budget": actual_h > projected * BUDGET_TOLERANCE,
        "cuda_device": summary.get("cuda_device"),
        "recorded_at": _now(),
    }


# ------------------------------------------------------------------ execution


def run_matrix(
    *,
    out_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    only: list[str] | None = None,
    force: bool = False,
    timeout: float | None = None,
) -> tuple[dict, bool]:
    """Run the (optionally filtered) matrix serially with the shared T0 cache.

    Stop-and-report (owner Q13): the first failing cell is recorded to the
    manifest, the manifest is flushed, and the run halts. Returns
    ``(manifest, ok)`` — ``ok`` is False iff a cell failed. Never raises for a
    cell failure (the caller reports); still raises for programmer errors
    (unknown selector, missing DATA_ROOT).
    """
    root = Path(out_root) if out_root is not None else launch.default_out_root()
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir(root)
    cache.mkdir(parents=True, exist_ok=True)

    cells = enumerate_cells()
    if only:
        wanted = set(only)
        by_sel = {c.selector: c for c in cells}
        by_key = {c.key: c for c in cells}
        unknown = [s for s in wanted if s not in by_sel and s not in by_key]
        if unknown:
            raise ValueError(
                f"unknown cell selector(s) {unknown}; expected "
                f"{sorted(by_sel)} or {sorted(by_key)}"
            )
        cells = [c for c in cells if c.selector in wanted or c.key in wanted]

    manifest_path = root / "run_manifest.json"
    manifest = _load_manifest(manifest_path)
    manifest["matrix_size"] = len(enumerate_cells())

    for i, cell in enumerate(cells, 1):
        cache_file = pretrain_cache_file(cache, cell.arch, cell.seed)
        from_cache = cache_file.is_file()
        tag = "HIT" if from_cache else "MISS"
        print(f"[{i}/{len(cells)}] {cell.key}  (T0 cache {tag})", flush=True)
        t0 = time.perf_counter()
        try:
            launch.run_cell(
                cell.arch, cell.seed,
                sleep_on=cell.sleep_on, out_root=root,
                force=force, pretrain_cache=cache, timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 — record any cell failure
            manifest["cells"][cell.key] = {
                "arch": cell.arch, "seed": cell.seed, "variant": cell.variant,
                "path": str(cell.cell_dir(root)),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "runner_observed_seconds": round(time.perf_counter() - t0, 1),
                "recorded_at": _now(),
            }
            _write_manifest(manifest_path, manifest)
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            print("  stop-and-report (owner Q13): halting the matrix.",
                  flush=True)
            return manifest, False

        entry = _ok_entry(cell, cell.cell_dir(root), from_cache,
                          time.perf_counter() - t0)
        manifest["cells"][cell.key] = entry
        _write_manifest(manifest_path, manifest)
        flag = "  ** OVER BUDGET **" if entry["over_budget"] else ""
        print(f"  done: {entry['wall_time_hours']} h  "
              f"ltm={entry['final_ltm_size']}  "
              f"class_acc={entry['final_class_acc']}{flag}", flush=True)

    return manifest, True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-root", default=None,
                        help="pcmc_sleep root (default from roots.env)")
    parser.add_argument("--cache-dir", default=None,
                        help="shared T0 encoder cache (default <out-root>/"
                             "_pretrain_cache)")
    parser.add_argument("--only", nargs="+", default=None, metavar="SELECTOR",
                        help="run a subset: compact selectors like "
                             "resnet18_sleep_42, or cell keys")
    parser.add_argument("--force", action="store_true",
                        help="re-run cells even if their config matches")
    parser.add_argument("--timeout", type=float, default=None,
                        help="per-cell subprocess timeout (seconds)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the enumerated plan (with cache HIT/MISS "
                             "prediction) and exit — no GPU, no runs")
    args = parser.parse_args(argv)

    cells = enumerate_cells()
    if args.only:
        by_sel = {c.selector for c in cells}
        by_key = {c.key for c in cells}
        cells = [c for c in cells
                 if c.selector in set(args.only) or c.key in set(args.only)]
        unknown = [s for s in args.only if s not in by_sel and s not in by_key]
        if unknown:
            parser.error(f"unknown selector(s): {unknown}")

    if args.dry_run:
        root = (Path(args.out_root) if args.out_root
                else launch.default_out_root())
        cache = (Path(args.cache_dir) if args.cache_dir
                 else default_cache_dir(root))
        print(f"out-root: {root}")
        print(f"cache:    {cache}")
        # Simulate cache warming across the plan: a cell is a HIT if the
        # encoder is already on disk OR an earlier cell in this plan pretrains
        # the same (arch, seed).
        warm: set[tuple[str, int]] = set()
        for i, cell in enumerate(cells, 1):
            key = (cell.arch, cell.seed)
            hit = key in warm or pretrain_cache_file(
                cache, cell.arch, cell.seed).is_file()
            print(f"  [{i:2d}/{len(cells)}] {cell.key}  "
                  f"(predict T0 cache {'HIT' if hit else 'MISS'})")
            warm.add(key)
        return 0

    _manifest, ok = run_matrix(
        out_root=args.out_root, cache_dir=args.cache_dir,
        only=args.only, force=args.force, timeout=args.timeout,
    )
    if not ok:
        print("matrix halted on a cell failure — see run_manifest.json")
        return 1
    print("matrix complete — all requested cells ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
