"""T17 Phase 3: the lossy PCMC -> memory-dynamics adapter (owner Q10b).

F-PCMC's §7.3 memory-dynamics metrics are computed from its event log and
live in each archived cell's ``summary.json``. PCMC has no event log; this
adapter extracts the comparable signals from the Phase 2 driver's artifacts
— a documented LOSSY mapping (PLAN.md owner decision 3: §7.3 metrics are
secondary, reported one-sided for F-PCMC or through this adapter for PCMC).
Owner Q10b scoped it to the cheap artifacts only: no driver changes, no
per-step routing events.

Field-by-field mapping (sources are all driver.py outputs):

  ltm_size_per_step     logs/**/layer_L0/ltm_size_history.npy — their Layer
                        appends len(ltm) once per wake step
                        (pcmc_layer.py:760) and PCMC.plots re-saves the
                        array at every checkpoint eval (pcmc.py:296); the
                        longest saved array covers the run up to the last
                        eval. Index i = LTM size AFTER wake step i.
  ltm_size_at_checkpoints  the above sampled at the checkpoint steps — the
                        analog of the F-PCMC checkpoint records' n_ltm.
  sleep_steps_executed  summary.json (driver-recorded actual sleep steps).
  eval_wall_times       checkpoints/*.json eval_wall_time_s per checkpoint.
  snapshots             phase-end snapshots/step_NNNNN.pt (owner Q4 cadence):
                        distance_threshold (their novelty threshold — the
                        loose analog of F-PCMC's tau distribution), stm/ltm
                        buffer sizes, sleep_cycles. Needs torch (CPU load);
                        imported lazily so everything else stays torch-free.

Lossy by construction (documented, not silently absorbed):
  * no per-step routing/assignment events -> no streaming-detection
    (AUROC/FPR) analog, no per-concept match/eviction/merge accounting;
  * ltm_size_per_step ends at the last checkpoint eval that re-saved it;
  * distance_threshold is only observable at phase ends + final.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def ltm_size_per_step(cell_dir: str | Path) -> np.ndarray:
    """The longest saved per-wake-step LTM size history (see module
    docstring for provenance/indexing). Empty array if none saved yet."""
    best = np.array([], dtype=np.int64)
    for p in Path(cell_dir).glob("logs/**/layer_L0/ltm_size_history.npy"):
        arr = np.load(p).astype(np.int64).ravel()
        if arr.size > best.size:
            best = arr
    return best


def ltm_size_at_checkpoints(cell_dir: str | Path) -> dict[int, int | None]:
    """LTM size after each checkpoint step (None where the saved history is
    shorter — possible only if the run died between evals)."""
    sizes = ltm_size_per_step(cell_dir)
    out: dict[int, int | None] = {}
    for rec in checkpoint_records(cell_dir):
        step = rec["step"]
        if step < 0:
            continue  # t0 pre-stream: no wake step yet
        out[step] = int(sizes[step]) if step < sizes.size else None
    return out


def checkpoint_records(cell_dir: str | Path) -> list[dict]:
    """All checkpoint JSONs (t0 first, then by step)."""
    cp = Path(cell_dir) / "checkpoints"
    records = [json.loads(p.read_text()) for p in sorted(cp.glob("*.json"))]
    return sorted(records, key=lambda r: r["step"])


def eval_wall_times(cell_dir: str | Path) -> dict[int, float]:
    return {
        r["step"]: r["eval_wall_time_s"]
        for r in checkpoint_records(cell_dir)
        if "eval_wall_time_s" in r
    }


def sleep_steps_executed(cell_dir: str | Path) -> list[int]:
    summary = json.loads((Path(cell_dir) / "summary.json").read_text())
    return list(summary["sleep_steps_executed"])


def snapshot_dynamics(cell_dir: str | Path) -> list[dict]:
    """Phase-end snapshot fields, in step order (lazy torch import; CPU
    map_location); final_state.pt rides along with step key 'final'."""
    import torch  # CPU wheel in the root env; lazy so the rest stays torch-free

    out = []
    paths = sorted(Path(cell_dir).glob("snapshots/step_*.pt"))
    final = Path(cell_dir) / "final_state.pt"
    entries = [(int(p.stem.split("_")[1]), p) for p in paths]
    if final.is_file():
        entries.append(("final", final))
    for step, path in entries:
        state = torch.load(path, map_location="cpu", weights_only=False)
        layer = state["layers"][0]
        out.append({
            "step": step,
            "distance_threshold": float(layer["distance_threshold"]),
            "sleep_cycles": int(layer["sleep_cycles"]),
            "n_ltm": int(layer["ltm"].shape[0]),
            "n_stm": int(layer["stm"].shape[0]),
        })
    return out


def dynamics(cell_dir: str | Path, *, with_snapshots: bool = True) -> dict:
    """The full lossy memory-dynamics record for one PCMC cell."""
    cell_dir = Path(cell_dir)
    out = {
        "cell_dir": str(cell_dir),
        "sleep_steps_executed": sleep_steps_executed(cell_dir),
        "ltm_size_at_checkpoints": ltm_size_at_checkpoints(cell_dir),
        "eval_wall_times_s": eval_wall_times(cell_dir),
        "lossy_note": (
            "PCMC has no event log: no streaming-detection analog, no "
            "per-concept accounting; see pcmc_dynamics.py docstring"
        ),
    }
    if with_snapshots:
        out["snapshots"] = snapshot_dynamics(cell_dir)
    return out
