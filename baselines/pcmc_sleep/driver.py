"""T17 Phase 2: the sleep-retrained-PCMC driver (replaces upstream main.py).

Runs INSIDE the GPU env (`baselines/pcmc_sleep/env/.venv`) — never the repo's
pinned CPU env. Executes one cell: T0 contrastive pretrain -> 21,538-step P2
wake loop with phase-midpoint sleeps -> their eval at every checkpoint ->
persisted artifacts, with run_matrix.run_cell's resumability conventions
(resolved_config.yaml + summary.json, skip-iff-config-matches, --force).

PCMC itself runs as published: the vendored files are byte-identical
(test_pcmc_vendor_untouched) and this driver only plumbs streams, configs and
persistence around them. The two deliberate driver-side interventions, both
HANDOFF §5 facts:

  * DataLoader force-patch to in-process loading (num_workers=0, worker-only
    kwargs stripped). Their Layer.pretrain builds a persistent-workers loader
    that init_memory abandons early (pcmc_layer.py:341-343) — observed live
    deadlock (workers exited, main blocked in futex_do_wait). A config-only
    fix is impossible: n_workers=0 with persistent_workers=True raises in
    torch. Same proven patch as the Phase 0.3 spike; the smoke test runs the
    driver under a hard timeout so any regression fails loudly.
  * Sleep-schedule steering (owner Q2): their trigger reads
    ``layer.sleep_start``/``sleep_freq`` each step; the driver sets them so
    the next sleep lands exactly on the recorded phase-midpoint step list.
    Their trigger code is untouched; with the sentinel freq the modulo clause
    cannot fire on its own.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "vendor"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml

# ---- deadlock guard: force in-process data loading BEFORE any vendored
# import can bind torch.utils.data.DataLoader (see module docstring).
_ORIG_DATALOADER = torch.utils.data.DataLoader


def _inprocess_dataloader(*args, **kwargs):
    kwargs["num_workers"] = 0
    kwargs.pop("persistent_workers", None)
    kwargs.pop("prefetch_factor", None)
    return _ORIG_DATALOADER(*args, **kwargs)


torch.utils.data.DataLoader = _inprocess_dataloader

from baselines.pcmc_sleep.run_config import (
    SLEEP_FREQ_SENTINEL,
    build_run_config,
    sleep_steps,
    snapshot_steps,
)


def set_seed(seed: int) -> None:
    """Verbatim replica of upstream main.py::set_seed (seeded, not bitwise —
    accepted deviation, PLAN.md owner decision 1)."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def _steer_sleep(model, pending: deque) -> None:
    """Point their per-layer trigger at the next scheduled sleep. Layer.step
    is stream_index + 1 at trigger time, hence the +1."""
    target = (pending[0] + 1) if pending else SLEEP_FREQ_SENTINEL
    for layer in model.layers:
        layer.sleep_start = target
        layer.sleep_freq = SLEEP_FREQ_SENTINEL


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _snapshot(model, path: Path, *, with_examples: bool) -> None:
    """Model + centroid memories. Example patch memories (the sleep replay
    buffer, ~raw pixels) only ride along on the FINAL snapshot — phase-end
    snapshots stay centroid-sized (owner Q4)."""
    layers = []
    for layer in model.layers:
        entry = {
            "state_dict": layer.model.state_dict(),
            "ltm": layer.ltm.detach().cpu(),
            "ltm_task": layer.ltm_task.detach().cpu(),
            "stm": layer.stm.detach().cpu(),
            "stm_matches": layer.stm_matches.detach().cpu(),
            "stm_ages": layer.stm_ages.detach().cpu(),
            "distance_threshold": float(layer.distance_threshold),
            "sleep_cycles": int(layer.sleep_cycles),
            "cent_g": getattr(layer, "cent_g", None),
        }
        if with_examples:
            entry["ltm_examples"] = [
                [p.detach().cpu() for p in ex] for ex in layer.ltm_examples
            ]
            entry["stm_examples"] = [
                [p.detach().cpu() for p in ex] for ex in layer.stm_examples
            ]
        layers.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"layers": layers}, path)


def _pretrain_model_file(config) -> Path:
    """Where Layer.pretrain saves/loads the T0 encoder, relative to the cell
    dir (their smart_dir layout; layer name is L0)."""
    return Path(
        f"logs/{config.pretrain_log}/{config.dataset.name}"
        "/task_0/layer_L0/saved_models/final_model.pkl"
    )


def _run_eval(model, stream, task: int, step: int, out_dir: Path) -> dict:
    """Their eval (supervise -> classify -> cluster) + the Q3 auxiliary
    synthetic-inclusive clustering, persisted as one checkpoint record."""
    t_start = time.perf_counter()
    sup, evl = stream.eval_loaders(task)
    class_acc, class_pc, clust_acc, clust_pc = model.eval(sup, evl, task, step)
    record = {
        "step": step,
        "task": task,
        "class_acc": _to_jsonable(class_acc),
        "class_pc_acc": _to_jsonable(class_pc),
        "clust_acc": _to_jsonable(clust_acc),
        "clust_pc_acc": _to_jsonable(clust_pc),
        "aux_clust_acc": None,
        "aux_clust_pc_acc": None,
    }
    aux_loader = stream.cluster_loader(task)
    if aux_loader is not None:
        aux_acc, aux_pc = model.cluster(aux_loader, task, f"{step}_all")
        record["aux_clust_acc"] = _to_jsonable(aux_acc)
        record["aux_clust_pc_acc"] = _to_jsonable(aux_pc)
    record["eval_wall_time_s"] = time.perf_counter() - t_start
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "t0" if step < 0 else f"step_{step:05d}"
    (out_dir / f"{name}.json").write_text(json.dumps(record, indent=1))
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", required=True, choices=["resnet18", "resnet50"])
    parser.add_argument("--seed", required=True, type=int, choices=[42, 43, 44])
    parser.add_argument("--no-sleep", action="store_true",
                        help="model.sleep_on=False (their §4.4 ablation)")
    parser.add_argument("--out", required=True, help="cell directory (absolute)")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny-budget end-to-end run (see run_config)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pretrain-cache", default=None, metavar="DIR",
                        help="optional per-(arch,seed) T0 encoder cache; uses "
                             "their released load_pretrain mechanism. OFF by "
                             "default (Phase 4 owner decision pending).")
    args = parser.parse_args(argv)

    cell_dir = Path(args.out).resolve()
    config_dict = build_run_config(
        args.arch, args.seed, sleep_on=not args.no_sleep, smoke=args.smoke
    )

    # --- build the stream first: the schedule is part of the resolved config.
    from omegaconf import OmegaConf

    from fpcmc.config import FPCMCConfig
    from fpcmc.data import load_all_pools
    from fpcmc.protocols import build_p2
    from baselines.pcmc_sleep.stream_mirror import P2PixelMirror
    from baselines.pcmc_sleep.p2_stream import P2UPLStream

    fpcmc_config = FPCMCConfig.from_yaml(REPO_ROOT / "configs" / "fpcmc_default.yaml")
    pools = load_all_pools()
    mirror = P2PixelMirror(build_p2(fpcmc_config, args.seed, pools), pools)

    if args.smoke:
        smoke = config_dict["smoke"]
        schedule = {
            "sleep_steps": list(smoke["sleep_steps"]),
            "checkpoint_steps": list(smoke["checkpoint_steps"]),
            "snapshot_steps": [],
            "max_stream_steps": int(smoke["max_stream_steps"]),
        }
    else:
        schedule = {
            "sleep_steps": sleep_steps(mirror.phases),
            "checkpoint_steps": [int(s) for s in mirror.checkpoint_steps],
            "snapshot_steps": snapshot_steps(mirror.phases),
            "max_stream_steps": len(mirror),
        }
    config_dict["schedule"] = schedule
    resolved_yaml = yaml.safe_dump(config_dict, sort_keys=True)

    # --- run_cell resumability conventions.
    summary_path = cell_dir / "summary.json"
    resolved_path = cell_dir / "resolved_config.yaml"
    if (
        not args.force
        and summary_path.is_file()
        and resolved_path.is_file()
        and resolved_path.read_text(encoding="utf-8") == resolved_yaml
    ):
        print(f"cell up to date, skipping: {cell_dir}")
        return 0

    cell_dir.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(resolved_yaml, encoding="utf-8")
    if summary_path.is_file():
        summary_path.unlink()  # stale summary must not survive a failed rerun

    config = OmegaConf.create(config_dict)

    # Optional shared T0 encoder (their load_pretrain path, driver-side copy).
    cache_file = None
    if args.pretrain_cache:
        cache_key = (
            f"{args.arch}_seed{args.seed}_ep{config.model.init_epochs}"
            f"_img{config.dataset.img_size}.pkl"
        )
        cache_file = Path(args.pretrain_cache).resolve() / cache_key
        if cache_file.is_file():
            target = cell_dir / _pretrain_model_file(config)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cache_file, target)
            config.load_pretrain = True
            print(f"using cached T0 encoder: {cache_file}")

    # Their ungated smart_dir('logs/...') side effects land in the cell dir.
    os.chdir(cell_dir)
    set_seed(args.seed)

    stream = P2UPLStream(mirror, config, max_steps=schedule["max_stream_steps"])

    from core.models.pcmc.pcmc import PCMC

    t_run = time.perf_counter()
    model = PCMC(config)
    print(f"pretrain: {len(stream.pretrain_dataloader.dataset)} T0 images, "
          f"epochs={config.model.init_epochs}")
    model.pretrain(stream.pretrain_dataloader)
    if cache_file is not None and not cache_file.is_file():
        produced = cell_dir / _pretrain_model_file(config)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(produced, cache_file)
        print(f"cached T0 encoder: {cache_file}")

    checkpoints_dir = cell_dir / "checkpoints"
    records = [_run_eval(model, stream, 0, -1, checkpoints_dir)]

    pending = deque(schedule["sleep_steps"])
    checkpoint_set = set(schedule["checkpoint_steps"])
    snapshot_set = set(schedule["snapshot_steps"])
    sleep_times: list[int] = []

    for it, (data, _label, t) in enumerate(stream):
        _steer_sleep(model, pending)
        model(data, t)
        if pending and it == pending[0]:
            pending.popleft()
            sleep_times.append(it)
            print(f"sleep at step {it} (task {t}, sleep_on={config.model.sleep_on})")
        if it in checkpoint_set:
            print(f"checkpoint eval at step {it} (task {t})")
            records.append(_run_eval(model, stream, t, it, checkpoints_dir))
        if it in snapshot_set:
            _snapshot(model, cell_dir / "snapshots" / f"step_{it:05d}.pt",
                      with_examples=False)

    _snapshot(model, cell_dir / "final_state.pt", with_examples=True)

    summary = {
        "cell": {
            "system": f"pcmc_{args.arch}" + ("_nosleep" if args.no_sleep else "_sleep"),
            "protocol": "p2",
            "seed": args.seed,
            "smoke": bool(args.smoke),
        },
        "n_stream_steps": len(stream),
        "sleep_steps_executed": sleep_times,
        "checkpoints_written": len(records),
        "final_ltm_size": int(len(model.layers[0].ltm)),
        "final_class_acc": records[-1]["class_acc"],
        "final_clust_acc": records[-1]["clust_acc"],
        "wall_time_seconds": time.perf_counter() - t_run,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
    }
    summary_path.write_text(json.dumps(summary, indent=1))
    print(f"done: {cell_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
