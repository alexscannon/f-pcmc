"""T17 Phase 3: score archived F-PCMC cells under the paper protocol (CLI).

CPU-side, root env, no GPU. For each requested cell (system x seed):

 1. read the archived ``events.jsonl`` config_header (the authoritative
    resolved config — archived cells are never regenerated, HANDOFF §5);
 2. load that encoder's pools and rebuild the P2 stream + initial store
    exactly as the live run did (``run_matrix._init_store`` — FR-2.1 init on
    the frozen 80-class split + the frozen global prior);
 3. walk ``fpcmc.replay.iter_checkpoint_states`` over the log (owner Q8a),
    verifying EVERY reconstructed state against its own checkpoint record
    (counts + full tau snapshot — replay is bit-reproducible, so any
    mismatch aborts the cell);
 4. score the task-0 state + all 44 checkpoints with
    ``fpcmc_scorer.score_checkpoint`` (owner Q7: tier-1 headline fields +
    ltm_only block) and persist records in the Phase 2 driver's
    ``checkpoints/*.json`` shape (``t0.json`` with step=-1/task=0,
    ``step_NNNNN.json`` otherwise — same field names, same step keys, so
    Phase 5 reads both systems uniformly; NaN -> JSON null);
 5. mirror ``run_matrix.run_cell`` resumability: ``scoring_config.yaml`` +
    ``summary.json``, skip iff both exist and the config matches, --force
    re-scores; plus a root ``manifest.json`` of everything scored.

Outputs (owner Q10c):
``${DATA_ROOT}/evaluation/f_pcmc_runs/pcmc_sleep/paper_protocol/<system>/
p2_seed<seed>/``.

Usage (repo root):
    uv run python baselines/pcmc_sleep/score_fpcmc.py            # all 6 cells
    uv run python baselines/pcmc_sleep/score_fpcmc.py --systems fpcmc_default \
        --seeds 42 --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

from baselines.pcmc_sleep.fpcmc_scorer import (
    score_checkpoint,
    verify_checkpoint_record,
)
from baselines.pcmc_sleep.run_config import CLUST_SIZE, RHO, SUP_SIZE, TEST_SIZE
from fpcmc.config import FPCMCConfig
from fpcmc.replay import iter_checkpoint_states, read_log

SYSTEMS = ("fpcmc_default", "a6_resnet50")
SEEDS = (42, 43, 44)


def scoring_config(system: str, seed: int, source: Path) -> dict:
    """What fully determines a scoring run (the resumability key)."""
    return {
        "system": system,
        "seed": int(seed),
        "protocol": "p2",
        "source_events": str(source),
        "sup_size": SUP_SIZE,
        "test_size": TEST_SIZE,
        "clust_size": CLUST_SIZE,
        "rho": RHO,
        "clustering": "analytic_collapse",  # owner Q9
        "populations": ["tier1", "ltm_only"],  # owner Q7
    }


def _task_of_step(phases, step: int) -> int:
    for i, p in enumerate(phases):
        if p.start <= step < p.end:
            return i + 1
    raise ValueError(f"step {step} outside every phase")


def score_cell(
    system: str,
    seed: int,
    *,
    cells_root: Path,
    out_root: Path,
    force: bool = False,
    pools_cache: dict | None = None,
) -> Path:
    """Score one archived cell; return its output directory."""
    from run_matrix import _init_store, _jsonable, embeddings_dir_for_encoder
    from fpcmc.data import load_all_pools
    from fpcmc.protocols import build_p2

    source = cells_root / system / f"p2_seed{seed}" / "events.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"archived cell log not found: {source}")
    out_dir = out_root / system / f"p2_seed{seed}"
    sconfig = scoring_config(system, seed, source)
    sconfig_yaml = yaml.safe_dump(sconfig, sort_keys=True)

    summary_path = out_dir / "summary.json"
    sconfig_path = out_dir / "scoring_config.yaml"
    if (
        not force
        and summary_path.is_file()
        and sconfig_path.is_file()
        and sconfig_path.read_text(encoding="utf-8") == sconfig_yaml
    ):
        print(f"up to date, skipping: {out_dir}")
        return out_dir

    records = read_log(source)
    header = records[0]
    config = FPCMCConfig.from_yaml_text(yaml.safe_dump(header["config"]))
    assert config.seed == seed, (config.seed, seed)

    key = config.encoder
    if pools_cache is not None and key in pools_cache:
        pools = pools_cache[key]
    else:
        pools = load_all_pools(embeddings_dir_for_encoder(config.encoder))
        if pools_cache is not None:
            pools_cache[key] = pools

    stream = build_p2(config, seed, pools)
    assert list(stream.checkpoint_steps) == list(header["checkpoint_steps"])
    store, prior = _init_store(stream, pools, config)
    cp_records = {r["step"]: r for r in records if r["type"] == "checkpoint"}

    out_dir.mkdir(parents=True, exist_ok=True)
    sconfig_path.write_text(sconfig_yaml, encoding="utf-8")
    if summary_path.is_file():
        summary_path.unlink()  # stale summary must not survive a failed rerun
    cp_dir = out_dir / "checkpoints"
    cp_dir.mkdir(exist_ok=True)

    t_run = time.perf_counter()
    written = []
    for step, state in iter_checkpoint_states(source, stream.x, store, prior):
        t_eval = time.perf_counter()
        if step < 0:
            task = 0
        else:
            verify_checkpoint_record(state, cp_records[step])
            task = _task_of_step(stream.phases, step)
        record = {"step": int(step), "task": task}
        record.update(score_checkpoint(state, stream, pools, task, seed))
        record["eval_wall_time_s"] = time.perf_counter() - t_eval
        name = "t0" if step < 0 else f"step_{step:05d}"
        (cp_dir / f"{name}.json").write_text(
            json.dumps(_jsonable(record), indent=1) + "\n", encoding="utf-8"
        )
        written.append(record)
        print(
            f"  {system}/p2_seed{seed} {name}: task {task}, "
            f"class_acc {record['class_acc']:.2f}, "
            f"clust_acc {record['clust_acc']:.2f}, "
            f"n_tier1 {record['n_tier1']} ({record['eval_wall_time_s']:.1f}s)"
        )

    assert len(written) == len(cp_records) + 1, "missed a checkpoint"
    summary = {
        "cell": {"system": f"{system}_paper_protocol", "protocol": "p2",
                 "seed": int(seed)},
        "source_events": str(source),
        "checkpoints_written": len(written),
        "final_class_acc": written[-1]["class_acc"],
        "final_clust_acc": written[-1]["clust_acc"],
        "final_class_acc_ltm_only": written[-1]["ltm_only"]["class_acc"],
        "final_n_tier1": written[-1]["n_tier1"],
        "final_n_ltm": written[-1]["n_ltm"],
        "wall_time_seconds": time.perf_counter() - t_run,
    }
    summary_path.write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"done: {out_dir} ({summary['wall_time_seconds']:.0f}s)")
    return out_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS),
                        choices=list(SYSTEMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS),
                        choices=list(SEEDS))
    parser.add_argument("--cells-root", default=None,
                        help="archived F-PCMC cells root "
                             "(default ${DATA_ROOT}/evaluation/f_pcmc_runs)")
    parser.add_argument("--out-root", default=None,
                        help="output root (default <cells-root>/pcmc_sleep/"
                             "paper_protocol — owner Q10c)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    from run_matrix import default_out_root

    cells_root = Path(args.cells_root) if args.cells_root else default_out_root()
    out_root = (
        Path(args.out_root) if args.out_root
        else cells_root / "pcmc_sleep" / "paper_protocol"
    )

    pools_cache: dict = {}
    scored = {}
    for system in args.systems:
        for seed in args.seeds:
            out = score_cell(
                system, seed,
                cells_root=cells_root, out_root=out_root,
                force=args.force, pools_cache=pools_cache,
            )
            summary = json.loads((out / "summary.json").read_text())
            scored[f"{system}/p2_seed{seed}"] = {
                "path": str(out),
                "checkpoints": summary["checkpoints_written"],
                "final_class_acc": summary["final_class_acc"],
                "final_clust_acc": summary["final_clust_acc"],
            }

    manifest_path = out_root / "manifest.json"
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    manifest.update(scored)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(f"manifest: {manifest_path} ({len(manifest)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
