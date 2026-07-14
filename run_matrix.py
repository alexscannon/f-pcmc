"""T15 run matrix: {system x protocol x seed} execution with resumability,
plus the PRD §8 sweep runner (TASKS T15; PRD §7.4, §8, NFR-2).

Systems are the PRD §7.4 run rows, one committed YAML each under configs/
(owner-approved inventory 2026-07-13): the main F-PCMC run (fpcmc_default),
the three baselines (b1_v1, b2_batch, b3_oracle) and the six ablations
(a1_global_tau, a2_no_stm, a3_no_recurrence, a4_no_merge, a5_knn_ref /
a5_vmf, a6_resnet50). Protocol (p1/p2) and seed ({42, 43, 44}) are matrix
axes supplied here, never config keys.

Owner-approved T15 decisions (Q&A 2026-07-13, recorded in docs/CHANGES.md):

  * B1 x P2 is UNSUPPORTED BY CONSTRUCTION: the vendored v1 pipeline builds
    its own P1 stream internally and cannot consume a P2 stream without
    modifying the frozen port (whose regression pin covers P1/seed 42 only).
    ``plan_matrix`` marks the cell, ``run_cell`` raises ``CellUnsupported``,
    and the T16 scorecard reports the gap.
  * P2 window re-derivation (PRD §11: "m_windows/W chosen so a phase spans
    >= 6 windows; window params may be re-derived from phase length"; flagged
    at T12): every P2 cell runs with ``window_W = P2_WINDOW_W`` (= 50 — the
    shortest P2 phase is the 357-step near-OOD phase, and 357 // 50 = 7 >= 6
    windows; m_windows stays at its §8 default). The override is applied to
    the resolved config recorded in every P2 cell; the committed configs stay
    protocol-neutral.
  * A6 encoder mapping: ``encoder`` selects the embeddings directory — the
    named sibling of the roots.env-resolved EMBEDDINGS_DIR (docs/ASSETS.md
    §1: DINOv3_large_32px primary, ResNet50_32px for A6, same schema).
  * Resumability: a cell is skipped iff its ``summary.json`` exists AND its
    recorded ``resolved_config.yaml`` matches the config this invocation
    would run — a changed config re-runs rather than silently reusing stale
    results; ``force=True`` always re-runs. (B1 additionally honours v1's own
    resume semantics inside the cell.)
  * Sweeps (``run_sweep``) are limited to the three PRD §8 sweep parameters
    (``SWEEP_PARAMS``), on P1, seed 42, the fpcmc_default system only —
    "sweeps limited to the three marked parameters, on P1 only, seed 42
    only" is the PRD §8 literal, enforced structurally here
    (test_sweep_scope_guard).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = REPO_ROOT / "configs"

#: PRD §7.4 run rows -> committed config files (owner-approved inventory).
SYSTEM_CONFIGS: dict[str, Path] = {
    name: CONFIGS_DIR / f"{name}.yaml"
    for name in (
        "fpcmc_default",
        "b1_v1",
        "b2_batch",
        "b3_oracle",
        "a1_global_tau",
        "a2_no_stm",
        "a3_no_recurrence",
        "a4_no_merge",
        "a5_knn_ref",
        "a5_vmf",
        "a6_resnet50",
    )
}

PROTOCOLS = ("p1", "p2")
SEEDS = (42, 43, 44)

#: The ONLY sweepable keys (PRD §8: "sweeps limited to the three marked
#: parameters"). test_sweep_scope_guard pins this.
SWEEP_PARAMS = ("stm_capacity", "theta_promote", "min_cohesion_ratio")

#: P2 window_W re-derivation (PRD §11; owner-approved 2026-07-13): the
#: shortest P2 phase (near-OOD, 357 steps at seed 42) must span >= 6 windows
#: of length W -> W <= 59; 50 gives 7 windows. m_windows keeps its §8 default.
P2_WINDOW_W = 50

#: encoder key -> embeddings directory name (docs/ASSETS.md §1).
ENCODER_DIRS = {"dinov3_vitl16": "DINOv3_large_32px", "resnet50": "ResNet50_32px"}


class CellUnsupported(RuntimeError):
    """A matrix cell that cannot be run by construction (B1 x P2)."""


# ----------------------------------------------------------------- resolution


def embeddings_dir_for_encoder(encoder: str) -> Path:
    """The pool directory for an encoder: the ENCODER_DIRS-named sibling of
    the roots.env-resolved EMBEDDINGS_DIR (raises EmbeddingsUnavailable if
    the directory does not exist on this machine)."""
    from fpcmc.data import EmbeddingsUnavailable, resolve_embeddings_dir

    resolved = resolve_embeddings_dir()
    wanted = ENCODER_DIRS[encoder]
    path = resolved if resolved.name == wanted else resolved.parent / wanted
    if not path.is_dir():
        raise EmbeddingsUnavailable(
            f"encoder {encoder!r} needs embeddings at {path}, which does not exist"
        )
    return path


def default_out_root() -> Path:
    """${DATA_ROOT}/evaluation/f_pcmc_runs — run artifacts never live in-repo."""
    from fpcmc.data import EmbeddingsUnavailable, read_roots_env

    env = read_roots_env()
    data_root = env.get("DATA_ROOT")
    if not data_root:
        raise EmbeddingsUnavailable("roots.env does not define DATA_ROOT")
    return Path(data_root) / "evaluation" / "f_pcmc_runs"


def resolve_cell_config(system: str, protocol: str, seed: int):
    """The exact FPCMCConfig a cell runs: committed YAML + the seed axis +
    the P2 window override. This is what resolved_config.yaml records."""
    from fpcmc.config import FPCMCConfig

    config = FPCMCConfig.from_yaml(SYSTEM_CONFIGS[system])
    config = dataclasses.replace(config, seed=int(seed))
    if protocol == "p2":
        config = dataclasses.replace(config, window_W=P2_WINDOW_W)
    return config


def plan_matrix(
    systems: Optional[Sequence[str]] = None,
    protocols: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
) -> list[dict]:
    """All requested cells in deterministic order, unsupported ones marked."""
    cells = []
    for system in systems or SYSTEM_CONFIGS:
        if system not in SYSTEM_CONFIGS:
            raise ValueError(f"unknown system {system!r}; known: {sorted(SYSTEM_CONFIGS)}")
        for protocol in protocols or PROTOCOLS:
            if protocol not in PROTOCOLS:
                raise ValueError(f"unknown protocol {protocol!r}; known: {PROTOCOLS}")
            for seed in seeds or SEEDS:
                cells.append({
                    "system": system,
                    "protocol": protocol,
                    "seed": int(seed),
                    "supported": not (system == "b1_v1" and protocol == "p2"),
                })
    return cells


# ------------------------------------------------------------- serialization


def _jsonable(obj):
    """Strict-JSON view of a metrics report: NaN/inf -> null, numpy -> python."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _write_summary(cell_dir: Path, summary: dict) -> None:
    (cell_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ------------------------------------------------------------ cell execution


def run_cell(
    system: str,
    protocol: str,
    seed: int,
    *,
    out_root: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Run one matrix cell; return its directory (containing summary.json).

    Resumable: skipped iff summary.json exists and the recorded resolved
    config matches what this invocation would run (force=True re-runs).
    """
    if system not in SYSTEM_CONFIGS:
        raise ValueError(f"unknown system {system!r}; known: {sorted(SYSTEM_CONFIGS)}")
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; known: {PROTOCOLS}")
    if system == "b1_v1" and protocol == "p2":
        raise CellUnsupported(
            "B1 x P2: the vendored v1 pipeline builds its own P1 stream and "
            "cannot consume a P2 stream without modifying the frozen port "
            "(owner ruling 2026-07-13); report the gap, do not shim it"
        )

    root = Path(out_root) if out_root is not None else default_out_root()
    cell_dir = root / system / f"{protocol}_seed{int(seed)}"
    config = resolve_cell_config(system, protocol, seed)
    resolved_yaml = config.to_yaml()

    summary_path = cell_dir / "summary.json"
    resolved_path = cell_dir / "resolved_config.yaml"
    if (
        not force
        and summary_path.is_file()
        and resolved_path.is_file()
        and resolved_path.read_text(encoding="utf-8") == resolved_yaml
    ):
        return cell_dir

    cell_dir.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(resolved_yaml, encoding="utf-8")

    if system == "b1_v1":
        summary = _run_b1(cell_dir, config, force=force)
    elif system == "b2_batch":
        summary = _run_b2(cell_dir, config, protocol)
    elif system == "b3_oracle":
        summary = _run_b3(cell_dir, config, protocol)
    else:
        summary = _run_fpcmc(cell_dir, config, protocol)

    summary["cell"] = {"system": system, "protocol": protocol, "seed": int(seed)}
    _write_summary(cell_dir, summary)
    return cell_dir


def _build_protocol(protocol: str, config, pools):
    from fpcmc.protocols import build_p1, build_p2

    if protocol == "p1":
        return build_p1(config, config.seed, pools)
    return build_p2(config, config.seed, pools)


def _init_store(stream, pools, config):
    """FR-2.1 LTM init from the protocol's T0 classes (P1: all 100; P2: the
    frozen 80-class split carried on the built stream)."""
    from fpcmc.init import initialize_ltm
    from fpcmc.thresholds import compute_global_prior

    ref = pools["ind_reference"]
    labels = np.asarray(ref.subclass_names, dtype=str)
    mask = np.isin(labels, np.asarray(stream.t0_classes, dtype=str))
    store = initialize_ltm(ref.x[mask], labels[mask], config)
    prior = compute_global_prior(store.ltm, config)
    return store, prior


def _run_fpcmc(cell_dir: Path, config, protocol: str) -> dict:
    """F-PCMC (main system or A1-A6 ablation): protocol -> LTM init ->
    StreamRunner -> §7.3 report from the log alone."""
    from eval.gt import StreamGroundTruth
    from eval.harness import evaluate_run
    from fpcmc.data import load_all_pools
    from fpcmc.stream import StreamRunner

    pools = load_all_pools(embeddings_dir_for_encoder(config.encoder))
    stream = _build_protocol(protocol, config, pools)
    store, prior = _init_store(stream, pools, config)

    log_path = cell_dir / "events.jsonl"
    runner = StreamRunner(
        config, store, prior,
        log_path=log_path,
        checkpoint_steps=stream.checkpoint_steps,
    )
    runner.run(stream.x)

    gt = StreamGroundTruth.from_manifest(stream)
    return evaluate_run(log_path, gt)


def _run_b1(cell_dir: Path, config, *, force: bool) -> dict:
    """B1: the vendored v1 subprocess (P1 only), outputs adapted to the T13
    schema so the same harness scores it."""
    from baselines.v1_stream import run_v1, v1_ground_truth, v1_run_to_jsonl
    from eval.harness import evaluate_run

    v1_dir = run_v1(cell_dir / "v1", seed=config.seed, force=force)
    log_path = v1_run_to_jsonl(v1_dir, cell_dir / "events.jsonl")
    gt = v1_ground_truth(v1_dir)
    return evaluate_run(log_path, gt)


def _run_b2(cell_dir: Path, config, protocol: str) -> dict:
    """B2: the static batch knn_vmf detector at each protocol checkpoint."""
    from baselines.batch_knn_vmf import evaluate_batch_checkpoints
    from fpcmc.data import load_all_pools

    pools = load_all_pools(embeddings_dir_for_encoder(config.encoder))
    stream = _build_protocol(protocol, config, pools)
    return evaluate_batch_checkpoints(stream, pools)


def _run_b3(cell_dir: Path, config, protocol: str) -> dict:
    """B3: the ground-truth oracle ceiling through the unmodified harness."""
    from baselines.oracle import run_oracle
    from eval.gt import StreamGroundTruth
    from eval.harness import evaluate_run
    from fpcmc.data import load_all_pools

    pools = load_all_pools(embeddings_dir_for_encoder(config.encoder))
    stream = _build_protocol(protocol, config, pools)
    gt = StreamGroundTruth.from_manifest(stream)
    log_path = run_oracle(
        gt, config, cell_dir / "events.jsonl",
        checkpoint_steps=stream.checkpoint_steps,
    )
    return evaluate_run(log_path, gt)


# ------------------------------------------------------------------- sweeps


def run_sweep(
    param: str,
    values: Sequence,
    *,
    out_root: str | Path | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """The PRD §8 sweep: one parameter over its values, fpcmc_default on
    P1/seed 42 only. Any parameter outside SWEEP_PARAMS is rejected — that
    scope guard is the point (test_sweep_scope_guard)."""
    if param not in SWEEP_PARAMS:
        raise ValueError(
            f"{param!r} is not a sanctioned sweep parameter — PRD §8 limits "
            f"sweeps to {SWEEP_PARAMS} (on P1, seed 42 only)"
        )
    plan = [
        {"system": "fpcmc_default", "protocol": "p1", "seed": 42,
         "param": param, "value": v}
        for v in values
    ]
    if dry_run:
        return plan

    root = Path(out_root) if out_root is not None else default_out_root()
    for cell in plan:
        config = resolve_cell_config("fpcmc_default", "p1", 42)
        config = dataclasses.replace(config, **{param: type(getattr(config, param))(cell["value"])})
        cell_dir = root / "sweep" / f"{param}={cell['value']}" / "p1_seed42"
        resolved_yaml = config.to_yaml()
        summary_path = cell_dir / "summary.json"
        resolved_path = cell_dir / "resolved_config.yaml"
        if (
            not force
            and summary_path.is_file()
            and resolved_path.is_file()
            and resolved_path.read_text(encoding="utf-8") == resolved_yaml
        ):
            cell["dir"] = str(cell_dir)
            continue
        cell_dir.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(resolved_yaml, encoding="utf-8")
        summary = _run_fpcmc(cell_dir, config, "p1")
        summary["cell"] = {k: cell[k] for k in ("system", "protocol", "seed", "param", "value")}
        _write_summary(cell_dir, summary)
        cell["dir"] = str(cell_dir)
    return plan


# ---------------------------------------------------------------------- CLI


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--systems", help=f"comma list of {sorted(SYSTEM_CONFIGS)}")
    parser.add_argument("--protocols", help="comma list of p1,p2")
    parser.add_argument("--seeds", help="comma list, default 42,43,44")
    parser.add_argument("--out", help="output root (default ${DATA_ROOT}/evaluation/f_pcmc_runs)")
    parser.add_argument("--force", action="store_true", help="re-run cells even if resumable")
    parser.add_argument(
        "--sweep", metavar="PARAM=V1,V2,...",
        help=f"sweep mode: one of {SWEEP_PARAMS} on P1/seed 42 (exclusive with matrix args)",
    )
    args = parser.parse_args(argv)

    if args.sweep:
        if args.systems or args.protocols or args.seeds:
            parser.error("--sweep is exclusive with --systems/--protocols/--seeds")
        param, _, raw = args.sweep.partition("=")
        if not raw:
            parser.error("--sweep needs PARAM=V1,V2,...")
        values = [float(v) if "." in v else int(v) for v in raw.split(",")]
        for cell in run_sweep(param, values, out_root=args.out, force=args.force):
            print(f"DONE sweep {cell['param']}={cell['value']} -> {cell['dir']}")
        return 0

    systems = args.systems.split(",") if args.systems else None
    protocols = args.protocols.split(",") if args.protocols else None
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    for cell in plan_matrix(systems, protocols, seeds):
        label = f"{cell['system']} x {cell['protocol']} x seed{cell['seed']}"
        if not cell["supported"]:
            print(f"SKIP (unsupported by construction) {label}: vendored v1 "
                  f"builds its own P1 stream; B1 x P2 is reported as a gap")
            continue
        cell_dir = run_cell(
            cell["system"], cell["protocol"], cell["seed"],
            out_root=args.out, force=args.force,
        )
        print(f"DONE {label} -> {cell_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
