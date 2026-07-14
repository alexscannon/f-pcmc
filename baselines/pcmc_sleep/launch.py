"""T17 Phase 2: CPU-side launcher for the PCMC GPU driver.

Runs in the repo's pinned CPU env; the actual work happens in a subprocess
using the separate GPU env's interpreter (``env/.venv/bin/python`` — see
``env/README.md``). Mirrors run_matrix.run_cell's shape: one cell per
(arch, sleep-variant, seed) under
``${DATA_ROOT}/evaluation/f_pcmc_runs/pcmc_sleep/``; the driver itself owns
the resolved-config/summary resumability check.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
ENV_PYTHON = HERE / "env" / ".venv" / "bin" / "python"
DRIVER = HERE / "driver.py"

ARCHS = ("resnet18", "resnet50")
SEEDS = (42, 43, 44)


def gpu_env_available() -> tuple[bool, str]:
    """(ok, reason) — cheap checks first, CUDA probe last (subprocess)."""
    if not ENV_PYTHON.is_file():
        return False, f"GPU env not built: {ENV_PYTHON} missing (env/README.md)"
    probe = subprocess.run(
        [str(ENV_PYTHON), "-c",
         "import torch; raise SystemExit(0 if torch.cuda.is_available() else 3)"],
        capture_output=True, text=True, timeout=120,
    )
    if probe.returncode == 3:
        return False, "GPU env present but CUDA unavailable"
    if probe.returncode != 0:
        return False, f"GPU env python failed: {probe.stderr.strip()[:200]}"
    return True, ""


def default_out_root() -> Path:
    from fpcmc.data import read_roots_env

    env = read_roots_env()
    root = env.get("DATA_ROOT")
    if not root:
        raise FileNotFoundError("roots.env does not define DATA_ROOT")
    return Path(root) / "evaluation" / "f_pcmc_runs" / "pcmc_sleep"


def cell_dir(out_root: Path, arch: str, sleep_on: bool, seed: int) -> Path:
    system = f"pcmc_{arch}_" + ("sleep" if sleep_on else "nosleep")
    return Path(out_root) / system / f"p2_seed{int(seed)}"


def run_cell(
    arch: str,
    seed: int,
    *,
    sleep_on: bool = True,
    out_root: str | Path | None = None,
    smoke: bool = False,
    force: bool = False,
    pretrain_cache: str | Path | None = None,
    timeout: float | None = None,
) -> Path:
    """Launch one driver run; returns the cell dir. Raises CalledProcessError
    on a failed run and TimeoutExpired on a hung one (the smoke test uses the
    timeout to make the DataLoader-deadlock failure mode loud)."""
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; known: {ARCHS}")
    if seed not in SEEDS:
        raise ValueError(f"seed {seed} not in the T17 matrix {SEEDS}")
    ok, reason = gpu_env_available()
    if not ok:
        raise RuntimeError(reason)

    root = Path(out_root) if out_root is not None else default_out_root()
    cell = cell_dir(root, arch, sleep_on, seed)
    cmd = [
        str(ENV_PYTHON), str(DRIVER),
        "--arch", arch, "--seed", str(int(seed)), "--out", str(cell),
    ]
    if not sleep_on:
        cmd.append("--no-sleep")
    if smoke:
        cmd.append("--smoke")
    if force:
        cmd.append("--force")
    if pretrain_cache is not None:
        cmd += ["--pretrain-cache", str(pretrain_cache)]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, timeout=timeout)
    return cell


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", required=True, choices=list(ARCHS))
    parser.add_argument("--seed", required=True, type=int, choices=list(SEEDS))
    parser.add_argument("--no-sleep", action="store_true")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pretrain-cache", default=None)
    args = parser.parse_args(argv)
    cell = run_cell(
        args.arch,
        args.seed,
        sleep_on=not args.no_sleep,
        out_root=args.out_root,
        smoke=args.smoke,
        force=args.force,
        pretrain_cache=args.pretrain_cache,
    )
    print(cell)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
