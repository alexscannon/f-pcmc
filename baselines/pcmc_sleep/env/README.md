# GPU environment for the T17 PCMC baseline

Self-contained uv project for `baselines/pcmc_sleep/driver.py` — the only
code in this repo that touches a GPU. The repo root project (CPU-only torch
2.11.0, python 3.14) is untouched by this; never `uv add` PCMC deps there.

## Setup (one-time)

```bash
cd baselines/pcmc_sleep/env
uv sync            # creates ./.venv with python 3.11 + torch 2.5.1+cu121
```

Validated against NVIDIA driver 580.173.02 / RTX 3090 (Phase 0.2,
2026-07-14). The interpreter the launcher and the smoke test look for is
`baselines/pcmc_sleep/env/.venv/bin/python`.

## Before running anything

- **The 3090 is shared**: a llama.cpp server (`llama-server`, port 9401)
  usually holds ~17.7 GB of the 24 GB. Check `nvidia-smi` first and budget
  PCMC for the remaining ~6 GB (RN18 @ bs 256 @ 120 px fits; RN50 may not).
  Raise GPU scheduling with the owner before production runs.
- **KeOps JIT-compiles its CUDA kernels on first use** (~30 s warm-up,
  cached under `~/.cache/keops*`). The first `init_memory` k-means of a
  fresh env is expected to stall there once.

## Running the driver

Prefer the repo-side launcher (CPU env), which shells out to this venv:

```bash
uv run python baselines/pcmc_sleep/launch.py --arch resnet18 --seed 42 --smoke
```

Direct invocation from the repo root works too:

```bash
baselines/pcmc_sleep/env/.venv/bin/python baselines/pcmc_sleep/driver.py \
    --arch resnet18 --seed 42 --out <cell_dir> --smoke
```
