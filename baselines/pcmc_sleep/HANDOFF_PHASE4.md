# HAND-OFF: T17 Phase 4 — the production run matrix (12 PCMC GPU cells)

You are implementing **Phase 4 of task T17** in this repository. Read this
file completely, then `CLAUDE.md` (repo root, binding), then
`baselines/pcmc_sleep/PLAN.md` (task plan, Phase 0–3 findings, the full
owner-decision log), then `baselines/pcmc_sleep/HANDOFF_PHASE2.md` §5 and
`HANDOFF_PHASE3.md` §5 (hard-won upstream code facts — still binding). Work
on branch `task/T17-sleep` (exists; do not rebase or merge). **Post a short
plan and STOP to ask the owner the open questions in §8 before launching any
production run** — the Q&A-before-code workflow is mandatory (CLAUDE.md
"Per-task workflow"); Phases 0–3 all followed it and their answers are
recorded verbatim in PLAN.md. Phase 4's open questions are almost entirely
about **GPU budget and scheduling on a shared card** — this is the one phase
where launching before the owner signs off can burn real days of wall-clock.

## 1. Mission context (why this phase exists)

T17 closes the 2×2 {encoder}×{sleep} comparison on the shared CIFAR-100 P2
stream:

| | Sleep-retrained | Frozen |
|---|---|---|
| ResNet-18/50 | **Phase 4 — this work** | A6 (T16 archive, rescored in Phase 3) |
| DINOv3 | n/a by construction | fpcmc_default (T16 archive, rescored in Phase 3) |

plus the paper's own **no-sleep** PCMC variant (`model.sleep_on=False`, their
§4.4 ablation) as the bridge cell.

**Pre-registered decision rule** (PLAN.md top): "sleep is removable" iff
F-PCMC/DINOv3 ≥ PCMC-sleep/RN{18,50} on the shared primary metrics at every
P2 checkpoint (3 seeds, mean ± std); "the encoder is what removes it" iff
additionally PCMC-no-sleep/RN sits well below both while A6 shows frozen-RN50
failing inside the F-PCMC machinery.

**Phase 4 produces the missing left column**: run the actual 2024-paper PCMC
(sleep-retrained and no-sleep, RN18 and RN50, 3 seeds) on our P2 stream, with
their Eq. 6–7 eval running IN-RUN at the 44 P2 checkpoints (the Phase 2
driver already does this; Phase 3 settled the eval sizing that makes it
feasible). Phase 4 does **not** build the workbook/report (Phase 5) and does
**not** touch the already-scored F-PCMC/A6 cells.

## 2. The matrix — exactly what to produce

**12 cells** = {resnet18, resnet50} × {sleep, no-sleep} × seeds {42, 43, 44}.
The launcher owns the naming; each cell lands at:

```
${DATA_ROOT}/evaluation/f_pcmc_runs/pcmc_sleep/
    pcmc_{arch}_{sleep|nosleep}/p2_seed{N}/
```

e.g. `pcmc_resnet18_sleep/p2_seed42`, `pcmc_resnet50_nosleep/p2_seed44`.
(This root is a sibling of Phase 3's `pcmc_sleep/paper_protocol/…` tree —
keep them separate; Phase 5 reconciles the two.)

Per cell, the driver writes (Phase 2, verified): `resolved_config.yaml`
(embeds the computed `schedule` block), `summary.json`, `checkpoints/t0.json`
(step=-1, task 0) + `checkpoints/step_NNNNN.json` × 44 (fields: `step`,
`task`, `class_acc`, `class_pc_acc`, `clust_acc`, `clust_pc_acc`,
`aux_clust_acc`, `aux_clust_pc_acc`, `eval_wall_time_s`), phase-end
`snapshots/step_NNNNN.pt` × 11, and `final_state.pt` (state_dict + ltm/stm
centroids + `cent_g`; final also carries example patch memories). **This is
the same checkpoint-JSON shape Phase 3 emitted for the F-PCMC side**, by
design, so Phase 5 reads both systems uniformly.

**No-sleep is not cheap.** Both variants pay the full T0 contrastive
pretrain; `--no-sleep` only sets `model.sleep_on=False`, which skips the 11
in-stream sleep-retraining cycles. So per (arch, seed) the sleep and no-sleep
cells differ **only** in the sleep cycles and can legitimately branch from
the *identical* T0 encoder — see §8 Q11.

## 3. What already exists (verified 2026-07-14, through commit `6539f48`)

- **The driver + launcher are built and smoke-green** (Phase 2, 12 tests in
  `tests/test_pcmc_driver.py`; GPU smoke ~86 s on the 3090):
  - `baselines/pcmc_sleep/launch.py` — CPU-side launcher. Public API:
    `run_cell(arch, seed, *, sleep_on=True, out_root=None, smoke=False,
    force=False, pretrain_cache=None, timeout=None)` and a `__main__` CLI
    (`--arch --seed [--no-sleep] [--out-root] [--smoke] [--force]
    [--pretrain-cache DIR]`). It shells out to the GPU env interpreter
    (`env/.venv/bin/python driver.py …`), raising `CalledProcessError` on a
    failed run and `TimeoutExpired` on a hung one. `gpu_env_available()`
    returns `(ok, reason)`; `default_out_root()` →
    `${DATA_ROOT}/evaluation/f_pcmc_runs/pcmc_sleep`; `cell_dir(...)` gives
    the exact per-cell path.
  - `baselines/pcmc_sleep/driver.py` (GPU env) — one cell end-to-end: T0
    pretrain → 21,538-step wake loop with phase-midpoint sleeps → their eval
    at each checkpoint → artifacts. **Resumable** (run_matrix.run_cell
    conventions): skips iff `summary.json` + `resolved_config.yaml` exist and
    the resolved config matches; `--force` re-runs. Owns two driver-side
    interventions (both HANDOFF_PHASE2 §5 facts): the global DataLoader
    force-patch to in-process loading (deadlock guard) and sleep-schedule
    steering (`layer.sleep_start`/`sleep_freq`; trigger code untouched). It
    also has an OFF-by-default `--pretrain-cache DIR` implementing their
    released `load_pretrain` mechanism as a driver-side file copy (§8 Q11).
  - `run_config.py` (torch-free) resolves the paper-faithful config:
    `pretrained=False`, both epoch knobs = 500, their-120 geometry
    (img 120 / patch 60 / stride 30), θ=30 Δ=400 α=0.1 β=0.99 M=30, CIFAR
    mean/std, `feat_size` 512 RN18 / 2048 RN50, and the **Phase 3 Q6 sizing**
    (`SUP_SIZE=TEST_SIZE=100`, `CLUST_SIZE=25`, `RHO=0.25`).
  - GPU env at `baselines/pcmc_sleep/env/` (`uv sync` inside it; py 3.11,
    torch 2.5.1+cu121 + the Phase 0.2 recipe; `uv.lock` committed). See its
    README.
- **Phase 3 unblocked the in-run eval cost** (owner decision Q6, PLAN.md):
  their clustering is O(N²) python-level jaccard + spectral; at full 100/class
  it costs ~2.9 h/run, infeasible ×12. Q6 subsamples the *clustering* eval to
  25/class (classification stays 100/100), measured **0.19 h/run**, worst
  checkpoint ~24 s. This is baked into `run_config.CLUST_SIZE` and the
  `P2UPLStream.clust_loader`/`cluster_loader`; the driver feeds those to
  `model.cluster()`. **The sizing is final — it was the Q6 gate on this
  phase. Do not change it without re-running the Phase 3 Q&A.**
- **Phase 3 already scored the frozen-side cells** under the identical paper
  protocol (headline tier-1, PLAN.md): fpcmc_default (DINOv3) t0 91.5 class /
  91.9 purity → final ~78–81; a6_resnet50 (frozen RN50) t0 63.9 / 63.3 →
  final ~51. Those are archived under `pcmc_sleep/paper_protocol/`. **Phase 4
  does not re-touch them** — it only adds the 12 PCMC cells.

## 4. Phase 4 deliverables

1. **The 12 PCMC cells run to completion** and archived under
   `pcmc_sleep/pcmc_{arch}_{variant}/p2_seed{N}/`, each with its full
   artifact set (§2), resumable and idempotent (re-invoking a completed cell
   is a no-op).
2. **A batch runner** (thin, ours; may live in `launch.py` as a `--matrix`
   mode or a small `run_pcmc_matrix.py`) that iterates the 12 cells in a
   defined order, honours resumability, records per-cell wall time, and
   implements whatever GPU-scheduling / pretrain-sharing the owner picks in
   §8. Small [U] test for the cell enumeration + order (no GPU); the actual
   runs are the [I] deliverable, not a CI test.
3. **A run manifest / provenance record** (JSON under the `pcmc_sleep/` root)
   listing each cell, its resolved-config hash, wall time, sleep steps
   executed, final LTM size, and final class/clust acc — the cheap index
   Phase 5 and the owner read without opening 12 dirs.
4. **NFR-1 budget check**: record measured per-cell wall time against the
   projection (§5) and surface any cell that blew the budget. (T17's GPU runs
   are exempt from the CPU NFR-1 numbers, but the owner wants the actuals.)
5. **PLAN.md updated** (Phase 4 checked off, §8 Q&A recorded verbatim, as-built
   run table with wall times) + session report. **This is the last phase
   before Phase 5**; still **no `docs/CHANGES.md` entry** (that lands at TASK
   completion, i.e. end of Phase 5 — CLAUDE.md standing exception is
   per-*task*, not per-phase).

## 5. Cost reality — read before you schedule anything

- **T0 pretrain dominates and is the scheduling problem.** Paper-faithful =
  500 epochs × 40k T0 images / bs 256 ≈ 78k batches; from the Phase 0.3 spike
  timings that projects to **~30 GPU-h per run for pretrain alone** (Phase 2
  flagged this explicitly — it supersedes the older, pre-measurement "8–15
  GPU-h/run" estimate in PLAN.md's phase list). Twelve independent pretrains
  ≈ **360 GPU-h**. The `--pretrain-cache` sharing in §8 Q11 is the lever that
  turns 12 pretrains into **6** (one per (arch, seed); sleep and no-sleep
  share), i.e. ~180 GPU-h of pretrain — and it is arguably the *cleaner*
  controlled comparison (both sleep variants branch from the byte-identical
  encoder). Sleep-retraining adds 11 cycles × `sleep_epochs=300` over the
  replay buffer on top of the sleep runs only.
- **The wake loop + in-run eval are now cheap** thanks to Q6: 21,538 wake
  steps (stream_bs=1) + 44 evals at 25/class clustering ≈ **0.2 GPU-machine-h
  of eval per run** (measured Phase 3), negligible beside pretrain.
- **The 3090 is SHARED.** A llama.cpp server (`llama-server`, port 9401)
  usually holds ~17.7 GB of the 24 GB; **it happened to be down at 2026-07-14
  17:00** (`nvidia-smi` showed 15 MiB used), but do not assume that — check
  every time and coordinate. RN18 @ bs 256 @ 120 px fits the ~6 GB llama
  leaves; **RN50 @ 120 px may not** — this is a hard §8 question, not a
  runtime surprise to discover at hour 3.

## 6. Environment / commands

- **Repo/CPU side** (planning, launcher, tests, manifest): root env —
  `uv run pytest -m "not slow"` (fast) / `uv run pytest` (full). Suite
  baseline you inherit: fast = **122 passed**, full = **148 passed** (needs
  roots.env + live data + the GPU env for the Phase 2 smoke; skips cleanly
  without them — keep it that way). **No `uv add` in the root project.**
- **GPU side**: `baselines/pcmc_sleep/env/.venv/bin/python`. One-time
  `cd baselines/pcmc_sleep/env && uv sync`. Never `uv add` PCMC deps into the
  root project.
- **Launch one cell** (CPU launcher shells out to the GPU env):
  ```
  uv run python baselines/pcmc_sleep/launch.py --arch resnet18 --seed 42
  uv run python baselines/pcmc_sleep/launch.py --arch resnet50 --seed 42 --no-sleep
  ```
  Add `--pretrain-cache <DIR>` once §8 Q11 is decided; `--force` to re-run a
  completed cell; `--smoke` for the tiny-budget plumbing run.
- **Before every launch**: `nvidia-smi` (shared card), and confirm
  `launch.gpu_env_available()` is `(True, "")`.
- Run artifacts live under `${DATA_ROOT}/evaluation/f_pcmc_runs/pcmc_sleep/`,
  **never in-repo** (roots.env → `DATA_ROOT=/home/alex/data`).

## 7. Fidelity anchors (when in doubt, conform in this order)

1. The 2024 paper's §2–4 text (`research_papers/2024_Patch_Based_…md`) — the
   sleep/no-sleep ablation is their §4.4; the geometry (patch 60) their §4.4.
2. The vendored code's actual behavior — it RUNS verbatim
   (`test_pcmc_vendor_untouched`); never edit a file under `vendor/`.
3. The paper-faithful config already resolved in `run_config.py`
   (`pretrained=False`, both epoch knobs = 500, their-120, M≤θ, per-arch
   feat_size). **Table-2 reproduction is NOT possible on this machine** (no
   ImageNet-40/Places365 raw data — HANDOFF_PHASE2 §5); fidelity rests on
   paper-conformant settings + byte-identical code, as recorded.

Never "tune" PCMC to close the gap — the risk register (PLAN.md) pre-registers
the tuning budget (geometry from the 0.3 spike; α/θ/β from paper/STAM
conventions) and requires publishing it with results. The only legitimate new
code is batch orchestration, scheduling, and the manifest.

## 8. OPEN DESIGN QUESTIONS — ask the owner BEFORE launching

Q11. **Pretrain sharing & budget (GATES THE WHOLE PHASE).** 500-epoch
     paper-faithful pretrain ≈ 30 GPU-h/run × 12 ≈ 360 GPU-h on one shared
     card. Options: **(a)** `--pretrain-cache` per (arch, seed) so sleep and
     no-sleep branch from the byte-identical T0 encoder — 6 pretrains not 12
     (~180 GPU-h), and a *cleaner* controlled sleep-vs-no-sleep comparison
     (recommend); **(b)** independent pretrains for all 12 (maximal
     independence, ~360 GPU-h); **(c)** reduce `init_epochs` below 500 (a
     documented deviation from paper-faithful — needs explicit sign-off and a
     recorded rationale). Present measured pretrain wall time from the first
     real RN18 pretrain before committing the other 11. **Note the cache key**
     (`{arch}_seed{seed}_ep{epochs}_img{size}.pkl`) already scopes sharing to
     exactly (arch, seed) — sleep/no-sleep collide by construction, which is
     the intent under (a).
Q12. **GPU scheduling on the shared card.** RN50 @ 120 px may exceed the ~6 GB
     the llama-server leaves. Decide: (i) run RN50 cells only when the
     llama-server is down / paused (owner coordinates), (ii) drop RN50 batch
     size (a config change — record it; it changes the LR-horizon quirk
     accounting per HANDOFF_PHASE2 §5), or (iii) split the matrix across time
     windows. Also settle **run ordering / parallelism** (one card ⇒ serial;
     12 cells over days) and whether to checkpoint-resume across sessions.
Q13. **Batch-runner placement & failure policy.** `launch.py --matrix` mode
     vs a small `run_pcmc_matrix.py`; and on a mid-matrix failure — stop and
     report, or skip-and-continue logging the failure to the manifest?
     (Resumability makes either safe to restart; the question is autonomy.)
Q14. **Snapshot disk budget.** Phase-end snapshots are centroid-sized, but
     `final_state.pt` carries example patch memories; RN50 state ~100 MB ×
     12 cells. Confirm the Q4 phase-end cadence is still fine at full matrix
     scale, or prune (final-only) for the no-sleep cells where nothing
     changes mid-stream.

## 9. Definition of done for Phase 4

- All 12 PCMC cells complete and archived under
  `pcmc_sleep/pcmc_{arch}_{variant}/p2_seed{N}/`, each with the full Phase 2
  artifact set and a matching `resolved_config.yaml`; re-invoking any cell is
  a no-op (resumability holds).
- Batch runner + run manifest committed; per-cell wall times recorded and
  checked against the §5 projection; the pretrain-sharing / scheduling
  decisions (§8) implemented as the owner chose.
- Full repo suite green (fast + slow), **zero existing tests modified**
  (additive `test_repo_layout` / new [U] entries only, per precedent).
- PLAN.md updated (Phase 4 checked off; §8 Q&A verbatim; as-built run table);
  session report written; owner answers recorded verbatim.
- **Hand it to Phase 5**: both sides now have `checkpoints/*.json` in the same
  shape — PCMC under `pcmc_sleep/pcmc_*`, F-PCMC/A6 under
  `pcmc_sleep/paper_protocol/*`. Phase 5 extends the workbook with the 2×2 +
  bridge cell, per-checkpoint curves, the decision-rule verdict, and the PCMC
  tuning-budget sensitivity note, and writes the `docs/CHANGES.md` T17 entry.
