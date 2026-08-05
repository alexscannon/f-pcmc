# HAND-OFF: T17 Phase 2 — the sleep-retrained PCMC driver

You are implementing **Phase 2 of task T17** in this repository. Read this
file completely, then `CLAUDE.md` (repo root, binding), then
`baselines/pcmc_sleep/PLAN.md` (the task plan + Phase 0/1 findings), before
writing any code. Work on branch `task/T17-sleep` (exists; do not rebase or
merge). **Post a short plan and STOP to ask the owner the open design
questions in §8 before implementing** — that Q&A-before-code workflow is
mandatory here (CLAUDE.md "Per-task workflow"), and every prior task followed
it.

## 1. Mission context (why this exists)

The project (F-PCMC) answers: *can PCMC's sleep-retrained encoder be replaced
by a frozen DINOv3?* T16 completed the frozen-side experiment matrix. T17
fills the missing comparison cell: **the actual 2024-paper PCMC (sleep-
retrained ResNet-18/50, patch-based contrastive) run on OUR CIFAR-100 P2
stream**, so the 2×2 {encoder}×{sleep} closes on one shared benchmark:

| | Sleep-retrained | Frozen |
|---|---|---|
| RN18/RN50 | **T17 (this work)** | A6 in the T16 archive |
| DINOv3 | n/a by construction | fpcmc_default×P2 in the T16 archive |

plus PCMC-no-sleep (`model.sleep_on=False`) as the bridge cell. Pre-registered
decision rule and phase list: `PLAN.md`. **Phase 2 builds and smoke-tests the
driver. It does NOT run the full 12-run matrix (that is Phase 4) and does NOT
build the evaluation-parity scorer (Phase 3).**

Owner decisions already made (2026-07-14, do not re-ask): RTX 3090 on this
machine; backbones RN18 **and** RN50; primary metric conforms to the PCMC
paper's protocol; GPU work confined to `baselines/pcmc_sleep/`.

## 2. Repository rules that bind you (distilled; CLAUDE.md is authoritative)

- `reference/pcmc` (submodule, pinned `e77f5f7ae8fa`) is **read-only**. You
  may read it; nothing under `fpcmc/`, `eval/`, or `tests/` may import from
  it (`test_no_reference_imports` enforces). The T14 precedent you follow:
  **vendor byte-identical copies** of the needed files, record every git blob
  hash in `lib/PROVENANCE.md` (owner-approved location for vendored-file
  hashes), and land an untouched-checksum test. Shims/drivers are separate,
  clearly-ours files — never edit a vendored file.
- **Never touch** `docs/` (except appending `docs/CHANGES.md` at task
  completion — not at end of Phase 2), `tests/reference_numbers.yaml`,
  `configs/golden_run.yaml`, `lib/`, the root `pyproject.toml`/`uv.lock`
  (pins CPU-only torch 2.11.0 — your GPU env is separate, see §6), or
  anything under `fpcmc/` (its no-learning/CPU invariants are AST-enforced).
- **Never weaken, skip, or delete an existing test.** Full suite must stay
  green: `uv run pytest -m "not slow"` (fast, currently 101 passed) on every
  meaningful change; `uv run pytest` (full, currently 123 passed, needs
  roots.env + live data) before declaring done. New GPU-dependent tests must
  **skip cleanly with a clear message** when the GPU env / CUDA / vendored
  env is absent — the repo's CPU-only CI must never fail for lack of a 3090.
- Commit in small logical units on `task/T17-sleep`; message style
  `T17: <what>`; end the session with a report (built/tests/deviations/next).
- Data resolves via `roots.env` (`DATA_ROOT=/home/alex/data` here). Run
  artifacts live under `${DATA_ROOT}/evaluation/f_pcmc_runs/pcmc_sleep/`,
  never in-repo.

## 3. What already exists (Phase 0/1, commits `0cf9707`, `6c83412`)

- `baselines/pcmc_sleep/PLAN.md` — the plan, decision rule, and the Phase 0
  findings inventory. **If a "0.3 geometry" result is recorded there, use it;
  if still marked in-flight, geometry is open question Q1 in §8.**
- `baselines/pcmc_sleep/stream_mirror.py` — `P2PixelMirror`: pixel-space
  replay of the exact P2 stream. API you build on:
  - constructed as `P2PixelMirror(build_p2(config, seed, pools), pools)`
    (`fpcmc.protocols.build_p2`, `fpcmc.data.load_all_pools`);
  - `len(mirror)` = 21,538 (seed 42); `manifest` (parallel arrays: `pool`,
    `within_pool_index`, `true_class`, `true_superclass`, `phase`);
    `checkpoint_steps` (44 of them); `t0_classes` (the frozen 80);
  - `image_array(i)` → (32,32,3) uint8; `image_pil(i)`; `source_ref(i)`;
  - `t0_image_refs()` → 40,000 `(class, "cifar", "train:NNNNN")` refs for T0
    pretraining (class is for bookkeeping only — pretraining is unsupervised).
- `tests/test_pcmc_sleep.py` — 4 [I] tests proving full-stream alignment.
  Note its determinism test documents a T12 fact you must respect: the P2
  interleave consumes a seed-dependent subset of ind_test; only non-ind_test
  composition is seed-invariant.
- T16 archive (`${DATA_ROOT}/evaluation/f_pcmc_runs/`) — frozen-side cells,
  workbook machinery (`run_matrix.py`, `eval/workbook.py`) whose
  artifact/resumability conventions you should mirror (per-cell dir,
  `resolved_config` + `summary.json`, skip-iff-config-matches, `--force`).

## 4. Phase 2 deliverables

1. **Vendored PCMC** under `baselines/pcmc_sleep/vendor/` — byte-identical
   copies from `reference/pcmc` @ `e77f5f7ae8fa` of exactly the import
   closure the driver needs (expected, verify precisely: `core/__init__.py`,
   `core/utils.py`, `core/models/pcmc/{__init__,pcmc,pcmc_layer,encoders,
   sleep_algos}.py`, `core/stream/{__init__,collate,samplers}.py`, and the
   config YAMLs `config/main.yaml`, `config/model/pcmc.yaml` as fidelity
   reference). Do NOT vendor `main.py` (imports neptune + every other model;
   your driver replaces it). Blob hashes → `lib/PROVENANCE.md`; add the
   untouched-checksum test mirroring `test_v1_untouched`.
2. **GPU environment**, self-contained: `baselines/pcmc_sleep/env/`
   (own uv project or requirements file + README). Validated recipe (Phase
   0.2, driver 580.173.02): python 3.11; `torch==2.5.1+cu121` +
   `torchvision` via `--index-url https://download.pytorch.org/whl/cu121`;
   `lightly`, `pykeops`, `numba`, `scipy`, `scikit-learn`, `matplotlib`,
   `tqdm`, `hydra-core`, `omegaconf`, `pyyaml`, `seaborn`, `pandas`.
   Upstream `requirements.txt` is an unusable machine freeze — ignore it.
3. **Driver** (`baselines/pcmc_sleep/driver.py`, ours, runs inside the GPU
   env): builds the mirror, constructs their `PCMC(config)` from vendored
   code, and executes T0 pretrain → 21,538-step wake loop with sleeps →
   their eval at each of the 44 checkpoint steps → persisted artifacts.
   Components:
   - `P2UPLStream` shim satisfying the interface their `main.py` consumes
     (`__iter__/__next__` → `(data, label, t)`, `pretrain_dataloader`,
     `eval_loaders(t)`, `task_bounds`, `eval_times`, `__len__`), fed from
     `P2PixelMirror`. `t` = the manifest `phase` mapped to their task index.
   - A `cifar100-p2` dataset config + per-run model config (OmegaConf; you
     do not need the hydra runtime — `PCMC(config)` only reads attributes).
     Paper-faithful settings: `model.pretrained=False` (see §5!), geometry
     per the 0.3 decision, θ=30, Δ=400, α=0.1, β=0.99, M per §5's M≤θ
     invariant, CIFAR-100 mean/std `[0.5071,0.4865,0.4409]/[0.2673,0.2564,
     0.2762]`, `arch ∈ {resnet18, resnet50}` (verify `encoders.py` accepts
     resnet50 — it maps arch names to torchvision backbones).
   - `--no-sleep` → `model.sleep_on=False`; `--arch`, `--seed {42,43,44}`,
     `--out <cell_dir>`, resumability via recorded resolved config.
   - Checkpoint persistence: their eval returns `(class_acc, class_pc_acc,
     clust_acc, clust_pc_acc)`; persist per checkpoint as JSON, plus final
     model state_dict + centroid memories. (Cadence of *mid-stream* model
     snapshots is open question Q4.)
4. **CPU-side launcher + tests**: a thin repo-side entry (mirrors
   `run_matrix.run_cell` conventions) that shells out to the GPU env's
   python for `driver.py`; [U] tests for pure logic (config resolution,
   shim index/task mapping — testable with fakes in the CPU env), [I][slow]
   driver smoke: a tiny-budget run (few hundred stream steps, reduced
   epochs, one forced sleep, one checkpoint eval) end-to-end on the 3090,
   skipping cleanly without GPU/env.
5. **PLAN.md updated** (Phase 2 section checked off, decisions recorded) +
   end-of-session report. No `docs/CHANGES.md` entry yet (T17 incomplete).

## 5. Hard-won code facts — violating any of these costs you hours

All verified 2026-07-14 against `reference/pcmc` @ `e77f5f7ae8fa`:

- **`pretrained: True` (their released default) silently SKIPS contrastive
  T0 training** (`pcmc_layer.py:488` — `elif self.pretrained: pass`; the
  backbone keeps torchvision-pretrained weights). The paper describes
  500-epoch patch-contrastive training. Owner ruling "conform to the paper"
  ⇒ `pretrained=False`. Note: Table-2 reproduction is NOT possible on this
  machine (no ImageNet-40/Places365 raw data) — fidelity is established by
  conforming to paper text + released defaults, recorded per §8 Q&A.
- **Epoch semantics** (`streams.py:100`, `pcmc_layer.py:491-516`): the
  pretrain loop makes exactly ONE pass over its dataloader; epochs =
  `ExtendedSampler(inds, shuffle=True, repeats=config.model.init_epochs)`.
  The layer-level `init_epochs` only sets the cosine-LR horizon
  (`len(trainloader) × init_epochs` — already-repeated length, i.e. the
  schedule is stretched ~epochs× beyond the steps that run; released-code
  behavior, conform to it) and the logging cadence, and **must satisfy
  `len(trainloader) // layer.init_epochs ≥ 1` or pretrain crashes with
  ZeroDivisionError**. Their release mismatches the two knobs (model 300 vs
  layer 500; paper says 500). Set both to the same value; record it.
- **Wake input must be CPU tensors** shaped like `DefaultCollateFunction`
  output (`collate.py`); layers `.cuda()` internally. Feeding CUDA tensors
  poisons `stm_examples` device bookkeeping and crashes the sleep-time
  `save_image` (mixed-device `make_grid`).
- **`M ≤ θ` invariant** (`pcmc_layer.py:753`): STM→LTM promotion copies
  `M` stored patches; a cluster promotes with ~θ — with M > θ it's an
  IndexError. Their configs use 30/30.
- **`init_memory` hard-codes a 2,000-patch sample** (`pcmc_layer.py:370`,
  `replace=False`) — T0 must yield ≥ 2,000 patches (trivially true at 40k
  images; matters only for smoke-scale configs).
- **Ungated filesystem side-effects**: `smart_dir(f'logs/...')` writes PNGs/
  pickles relative to CWD on every pretrain/sleep/eval — the driver must
  `chdir` into its cell directory. `plot=False` gates only some plots
  (`pcmc_layer.py:570` runs regardless).
- **Their eval loaders use `batch_size=1`** (`pcmc.py:76-84` does
  `y.item()`), and eval is CPU-heavy/slow — budget for it in the smoke.
- **KeOps JIT-compiles CUDA kernels on first use** (~30 s warm-up, cached
  in `~/.cache/keops*`); GPU k-means (`core/utils.py::KMeans_cosine`)
  requires `pykeops`.
- Sleep triggers inside `Layer.__call__` (`pcmc_layer.py:757`):
  `step == sleep_start`, then every `sleep_freq` steps. `stream_bs=1`.
- Seeding: `main.py::set_seed` pattern (numpy/random/torch/cuda +
  `cudnn.deterministic=True`) — replicate in the driver; GPU contrastive
  training is seeded-but-not-bitwise (accepted deviation, PLAN.md §Owner
  decisions).
- **DataLoader deadlock, WILL hit production runs** (observed live,
  2026-07-14): `Layer.pretrain` builds its loader with `num_workers>0` +
  `persistent_workers=True`, then `init_memory` re-iterates it and BREAKS
  after `len(trainloader) // init_epochs` batches (`pcmc_layer.py:341-343`
  — ~1 epoch of a repeats-expanded loader, so at real epoch counts the
  iterator is always abandoned early). Observed end state: worker processes
  exited, main python blocked forever in `futex_do_wait` on the worker
  queue, 0% CPU/GPU. Mitigation used in the Phase 0.3 spike (driver-side,
  their files untouched): monkeypatch `torch.utils.data.DataLoader` to
  force `num_workers=0` and strip `persistent_workers`/`prefetch_factor`
  (see the scratchpad `pcmc_geometry_spike.py`). Decide in your plan (and
  say so): same patch (slower, simple, proven) vs a scoped alternative —
  and if you keep workers anywhere, prove the abandonment path with a
  timeout in the smoke.
- **The 3090 is SHARED**: the owner runs a llama.cpp server
  (`llama-server`, port 9401) holding ~17.7 GB of the 24 GB. Budget PCMC
  for the remaining ~6 GB (RN18 @ bs 256 fits; RN50 and/or 120px patches
  may not) — check `nvidia-smi` before sizing batches, and raise GPU-memory
  scheduling with the owner before the Phase 4 production runs rather than
  assuming the card is free.

## 6. Environment / commands

- Repo test env (CPU, pinned): `uv run pytest -m "not slow"` / `uv run
  pytest`. **Never** `uv add`/upgrade in the root project.
- The Phase 0 spike env lives at
  `/tmp/claude-1000/-home-alex-projects-f-pcmc/*/scratchpad/pcmc_env` with
  spike scripts (`pcmc_smoke.py`, `pcmc_geometry_spike.py`) beside it —
  session-scratch, may vanish; treat as reference, recreate per §4.2.
- `roots.env` at repo root (gitignored) already points at live data. Raw
  images: `${DATA_ROOT}/cifar100/cifar-100-python/{train,test,meta}`
  (canonical pickles), `${DATA_ROOT}/ms_cifar100_genai_ind_32x32/`,
  `${DATA_ROOT}/ms_cifar100_genai_novel_32x32/novel_{sub,super}classes/`.
  Alignment to embedding rows is PROVEN (PLAN.md Phase 0.1) — do not re-derive
  the mapping, use `P2PixelMirror`.

## 7. Fidelity anchors (when in doubt, conform in this order)

1. The 2024 paper's §2–3 text (`research_papers/2024_Patch_Based_
   Contrastive_Learning_and_Memory_Consolidation_...md`).
2. The released code's actual behavior (vendored, byte-identical).
3. The released config defaults — EXCEPT where they contradict the paper
   (`pretrained`, patch size), which is settled by §8 Q&A with the owner.

Never "fix" or improve their algorithm; PCMC must be run as published. The
only legitimate new code is stream/data/config plumbing and persistence.

## 8. OPEN DESIGN QUESTIONS — ask the owner BEFORE implementing

Q1. **Patch geometry — RESOLVED, do not re-ask** (PLAN.md Phase 0.3):
    their-120 (upscale to 120×120, patch 60, stride 30) won the same-budget
    spike 70.8/71.4 vs 63.4/51.4 (class acc / clust purity) and is the
    paper's own geometry. Pre-registered for all T17 runs.
Q2. **Sleep schedule on P2**: P2 phases are unequal (4×4,286 / 2×357 /
    5×641–846 steps). Their mechanism is fixed-interval (`sleep_start`,
    `sleep_freq`). Options: (a) fixed interval ≈ one sleep per mean phase
    length (mechanism-faithful); (b) drive `sleep_start/sleep_freq`
    per-phase so a sleep lands mid-phase (paper's "sleep-middle" intent,
    needs shim-side step accounting). Recommend (b) with the computed step
    list recorded in the run config.
Q3. **Their eval sets on P2 novel classes**: the paper protocol needs 100
    labeled + 100 test images per class seen so far. CIFAR classes have
    clean train/test splits; the synthetic near/far classes have NO held-out
    split (the pools are fully consumed by the stream: near ~83/class, far
    ~60/class). Options: (a) restrict their classification protocol to
    CIFAR classes (initial + held-out) and cover novel classes via their
    clustering-purity protocol only; (b) split each synthetic class's pool
    rows into sup/test halves and exclude those rows' stream arrivals from
    training-time influence claims (leaky — PCMC saw them in the stream);
    (c) sup/test from stream-seen images, documented as in-stream
    evaluation. This choice shapes the headline comparison — do not decide
    unilaterally.
Q4. **Mid-stream model snapshot cadence** (disk: RN50 state ~100 MB ×44
    checkpoints ×12 runs ≈ 50 GB): every checkpoint, every phase end, or
    final-only + eval JSONs. Recommend phase-end (11/run).
Q5. **Vendored-hash location**: confirm appending to `lib/PROVENANCE.md`
    (T14 precedent) vs a new `baselines/pcmc_sleep/PROVENANCE.md`.

## 9. Definition of done for Phase 2

- Vendored files byte-identical with hashes recorded + checksum test green.
- GPU env recreatable from committed files (no machine-freeze artifacts).
- Driver smoke [I]: tiny-budget end-to-end run on the 3090 — T0 pretrain
  (reduced), ≥200 wake steps, ≥1 sleep cycle, ≥1 checkpoint eval, artifacts
  + resume check — green, and skipping cleanly on machines without the env.
- Full repo suite green (fast + slow), zero existing tests modified (the
  only allowed edit: additive `test_repo_layout` entries, per precedent).
- PLAN.md updated; session report written; open-question answers recorded
  verbatim in PLAN.md's owner-decision log.
