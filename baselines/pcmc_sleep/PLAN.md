# T17 — Sleep-retrained PCMC on the CIFAR-100 P2 stream (`task/T17-sleep`)

**Research question:** can sleep-retraining be removed given a very well-trained
frozen DINOv3 encoder? Answered by a 2×2 over {encoder} × {sleep} on the SAME
P2 stream:

| | Sleep-retrained | Frozen |
|---|---|---|
| ResNet-18/50 | **NEW — this task** | A6 (T16 archive: 0 promotions, strict 0.42) |
| DINOv3 | n/a by construction | fpcmc_default P2 (T16 archive) |

plus the paper's own **no-sleep** PCMC variant (`model.sleep_on=False`, their
§4.4 ablation) as the bridge cell.

**Pre-registered decision rule.** "Sleep is removable" is supported iff
F-PCMC/DINOv3 ≥ PCMC-sleep/RN{18,50} on the shared primary metrics at every P2
checkpoint (3 seeds, mean ± std); "the encoder is what removes it" iff
additionally PCMC-no-sleep/RN sits well below both while A6 shows frozen-RN50
failing inside the F-PCMC machinery.

## Owner decisions (2026-07-14, pre-implementation)

1. **GPU**: RTX 3090 (24 GB), this machine. GPU work is confined to
   `baselines/pcmc_sleep/` — the `fpcmc/` no-learning/CPU-only invariants are
   untouched; PCMC training is seeded but NOT bit-deterministic (deviation
   from NFR-3 scoped to this baseline; model checkpoints are archived so all
   scoring is replayable).
2. **Backbones**: RN18 + RN50 (the paper's Table 2 pairing).
3. **Primary metric: conform to the PCMC paper** — their Eq. 6–7 labeled
   patch-vote classification (100 labels/class, eval-side only) and their
   clustering purity, applied identically to PCMC and to F-PCMC concepts at
   the 44 P2 checkpoints. Our §7.3 open-world metrics are secondary and
   reported one-sided (F-PCMC) or via a documented lossy adapter (PCMC).
4. **Branch**: `task/T17-sleep` (renamed from `tasks/T17-sleep` to match the
   CLAUDE.md convention).

## Owner decisions (2026-07-14, Phase 2 Q&A — HANDOFF_PHASE2.md §8, verbatim)

Asked pre-implementation as mandated; answers recorded exactly as given:

- **Q1 (patch geometry)**: resolved pre-Q&A by the Phase 0.3 spike —
  **their-120** (img 120, patch 60, stride 30). Not re-asked.
- **Q2 (sleep schedule on P2)**: *"(b) Mid-phase per phase"* — one sleep at
  each P2 phase's midpoint (11 sleeps/run); the driver steers their
  `sleep_start`/`sleep_freq` attributes between steps (their trigger code
  untouched) and the computed step list is recorded in the resolved config.
- **Q3 (their eval sets on P2 novel classes)**: *"(a) CIFAR-only
  classification"* — the classification protocol is restricted to the 100
  CIFAR classes (T0 80 + held-out 20, test images from the canonical CIFAR
  test split); synthetic near/far classes are covered via clustering purity
  only, fed stream-seen rows (documented). Two facts recorded with the
  decision: (i) their own released protocol draws SUPERVISE images from
  streamed data (streams.py:97), so sup-from-stream is protocol-conformant;
  (ii) **P2 streams the held-out classes' ind_test rows** (build_p2 gives
  each held-out class ALL of its train+test rows) and consumes ~80% of the
  T0 ind_test pool as interleave — so CIFAR test images may have been seen
  UNLABELED in-stream. That matches released upstream behavior too
  (streams.py sets `cls_inds_test = cls_inds_train` for most datasets); it
  is documented, not silently absorbed.
- **Q4 (mid-stream snapshot cadence)**: *"Phase-end (11/run)"* — model +
  centroid memories at each phase end plus a final snapshot (the final one
  also carries the example-patch memories); eval-metric JSONs at every
  checkpoint regardless.
- **Q5 (vendored-hash location)**: *"Append to lib/PROVENANCE.md"* — new
  `## baselines/pcmc_sleep/vendor/` section (placed BEFORE the v1 section,
  whose test parses from its own marker to end-of-file).

## Phase 0 findings (2026-07-14)

- **0.1 Alignment: PROVEN, 0 mismatches over all 63,326 rows.** Every pool
  carries `image_paths`: CIFAR rows are index-keyed into the canonical
  pickles (`cifar100_train_00000`…, labels verified row-by-row against
  `${DATA_ROOT}/cifar100/cifar-100-python`); all 3,326 synthetic rows resolve
  to existing files under `ms_cifar100_genai_{ind,novel}_32x32/` with
  path-derived labels matching row labels. (Label/index-level proof; a
  byte-level re-embed spot check rides along once the GPU env is wired in.)
- **0.2 PCMC smoke: PASSED** on the 3090 (torch 2.5.1+cu121, lightly,
  pykeops, hydra; py3.11 — its own env, never the repo's pinned CPU-torch
  venv). Full cycle exercised at CIFAR scale: contrastive T0 pretrain
  (`pretrained=False`), KeOps GPU k-means memory init, wake steps with
  novelty detection + STM→LTM promotions, one full sleep (encoder retrain +
  centroid re-embed). `requirements.txt` upstream is an unusable machine
  freeze — a curated minimal env is part of Phase 2.
- **0.3 geometry spike: DECIDED — their-120 geometry** (upscale CIFAR to
  120×120, patch 60, stride 30). Same-budget head-to-head (5 real CIFAR
  classes, 50 epochs, RN18, `pretrained=False`, their supervise/classify
  eval): their-120 scored **70.8% classification / 71.4% clustering purity**
  vs native-32 (img 32, patch 16, stride 8) at **63.4% / 51.4%** — B wins
  on both metrics by wide margins (+7.4 acc, +20.0 purity) at ~1.6× train
  cost (679 s vs 429 s). Fidelity and fairness agree: their-120 is also the
  paper's own pipeline geometry (patch 60 per their §4.4 ablation). Small-
  scale proxy, but unambiguous and consistent with the paper's finding that
  small patches hurt. Pre-registered for all T17 runs. (Bonus data point:
  RN18 @ bs 256 @ 120px trained inside the ~6 GB GPU share — see the
  shared-GPU note in HANDOFF_PHASE2.md.)
  The spike also surfaced the `init_memory` persistent-workers deadlock —
  mitigation and evidence recorded in HANDOFF_PHASE2.md §5.

### Code facts that bind later phases (from reading `reference/pcmc`)

- Integration seam: `main.py` needs only a Stream object exposing
  `__iter__/__next__ -> (data, label, t)` (CPU tensors — layers move to CUDA
  internally; feeding CUDA tensors corrupts their stored-example device
  bookkeeping), `pretrain_dataloader`, `eval_loaders(t)`, `task_bounds`,
  `eval_times`, `__len__`. A `P2UPLStream` shim replays our exact
  `StreamManifest` order; their model code stays verbatim.
- `model.sleep_on=False` is the built-in no-sleep ablation. Sleep triggers
  inside `Layer.__call__` at `step == sleep_start` then every `sleep_freq`.
- **Config/paper discrepancies to resolve by reproducing Table-2 behavior**
  (owner ruling "conform to the paper"): released `model/pcmc.yaml` has
  `pretrained: True` (SKIPS contrastive T0 training — paper §3.2 describes
  500-epoch training) and `patch_size: 90` (paper chose 60). Default to the
  paper-faithful settings (`pretrained: False`, patch per §4.4); record both.
- Hard invariants in their code: promoting cluster must hold ≥ M patches
  (keep M ≤ θ); `init_memory` draws a hardcoded 2,000-patch sample (T0 must
  supply ≥ 2,000 patches — trivially true at 40k images); ungated plotting
  side-effects write PNGs every sleep/eval (accept the I/O; plot=False gates
  only some of them).
- **Epoch semantics**: `Layer.pretrain` makes exactly ONE pass over the
  loader; "epochs" are index repetition via
  `ExtendedSampler(inds, shuffle=True, repeats=config.model.init_epochs)`
  (streams.py:100). The layer-level `init_epochs` only stretches the cosine
  LR horizon (`len(trainloader) × init_epochs` where `len(trainloader)`
  already includes the repeats) and sets logging cadence — and it must
  satisfy `len(trainloader) // init_epochs ≥ 1` or pretrain crashes. The
  released config mismatches the two (model 300 vs layer 500, paper says
  500); our runs set both to the same value and record the LR-horizon quirk
  as released-code behavior we conform to.

## Remaining phases

1. **P2 pixel mirror** (`stream_mirror.py`): image-space replay of the exact
   T12 `StreamManifest` per seed {42,43,44}; T0 = raw train images of the 80
   split classes. [U] tests: per-index identity with the embedding stream.
   **DONE 2026-07-14** (commit `6c83412`): 4 [I] tests green on live data.
2. **Driver**: vendor needed upl-benchmark modules byte-identical (blob
   hashes in `lib/PROVENANCE.md`, untouched-checksum test — T14/v1
   precedent; nothing imports `reference.*`), plus a non-verbatim shim:
   `P2UPLStream`, curated env (`pcmc_sleep` own pyproject), sleep schedule
   mapped to P2 phases (their sleep-middle default), checkpoint artifacts at
   the 44 P2 checkpoints, `--no-sleep` flag. Resolve the two paper/config
   discrepancies against a Table-2 reproduction smoke.
   **DONE 2026-07-14.** As built:
   - `vendor/` = the verified import closure of `PCMC` (12 files,
     byte-identical from the pinned submodule commit's blobs; hashes in
     `lib/PROVENANCE.md`, `test_pcmc_vendor_untouched` green). Verified
     deviations from the HANDOFF §4.1 expectation: `sleep_algos.py` is empty
     and imported by nothing (excluded); upstream has no
     `core/stream/__init__.py` (namespace subpackage — none invented);
     `core/stream/dataset.py` IS needed (`NumpyDataset`); `torchmetrics` is
     an undocumented upstream import, pinned explicitly in the env.
   - `env/` = own uv project (py 3.11, torch 2.5.1+cu121 + the Phase 0.2
     recipe, versions pinned to the validated spike env; `uv.lock`
     committed). Table-2 reproduction is NOT possible on this machine (no
     ImageNet-40/Places365 raw data — HANDOFF §5); fidelity is established
     by paper-conformant settings + byte-identical code instead.
   - `run_config.py` (torch-free, CPU-unit-tested) resolves the paper-
     faithful config: `pretrained=False`, both epoch knobs = 500, their-120
     geometry, θ=30 Δ=400 α=0.1 β=0.99 M=30, CIFAR-100 mean/std, and
     `feat_size` per arch — **512 RN18 / 2048 RN50** (their 512 default
     would shape-crash resnet50's stm/ltm buffers).
   - `p2_stream.py::P2UPLStream` satisfies exactly the upstream main.py
     stream interface, fed from `P2PixelMirror`; labels are contiguous in
     seen order (their supervise() indexes score columns by raw label).
     Eval sets per owner Q3(a); supervise draws seeded per class via
     `fpcmc.rng.make_rng`, checkpoint-independent.
   - `driver.py` (GPU env) replaces main.py: set_seed replica; the proven
     global DataLoader force-patch (num_workers=0, worker-only kwargs
     stripped) — chosen over a scoped patch because a config-only fix is
     impossible (persistent_workers with 0 workers raises) and the sleep
     loaders share the abandonment-prone pattern; the smoke runs under a
     hard subprocess timeout so a deadlock regression fails loudly. Sleep
     steering per owner Q2 (sets `layer.sleep_start/sleep_freq` to the next
     recorded phase-midpoint target; trigger code untouched; sentinel freq
     blocks the python-modulo early-fire path). Their eval at every
     checkpoint + the Q3 auxiliary synthetic-inclusive `model.cluster()`
     call; phase-end snapshots per Q4; `resolved_config.yaml`/`summary.json`
     resumability mirroring `run_matrix.run_cell`; `--no-sleep`, `--smoke`,
     `--force`, and an OFF-by-default `--pretrain-cache` (their released
     `load_pretrain` mechanism, driver-side file copy) for the Phase 4
     shared-T0 decision.
   - `launch.py` = thin CPU-side subprocess launcher + `gpu_env_available`.
   - Tests: 11 CPU-fast (vendor checksum + run_config logic) + 1 [I][slow]
     GPU smoke (tiny budget: 1,024 T0 images × 2 epochs, 240 wake steps,
     forced sleep at 150, checkpoint eval at 200, resume check; 77 s wall on
     the 3090, 88 s as a test). Smoke accuracies are near-random BY DESIGN
     (2-epoch encoder) — the smoke validates plumbing, not performance.
   - **Flagged for Phase 4 (not decided)**: at paper-faithful 500 epochs ×
     40k T0 images ≈ 78k batches, T0 pretrain alone projects to ~30 GPU-h
     per run from spike timings (in-process loading) — pretrain sharing per
     (arch, seed) via `--pretrain-cache` (also arguably a cleaner controlled
     comparison: sleep and no-sleep branch from the identical encoder) and
     GPU scheduling need owner sign-off before the 12-run matrix. **Flagged
     for Phase 3**: their clustering eval is O(N²) python-level Jaccard +
     spectral clustering — full 100×100 test sets (10k images) × 44
     checkpoints is infeasible; eval sizing must be decided there (the
     driver reads `dataset.sup_size`/`test_size` from config).
3. **Evaluation parity**: their Eq. 6–7 supervise/classify + clustering
   purity as an eval-side scorer applied to BOTH systems (F-PCMC concepts
   score patches?? No — F-PCMC is whole-image: its "centroids" for their
   protocol are concept centroids over CLS embeddings; the scorer associates
   concepts↔classes with 100 labels/class and votes 1-per-image — the
   whole-image degenerate case of their Eq. 6–7, documented). Secondary: the
   lossy JSONL adapter for memory-dynamics metrics.
4. **Runs**: {RN18, RN50} × {sleep, no-sleep} × seeds {42,43,44} (12 GPU
   runs; est. 8–15 GPU-h per sleep run, init-only for no-sleep), archived
   under `${DATA_ROOT}/evaluation/f_pcmc_runs/pcmc_sleep/`. F-PCMC/A6 cells
   reused from the T16 archive + rescored under the paper protocol (CPU).
5. **Report**: extend the workbook with the 2×2 + bridge cell, per-checkpoint
   curves, the decision-rule verdict, and a PCMC tuning-budget sensitivity
   note.

## Risks (ranked)

1. Under-tuned PCMC ⇒ hollow win — pre-registered tuning budget (patch
   geometry via the 0.3 spike; α/θ/β from paper/STAM CIFAR conventions),
   published with results.
2. RN50-on-small-patches degeneracy — the 0.3 geometry decision de-risks;
   worst case report RN18 (their primary) alongside.
3. upl-benchmark adaptation surface — Phase 0.2 bounds it: the stream shim +
   dataset config are the only integration points found so far.
