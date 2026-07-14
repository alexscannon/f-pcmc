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
- **0.3 geometry spike**: in flight (native-32/patch-16/stride-8 vs
  upscale-120/patch-60/stride-30, same budget, their eval protocol on 5 real
  CIFAR classes).

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
2. **Driver**: vendor needed upl-benchmark modules byte-identical (blob
   hashes in `lib/PROVENANCE.md`, untouched-checksum test — T14/v1
   precedent; nothing imports `reference.*`), plus a non-verbatim shim:
   `P2UPLStream`, curated env (`pcmc_sleep` own pyproject), sleep schedule
   mapped to P2 phases (their sleep-middle default), checkpoint artifacts at
   the 44 P2 checkpoints, `--no-sleep` flag. Resolve the two paper/config
   discrepancies against a Table-2 reproduction smoke.
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
