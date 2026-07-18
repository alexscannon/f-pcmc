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

## Owner decisions (2026-07-14, Phase 3 Q&A — HANDOFF_PHASE3.md §8, verbatim)

Asked pre-implementation as mandated; answers recorded exactly as given
(option labels as selected). Measured evidence presented with the questions:
their exact `cluster()` hot path times at 1.11 µs/jaccard-pair (quadratic) +
spectral ~N^1.8, so full 100/class clustering = 2.88 h/run (×12 ≈ 35 h,
worst checkpoint 5.8 min ≈ 13,500-image final aux set); and on singleton
(binary-affinity) sets their SpectralClustering does NOT agree with the
analytic collapse (arbitrary merges when concepts > 2×classes: purity 43–54%
vs 79–88%; arbitrary splits in F-PCMC's actual regime of concepts <
2×classes: totals close but per-class purity off by up to 56pp, nan patterns
disagree, seed-unstable).

- **Q6 (eval sizing, gates Phase 4)**: *"25/class clustering (Recommended)"*
  — classification stays 100 sup + 100 test/class at all 44 checkpoints;
  clustering set = first 25 of each class's seeded test draw (CIFAR) / first
  25 stream arrivals (synthetic). Measured 0.19 h/run, worst checkpoint 24 s.
  One constant set in `run_config.py` (`CLUST_SIZE`), used byte-identically
  by both systems.
- **Q7 (F-PCMC centroid population)**: *"Both, tier-1 headline
  (Recommended)"* — every checkpoint scored twice, tier-1 (LTM ∪ mature STM,
  exactly `ConceptStore._tier1_stm`'s maturity-only view) and LTM-only; the
  headline comparison uses tier-1 (the functional analog of their answering
  memory `self.ltm`), LTM-only recorded alongside in each checkpoint JSON.
  Centroid = `Concept.centroid` ((D,) L2-normalized; EMA for STM, frozen for
  LTM). Context recorded with the decision: final checkpoint seed 42 has
  fpcmc_default 84 LTM + 99 STM (4 promotions), a6_resnet50 80 LTM + 100 STM
  (0 promotions) — LTM-only leaves novel classes structurally unrepresented.
- **Q8 (checkpoint-state reconstruction)**: *"Additive replay iterator
  (Recommended)"* — `fpcmc/replay.py` refactored additively (per-record
  application extracted; a generator yields the store after init and after
  each checkpoint record); `replay()` behavior byte-identical, T11 tests
  untouched; every yielded state cross-checked against its checkpoint
  record's `n_ltm`/`n_stm`/`taus`.
- **Q9 (whole-image clustering)**: *"Analytic collapse (Recommended)"* —
  cluster assignment = matched concept id; purity via their `purity_score`
  math verbatim on that assignment. The degenerate binary affinity already
  IS a partition; spectral on it was measured non-equivalent and
  seed-unstable (numbers above). Slightly conservative for F-PCMC (≤ ~183
  clusters vs the k = 2×classes ≈ 200 PCMC gets).
- **Q10a (placement)**: *"baselines/pcmc_sleep/ (Recommended)"* — scorer +
  CLI live beside the rest of the T17 protocol machinery.
- **Q10b (adapter scope)**: *"Cheap artifacts only (Recommended)"* —
  per-checkpoint LTM size, sleep step times, eval wall times,
  phase-end-snapshot fields, read from the Phase 2 artifacts; no driver
  changes; documented field-by-field as a lossy mapping.
- **Q10c (output location)**: *"pcmc_sleep/paper_protocol/ (Recommended)"* —
  `${DATA_ROOT}/evaluation/f_pcmc_runs/pcmc_sleep/paper_protocol/<system>/
  p2_seed<N>/checkpoints/*.json`; scored cells fpcmc_default × {42,43,44} +
  a6_resnet50 × {42,43,44}.

## Owner decisions (2026-07-14, Phase 4 Q&A — HANDOFF_PHASE4.md §8, verbatim)

Asked pre-launch as mandated (Phase 4 §8 GATES the production matrix — the one
phase where launching before sign-off can burn real days of shared-card
wall-clock). Answers recorded exactly as selected. Machine state at ask time:
GPU free (`nvidia-smi` 15 MiB used, llama-server down); DATA_ROOT 1.4 TB free.

- **Q11 (pretrain sharing & budget — GATES the phase)**: *"Share per
  (arch,seed)"* — use `--pretrain-cache` so the sleep and no-sleep cells of
  each (arch, seed) branch from the byte-identical T0 encoder: **6 pretrains,
  not 12** (~180 GPU-h of pretrain), and the cleaner controlled comparison
  (the bridge cell then isolates only the sleep cycles). Cache key
  `{arch}_seed{seed}_ep{epochs}_img{size}.pkl` scopes sharing to exactly
  (arch, seed); sleep/no-sleep collide by construction. **Protocol commitment:
  the first real RN18 pretrain's measured wall time is reported to the owner
  before the other 5 pretrains are committed.**
- **Q12 (GPU scheduling / RN50 sizing)**: *"Serial, bs256, measure RN50
  first"* — one card ⇒ cells run serially; keep the config's `pretrain_bs=256`
  while the card is free; capture peak VRAM on the FIRST RN50 pretrain and drop
  the batch size only if it OOMs (recording the change + its LR-horizon-quirk
  effect). Cross-session resume already holds at cell granularity.
- **Q13 (mid-matrix failure policy)**: *"Stop-and-report"* — on the first cell
  failure the batch runner records the failure to the manifest, writes it, and
  halts the whole matrix rather than skipping ahead; resumability makes the
  restart cheap once the cause is understood.
- **Q14 (snapshot cadence at full-matrix scale)**: *"Keep phase-end for all
  cells"* — uniform Q4 cadence (11 phase-end snapshots + final) for both sleep
  and no-sleep cells; disk is ample (~15 GB for all 12) and no-sleep memory
  buffers still evolve mid-stream (promotions happen without sleep), so the
  phase-end snapshots are not redundant. No driver change.

## Phase 4 status (2026-07-14, IN PROGRESS — the 12-cell production matrix)

The batch runner + manifest are built, tested, and committed (commit
`7e759a6`); the matrix runs are a multi-day, resumable effort that continues
across sessions (one shared card, ~180 GPU-h with the Q11 pretrain sharing).

As-built (CPU-side, committed):
- `run_pcmc_matrix.py` — orders the 12 cells arch→seed→**sleep-before-nosleep**
  so each (arch,seed)'s sleep cell populates the `--pretrain-cache` its nosleep
  twin reuses (Q11 ⇒ 6 pretrains). Serial (Q12); **stop-and-report** on the
  first cell failure (Q13); writes `run_manifest.json` at the pcmc_sleep root
  (per-cell resolved-config sha256, wall time, sleep steps, final LTM + class/
  clust acc, NFR-1 over-budget flag). CLI: `--only`, `--dry-run`, `--force`,
  `--timeout`, `--cache-dir`, `--out-root`. Shared cache at
  `pcmc_sleep/_pretrain_cache/{arch}_seed{seed}_ep500_img120.pkl`.
- Tests: 3 [U] (enumeration order + naming, cache-key parity with driver.py,
  budget projection) added to `tests/test_pcmc_driver.py`; `run_pcmc_matrix.py`
  added to `test_layout.py`. Fast suite green (127 passed); GPU driver smoke
  re-verified green on the free card (87.6 s).

### Incident record + owner Q&A (2026-07-16) — two lost cell-1 attempts

Cell 1 was launched twice and lost both times **before** its T0 pretrain could
finish. **Their pretrain is all-or-nothing**: `Layer.pretrain`
(vendor/core/models/pcmc/pcmc_layer.py:491-530) makes ONE pass over the
78,125-batch repeats-expanded loader and `torch.save`s the encoder only on the
final line; `load_pretrain` only ever loads that finished file. **There is no
mid-pretrain checkpoint and no resume** — any interruption inside the ~28 h
window loses the whole pass.

- Attempt 1 (2026-07-15 03:10Z, `Bash run_in_background`): killed at batch
  27,384/78,125 (~9.9 h) when the harness reaped the background task, which
  took the child `driver.py` with it. **Lesson: harness-tracked background
  tasks are not a durable host for multi-hour GPU runs.**
- Attempt 2 (2026-07-15 13:08Z, detached **tmux** session — durable against
  task reaping): ran 22 h 40 m (~81% of the pretrain, ≥ batch ~63k) and died to
  an **unclean shutdown** (`last -x`: `tmux(156737) … - crash`; boot
  2026-07-16 06:49 local). Nothing salvaged: empty `_pretrain_cache/`, no
  `.pkl`, no `summary.json`. **Lesson: the scratchpad log lived in `/tmp` and
  was wiped by the reboot — run logs now go to
  `${DATA_ROOT}/…/pcmc_sleep/_logs/` with the other durable artifacts.**
- Cost: ~9.9 + ~22.7 ≈ **32 GPU-h burned, 0 cells completed.**

Owner Q&A (2026-07-16, verbatim option labels), asked with the evidence above
plus the measured CPU-bound suspicion (every `nvidia-smi` sample read 0% util
at 5 GB while batches ticked at a steady 1.30 s/it; the driver's deadlock guard
forces `num_workers=0`, so SimCLR augmentation — 9 patches × 2 views × 256
images @ 120 px — runs serially on 1 of 16 cores) and the recent uptime
distribution (only 3 of 11 completed boot sessions reached 28 h):

- **Recovery**: *"Just retry as-is"* — relaunch the identical paper-faithful
  28 h pretrain; **no** dataloader change, **no** `init_epochs` reduction, **no**
  checkpoint/resume machinery. Fidelity is preserved exactly; the cost of a
  crash is accepted.
- **Stability**: *"One-off outage"* — the 2026-07-16 crash was an isolated
  power event, not a recurring risk; the machine normally holds 30 h+ (the
  preceding session ran 86 h clean, and the 2026-07-10/11 short-session cluster
  was planned 6.11→6.17 kernel upgrades). Plan for long runs; relaunch after
  any rare crash.

**Not adopted (recorded so a later phase need not re-derive them):** the
`num_workers>0` + `persistent_workers=False` scoped dataloader alternative
(HANDOFF_PHASE2 §5 leaves this door open if the abandonment path is re-proven
under the smoke's hard timeout) — a possible 2–4× pretrain speedup at zero
fidelity cost, declined in favour of not touching the proven deadlock guard;
reducing `init_epochs` (Q11 option (c)) — declined, it feeds PLAN risk #1
(under-tuned PCMC ⇒ hollow win).

### Deadline replan + owner Q&A (2026-07-17) — results needed in 2 days

Owner set a hard deadline: **results by ~2026-07-19 21:00Z**. Full-matrix
reality at that point: cell 1 attempt 3 healthy — its 28.2 h pretrain DONE and
durably cached (`_pretrain_cache/resnet18_seed42_ep500_img120.pkl`), sleep
tail running. Corrected cost model: a sleep cell pays ~9–10 h of sleep
retraining (11 cycles × ~50 min) on top of pretrain — the earlier "~0.2 h
tail" figure was the frozen-side eval cost only. Serial full matrix ≈ ~300
GPU-h ⇒ cannot fit 48 h; measured evidence gathered for the replan: **GPU 75%
idle (avg ~13% util) during training — the workload is CPU-augmentation-bound
(driver ~7 of 16 cores)**; 16 GB VRAM free; but system RAM is the real
constraint (30 GB total; the wake/sleep-phase driver accumulates ~20.5 GB RSS,
pretrain-phase only ~2.8 GB).

Owner Q&A (2026-07-17, verbatim option labels):

- **48h plan**: *"A: Concurrent RN18×3 (Recommended)"* — launch seeds 43+44
  RN18 pretrains in parallel with cell 1's tail; verify contention at 30 min
  (abort one if >30% slowdown). Target: all 6 RN18 cells inside the window ⇒
  the pre-registered 3-seed decision rule evaluable on the paper's primary
  arch. Zero fidelity cost; **overrides the Q12 "serial" scheduling decision
  only** (Q12's bs256 + measure-RN50-first stand).
- **Speedup fix**: *"Benchmark; apply only if needed (Recommended)"* — the
  `num_workers>0` dataloader change is benchmarked, applied to remaining
  pretrains ONLY if the contention check shows the deadline slipping; recorded
  as plumbing heterogeneity (seed 42 pretrained with workers=0) + deadlock
  re-proof via the smoke's hard timeout before use.
- **RN50**: *"Defer, full fidelity (Recommended)"* — the RN50 half runs after
  the deadline at the unchanged 500-epoch config, reported as an addendum;
  the 2-day deliverable is explicitly RN18-anchored.

Follow-up Q&A (2026-07-17 ~23:30Z, after the 30-min contention check):
measured basis presented — dataloader benchmark on their exact pretrain
pipeline (transforms mirrored from pcmc_layer.py:73,86-111, run under live
contention): workers=0 **1.876 s/batch** vs workers=8 **0.264 s/batch** (7.1×);
and cell 1's sleep cycles GROW (cycle 1 = 4,691 iters, cycle 2 = 8,972 — the
replay buffer expands), so sleep tails at workers=0 may miss the deadline on
their own if growth doesn't plateau.

- **Fix scope**: *"All remaining + cell-1 contingency (Recommended)"* —
  workers=8 for seeds 43 (restart) + 44 + nosleep cells, pretrain AND sleep
  tails; if cycle 3 confirms runaway sleep growth, cell 1 is killed and re-run
  from its cached 28 h encoder at workers=8 rather than miss the deadline.
  Conditioned explicitly on the deadlock re-proof smoke passing; **"if it
  fails, everything stays at workers=0."** Seed-42's pretrain remains the
  paper-faithful workers=0 artifact either way.
- **Gate result v1 (FAILED)**: the blanket workers=8 smoke HUNG (2,400 s hard
  timeout; ~87 s expected). Filesystem forensics: `final_model.pkl` was saved
  — pretrain itself ran fine under workers — the hang is post-pretrain, where
  their eval path constructs MANY tiny batch_size=1 loaders (pcmc.py:76-84):
  a worker pool per eval loader is a fork storm. v1's blanket force was the
  defect, not the workers mechanism per se.
- **Gate v2 (bounded refinement, same rule)**: workers apply ONLY to loaders
  with batch_size ≥ 64 (pretrain bs=256, sleep bs=512); every small loader
  stays in-process. Re-gated by its own hard-timeout smoke; if red, hard
  fallback to workers=0 everywhere and the timeline is re-planned honestly.

Implementation (committed regardless of gate outcome; default OFF): driver
`--workers N` sets the guard's forced worker count (persistent_workers forced
False; big-loader-only per v2), threaded through `launch.run_cell(workers=)`
and `run_pcmc_matrix --workers`; recorded per cell in `summary.json`
(`dataloader_workers`) + manifest, deliberately NOT in resolved_config.yaml so
completed cells' resumability is unaffected (execution knob, not run
semantics; augmentation-RNG draw order moves into workers — inside the
accepted "seeded, not bitwise" deviation, recorded). [U] test
`test_dataloader_guard_workers_modes` pins both patch modes.

**Workers gate: FINAL VERDICT RED (2026-07-18 ~01:20Z) — workers=0 is
permanent for T17.** Gate history: v1 blanket (hung post-pretrain), v2
big-loaders-only (hung in eval; both confounded by CPU/RAM contention), v3
take-3 on quiet hardware with /proc classification: **CPU jiffies delta 0
over 10 s, worker children 0 (died at spin-up), driver blocked in
futex_do_wait** — the HANDOFF §5 deadlock signature exactly. Root cause:
their `Layer.__init__` touches CUDA before any loader exists
(pcmc_layer.py:113 `.to('cuda')`), so every forked DataLoader worker inherits
a poisoned CUDA context and dies **nondeterministically** (v1/v2's workers
survived a full pretrain; v3's died at first batch). A nondeterministic hang
at any of dozens of loader spin-ups per run is unshippable; spawn-context
would avoid the fork but costs ~8× dataset RAM per loader — infeasible at
30 GB. The `--workers` flag stays in the code, default 0, as the record of
the investigation.

**Casualty (2026-07-18 ~01:15Z): cell-1 attempt 3 OOM-killed during gate
take-1** — the gate froze the 20.5 GB wake-phase driver (SIGSTOP) while the
smoke's 8 workers ballooned RAM on a 3.6 GB-available box; the kernel OOM
killer chose the largest (frozen) process. Stop-and-report recorded it to the
manifest as designed. The 28 h pretrain was already durably cached, so the
loss is ~5 h of wake/sleep tail. Lessons now operational: never SIGSTOP a
wake-phase driver while anything RAM-hungry launches; a persistent RAM-guard
monitor (<2 GB available → alert) runs during all further scheduling.

**Sleep-cycle runaway growth is CONFIRMED and quantified** (cell-1 attempt 3):
cycle 1 = 4,691 iters, cycle 2 = 8,972, cycle 3 = 14,331 (~+4.7 k/cycle,
linear in the growing replay buffer) ⇒ a full 11-cycle tail ≈ 65–70 h at
workers=0. **Consequence: no sleep cell can complete inside any 48 h window;
the sleep column completes only past the deadline.** Deadline replan under
workers=0 (launched 2026-07-18 01:30Z): cell-1 restarted from cache
(attempt 4, wake+tail, accumulates checkpoint JSONs incrementally — partial
sleep curve at the deadline, completion ~Jul 20); **nosleep-44 launched
FIRST for seed 44** (order inversion: either variant can produce the shared
cache; the nosleep cell's ~4 h post-pretrain tail is completable, and
sleep-44 reuses the cache post-deadline); seed-43's sleep cell continues its
pretrain (cache lands ~Jul 19 02:30Z, decision point: nosleep-43 vs letting
sleep-43's tail start — one wake-phase driver at a time, RAM-bound).

**Second OOM + the init_memory RAM constraint (2026-07-18 ~02:00–02:20Z).**
Cell-1 attempt 4 (cache-HIT restart) was OOM-killed 30 min in, **during
`init_memory`**: kern.log shows the driver itself at **anon-rss 26.2 GB** —
the load-pretrain path goes straight into the vendored init_memory, which
materializes ~360 k patches (156 in-process batches × 256 imgs × 9 patches)
before drawing its hardcoded 2,000-sample; attempt 3 survived the same spike
only because the box was empty. **Operational rule: a cell's first ~40 min
(init_memory + t0 eval) needs a near-empty box (~26 GB available); the
allocator retains ~20 GB RSS afterwards for the life of the run.** As-executed
recovery: ns44 (1 h invested) paused to free headroom → cell-1 attempt 5
launched (passed init, fresh t0.json tripwire) → ns44 relaunched. Known
wrapper nit: the matrix runner's stop-and-report path printed `EXIT 0` from
the tmux wrapper (launcher exit-code propagation) — runner-internal manifest
status is authoritative; fix folded into the Phase-4 wrap-up.

Concurrency launch discipline (RAM-driven, as-executed): one driver added at a
time with RSS/available-RAM checks between; two sleep-phase tails (~20 GB
each) can NOT coexist on 30 GB — tails stagger. Concurrent cells are launched
via `run_pcmc_matrix.py --only <cell>` in separate tmux sessions (durable logs
under `${DATA_ROOT}/…/pcmc_sleep/_logs/`); the manifest's read-modify-write
race between simultaneously-finishing cells is accepted — entries derive from
each cell's durable `summary.json` and are reconstructible by re-invoking
`--only` (resume = no-op).

Runs:
- **Cell 1/12 `pcmc_resnet18_sleep/p2_seed42` LAUNCHED** (2026-07-15 ~03:10Z).
  **Q11 gate measurement (RN18 T0 pretrain): 78,125 batches @ ~1.30 s/it ⇒
  ~28 h**, confirming the §5 ~30 GPU-h/pretrain projection (no cost surprise).
  RN18 @ bs256 @ 120px uses **~5 GB** GPU — fits the shared-card budget (Q12).
- Remaining 11 cells pending cell-1 completion; cell 2 (rn18 nosleep seed42)
  reuses the cached encoder (cache HIT). RN50 peak VRAM to be measured on the
  first RN50 cell (Q12) before its batch size is touched.

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
   **DONE 2026-07-14** (owner Q6–Q10 answers above). As built:
   - `fpcmc_scorer.py` — the whole-image degenerate case, documented
     function-by-function against vendored line numbers (supervise
     pcmc_layer.py:211-274 → `supervise_cent_g`; classify 277-312 +
     pcmc.py:106-181 → `classify`; embed 314-323 → `match_concepts`;
     cluster pcmc.py:221-268 → `cluster_analytic` per Q9, with their
     `purity_score` math verbatim including the 0/0→nan per-class quirk,
     serialized as JSON null). [U]-tested against LITERAL transliterations
     of their per-image loops (tests/test_paper_protocol.py).
   - Eval sets byte-identical to the PCMC side (§5 parity contract):
     `cifar_eval_rows` reproduces `p2_stream._cifar_eval_items` through the
     same `make_rng` substreams; pool-row ↔ mirror-row identity asserted
     over all 100 classes in the [I] test. Q6 clustering subset = per-class
     prefix of the test draw; synthetic classes = first CLUST_SIZE stream
     arrivals, labels from `cluster_label_map`.
   - `fpcmc/replay.py` gained the additive `iter_checkpoint_states` (Q8a):
     per-record application extracted into a shared `_apply_record`,
     `replay()` behavior unchanged, T11 replay tests untouched and green.
   - PCMC-side Q6 wiring: `dataset.clust_size` (25; smoke 3 — must stay > 2,
     their SpectralClustering needs n_clusters < n_samples),
     `clust_loader`/`cluster_loader` subsampling, and driver-side eval
     composition (the same three vendored calls `PCMC.eval` makes, with
     cluster() fed the subset loader). GPU driver smoke re-run green
     (86.5 s on the 3090).
   - `score_fpcmc.py` scored fpcmc_default + a6_resnet50 × seeds {42,43,44}
     — 45 records/cell (t0 + 44 checkpoints), EVERY reconstructed state
     verified exactly against its checkpoint record (counts + full tau
     snapshot) — to `${DATA_ROOT}/evaluation/f_pcmc_runs/pcmc_sleep/
     paper_protocol/<system>/p2_seed<N>/` in the Phase 2 checkpoint-JSON
     shape (+ Q7 `ltm_only` block and n_tier1/n_ltm/n_stm context;
     manifest.json at the root). ~25 s per cell.
   - `pcmc_dynamics.py` — the Q10b lossy adapter (per-step LTM size history
     sampled at checkpoints, sleep steps, eval wall times, phase-end
     snapshot fields; losses documented field-by-field).
   - Headline (tier-1) numbers, fpcmc_default: t0 91.5 class / 91.9 purity
     (80 classes) declining to 78.2 / 81.2 / 80.7 class_acc at the final
     checkpoint (100 classes; seeds 42/43/44), LTM-only 76.1 / 74.8 / 79.2;
     149-class aux purity falls to ~53 (seed 42) as the 49 synthetic
     classes accumulate largely unrepresented (60 nan entries).
     a6_resnet50 (frozen RN50, 0 promotions): t0 63.9 / 63.3 → final 51.7 /
     50.9 / 51.9 with n_tier1 only 80→86 — the frozen-RN50-fails-inside-
     F-PCMC cell of the 2×2, now on the paper's own metric.
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
