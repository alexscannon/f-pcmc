# HAND-OFF: T17 Phase 3 — evaluation parity (the paper-protocol scorer)

You are implementing **Phase 3 of task T17** in this repository. Read this
file completely, then `CLAUDE.md` (repo root, binding), then
`baselines/pcmc_sleep/PLAN.md` (task plan, Phase 0–2 findings, owner-decision
log), then `baselines/pcmc_sleep/HANDOFF_PHASE2.md` §5 (hard-won upstream
code facts — still binding). Work on branch `task/T17-sleep` (exists; do not
rebase or merge). **Post a short plan and STOP to ask the owner the open
design questions in §8 before implementing** — the Q&A-before-code workflow
is mandatory (CLAUDE.md "Per-task workflow"); Phases 0–2 all followed it and
their Q&A answers are recorded verbatim in PLAN.md.

## 1. Mission context (why this phase exists)

T17 closes the 2×2 {encoder}×{sleep} comparison on the shared CIFAR-100 P2
stream. The owner's primary-metric ruling (PLAN.md owner decision 3):
**conform to the PCMC paper's protocol** — their Eq. 6–7 labeled patch-vote
classification (100 labels/class, eval-side only) + their clustering purity —
**applied identically to PCMC and to F-PCMC** at the 44 P2 checkpoints.

Phase 2 built the PCMC side: the driver runs THEIR eval code verbatim in-run
and persists per-checkpoint JSONs (§3). **Phase 3 builds the F-PCMC side**:
an eval-side scorer that applies the same protocol to F-PCMC concept states —
the *whole-image degenerate case* of Eq. 6–7 (PLAN.md phase 3: F-PCMC is
whole-image; its "centroids" are concept centroids over CLS embeddings; the
scorer associates concepts↔classes with the labeled set and votes
1-per-image) — **on byte-identical eval sets at identical checkpoint steps**,
plus the secondary lossy adapter for memory-dynamics metrics. Phase 3 also
**settles eval sizing (§8 Q6), which GATES Phase 4**: PCMC's eval runs
*inside* the production GPU runs, so the sizing must be final before any
12-run matrix cell launches.

Phase 3 does NOT run the production matrix (Phase 4) and does NOT write the
report/workbook extension (Phase 5).

## 2. Repository rules that bind you (distilled; CLAUDE.md is authoritative)

- **Never touch** `baselines/pcmc_sleep/vendor/` (byte-identical vendored
  PCMC; `test_pcmc_vendor_untouched` asserts every hash in
  `lib/PROVENANCE.md`), `lib/`, `docs/` (except appending `docs/CHANGES.md`
  at TASK completion — not at end of Phase 3), `tests/reference_numbers.yaml`,
  `configs/golden_run.yaml`, root `pyproject.toml`/`uv.lock` (**the F-PCMC
  scorer must run in the pinned root CPU env with NO new dependencies** —
  numpy/sklearn/scipy are already pinned there), `reference/`.
- **Never weaken, skip, or delete an existing test.** Suite baseline you
  inherit: fast `uv run pytest -m "not slow"` = **112 passed**; full
  `uv run pytest` = **135 passed** (needs roots.env + live data + the GPU env
  for the Phase 2 smoke — it skips cleanly without them, keep it that way).
- **No ground-truth leakage into `fpcmc/`** (AST-enforced): the paper
  protocol is *eval-side supervised by design*, so the scorer lives outside
  `fpcmc/` (placement is §8 Q10) and labels never flow into runtime code.
- **fpcmc/ mechanisms are frozen as-built** (scorers T2, Concept T3,
  thresholds T4, store T5, memory T7–T10 — see CLAUDE.md bullets). If you
  need checkpoint-state reconstruction (§8 Q8), any change to
  `fpcmc/replay.py` must be additive and behavior-preserving with T11's
  replay tests green.
- GPU work stays confined to `baselines/pcmc_sleep/` — but Phase 3's scorer
  is CPU/embedding-space; you should not need the GPU except to re-run the
  Phase 2 driver smoke.
- Commit small units on `task/T17-sleep`, messages `T17: <what>`; end with a
  session report; record owner Q&A answers verbatim in PLAN.md.

## 3. What already exists (verified 2026-07-14, commits `3916363`, `f61d9be`)

### Phase 2 deliverables (all tested; 12 tests in `tests/test_pcmc_driver.py`)

- `baselines/pcmc_sleep/run_config.py` — **torch-free** config/schedule/label
  logic. You will reuse: `eval_label_map`, `cluster_label_map`,
  `eval_classes_at`, `phase_task_index`, `SUP_SIZE`/`TEST_SIZE` (100/100
  paper defaults — the Q6 sizing decision lands HERE), constants. Owner
  decisions Q1–Q5 are documented in its module docstring.
- `baselines/pcmc_sleep/stream_mirror.py` — `P2PixelMirror` (torch-free hot
  path): `manifest`, `phases`, `t0_classes`, `checkpoint_steps`,
  `cifar_rows(split, cls)` (ascending canonical pickle rows of a class),
  `cifar_image(split, index)`, `image_pil(i)`, `t0_image_refs()`.
- `baselines/pcmc_sleep/p2_stream.py` — GPU-env-only `P2UPLStream`; its
  `_cifar_eval_items` is the reference implementation of the eval-set draw
  (§5 recipe) that your scorer must reproduce byte-identically.
- `baselines/pcmc_sleep/driver.py` — the GPU driver. Per checkpoint it
  writes `checkpoints/t0.json` (T0 eval, `step=-1`, task 0) and
  `checkpoints/step_NNNNN.json` with fields: `step`, `task`, `class_acc`,
  `class_pc_acc` (list, one per eval class), `clust_acc`, `clust_pc_acc`,
  `aux_clust_acc`, `aux_clust_pc_acc` (null until a synthetic class is
  visible), `eval_wall_time_s`. Plus `resolved_config.yaml` (embeds the
  `schedule` block), `summary.json`, phase-end `snapshots/step_NNNNN.pt`
  and `final_state.pt` (state_dict + ltm/stm centroids + `cent_g`; final
  also carries example memories). **Your F-PCMC records should mirror this
  shape so Phase 5 can consume both sides uniformly.**
- `baselines/pcmc_sleep/launch.py` — CPU-side launcher; `gpu_env_available`.
- GPU env at `baselines/pcmc_sleep/env/` (`uv sync` inside it; see its
  README). Driver smoke: ~80 s on the 3090
  (`uv run pytest tests/test_pcmc_driver.py -q` runs everything).

### The F-PCMC side you must score (T16 archive, verified on live data)

- Cells at `${DATA_ROOT}/evaluation/f_pcmc_runs/fpcmc_default/p2_seed{42,43,44}/`
  (and `a6_resnet50/p2_seed*/` for the A6 cell): each holds `events.jsonl`
  (schema-v2; first record is a `config_header` carrying the full resolved
  config AND `checkpoint_steps` — the real 44 for seed 42 begin 1070, 2142,
  3213, 4285, 5356, ... and end 21537), `resolved_config.yaml`,
  `summary.json` (keys: cell, checkpoint_steps, checkpoints, config,
  detection, end_of_stream, n_excluded, n_steps, wall_time_seconds — the
  §7.3 secondary metrics live here already).
- `fpcmc/replay.py::replay(log_path, stream_x, store, prior)` reconstructs
  the FINAL `ConceptStore` by bit-reproducible event application (records
  carry `"step"`; kinds: assign/seed/evict/promote/merge/checkpoint).
  Checkpoint-state reconstruction = applying records with step ≤ s to a
  fresh initial store — §8 Q8 decides the mechanism. The initial store is
  built like the live run's (`fpcmc.init.initialize_ltm` on the T0 pool +
  `compute_global_prior`); `run_matrix.py::_run_fpcmc` shows the exact
  construction and the P2 `window_W=50` override.
- **Every Q3(a) eval image has an embedding row**: the pools total 63,326
  rows = `ind_reference` 50,000 (FULL cifar train, all 100 classes) +
  `ind_test` 10,000 (FULL test) + 3,326 synthetic. Map pickle index ↔ pool
  row via `Pool.image_paths` (`"cifar100_train_00037"` naming; row-by-row
  alignment PROVEN in Phase 0.1, 0 mismatches). Synthetic aux rows resolve
  via `manifest.pool`/`within_pool_index` → `pools[pool].x[wpi]`.

## 4. Phase 3 deliverables

1. **The F-PCMC paper-protocol scorer** (placement per §8 Q10): given a
   ConceptStore state at a checkpoint + the eval sets, compute
   (class_acc, class_pc_acc, clust_acc, clust_pc_acc) and the aux
   synthetic-inclusive clustering — the whole-image degenerate case of §5's
   math, documented function-by-function against the upstream line numbers.
2. **Checkpoint-state supply** for the archived P2 cells (per §8 Q8):
   fpcmc_default × seeds {42,43,44} + a6_resnet50, at all 44 checkpoints +
   the task-0 (post-init) state.
3. **Per-cell output artifacts** mirroring the PCMC side's
   `checkpoints/*.json` shape (same fields, same step keys) so both systems
   read identically in Phase 5 — plus a small manifest of what was scored.
4. **The eval-sizing decision (Q6) implemented**: whatever the owner picks
   lands in `run_config.py` (PCMC side, BEFORE Phase 4) and in the scorer
   (F-PCMC side) — one constant set, used by both.
5. **Secondary lossy adapter** (scope per §8 Q9/Q10 answer): the documented
   mapping from PCMC run artifacts to the memory-dynamics metrics reported
   one-sided for F-PCMC (PLAN.md owner decision 3).
6. **Tests**: [U] scorer math on synthetic fixtures (`tests/fixtures/
   vmf_world.py` — consume unmodified) including degenerate-case collapse
   checks; [I][slow] against the archived seed-42 cell (skip cleanly without
   roots.env/data); everything green alongside the existing 135.
7. **PLAN.md updated** (Phase 3 checked off, Q&A recorded verbatim) +
   session report. No `docs/CHANGES.md` entry yet (T17 incomplete).

## 5. Hard-won facts — their eval math and the parity contract

All line references are into `baselines/pcmc_sleep/vendor/` (byte-identical
to `reference/pcmc` @ `e77f5f7`); read them before implementing.

- **supervise** (`core/models/pcmc/pcmc.py:76`, `pcmc_layer.py:211-274`):
  two passes over the labeled set. Pass 1: `D` = mean over images of (mean
  over patches of nearest-centroid cosine DISTANCE, 1−sim). Pass 2: per
  image, per patch, nearest centroid; a sparse matrix gets
  `exp(-dist/D)` at (patch, matched centroid); `fz` = column max (best match
  per centroid per image); accumulate `sum_fz_pool[:, int(Y[it])]` — **raw
  integer label indexes the class column**, which is why eval labels MUST be
  contiguous 0..C−1 in seen order (`run_config.eval_label_map` provides
  this; a violated invariant fails silently with garbage columns). Rows are
  then normalized (+1e-5) → `cent_g` (their g-values).
- **classify** (`pcmc_layer.py:277-312`): per patch, nearest LTM centroid;
  vote weight = `max(cent_g[c])` zeroed when `< rho` (rho=0.25); vote class
  = `argmax(cent_g[c])`; votes sum into a per-image class vector; prediction
  = argmax. Accuracy ×100; per-class from the confusion matrix.
- **cluster** (`pcmc.py:221-268`): per test image, `layer.embed` returns the
  SET of matched centroid indices (one per patch); pairwise **Jaccard
  distance** via `sklearn.pairwise_distances(metric=python_callable)`;
  `SpectralClustering(n_clusters=2×num_classes, affinity='precomputed')` —
  NOTE they pass the DISTANCE matrix as a precomputed affinity (released
  quirk; conform, do not fix). Purity via their `purity_score`
  (`pcmc.py:194-217`): per-cluster majority class, per-class averaging that
  divides by per-class cluster counts — classes never winning a cluster
  yield 0/0 → nan entries in `clust_pc_acc` (released behavior; your JSON
  serialization must tolerate it — the Phase 2 driver already emits nulls).
- **Whole-image degenerate case** (what your scorer implements for F-PCMC):
  one "patch" per image = its CLS embedding; the concept centroids play the
  LTM-centroid role. Then D = mean nearest-concept distance over the labeled
  set; supervise's fz collapses to a single exp(-d/D) at the matched
  concept; classify collapses to ONE vote per image (prediction = the
  matched concept's argmax class, weight-thresholded at rho); cluster's
  per-image set is a singleton → Jaccard distance is binary (0 same concept,
  1 different). Document each collapse next to the code; §8 Q9 settles
  whether the clustering step runs their spectral pipeline verbatim on the
  binary matrix or the analytic equivalent.
- **Which concept centroids** (§8 Q7): `Concept` maintains a cached centroid
  (kept current by `add_observation`; verify the attribute in
  `fpcmc/concepts.py` before use). Candidate populations: LTM-only vs
  tier-1 (LTM ∪ mature STM — `ConceptStore._tier1_stm` is maturity-only, see
  its docstring). This choice changes the headline — owner decides.
- **The eval-set recipe (parity-critical — reproduce EXACTLY).** For CIFAR
  class `cls` at seed `seed` (implemented in `p2_stream.py::_cifar_eval_items`,
  which the GPU driver uses; reproduce it CPU-side, do not import torch):
  - candidates: `mirror.cifar_rows("train"|"test", cls)` (ascending);
  - supervise: `rng = fpcmc.rng.make_rng(seed, f"pcmc_sleep/sup/{cls}")`;
    `pick = np.sort(rng.choice(rows.size, size=sup_size, replace=False))`;
    take `rows[pick]`;
  - test: all rows if `test_size >= rows.size`, else same pattern with
    substream `f"pcmc_sleep/test/{cls}"`;
  - class universe at task t: `run_config.eval_classes_at(t0_classes,
    phases, t)` — classes in label order, current task included;
  - aux clustering adds, per visible synthetic class, its FIRST
    `test_size` stream arrivals (stream order, `manifest.true_class`
    match), labels from `cluster_label_map`.
  Any sizing change (Q6) changes these draws for BOTH systems — that is the
  point of routing both through the same constants.
- **Checkpoint identity**: checkpoints are 0-based post-step stream indices;
  PCMC evals when `it in checkpoint_steps`; F-PCMC's runner emits its
  checkpoint records at the same steps (`fpcmc/stream.py:176`). The T0
  record (`step=-1`, task 0) is the post-init, pre-stream state on both
  sides. Task index of a step: `run_config.phase_task_index` + the
  manifest's phase column.
- **Cost reality (drives Q6)**: their cluster() is O(N²) *python-level*
  jaccard calls + spectral on N×N. N = classes-seen × test_size: at the
  final checkpoints with 100 classes × 100 = 10,000 images → 10⁸ calls per
  checkpoint × 44 checkpoints × 12 runs — infeasible. Phase 2 smoke measured
  85 classes × 4/class (~680 images total) ≈ tens of seconds per eval.
  Classification is patch-embed + argmax and scales fine; clustering is the
  bottleneck. Remember the PCMC side pays this cost IN-RUN on the GPU
  machine's CPU — sizing gates Phase 4 scheduling.
- The archived F-PCMC events.jsonl is **authoritative and byte-deterministic**
  — never regenerate archived cells to make scoring easier; reconstruct
  state FROM them (or get owner approval for a re-run with snapshots).

## 6. Environment / commands

- Scorer + tests: root env — `uv run pytest -m "not slow"` (fast loop),
  `uv run pytest` (full; needs `roots.env`, live pools, and skips the GPU
  smoke without the env). **No `uv add` in the root project.**
- Data: `roots.env` → `DATA_ROOT=/home/alex/data`. Pools load via
  `fpcmc.data.load_all_pools()`; streams via
  `fpcmc.protocols.build_p2(FPCMCConfig.from_yaml("configs/fpcmc_default.yaml"),
  seed, pools)`; mirror via `P2PixelMirror(stream, pools)`.
- PCMC smoke (only if you need to regenerate a sample cell):
  `uv run python baselines/pcmc_sleep/launch.py --arch resnet18 --seed 42
  --smoke --out-root <dir>` (~80 s on the 3090; check `nvidia-smi` first —
  the card is SHARED, a llama-server often holds ~17.7 GB).
- Scorer outputs (real cells) belong under
  `${DATA_ROOT}/evaluation/f_pcmc_runs/pcmc_sleep/` or beside the scored
  F-PCMC cells — propose in your plan; never in-repo.

## 7. Fidelity anchors (when in doubt, conform in this order)

1. The 2024 paper's §2–3 (Eq. 6–7) text (`research_papers/2024_Patch_Based_
   Contrastive_Learning_and_Memory_Consolidation_...md`).
2. The vendored code's actual behavior — for PCMC it RUNS verbatim; for
   F-PCMC your scorer mirrors its computation in the documented
   whole-image degenerate case (including quirks: distance-as-affinity,
   purity nan entries, rho threshold).
3. Parity beats elegance: identical eval sets, identical checkpoint steps,
   identical output schema. When their quirk is ugly, conform and document;
   never "fix" their metric.

## 8. OPEN DESIGN QUESTIONS — ask the owner BEFORE implementing

Q6. **Eval sizing (GATES PHASE 4).** Their clustering eval is infeasible at
    100 labels + 100 test/class × 44 checkpoints (§5 cost facts). Options:
    (a) keep classification at 100/100 but subsample the clustering eval set
    (e.g. 25/class, seeded); (b) run clustering only at phase-end
    checkpoints (11 of 44) at fuller size; (c) reduce test_size globally.
    Whatever is chosen must be implemented in `run_config.py` (PCMC,
    in-run) AND the scorer (F-PCMC) before any Phase 4 cell runs — present
    measured timing estimates with the options.
Q7. **F-PCMC centroid population for the protocol**: LTM-only, or tier-1
    (LTM ∪ mature STM) as routing sees it? (And confirm the centroid
    attribute/definition against `fpcmc/concepts.py`.) This shapes the
    headline comparison — do not decide unilaterally.
Q8. **Checkpoint-state reconstruction**: (a) additive, behavior-preserving
    `max_step` support around `fpcmc/replay.py` primitives (T11 replay tests
    must stay green), vs (b) partial application in the scorer module using
    `read_log` + the documented event semantics, vs (c) owner-approved
    re-run of the F-PCMC P2 cells with a snapshot hook. Recommend (a) or
    (b); (c) touches archived cells and needs explicit approval.
Q9. **Whole-image clustering**: run their SpectralClustering verbatim on the
    binary Jaccard matrix (max fidelity, wasteful), or the documented
    analytic collapse (cluster = matched concept id, purity computed by
    their purity_score on that assignment)? Verify empirically on a small
    set whether the two agree before recommending.
Q10. **Placement + secondary-adapter scope**: scorer under `eval/` (CPU
    eval machinery lives there) vs `baselines/pcmc_sleep/`? And how far the
    lossy PCMC→memory-dynamics adapter goes (per-checkpoint LTM sizes /
    sleep times from the Phase 2 artifacts are cheap; per-step routing
    events would need driver changes — out of Phase 3 scope unless the
    owner says otherwise).

## 9. Definition of done for Phase 3

- Scorer implemented with the degenerate-case math documented against
  upstream line numbers; eval sets byte-identical to the Phase 2 recipe
  (assert this in a test using both implementations' outputs where
  practical — e.g. reproduce `p2_stream`'s draws from the CPU side).
- Checkpoint records for fpcmc_default × {42,43,44} + a6_resnet50 at task-0
  + all 44 checkpoints, in the Phase 2 JSON shape, archived out-of-repo.
- Q6 sizing decision recorded verbatim in PLAN.md and implemented in
  `run_config.py` + scorer (one constant set).
- [U] tests on fixtures + [I][slow] test against the archived seed-42 cell,
  skipping cleanly without data; **full repo suite green** (135 baseline +
  yours), zero existing tests modified (additive `test_repo_layout` entries
  allowed).
- PLAN.md updated; session report written; owner answers recorded verbatim.
