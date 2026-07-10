# CLAUDE.md — Operating Instructions for the F-PCMC Implementation

You are implementing F-PCMC, a frozen-encoder online unsupervised continual learning system. 
This file governs how you work in this repository across all sessions.

## Source of truth (in priority order)

1. `docs/TASKS_frozen_encoder_pcmc.md` — the task plan. Implement tasks in dependency order. Do not start a task whose dependencies are incomplete.
2. `docs/PRD_frozen_encoder_pcmc.md` — the product spec. Every FR/NFR/§ reference in the task plan resolves here.
3. `docs/ASSETS.md` — filled in: exact `.pt` file paths/schemas (verified against live data), the module-to-port inventory with source commit/blob hashes, and the data-access-mechanism decision (§ below). Do not guess paths; if an asset is missing or mismatched, STOP and report.
4. `tests/reference_numbers.yaml` — pinned known-good metrics, sourced from a **fresh seed-42 reproduction run** in the source project (not the rounded numbers in the paper drafts — one of those, v1 overall accuracy, was actually a transcription error; see the file's header). These values are ground truth. **Never edit this file to make a test pass.**
5. `configs/p2_class_split.yaml` — the P2 held-out class split is a frozen, human-decided list, not something to redraw. It deliberately excludes the `aquatic_mammals`/`fish`/`small_mammals` subclasses from the held-out draw (keeping them in T0) so the near-OOD condition stays anchored to classes the model actually knew at T0 — see the file header for the full rationale. `protocols.py`'s `build_p2` must consume this list verbatim.

If the PRD and task plan ever conflict, stop and ask; do not resolve specification conflicts yourself.

## Non-negotiable rules

- **One task per branch.** Branch name `task/TNN-short-name`. Do not mix tasks in one branch.
- **A task is done only when its listed tests pass AND `pytest` (full suite, including all prior tasks' tests) is green.** Never weaken, skip, mark xfail, or delete an existing test to make progress. If an existing test seems wrong, stop and report your reasoning instead of changing it.
- **Gate discipline.** Three tasks end in hard gates:
  - T6 `test_m1_gate` — LTM-only routing must match batch knn_vmf AUROC ±0.01
  - T11 `test_golden_stream_end_to_end` — executable spec of the full system
  - T14 `test_v1_regression_pin` — ported v1 must reproduce pinned numbers A red gate means the implementation is wrong, not the gate. Do not proceed past a red gate. Fix the implementation ("fix the port, never the pin"), or stop and report if you believe the pin itself is wrong.
- **No ground-truth leakage.** Nothing under `fpcmc/` may import from `eval/`'s ground-truth mapping or receive labels at runtime. The invariant tests enforce this; design accordingly.
- **No learning.** No torch autograd, optimizers, or model forward passes anywhere in `fpcmc/`. Embeddings are precomputed inputs.
- **Reference code is read-only.** `reference/stam` and `reference/pcmc` are populated git submodules (STAM, PCMC/`upl-benchmark`), pinned at fixed commits — do not `git submodule update --remote` them or edit their contents. You may read them (and the source papers at `research_papers/`) to understand conventions, but nothing in `fpcmc/`, `eval/`, or `tests/` may import from `reference/`. `test_no_reference_imports` enforces this.
- **`lib/` is a frozen vendored snapshot, read-only.** Verbatim byte-identical copies from the source project, blob hashes recorded in `lib/PROVENANCE.md`. Never edit files under `lib/`; extract/adapt the math at the consuming site (`fpcmc/`, `eval/`) with citation comments pointing back to the lib file.
- **Determinism everywhere.** All randomness flows through `fpcmc/rng.py::make_rng(seed, stream="")` (named substream per module), seeded from config. No module-level RNG, no unseeded library calls (UMAP must receive `random_state`). `test_byte_determinism` and per-module determinism tests enforce this.
- **The golden stream fixture is frozen.** `tests/fixtures/golden_stream.npz` + the pinned sha256 in `tests/fixtures/golden_stream.py` are gate-input data for T8/T11 — never regenerate or re-pin without owner approval. As built (owner-approved 2026-07-10) it carries 25 one-off distractor outliers beyond the TASKS-enumerated composition; the golden-run config must use `stm_capacity ≤ ~25` so the burst-eviction assertions have real LRU pressure.
- **Specs are immutable.** Never modify anything under `docs/` or `tests/reference_numbers.yaml`. Propose spec changes in your report instead. Standing exception: **appending the completed task's entry to `docs/CHANGES.md`** (owner-directed changelog, 2026-07-10). Any other `docs/` edit happens only on explicit owner instruction in-session (precedent: the dated 2026-07-10 as-built annotations in PRD §3/§9 and TASKS T0/T1/T8/T11/T12), and is recorded in `docs/CHANGES.md`.

## Per-task workflow

1. Read the task's entry in the task plan and every PRD section it references. Read the tests it requires before writing implementation code.
2. Post a short plan (files to create/modify, test list, open questions). If any open question is a design decision not answered by the PRD, STOP and ask before implementing.
3. Write the task's tests first where practical (they are specified in the task plan); then implement until green.
4. Run: `pytest -m "not slow"` on every meaningful change; `pytest` (full, including slow/integration) before declaring the task done.
5. Commit in small logical units. Final commit message: `TNN: <summary> — all tests green (<N> passed)`.
6. Append the completed task's entry to `docs/CHANGES.md` (what/why/commits/tests/approved deviations — see the T0 entry for the format).
7. End the session with a report: what was built, test counts, runtime of integration tests vs NFR-1 budgets, any deviations proposed (never silently applied), and anything the next task should know.

## Stop conditions (halt and report rather than improvise)

- A gate test fails and the cause is not a clear implementation bug.
- A real-data schema check fails (`test_real_pool_schemas`).
- A pinned reference number appears unreachable within tolerance.
- The PRD is ambiguous on a decision that changes behavior.
- Any action would require editing `docs/`, `tests/reference_numbers.yaml`, or code under `reference/`.

## Environment

- uv-managed: `uv sync` to set up; run tests as `uv run pytest -m "not slow"` (fast loop) / `uv run pytest` (full). Python pinned to 3.14.3 via `.python-version` — same interpreter as the source venv that produced `tests/reference_numbers.yaml`; don't change it.
- Pinned deps in `pyproject.toml`. Do not upgrade pinned versions — `docs/ASSETS.md` §4 records the exact upstream versions to match (`numpy==2.4.3`, `scikit-learn==1.8.0`, `scipy==1.17.1`, `umap-learn==0.5.11`). `torch==2.11.0` (CPU wheel via the `pytorch-cpu` index) matches the source venv and exists solely so `fpcmc/data.py` can `torch.load` the `.pt` pools — bare `import torch` is fine in `fpcmc/`; `torch.{nn,optim,autograd}` and `.backward()` are banned by `test_no_learning_in_fpcmc`.
- HDBSCAN is `sklearn.cluster.HDBSCAN`; the standalone `hdbscan` PyPI package is deliberately not a dependency (approved T0 deviation) — never add or import it.
- **Embeddings are never copied or symlinked into this repo.** Copy `roots.env.example` to `roots.env` and set `DATA_ROOT` before running anything that touches real data — `fpcmc/data.py` resolves the four `.pt` files from `roots.env`'s `EMBEDDINGS_DIR` at load time (`load_pool`/`load_all_pools`; skip helper `embeddings_available()` returns `(bool, reason)`). Full contract: `data/README.md`; provenance: `docs/ASSETS.md` §1. There is no local `data/embeddings/` directory to check for presence/absence — integration tests skip with a clear message if `roots.env` is missing/unset or the *resolved* path's files aren't found there (not if a literal `data/embeddings/` folder is missing). A task with [I] tests is not done until they have actually run green against real data.
- Unit tests build on the synthetic fixture world `tests/fixtures/vmf_world.py` (`from tests.fixtures.vmf_world import VMFWorld, Segment`; `tests/` is a package). Consume it unmodified — its sampling is a pure function of (world seed, stream label, class, n), so no call can perturb another. Frozen golden data loads via `tests.fixtures.golden_stream.load_golden()`.
- **The scorer interface is frozen as of T2** (`fpcmc/scorers.py`: `Scorer.{score, accepts, margin, score_detail, select}`) — later tasks import, never modify. Owner-approved deviations baked in (docs/CHANGES.md T2; dated notes at PRD FR-1/FR-4.3/FR-5): `Concept` carries per-sub-scorer thresholds `tau` (knn_ref) + `tau_vmf` (vmf); margins are `(τ−s)/|τ|` with lexicographic concept_id tie-break; the composed `knn_vmf` scalar score is its knn_ref sub-score (so composed acceptance is not derivable from the scalar). T4's LOO/shrinkage/global-prior machinery is per-sub-scorer. vMF numerics use the A&S 9.7.7 log-Bessel asymptotic as the production path at every D — never switch to `scipy.special.ive`, which underflows at D=1024; `VmfScorer` requires a finite cached `concept.kappa` once ref_set ≥ `n_vmf_min`.
- **`Concept` dynamics are as-built at T3** (owner-approved decisions, docs/CHANGES.md T3): cached κ self-maintained by `add_observation` on every ref_set-changing observation (T4's lazy ≥25% trigger governs τ/τ_vmf only, never κ); the seeding embedding counts in `ref_count_seen` but not `match_count`/`match_windows` (maturity/θ counts are post-seed matches; `last_matched_at = created_at` at seed, so LRU is defined from birth); each concept carries its own reservoir Generator (per-concept named substream) plus `window_W`/`k_max`/`alpha_ema` fixed at construction, so `add_observation(z, step)` takes no per-call config; `provenance ∈ {"initial", "seeded", "promoted"}` (T8 flips "seeded"→"promoted"; FR-8.3's never-merge rule reads "initial" only); `concept_id` mutation raises; a Concept owns its arrays (the reservoir replaces rows in place — never rely on aliasing arrays passed in). The reservoir draw discipline (exactly one `rng.random(2)` pair per post-fill observation) is load-bearing for `test_reservoir_uniformity`'s exact vectorized replay — changing it requires updating that test's simulator in the same commit.
- Everything is CPU-only. If a step is slow, optimize (vectorize) rather than reaching for GPU.