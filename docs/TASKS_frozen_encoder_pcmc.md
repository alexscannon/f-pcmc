# F-PCMC Implementation Task Plan

**Derived from:** PRD_frozen_encoder_pcmc.md v1.0 (all FR/NFR/§ references point to that document)
**Execution model:** Tasks are discrete units for a coding agent. Complete strictly in dependency order; parallel tracks are marked. **A task is not complete until all of its listed tests pass** and all tests from prior tasks still pass.
**Global rule:** Every module lands with its tests in the same task. No task may weaken, skip, or delete a prior task's test to pass.

---

## Testing Architecture (read before Task 0)

Three test tiers, marked per test below:

- **[U] Unit/property tests** — run on the synthetic vMF fixture (Task 1), no real data, < 1 s each, run on every commit (`pytest -m "not slow"`).
- **[I] Integration gates** — use real precomputed embeddings, marked `@pytest.mark.slow`, run at task completion and in CI nightly.
- **[G] Golden-stream tests** — end-to-end on a planted-novelty synthetic stream with analytically known correct behavior (built in Task 1, asserted from Task 11 onward).

Determinism is itself under test everywhere: any test that runs a seeded component twice must assert identical output.

---

## Dependency Graph

```
T0 ─► T1 ─► T2 ─► T3 ─► T4 ─► T5 ─► T6 [M1 gate]
                              │
                              ├─► T7 ─► T8 ─► T9
                              │              │
                              ├─► T10 ───────┤
                              │              ▼
T12 (parallel after T1) ────► T11 [golden gate] ─► T13 ─► T15 ─► T16
T14 (parallel after T0/T1) ─────────────────────────┘
```
Parallelizable: {T12}, {T14} can proceed alongside T2–T10. Everything else is sequential.

---

## Task 0 — Repository scaffold, config system, reference-code policy

**Depends on:** none  **PRD refs:** §9, NFR-2, NFR-4

**Scope**
1. Create repo layout exactly as PRD §9 (`fpcmc/`, `baselines/`, `configs/`, `tests/`, `eval/`), plus:
   - `lib/` — ported existing modules (scorers, UMAP+HDBSCAN wrapper, metrics utilities) copied verbatim from the current project; document source commit hash in `lib/PROVENANCE.md`.
   - `reference/` — **optional** git submodules for the STAM repo and PCMC repo, read-only, consultation only. If the user opts to exclude them, create `reference/README.md` citing both papers/repos instead. Either way the import guard below must exist.
   - `data/embeddings/` — gitignored; `data/README.md` documents the five expected `.pt` files and their schemas (PRD §3).
2. Config system: YAML → frozen dataclass `FPCMCConfig` containing every parameter from PRD §8, with schema validation (unknown keys are errors) and `to_yaml()` round-trip. Every run artifact embeds the resolved config.
3. Tooling: `pyproject.toml` (numpy, scipy, scikit-learn, umap-learn, hdbscan, pyyaml, pytest, pytest-mock; pin versions), `pytest.ini` with `slow` marker, deterministic-seed helper `fpcmc/rng.py` (single `np.random.Generator` factory; no module-level RNG anywhere).

**Tests**
- [U] `test_config_roundtrip` — load `configs/default.yaml`, serialize, reload, assert equality; assert all PRD §8 keys present with PRD default values.
- [U] `test_config_rejects_unknown_key` — YAML with a typo'd key raises a validation error naming the key.
- [U] `test_no_reference_imports` — static scan (AST walk) of `fpcmc/`, `eval/`, `tests/`: assert zero imports from `reference.*` or any path under `reference/`. **This is the vendoring guard.**
- [U] `test_rng_determinism` — two generators from the same seed produce identical 1,000-draw sequences; different seeds differ.
- [U] `test_repo_layout` — assert required directories/files exist (guards against agent drift from PRD §9).

**Done when:** CI runs `pytest -m "not slow"` green on the empty scaffold.

---

## Task 1 — Data layer: embedding I/O + synthetic fixture generator

**Depends on:** T0  **PRD refs:** §3, FR-1.2

**Scope**
1. `fpcmc/data.py` — loaders for the five real embedding pools: shape/dtype validation, parallel label arrays, class-name maps, on-load L2 normalization (idempotent), memory-mapped where possible.
2. `tests/fixtures/vmf_world.py` — **the synthetic test world used by all subsequent unit tests.** A generator that, given a seed, produces:
   - `k_known` known classes and `k_novel` novel classes, each a vMF distribution on the unit sphere in `D=32` dims with configurable mean directions (controlled pairwise separations) and per-class κ;
   - sampled "T0" pools, "IND test" pools, and "novel" pools with ground-truth labels;
   - a `make_stream(schedule)` method producing deterministic interleaved streams with phase schedules (used later by golden tests);
   - analytic helpers: true class means, true pairwise angular separations.
3. `tests/fixtures/golden_stream.py` — one frozen configuration of the vMF world (`seed=7, k_known=8, k_novel=3`, novel classes recurring across ≥4 windows, one additional "outlier burst" class that appears only in a single contiguous run of 15 examples). Serialize to a committed `.npz` so the golden stream is byte-stable across machines.

**Tests**
- [U] `test_fixture_determinism` — same seed ⇒ identical arrays; different seed ⇒ different.
- [U] `test_fixture_separations` — sampled class means match requested angular separations within tolerance; per-class sample mean direction within 5° of true mean for n=200.
- [U] `test_l2_on_load` — all loaded/generated embeddings have unit norm (atol 1e-6); double-normalization is a no-op.
- [U] `test_golden_stream_frozen` — hash of the committed `.npz` matches a pinned constant.
- [I] `test_real_pool_schemas` — for each of the five real `.pt` files: expected counts (50,000 / 10,000 / 250 / 500 / 2,576), D=1024, label arrays align with class maps, no NaNs. Skipped with a clear message if `data/embeddings/` absent.

**Done when:** all above green; fixture module documented well enough that later tasks use it without modification.

---

## Task 2 — Per-concept scorers

**Depends on:** T1  **PRD refs:** FR-4.1–4.3

**Scope**
`fpcmc/scorers.py`: common interface `Scorer.score(z, concept) -> float` (lower = more compatible) and `Scorer.accepts(z, concept) -> bool` using `concept.tau`. Implement `KnnRefScorer`, `VmfScorer` (Banerjee κ estimator; log-likelihood negated; `n_vmf_min` fallback to knn_ref), `KnnVmfScorer` (OR-accept; assignment margin `(τ_c − s)/τ_c` computed per sub-scorer, best margin wins). Reuse math from `lib/` where a verbatim port exists; otherwise implement fresh with citation comments. Note: a `Concept` stub (centroid, ref_set, tau, kappa fields only) is defined here in `fpcmc/concepts.py` and completed in T3.

**Tests**
- [U] `test_kappa_recovery` — sample n=500 from vMF(μ, κ) for κ ∈ {20, 100, 500} in D=32; Banerjee estimate within 15% relative error. Property-test over 5 seeds.
- [U] `test_kappa_monotone` — tighter fixture class ⇒ larger κ̂.
- [U] `test_knn_ref_monotonicity` — score strictly increases as query rotates away from a fixture class mean along a geodesic (5 waypoints).
- [U] `test_knn_ref_small_refset` — ref_set of size 1..4 clips k_ref correctly, no exception, sensible scores.
- [U] `test_vmf_fallback` — ref_set size < n_vmf_min ⇒ VmfScorer delegates to knn_ref (assert via mock/spy) and flags fallback in its return metadata.
- [U] `test_composed_or_logic` — construct cases where exactly one sub-scorer accepts; composed scorer accepts in both cases and rejects when neither does.
- [U] `test_composed_assignment_margin` — with two concepts both accepting, assignment goes to the larger normalized margin; exact tie broken by lexicographic concept_id (determinism), asserted explicitly.
- [U] `test_scorer_determinism` — identical inputs across two fresh scorer instances ⇒ identical floats.

**Done when:** scorer interface frozen (later tasks import, never modify).

---

## Task 3 — Concept dataclass, reservoir reference sets, centroid dynamics

**Depends on:** T2  **PRD refs:** FR-1.1–1.4, FR-3.2 (seeding shape only)

**Scope**
Complete `fpcmc/concepts.py`: full `Concept` dataclass per PRD FR-1; `add_observation(z, step)` implementing reservoir sampling at `K_max`, match/window bookkeeping (`match_windows` uses `step // window_W`), EMA centroid update for STM with re-normalization, frozen centroid for LTM; `Concept.seed(z, step, tau_prior)` constructor for singletons; merge-lineage fields.

**Tests**
- [U] `test_reservoir_uniformity` — stream 1,000 items into K_max=64 reservoir, repeat 2,000 trials; per-item inclusion frequency within 3σ of 64/1000 (chi-square p > 0.01). This validates the exact replacement rule in FR-1.1.
- [U] `test_reservoir_bound_and_count` — ref_set never exceeds K_max; `ref_count_seen` equals total observations regardless of reservoir state.
- [U] `test_ema_stm_centroid` — hand-computed 3-step EMA with α=0.1 on 2-D unit vectors matches implementation (atol 1e-8); result unit-norm after every step.
- [U] `test_ltm_centroid_frozen` — LTM concept centroid bit-identical before/after 100 observations.
- [U] `test_match_windows` — crafted step sequence spanning windows {0, 0, 2, 5} yields `match_windows == {0, 2, 5}` and `match_count == 4`.
- [U] `test_seed_singleton` — seeded concept: ref_set = [z], centroid = z, status STM, tau = tau_prior, provenance recorded.
- [U] `test_concept_id_immutable` — mutation attempts on `concept_id` raise.

---

## Task 4 — Per-concept adaptive thresholds

**Depends on:** T3  **PRD refs:** FR-5.1–5.4

**Scope**
`fpcmc/thresholds.py`: leave-one-out per-concept score percentile (`q=95` default); shrinkage `τ_c = w·τ_emp + (1−w)·τ_prior`, `w = n/(n+n_shrink)`; global prior computed once from pooled T0 LOO scores (FR-5.3); lazy recomputation trigger when ref_set changed ≥ 25% since last computation (track a dirty counter on `Concept`); `recompute_on_promotion(concept)` hook.

**Tests**
- [U] `test_loo_hand_case` — 5-point ref_set in 2-D with hand-computed knn_ref LOO scores (k_ref=1); percentile matches manual arithmetic exactly.
- [U] `test_loo_excludes_self` — with duplicate embeddings in ref_set, LOO never yields the trivial zero self-distance (guard against self-match bug).
- [U] `test_shrinkage_limits` — n=0 ⇒ τ = τ_prior exactly; n=10,000 ⇒ |τ − τ_emp| < 1e-3·τ_emp; w at n=n_shrink equals 0.5.
- [U] `test_prior_fixed_after_t0` — mutating concepts after prior computation does not change the stored prior (frozen).
- [U] `test_lazy_recompute_trigger` — with K_max=64: 15 new observations ⇒ no recompute (spy assert); 17th (≥25%) ⇒ recompute fires; counter resets.
- [U] `test_threshold_separates_fixture` — for a well-separated fixture class (κ=200, nearest neighbor class 60° away): ≥ 90% of held-out same-class samples accepted, ≥ 99% of other-class samples rejected under the computed τ. This is the semantic correctness test of the whole FR-5 stack.
- [U] `test_promotion_recompute` — promotion hook recomputes τ from full ref_set (value changes from the shrunk STM τ on a crafted case).

---

## Task 5 — ConceptStore and routing core

**Depends on:** T4  **PRD refs:** FR-9 (loop body only, no periodic hooks), FR-3.3 routing order

**Scope**
`fpcmc/concepts.py::ConceptStore` — holds LTM + STM registries, exposes `route(z, step) -> RoutingResult` implementing exactly the FR-9 decision cascade: (1) LTM ∪ mature-STM acceptance set → best-margin assignment; (2) else immature-STM acceptance → best-margin assignment, prediction "unknown"; (3) else seed new STM concept, prediction "unknown". `RoutingResult` carries: prediction, concept_id, tier used (1/2/3), score, margin — everything the event log needs. Vectorize scoring across concepts (single matrix op per tier) for NFR-1.

**Tests**
- [U] `test_routing_tier_order` — craft z accepted by both a mature-STM concept and an immature-STM concept with a *better* margin: tier-1 concept must win (immature cannot claim traffic; FR-3.3).
- [U] `test_routing_tier2` — z rejected by all tier-1, accepted by one immature: assigned there, prediction "unknown".
- [U] `test_routing_seeds` — z rejected everywhere: new STM concept exists, ref_set=[z], tau=tau_prior.
- [U] `test_promoted_participates_immediately` — promote a concept (manually flip via the T4 hook), route a near sample on the very next call: accepted at tier 1. **This is the promotion-aware-routing invariant, tested before the full loop exists.**
- [U] `test_route_updates_bookkeeping` — assignment updates match_count, last_matched_at, reservoir, windows on exactly one concept.
- [U] `test_vectorized_matches_loop` — vectorized routing output identical to a naive per-concept loop on 200 random queries (guards optimization correctness).
- [U] `test_routing_determinism` — full replay of 500 fixture queries twice ⇒ identical RoutingResult sequences.

---

## Task 6 — LTM initialization + **M1 sanity gate**

**Depends on:** T5  **PRD refs:** FR-2, PRD §9 M1 gate

**Scope**
`fpcmc/init.py::initialize_ltm(pool, labels, config) -> ConceptStore` — one LTM concept per T0 class: normalized class-mean centroid, reservoir-sampled ref_set, τ via FR-5.1, κ via FR-4.2; then global prior (FR-5.3).

**Tests**
- [U] `test_init_fixture` — 8-class fixture: 8 LTM concepts; each centroid within 3° of true class mean; every τ finite and positive; provenance "initial".
- [U] `test_init_determinism` — identical stores (including reservoir contents) across two runs, same seed.
- [I] **`test_m1_gate`** — initialize LTM from all 100 CIFAR-100 classes (real embeddings); score IND Test vs (near+far OOD) with min-over-concepts knn_vmf score as the novelty statistic; **AUROC within ±0.01 of the stored batch knn_vmf pipeline result** (pin the reference number in `tests/reference_numbers.yaml`, sourced from the existing project's results). Also assert near-OOD and far-OOD stratified AUROCs within ±0.015.
- [I] `test_init_runtime` — LTM initialization from 50k×1024 completes < 60 s (NFR-1 budget guard).

**Done when:** M1 gate green. **Do not proceed to T7 with a red M1 gate** — it means the routing/scoring stack does not reproduce known-good detection behavior.

---

## Task 7 — STM dynamics: capacity, LRU, maturity

**Depends on:** T6  **PRD refs:** FR-3.1–3.3

**Scope**
Extend `ConceptStore`: STM capacity `Δ`, LRU eviction on `last_matched_at` (ties: older `created_at` first), eviction log records (id, size, age, step), maturity transitions at `n_mature` matches.

**Tests**
- [U] `test_lru_eviction_order` — fill STM to Δ=5 with concepts matched at crafted steps; insertion #6 evicts exactly the least-recently-matched; repeat with a re-match that rescues a would-be victim.
- [U] `test_lru_tiebreak` — two concepts with equal `last_matched_at`: older `created_at` evicted (determinism).
- [U] `test_eviction_log_schema` — every eviction produces a log record with all fields; count matches evictions.
- [U] `test_maturity_transition` — concept at match_count = n_mature − 1 is tier-2; one more match ⇒ tier-1 on the next route call.
- [U] `test_capacity_invariant` — property test: random 2,000-step fixture stream, assert `len(STM) ≤ Δ` after every step.
- [U] `test_ltm_never_evicted` — LTM concepts exempt from capacity/eviction regardless of staleness.

---

## Task 8 — Promotion

**Depends on:** T7  **PRD refs:** FR-7.1–7.2, FR-5.4

**Scope**
`fpcmc/memory.py::PromotionEvaluator` — evaluates the four criteria (size θ, cohesion, separation vs every LTM τ, recurrence m_windows over window_W) against mature STM concepts on the periodic hook; atomic promotion (status flip, centroid freeze, τ recompute, STM accounting release, promotion log record with all PRD FR-7.2 fields).

**Tests**
- [U] `test_each_criterion_blocks` — parameterized over the four criteria: construct a candidate passing exactly three and failing one; assert no promotion and the log names the failing criterion. Four cases:
  - size: 29 matches, all else passing;
  - cohesion: two-lobe candidate (samples from two antipodal-ish fixture classes) with pairwise cos-sim below min_cohesion;
  - separation: candidate seeded inside a known LTM class (its centroid accepted by that LTM τ);
  - recurrence: 40 matches all within one window (the fixture "outlier burst" class).
- [U] `test_promotion_happy_path` — recurring fixture novel class: promoted; assert atomically (single hook call): status=LTM, centroid frozen thereafter, τ ≠ pre-promotion shrunk τ and equals FR-5.1 recompute, STM occupancy decremented, log record complete.
- [U] `test_outlier_burst_never_promotes` — golden-world burst class run through 2,000 steps: never promoted, eventually LRU-evicted (assert eviction record exists for it). **This is the recurring-novelty-vs-outlier discrimination test.**
- [U] `test_separation_uses_per_concept_tau` — same candidate promotes/blocks when only the nearest LTM concept's τ is tightened/loosened (proves criterion 3 reads per-concept thresholds, not a global one).
- [U] `test_promotion_idempotent` — evaluator on an already-promoted concept is a no-op.

---

## Task 9 — Merging

**Depends on:** T8  **PRD refs:** FR-8.1–8.3

**Scope**
`fpcmc/memory.py::MergeSweeper` — periodic STM↔STM (two-condition rule: centroid sim ≥ merge_sim AND cross-ref kNN ≤ 1.1× within-ref kNN), STM↔LTM folding, LTM↔LTM for promoted-only pairs; on-promotion check; survivor selection by match_count; ref_set union re-reservoired to K_max; centroid/τ/κ recompute for the survivor (STM survivor only — LTM survivor centroid stays frozen, ref_set/τ still updated); lineage map maintenance.

**Tests**
- [U] `test_merge_two_conditions` — three crafted pairs: (a) both conditions hold ⇒ merge; (b) centroids similar but bimodal ref_sets (cross-distance ratio > 1.1) ⇒ no merge; (c) tight ref_sets but centroid sim < merge_sim ⇒ no merge. Case (b) is the near-OOD-collapse guard from PRD §11.
- [U] `test_merge_survivor_and_lineage` — survivor is larger match_count; lineage records `{survivor: [absorbed]}`; absorbed id never reappears in routing; transitive merges (A←B then A←C) accumulate lineage.
- [U] `test_stm_ltm_fold` — STM candidate whose centroid is accepted by an LTM τ: folded, LTM centroid bit-identical, LTM ref_count_seen increased, candidate deleted.
- [U] `test_initial_initial_never_merges` — two provenance="initial" concepts moved artificially close: sweep refuses (FR-8.3).
- [U] `test_promoted_promoted_merge` — fragment the golden novel class into two promoted concepts (force via manual promotion of two halves); LTM↔LTM sweep merges them; fragmentation index for that class returns to 1.
- [U] `test_merged_refset_bound` — post-union ref_set ≤ K_max via reservoir subsample, deterministic under seed.

---

## Task 10 — Residual clustering (identity-preserving consolidation)

**Depends on:** T7 (can run parallel to T8–T9)  **PRD refs:** FR-6.1–6.2

**Scope**
`fpcmc/residual.py` — residual pool of embeddings whose singleton concepts failed to mature within `w_residual` steps; trigger every `T_cluster` steps when pool ≥ 30; wraps `lib/` UMAP+HDBSCAN; HDBSCAN groups over pool ∪ immature-STM centroids drive **merges of existing immature candidates** (never fresh anonymous clusters — identities preserved); noise points untouched.

**Tests**
- [U] `test_pool_aging` — singleton seeded at step s enters pool exactly at s + w_residual if still immature; matured singletons never enter.
- [U] `test_trigger_conditions` — no run below 30 pool items or off-schedule (spy on the clustering call).
- [U] `test_identity_preserving_merge` — mock HDBSCAN returning a known grouping over 4 immature candidates: candidates merged pairwise per the grouping via the T9 merge path (lineage recorded); no new concept_ids created by this pathway.
- [U] `test_noise_untouched` — mock noise labels: those candidates unchanged and still LRU-eligible.
- [I] `test_residual_consolidation_real` — under-segmentation scenario from real embeddings: seed 6 immature candidates from split halves of 3 near-OOD classes; real UMAP+HDBSCAN consolidation reduces them to 3 concepts with correct pairings (uses ground truth to verify pairing only).

---

## Task 11 — Stream runner, event log, periodic hooks + **golden gate**

**Depends on:** T8, T9, T10  **PRD refs:** FR-9, NFR-1, NFR-3

**Scope**
`fpcmc/stream.py::StreamRunner` — full wake loop wiring routing + hooks (residual clustering, merge sweep, promotion evaluation, eval checkpoints) on their schedules; JSONL event log with typed records (`assign`, `seed`, `evict`, `promote`, `merge`, `checkpoint`, `config_header`); `replay.py` that reconstructs final ConceptStore state from the log alone.

**Tests**
- [U] `test_hook_schedule` — spies confirm each hook fires at exactly the configured steps over a 2,000-step run.
- [U] `test_log_schema_complete` — every record validates against a JSON schema; every mutation of the store during a run has a corresponding record (instrument the store with a mutation counter and reconcile).
- [U] `test_byte_determinism` — two runs, same config+seed: byte-identical JSONL (NFR-3/FR-9.2). Third run with seed+1: differs.
- [U] `test_replay_reconstruction` — replayed final state equals live final state (concept ids, statuses, match_counts, lineage; centroids atol 1e-9).
- [G] **`test_golden_stream_end_to_end`** — run the frozen golden stream (8 known, 3 recurring novel, 1 outlier-burst class):
  - all 3 recurring novel classes promoted, each exactly once (fragmentation index = 1.0);
  - burst class: zero promotions, ≥ 1 eviction record;
  - end-of-stream purity of each promoted concept ≥ 0.95 against fixture ground truth;
  - post-promotion samples of promoted classes routed at tier 1 (promotion-aware routing, measured: ≥ 90% of that class's post-promotion arrivals);
  - known-class expanding accuracy ≥ 0.95 throughout;
  - "unknown" residual at end < 5% of novel-class examples.
  **This test is the executable specification of the whole system.** If any assertion fails, the responsible mechanism's task is reopened.
- [I] `test_runtime_budget` — P1-sized run (13,326 real embeddings) completes within NFR-1 budget; log wall-time per 1k steps in the report.

---

## Task 12 — Stream protocols P1 and P2 (parallel track, start after T1)

**Depends on:** T1  **PRD refs:** §7.1

**Scope**
`fpcmc/protocols.py` — `build_p1(config, seed)` reproducing the v1 stream construction exactly (1,000 IND warmup; shuffled interleave of 9,000 IND test + 250 synthetic IND + 500 near + 2,576 far); `build_p2(config, seed)` phased O-UCL stream per PRD (T0 = 80 classes; phase schedule for 20 held-out CIFAR classes, near-OOD, far-OOD by superclass; classes cease post-phase; 30% past-class interleave; 4 checkpoints/phase). Emits a `StreamManifest` (per-index: pool, class, phase) consumed by the runner and the eval harness.

**Tests**
- [U] `test_p2_fixture_schedule` — P2 builder on the fixture world: class introduction steps match schedule; zero occurrences of any class after its phase (hard assert); past-class interleave fraction = 0.30 ± 0.02 per phase; checkpoints at 1/4, 2/4, 3/4, 4/4 of each phase.
- [U] `test_protocol_determinism` — identical manifests for same seed; disjoint shuffles across seeds {42,43,44}.
- [I] `test_p1_matches_v1` — P1 composition: exact counts per pool; warmup contains only real IND test; total 13,326. If the original v1 code exposes its ordering (seed 42), assert index-level equality; otherwise assert distributional identity (counts per pool per 1k-step bucket).
- [I] `test_p2_real_partition` — the 80/20 CIFAR class partition is deterministic, disjoint, and covers all 100; near-OOD phases contain exactly the 6 near classes; far phases partition the 43 far classes by superclass with none repeated.

---

## Task 13 — Evaluation harness

**Depends on:** T11, T12  **PRD refs:** §7.2–7.3

**Scope**
`eval/` — ground-truth mapping (majority label per concept, eval-side only), strict/lenient "unknown" scoring, all §7.3 metrics: streaming detection AUROC/FPR@95 (stratified), expanding classification accuracy + forgetting curve, promotion-time vs end-of-stream purity, fragmentation index (post LTM↔LTM merge), coverage, STM occupancy/eviction composition, residual-unknown rate, threshold-health (post-hoc per-concept FPR/FNR), τ distribution. Figure/table generators reading only the JSONL log (NFR-3).

**Tests**
- [U] `test_metric_microcases` — hand-computed 10-example cases for: expanding accuracy, fragmentation index (incl. lineage-merged fragments counting as one), promotion vs end purity divergence, coverage. Exact equality.
- [U] `test_unknown_variants` — crafted sequence where a class is introduced at step 100 and promoted at step 300: "unknown" at step 50 correct in both variants; at step 200 correct only in lenient; at step 400 wrong in both. Assert both scorers.
- [U] `test_auroc_against_sklearn` — streaming AUROC on synthetic scores equals `sklearn.metrics.roc_auc_score` (atol 1e-9).
- [U] `test_gt_map_isolation` — static assertion (AST scan) that no module under `fpcmc/` imports the gt-mapping module; ground truth flows only through `eval/`. **Guards against label leakage into the pipeline.**
- [G] `test_eval_on_golden` — harness on the golden run reproduces the exact numbers asserted in T11's golden test (single source of truth for metric definitions).
- [U] `test_figures_from_log_only` — figure generation succeeds given only the JSONL file and manifest (no live objects), producing the PRD-listed plots without exceptions.

---

## Task 14 — Baselines: v1 port, batch wrapper, oracle (parallel track, start after T0/T1)

**Depends on:** T0, T1 (and T12/T13 for full comparability runs)  **PRD refs:** §7.4 B1–B3

**Scope**
- `baselines/v1_stream.py` — the existing v1 pipeline moved in **unmodified** except import-path shims; adapter emitting the same JSONL schema so the T13 harness scores it.
- `baselines/batch_knn_vmf.py` — wrapper invoking the existing batch pipeline at each P1/P2 checkpoint.
- `baselines/oracle.py` — ground-truth-labeled routing ceiling (existing oracle harness, adapted).

**Tests**
- [I] **`test_v1_regression_pin`** — v1 on P1, seed 42, reproduces its original headline numbers within tolerance, pinned in `tests/reference_numbers.yaml`: detection AUROC (all-OOD) 0.850 ± 0.005; overall accuracy 74.03% ± 0.5; promoted clusters 14 ± 0; end-of-stream median purity 0.61 ± 0.02; residual buffer 1,962 ± 0. **A red pin means the port changed behavior — fix the port, never the pin.** (Corrected 2026-07-10: the paper drafts state 72.4%, but a fresh seed-42 reproduction — byte-identical to the archived run — measured 0.7402822 (74.03%); 72.4% was a transcription error in `writeups/consolidated.md`/`05_experimental_results.md`, not a different pipeline state. See `tests/reference_numbers.yaml` header for full provenance.)
- [I] `test_batch_wrapper_matches_existing` — wrapper at the end-of-stream checkpoint reproduces the stored batch knn_vmf metrics within ±0.005.
- [U] `test_v1_untouched` — checksum of the v1 core module matches the recorded source hash from `lib/PROVENANCE.md` (only the shim file may differ).
- [U] `test_oracle_upper_bounds` — on the fixture world, oracle accuracy ≥ every F-PCMC golden-run accuracy metric (a ceiling that isn't a ceiling indicates a scoring bug).

---

## Task 15 — Ablation flags, run configs, sweep runner

**Depends on:** T11, T13, T14  **PRD refs:** §7.4 A1–A6, §8

**Scope**
Config-driven ablation switches: `A1 global_tau`, `A2 no_stm` (θ-count direct promotion), `A3 no_recurrence`, `A4 no_merge`, `A5 scorer ∈ {knn_ref, vmf}`, `A6 encoder=resnet50`. One YAML per run in §7.4 committed under `configs/`. `run_matrix.py` executing {system × protocol × seed} with resumability; sweep runner limited to the three PRD §8 sweep parameters on P1/seed 42.

**Tests**
- [U] `test_ablation_flags_bite` — parameterized per flag: run the golden stream with the flag on and off; assert a flag-specific behavioral delta (A1: all τ_c equal; A2: zero STM records in log; A3: burst class *does* promote — the pathology returns; A4: fragmentation index > 1 achievable on a crafted split; A5: sub-scorer identity visible in assign records). A flag that changes nothing is a wiring bug.
- [U] `test_config_matrix_complete` — every run row of PRD §7.4 has a config file; configs differ from default only in their declared ablation keys (diff-based assert).
- [U] `test_sweep_scope_guard` — sweep runner rejects any parameter outside the three PRD-sanctioned sweep keys.
- [I] `test_a6_resnet_smoke` — A6 config runs end-to-end on real ResNet-50 embeddings without error (no performance assertion — degraded results are the expected finding).

---

## Task 16 — Full experiment execution and success-criteria report

**Depends on:** T15  **PRD refs:** §7.5, NFR-1–3

**Scope**
Execute the full matrix (F-PCMC + B1–B3 + A1–A6, P1 + P2, seeds {42,43,44}); generate the results workbook: all §7.3 metrics mean ± std, the §7.5 success-criteria scorecard with explicit pass/fail per criterion vs B1, and the ablation attribution table. Archive all JSONL logs + resolved configs.

**Tests / acceptance**
- [I] `test_matrix_reproducibility` — re-running any single cell of the matrix from its archived config reproduces its archived headline metrics exactly (byte-determinism carried through).
- [I] `test_runtime_budgets` — every cell within NFR-1 budgets; report table of wall-times.
- [I] `test_scorecard_generated` — scorecard exists, contains all five §7.5 criteria with numeric evidence and pass/fail; **failing a research criterion does not fail this test** — the deliverable is the diagnostic scorecard, and the ablation table must then localize the underperforming mechanism (assert the attribution table is populated for any failed criterion).
- Manual gate: human review of the scorecard before results are cited anywhere.

---

## Cross-Cutting Invariant Tests (live in `tests/test_invariants.py`, extended as tasks land)

These run against every stream execution in the suite (fixture and golden) from the earliest task at which they're expressible:

1. **Single-pass:** each stream index processed exactly once (T11+).
2. **No label leakage:** AST guard from T13 plus runtime assertion that `ConceptStore` never receives a ground-truth label argument.
3. **STM capacity ≤ Δ at every step** (T7+).
4. **Concept-id uniqueness and immutability across the entire run, including merges** (T3+).
5. **No global threshold in the main path:** grep/AST assertion that the decision cascade references only `concept.tau` and `tau_prior` (the latter only in seeding/shrinkage code paths) (T5+).
6. **Frozen encoder:** no torch autograd, no optimizer, no model forward pass anywhere in `fpcmc/` (import + AST scan) (T0+).
7. **Reference-code isolation:** T0's `test_no_reference_imports`, re-run always.