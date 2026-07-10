# PRD: F-PCMC — Frozen-Encoder Online Unsupervised Continual Learning

**Version:** 1.0 (implementation spec for coding agent)
**Owner:** Alex Cannon
**Status:** Approved for implementation
**Target:** Minimal viable system in 2 weeks; full evaluation suite in 4 weeks

---

## 1. Context and Motivation

### 1.1 Research question

> Can the core PCMC/STAM online novelty-detection and memory-management machinery (Taylor et al., CoLLAs 2024; Smith et al., IJCAI 2021) perform open-world continual learning when the adaptive, sleep-trained encoder is replaced with a **frozen pretrained foundation-model encoder** (DINOv3 ViT-L/16)?

The hypothesis is that a frozen foundation encoder eliminates the need for sleep-based encoder retraining, but the **adaptive concept-memory layer** — STM/LTM hierarchy, per-concept novelty thresholds, outlier-vs-recurring-novelty discrimination, cluster promotion, duplicate merging — remains essential to realize the representation's capacity.

### 1.2 Empirical motivation (existing results, do not re-run)

Prior experiments on this codebase established:

1. **Representation ceiling is high, realized performance is low.** Oracle experiments show frozen DINOv3 embeddings support near-perfect near-OOD clustering (ARI 0.99) and strong far-OOD structure (NMI 0.90). The best no-oracle composed method (knn_vmf) recovers ~55% of the representational ceiling; Mahalanobis+HDBSCAN recovers ~18%.
2. **The encoder dominates.** An identical pipeline on frozen ResNet-50 fails qualitatively (0 promotions vs 14; Mahalanobis below chance on near-OOD).
3. **The v1 streaming pipeline (single global static threshold, stateless re-clustering, no promotion-aware thresholds) exhibits exactly the failure modes PCMC's memory layer was designed to prevent:**
   - FPR@95 degraded to 0.62 under a stale global threshold.
   - Median promoted-cluster purity drifted 1.0 → 0.61 (promoted classes had no thresholds of their own).
   - Class fragmentation: one ground-truth class promoted 4×; buffer re-clustered from scratch every trigger with no persistent cluster identity.
   - 1,962 residual buffer examples force-classified at 31% accuracy (no outlier/recurrence distinction).

F-PCMC replaces the v1 streaming pipeline. The v1 pipeline is retained **as a baseline only**.

### 1.3 What "PCMC minus sleep" means structurally

| PCMC component | F-PCMC disposition | Rationale |
|---|---|---|
| Contrastive encoder training (init + sleep) | **Removed** | Frozen DINOv3 replaces it. This removal is the paper's thesis. |
| Raw patch memories `M_j` per centroid | **Replaced by embedding reference sets** | PCMC stored raw patches only because embeddings went stale after each sleep. Frozen encoder → embeddings never stale → store embeddings directly. |
| Sleep-phase centroid re-embedding | **Removed** (no-op) | Embeddings stable by construction. |
| Sleep-phase memory consolidation (pruning) | **Replaced by reservoir sampling** on reference sets | Same goal (bounded memory), achievable online. |
| Pixel-patch hierarchy / patch-level clustering | **Replaced by whole-image CLS embeddings** in v1 | DINOv3 CLS token already encodes compositional semantics. Patch-token variant is a stretch goal (§10). |
| Online centroid learning (EMA update, Eq. 3 of PCMC) | **Kept** for STM; LTM centroids frozen | Direct port. |
| Novelty detection w/ sliding-window global threshold | **Upgraded to per-concept thresholds** | Explicit advisor requirement; fixes v1's dominant failure mode. |
| STM (capacity Δ, LRU eviction) | **Kept** | Direct port. |
| θ-match consolidation STM → LTM | **Kept, strengthened** with cohesion/separation/recurrence criteria | Distinguishes recurring novelty from bursty outliers. |
| Wake/sleep scheduler | **Removed** | Single continuous wake phase. |

---

## 2. Goals and Non-Goals

### 2.1 Goals

- G1: Implement an online, single-pass concept-memory system (F-PCMC) over frozen DINOv3 embeddings with: LTM of accepted concepts, STM of candidate concepts, per-concept adaptive novelty thresholds, persistent cluster identities, recurrence-aware promotion, immediate promotion-aware routing, duplicate-cluster merging, and per-concept reference sets.
- G2: Evaluate on the existing CIFAR-100 + synthetic-novel benchmark under two stream protocols (§7.1), with frequent mid-stream evaluation (O-UCL style), across ≥3 seeds.
- G3: Compare against retained baselines: v1 streaming pipeline, batch knn_vmf pipeline, oracle ceiling, and ablations (§7.4).
- G4: Reproduce all reported numbers deterministically from config files.

### 2.2 Non-Goals

- No encoder training, fine-tuning, adapters, prompts, or projection heads. Zero learned parameters anywhere.
- No re-generation of the synthetic dataset or re-extraction of embeddings (all precomputed; see §3).
- No GPU requirement at runtime — the entire system operates on cached embeddings (CPU, NumPy/scikit-learn).
- v1 does not use DINOv3 patch tokens (stretch goal only).
- No hyperparameter search beyond the sweeps explicitly listed in §8.

---

## 3. Existing Assets (reuse, do not rewrite)

The coding agent must integrate with, not duplicate, the following existing modules. Exact paths to be filled in from the current repo; the interfaces below are contracts. *(Filled in 2026-07-10: exact paths, schemas, and module inventory live in `docs/ASSETS.md`; the four `.pt` files resolve at load time via `roots.env` → `EMBEDDINGS_DIR` — contract in `data/README.md`. Ported snapshots live read-only under `lib/`.)*

| Asset | Contract |
|---|---|
| Precomputed embedding tensors | `.pt` files: IND Reference (50,000 × 1024, CIFAR-100 train), IND Test (10,000 × 1024), Synthetic IND (250 × 1024), Near-OOD (500 × 1024, 6 classes), Far-OOD (2,576 × 1024, 43 classes). Each with parallel integer label arrays and class-name mappings. |
| Scorer implementations | Mahalanobis (per-class, shared), min-cosine, kNN-density, **knn_vmf composed scorer**. Reuse as library functions; refactor into per-concept form per §5.4. |
| UMAP+HDBSCAN clustering module | Reuse for STM residual clustering (§5.6) with existing hyperparameters. |
| v1 streaming harness | Retain untouched as `baselines/v1_stream.py`. New system lives alongside. |
| Metrics utilities | AUROC/AUPR/FPR@95, ARI/NMI/purity/completeness, rolling-window trackers. |

---

## 4. System Overview

```
                          ┌──────────────────────────────────────────┐
 stream x_t ─► frozen ──► │             ROUTER                       │
               encoder    │  score z_t against ALL concepts          │
               (cached    │  (LTM ∪ mature STM), per-concept τ_c     │
                z_t)      └───────┬───────────────┬─────────────────┘
                                  │ match         │ no match (novel)
                                  ▼               ▼
                        ┌──────────────┐   ┌────────────────────┐
                        │ assign to    │   │ STM: seed/join      │
                        │ concept c*   │   │ candidate concept;  │
                        │ update stats │   │ LRU eviction        │
                        └──────┬───────┘   └─────────┬──────────┘
                               │                     │ θ matches +
                               │                     │ cohesion/separation/
                               │                     │ recurrence criteria
                               │                     ▼
                               │            ┌────────────────────┐
                               │            │ PROMOTE → LTM      │
                               │            │ (immediate routing │
                               │            │  participation)    │
                               │            └─────────┬──────────┘
                               │                      │
                               ▼                      ▼
                        ┌──────────────────────────────────┐
                        │ periodic MERGE sweep             │
                        │ (STM↔STM, STM↔LTM duplicates)    │
                        └──────────────────────────────────┘
```

There is no sleep phase. There is no global precision matrix and no global detection threshold in the main path (a global threshold survives only inside baselines and as a bootstrap prior, §5.5).

---

## 5. Functional Requirements

### 5.1 FR-1: Concept data structure

Implement a `Concept` dataclass. Every known class — original CIFAR-100 classes and promoted novel classes — is a `Concept`. No structural distinction between them other than provenance.

```python
@dataclass
class Concept:
    concept_id: str            # stable, never reused; e.g. "ltm_037", "stm_0142"
    status: Literal["STM", "LTM"]
    centroid: np.ndarray       # (D,) L2-normalized mean direction
    ref_set: np.ndarray        # (K, D) reference embeddings, K ≤ K_max
    ref_count_seen: int        # total assignments ever (for reservoir sampling)
    tau: float                 # per-concept novelty/acceptance threshold (§5.5)
    kappa: float               # vMF concentration estimate (§5.4)
    match_count: int           # total matched examples
    match_windows: set[int]    # distinct time-window indices with ≥1 match (§5.7)
    created_at: int            # stream step
    last_matched_at: int       # stream step (drives STM LRU)
    provenance: Literal["initial", "promoted"]
    gt_majority_label: Optional[str]  # eval-only; never used by the pipeline logic
```

Requirements:
- **FR-1.1** Reference sets are bounded at `K_max` (default 64) via reservoir sampling: each new assigned embedding replaces a uniformly random slot with probability `K_max / ref_count_seen` once full. This replaces PCMC's sleep-time pruning.
- **FR-1.2** All embeddings L2-normalized on ingestion; cosine geometry throughout.
- **FR-1.3** LTM centroids are frozen after creation (matching PCMC's frozen-LTM design and the STAM ablation showing dynamic LTM causes forgetting). STM centroids update by EMA: `c ← normalize((1-α)·c + α·z)`, α default 0.1.
- **FR-1.4** `concept_id` is persistent for the lifetime of the run. Merges record a lineage map `merged_from: {survivor_id: [absorbed_ids]}` for post-hoc analysis. This is the "persistent cluster identities" requirement.

*(As built at T2, owner-approved 2026-07-10: the dataclass carries per-sub-scorer thresholds — `tau: float` (knn_ref scale; also what FR-3.2 seeding and the single-scorer configs use) plus `tau_vmf: float` (vmf scale) — because FR-4.3's "respective per-concept threshold" cannot be one scalar across the two score scales (knn_ref: cosine distance in [0,2]; vmf: negative log-likelihood, ≈ −2400 and negative at D=1024). See docs/CHANGES.md T2.)*

### 5.2 FR-2: LTM initialization (task T0)

- **FR-2.1** Initialize one LTM concept per T0 class from the T0 training split (which classes constitute T0 depends on stream protocol, §7.1). Centroid = normalized class mean; reference set = reservoir sample of K_max class embeddings; τ_c and κ_c computed per §5.4–5.5.
- **FR-2.2** No pooled covariance, no precision matrix in the main path. (Mahalanobis survives only in baselines.)

### 5.3 FR-3: STM

- **FR-3.1** STM holds candidate concepts, capacity `Δ` (default 100 concepts). When full, evict least-recently-matched (LRU on `last_matched_at`). Evicted candidates are logged (id, size, age) — this is the "forgetting outliers" mechanism and must be measurable.
- **FR-3.2** A novel embedding (fails all concept thresholds, §5.5) seeds a new STM concept: centroid = the embedding, ref_set = {embedding}, τ bootstrapped from the global prior (§5.5.3).
- **FR-3.3** STM concepts become **routing-eligible** ("mature") once `match_count ≥ n_mature` (default 5). Immature STM concepts still receive assignments (a new point is checked against them during the novelty step) but do not "claim" points away from LTM: routing order is (1) test against LTM ∪ mature-STM thresholds; (2) if none accept, test against immature STM; (3) if none accept, seed new candidate. This prevents singleton noise from capturing stream traffic while preserving PCMC's create-cluster-from-one-novel-point behavior.

### 5.4 FR-4: Per-concept scoring (routing primitive)

Implement two interchangeable per-concept scorers behind a common interface `score(z, concept) -> float` (lower = more compatible). Selected by config; the composed default mirrors the validated knn_vmf paradigm:

- **FR-4.1 `knn_ref`:** mean cosine distance from `z` to its `k_ref` (default 5, clipped to ref_set size) nearest neighbors within `concept.ref_set`. Non-parametric; works from ref_set size 1 upward.
- **FR-4.2 `vmf`:** negative vMF log-likelihood under the concept's vMF(μ=centroid, κ). Estimate κ by the standard Banerjee et al. approximation from the ref_set: with `r̄ = ‖mean of ref embeddings‖`, `κ ≈ r̄(D − r̄²)/(1 − r̄²)`. Require ref_set size ≥ `n_vmf_min` (default 10); below that, fall back to `knn_ref`.
- **FR-4.3 `knn_vmf` (default):** route/accept if **either** knn_ref or vmf accepts under its respective per-concept threshold; assign to the concept with the best normalized margin `(τ_c − s(z,c)) / τ_c` among accepting concepts. Rationale: the composed paradigm was the strongest no-oracle method in prior experiments; kNN captures local multi-modal structure, vMF captures global directional concentration.

*(As built at T2, owner-approved 2026-07-10: (a) the normalized margin is computed as `(τ_c − s)/|τ_c|` — the literal formula inverts orientation when τ_c < 0, which the vmf thresholds are at D=1024; identical to the formula above for all τ_c > 0. Exact margin ties break to the lexicographically smallest concept_id. (b) The composed scorer's scalar `score(z, c)` is its knn_ref sub-score — FR-4.3 defines accept/assign but no composed scalar; this mirrors the pure-kNN detection statistic of the source paradigm behind the T6 pin — so composed acceptance is not derivable from the scalar alone. See docs/CHANGES.md T2.)*

### 5.5 FR-5: Per-concept adaptive thresholds

This is the core replacement for the v1 global static threshold. **There is no single global τ in the main path.**

- **FR-5.1 (LTM, well-populated):** τ_c = the `q`-th percentile (default q=95) of leave-one-out scores of the concept's own reference set members against the concept. Computed at initialization/promotion and recomputed lazily whenever the ref_set has changed by ≥25% since last computation.
- **FR-5.2 (STM / small ref_set):** shrinkage estimate `τ_c = w·τ_c^emp + (1−w)·τ_prior`, with `w = n/(n + n_shrink)` (n = ref_set size, n_shrink default 10), where `τ_c^emp` is the small-sample percentile and `τ_prior` is the global prior below.
- **FR-5.3 (global prior, bootstrap only):** τ_prior = the q-th percentile of leave-one-out per-concept scores pooled over all T0 classes at initialization. Fixed after T0. Used only as a shrinkage target and for singleton seeding — never as a decision boundary by itself.
- **FR-5.4** On promotion, the promoted concept's τ is recomputed from its full reference set (FR-5.1 rule). **This is what makes routing promotion-aware:** the promoted class immediately participates in novelty decisions with a calibrated boundary, so subsequent members of that class are absorbed rather than re-buffered — structurally eliminating the v1 purity-drift and re-promotion failure modes.

*(2026-07-10, consequence of the T2 per-sub-scorer thresholds: under `scorer=knn_vmf`, FR-5.1–5.4 apply per sub-scorer — LOO percentiles computed under knn_ref and under vmf separately, yielding τ_c and τ_vmf,c; the FR-5.3 global prior is likewise a per-sub-scorer pair, and FR-3.2 singleton seeding bootstraps both thresholds from their respective priors. See docs/CHANGES.md T2.)*

### 5.6 FR-6: STM residual clustering (assist mechanism)

Online single-linkage-to-centroid assignment can under-segment slowly. Retain a lightweight periodic clustering pass over **unattached recent novelty only** (not the whole buffer — there is no monolithic buffer anymore):

- **FR-6.1** Maintain a small residual pool of embeddings that seeded singleton STM concepts which have not matured after `w_residual` (default 500) steps. Every `T_cluster` (default 500) steps, if the pool ≥ 30, run the existing UMAP+HDBSCAN module on pool ∪ immature-STM centroids; use resulting clusters to **merge immature STM candidates** that HDBSCAN groups together (union ref_sets, recompute centroid/τ). This replaces v1's from-scratch re-clustering with an identity-preserving consolidation step.
- **FR-6.2** HDBSCAN noise points remain as-is (they will age out via LRU).

### 5.7 FR-7: Promotion (STM → LTM)

A mature STM concept is promoted iff ALL of:

1. **Size:** `match_count ≥ θ` (default 30, per PCMC).
2. **Cohesion:** mean pairwise cosine similarity within ref_set ≥ `min_cohesion` (default 0.55; sweep in §8).
3. **Separation:** margin to nearest existing LTM concept — `score(centroid_stm, c_ltm)` must exceed `sep_factor · τ_{c_ltm}` (default sep_factor 1.0) for all LTM concepts, i.e. the candidate's centroid would itself be rejected by every LTM concept. Prevents promoting duplicates of known classes.
4. **Recurrence:** matches span ≥ `m_windows` (default 3) distinct time windows of length `W` (default 250 steps). This operationalizes "recurring novelty vs isolated outlier burst": a one-shot cluster of near-duplicates arriving together fails recurrence and eventually LRU-evicts.

- **FR-7.1** Promotion is immediate and atomic: status→LTM, centroid frozen at current value, τ recomputed (FR-5.4), removed from STM capacity accounting.
- **FR-7.2** Log per-promotion: step, concept_id, size, cohesion, separation margin, window count, gt_majority_label + purity (eval-only fields).

### 5.8 FR-8: Duplicate-cluster merging

Periodic sweep every `T_merge` (default 500) steps, plus an on-promotion check:

- **FR-8.1 STM↔STM:** merge two STM concepts if centroid cosine similarity ≥ `merge_sim` (default 0.80) **and** cross-ref-set mean kNN distance ≤ 1.1× the mean of their within-ref-set kNN distances. Survivor = larger match_count; union ref_sets (re-reservoir to K_max); recompute centroid/τ/κ; record lineage.
- **FR-8.2 STM↔LTM:** if an STM candidate's centroid is accepted by an LTM concept's τ (i.e., it is not separated), fold the STM candidate's ref_set into the LTM concept's reservoir and delete the candidate. LTM centroid stays frozen.
- **FR-8.3 LTM↔LTM (promoted only):** check newly promoted concepts against previously promoted concepts with the FR-8.1 rule; never merge two `provenance="initial"` concepts. Merged promoted concepts share a routing identity thereafter. This addresses the residual fragmentation risk directly.

### 5.9 FR-9: Wake loop (single pass)

Per stream example, exactly one pass, O(|concepts|·k_ref) cost:

```
z = normalize(embedding[t])
accepts = {c in LTM ∪ matureSTM : score(z,c) ≤ τ_c}
if accepts:
    c* = argmax normalized margin; assign; update (match_count,
         last_matched_at, match_windows, reservoir; EMA centroid if STM)
    prediction[t] = c*.concept_id
else:
    accepts_imm = {c in immatureSTM : score(z,c) ≤ τ_c}
    if accepts_imm: assign to best (as above); prediction[t] = "unknown"
    else: seed new STM concept (FR-3.2); prediction[t] = "unknown"
run periodic hooks (residual clustering, merge sweep, promotion check,
eval checkpoint) on schedule
```

- **FR-9.1** Emission of `"unknown"` is a legitimate prediction (this is an open-world system); metrics must handle it (§7.3).
- **FR-9.2** Determinism: given a config + seed, byte-identical logs across runs.

---

## 6. Non-Functional Requirements

- **NFR-1** Pure CPU; full stream run (≈13k–60k examples) completes in < 10 min on a laptop-class machine excluding UMAP calls; < 30 min including them.
- **NFR-2** Config-driven (single YAML per experiment); every table in the eventual paper regenerable by `python run.py --config X.yaml --seed S`.
- **NFR-3** Structured JSONL event log (assignments, seeds, promotions, merges, evictions, checkpoint metrics) sufficient to reconstruct any figure without re-running.
- **NFR-4** Unit tests for: reservoir correctness, threshold shrinkage math, κ estimation, promotion criteria gating (each criterion independently blocking), merge lineage integrity, LRU eviction, determinism.

---

## 7. Evaluation Plan

### 7.1 Stream protocols

- **P1 — Compatibility stream (v1-comparable):** exact v1 construction. T0 = all 100 CIFAR-100 classes (LTM init from the 50k train split). Stream = 1,000-example IND warmup, then shuffled interleave of remaining IND test (9,000), synthetic IND (250), near-OOD (500), far-OOD (2,576). Purpose: direct head-to-head vs v1 numbers on identical data.
- **P2 — O-UCL phased stream (primary):** T0 = 80 CIFAR-100 classes (LTM init from their train split). Stream phases of equal length, each introducing new classes that then **stop appearing** after their phase (PCMC incremental protocol): phases introduce, in order, the 20 held-out CIFAR-100 classes (real images, 4 phases × 5 classes, drawn from their train+test splits), then near-OOD classes (2 phases × 3), then far-OOD classes (grouped by novel superclass, ~5 phases). Each phase interleaves 30% "past-class" IND test examples to measure retention. Four evaluation checkpoints per phase (PCMC protocol). Purpose: this is the stream on which the paper's claims rest — it tests novelty detection, discovery, retention, and forgetting jointly.
- Both protocols: seeds {42, 43, 44}; report mean ± std.

### 7.2 Ground-truth mapping for evaluation

Predictions are concept_ids. For scoring, map each concept to its majority ground-truth label over assigned examples **within the evaluation harness only** (identical to v1/PCMC evaluation practice). `"unknown"` predictions are correct iff the example's class has not yet been introduced OR has been introduced but not yet promoted — report both the strict variant (unknown = wrong once class introduced) and the lenient variant.

### 7.3 Metrics (per checkpoint and end-of-stream)

1. **Detection:** streaming AUROC / FPR@95 of the implicit novelty decision (accepted-by-any-concept vs not), stratified near/far OOD and by phase.
2. **Expanding classification accuracy** (PCMC protocol): accuracy over all classes seen so far, at each checkpoint; decomposed into initial-class accuracy (forgetting curve) and promoted-class accuracy.
3. **Discovery quality:** promoted-cluster purity at promotion and at end-of-stream (the v1 drift metric — target: end-of-stream median ≥ 0.85); discovered-vs-true class count; **fragmentation index** = promoted concepts per unique ground-truth novel class after LTM↔LTM merging (target ≤ 1.3); coverage = fraction of novel classes with ≥1 promoted concept.
4. **Memory dynamics:** STM occupancy over time, eviction counts, evicted-concept ground-truth composition (are evictions actually outliers?), residual "unknown" rate over time (target: v1's 1,962-example terminal buffer effectively eliminated — residual unknowns for promoted classes < 5%).
5. **Threshold health:** per-concept FPR/FNR estimated post-hoc from ground truth; distribution of τ_c across concepts.

### 7.4 Baselines and ablations (all on P1 and P2)

| Run | Description |
|---|---|
| B1 | v1 streaming pipeline (global static Mahalanobis threshold, stateless HDBSCAN buffer) — existing code, unmodified. |
| B2 | Batch knn_vmf pipeline (existing) applied offline at each checkpoint — "no memory management" reference. |
| B3 | Oracle ceiling: ground-truth-labeled routing (existing oracle harness). |
| A1 | F-PCMC with global threshold instead of per-concept τ (isolates FR-5). |
| A2 | F-PCMC without STM (direct promotion at θ matches; isolates outlier filtering). |
| A3 | F-PCMC without recurrence criterion (isolates FR-7 criterion 4). |
| A4 | F-PCMC without merging (isolates FR-8; measures fragmentation contribution). |
| A5 | F-PCMC with knn_ref-only and vmf-only scorers (isolates the composed scorer). |
| A6 | Full F-PCMC on frozen ResNet-50 embeddings (encoder ablation, existing embeddings). |

### 7.5 Success criteria (definition of done for the research milestone)

Relative to B1 on P1: end-of-stream promoted purity ≥ 0.85 median (v1: 0.61); fragmentation index ≤ 1.3 (v1: 14 concepts / 9 classes ≈ 1.6 pre-merge); residual unknown pool < 25% of v1's; overall accuracy ≥ v1 + 8 points. On P2: initial-class accuracy degrades < 5 points from first to last checkpoint (forgetting bound); ≥ 60% novel-class coverage. If criteria are unmet, ablation table must localize which mechanism underperforms — that is itself a reportable result.

---

## 8. Configuration and Default Hyperparameters

```yaml
encoder: dinov3_vitl16          # or resnet50 (A6)
scorer: knn_vmf                  # knn_ref | vmf | knn_vmf
k_ref: 5
n_vmf_min: 10
tau_percentile_q: 95
n_shrink: 10
alpha_stm_ema: 0.10
K_max_refset: 64
stm_capacity: 100                # sweep {50, 100, 200}
n_mature: 5
theta_promote: 30                # sweep {20, 30, 50}
min_cohesion: 0.55               # sweep {0.45, 0.55, 0.65}
sep_factor: 1.0
m_windows: 3
window_W: 250
T_cluster: 500
w_residual: 500
T_merge: 500
merge_sim: 0.80
umap: {dim: 50, n_neighbors: 15, min_dist: 0.0, metric: cosine}
hdbscan: {min_cluster_sizes: [10, 15, 20, 25, 30], selection: eom}
seed: 42
```

Sweeps limited to the three marked parameters, on P1 only, seed 42 only; final numbers use defaults or the single best sweep configuration, fixed before P2 runs.

## 9. Repository Layout and Milestones

```
fpcmc/
  concepts.py        # Concept, ConceptStore (routing, thresholds, reservoirs)
  scorers.py         # knn_ref, vmf, knn_vmf (per-concept interface)
  memory.py          # STM/LTM management, LRU, promotion, merging
  residual.py        # residual pool + UMAP/HDBSCAN consolidation (wraps existing module)
  stream.py          # wake loop, hooks, event log
  protocols.py       # P1, P2 stream builders
  eval/              # metrics, checkpointing, gt-mapping, figures
  baselines/         # v1_stream.py (moved, unmodified), batch_knn_vmf.py (wrapper)
  configs/           # one YAML per run in §7.4
  tests/
```

*As built at T0 (owner-confirmed 2026-07-10): `eval/`, `baselines/`, `configs/`, `tests/` are top-level siblings of `fpcmc/` — the nesting above was a formatting artifact — joined by top-level `lib/` (frozen vendored source-project modules), `reference/` (read-only pinned submodules), `docs/`, and `data/` (contract README only; embeddings resolve externally via `roots.env`, never a local `data/embeddings/`).*

- **M1 (days 1–3):** `concepts.py`, `scorers.py`, thresholds; unit tests green. LTM-only routing reproduces batch knn_vmf detection AUROC on P1 within ±0.01 (sanity gate).
- **M2 (days 4–6):** STM dynamics, promotion, wake loop; end-to-end P1 run with logging.
- **M3 (days 7–9):** merging, residual clustering, per-concept τ recomputation; ablation flags.
- **M4 (days 10–12):** P2 protocol, full metrics/figures, baselines wired, 3-seed runs.
- **M5 (days 13–14):** sweeps, ablation suite, results tables. ← "minimal version in two weeks" deliverable: M1–M4.

## 10. Stretch Goals (do not start before M5 complete)

- **S1 Patch-token variant:** route DINOv3 patch tokens (16×16 grid) through the same ConceptStore for a compositional, PCMC-faithful variant; whole-image decision by voting over patch-level concept matches (PCMC Eq. 6–7 analog).
- **S2 ImageNet-40 / Places365-40 streams:** replicate PCMC's original benchmarks for a direct comparison against published PCMC numbers (frozen encoder, no sleep, vs PCMC with sleep).
- **S3 Threshold drift adaptation:** slow online recalibration of τ_c from streaming accepted-score quantiles (P² estimator), reported as a separate ablation.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Online seeding under-segments adjacent novel classes (near-OOD pairs) | FR-6 residual consolidation + A5/A1 ablations localize it; HDBSCAN assist retained precisely for this. |
| Per-concept τ from small ref_sets is noisy → STM churn | Shrinkage (FR-5.2) + maturity gate (FR-3.3); monitor threshold-health metric. |
| Recurrence criterion too strict for phase-limited classes (class disappears before 3 windows) | m_windows/W chosen so a phase spans ≥ 6 windows; verify per-protocol in M4; window params may be re-derived from phase length. |
| Merging collapses genuinely distinct near-OOD classes | Two-condition merge rule (FR-8.1) requires both centroid similarity and cross/within distance ratio; A4 quantifies. |
| Success criteria unmet | Ablation matrix (§7.4) is designed to make failure diagnostic, not terminal — negative results per mechanism are reportable. |