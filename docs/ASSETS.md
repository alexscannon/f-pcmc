# Existing Assets — Paths, Schemas, and Modules to Port

Fills in PRD §3 ("Exact paths to be filled in from the current repo"). Source repo: `msproject_misc/` (git remote `origin/main`, HEAD `e723f028d84846b4a244fb3a30c9456f817b16c6` as of 2026-07-10; five files under `evaluation/continual/` have uncommitted local edits — see §4).

Machine paths below assume `roots.env`: `PROJECT_ROOT=/home/alex/projects/msproject_misc`, `DATA_ROOT=/home/alex/data`, `EMBEDDINGS_DIR=${DATA_ROOT}/embeddings/DINOv3_large_32px`. **These are machine-local, gitignored paths — do not hardcode them into `fpcmc/`.**

**Data access mechanism (decided):** F-PCMC's own `roots.env` (gitignored, `roots.env.example` tracked as the template — both now at the repo root, mirroring `msproject_misc`'s own convention) resolves `EMBEDDINGS_DIR` at config-load time; `fpcmc/data.py` reads the four `.pt` files directly from that external path. No copy, no symlink, no local `data/embeddings/` directory. Rejected copy (duplicates ~260MB of data the PRD already treats as frozen/never-regenerated, for no staleness benefit) and symlink (bakes an absolute path into the filesystem at creation time — breaks silently on any other machine, CI runner, or sandboxed agent worktree) in favor of this, since it's portable per-environment and matches a pattern the source project has already validated across 8 subprojects. Full contract and schema: `data/README.md`. Note this changes T1's `test_real_pool_schemas` skip condition — it's not "`data/embeddings/` absent," but "`roots.env` missing/unset or the resolved `EMBEDDINGS_DIR` path's files not found"; see `data/README.md`'s "Deviation from the literal PRD §9 layout" note.

## 1. Embedding `.pt` files

All five pools live in **four files** at `${DATA_ROOT}/embeddings/DINOv3_large_32px/` (a `ResNet50_32px/` sibling exists for the A6 ablation, 2048-dim, same schema). Verified 2026-07-10 by loading every file and checking shape/dtype/NaN/Inf/label-alignment (see §5).

| Pool (PRD §3 name) | File | Count | D | dtype | Notes |
|---|---|---|---|---|---|
| IND Reference | `real_cifar100.pt` (`sources=="cifar100_train"`) | 50,000 | 1024 | float32 | 100 classes, 500/class |
| IND Test | `real_cifar100.pt` (`sources=="cifar100_test"`) | 10,000 | 1024 | float32 | same 100 classes, 100/class |
| Synthetic IND | `ind.pt` | 250 | 1024 | float32 | 10 of the 100 subclasses only (partial synthetic generation — **not all 100**, see §6) |
| Near-OOD | `novel_subclasses.pt` | 500 | 1024 | float32 | 6 classes × ~83, superclasses `{aquatic_mammals, fish, small_mammals}` only |
| Far-OOD | `novel_superclasses.pt` | 2,576 | 1024 | float32 | 43 classes across 16 (of 20 planned) novel superclasses |

`real_cifar100.pt` is one file containing both IND Reference and IND Test; split by the `sources` field, not by file. All counts match the PRD table exactly.

### `.pt` file schema (identical across all four files)

```python
{
  "embeddings":       torch.FloatTensor (N, D),   # NOT L2-normalized on disk (norms ~11-18, not 1.0)
  "subclass_names":   list[str] (len N),           # per-image fine label
  "superclass_names": list[str] (len N),           # per-image coarse label
  "sources":          list[str] (len N),           # provenance tag, see below
  "image_paths":      list[str] (len N),
  "label_mappings": {
    "subclass_to_id": dict[str,int], "id_to_subclass": dict[int,str],
    "superclass_to_id": dict[str,int], "id_to_superclass": dict[int,str],
  },
}
```

`sources` values observed: `real_cifar100.pt` → `cifar100_train` | `cifar100_test`; `ind.pt` → `genai_ind`; `novel_subclasses.pt` → `genai_novel_subclass`; `novel_superclasses.pt` → `genai_novel_superclass`.

**Class-name maps**: `real_cifar100.pt`'s `label_mappings` is the canonical 100-subclass / 20-superclass CIFAR-100 map (also mirrored at `msproject_misc/cifar100_super_subclass_mapping.json`, superclass→[5 subclasses]). The three synthetic files carry their own `label_mappings` scoped to only the classes present in that file (`ind.pt`: 10 subclasses/2 superclasses; `novel_subclasses.pt`: 6/3; `novel_superclasses.pt`: 43/16) — **do not assume a shared global id space across files**; join on class-name strings, not on `label_mappings` ids.

**Normalization**: embeddings are raw DINOv3 CLS-token vectors, not unit-norm. F-PCMC's `fpcmc/data.py` loader must apply on-load L2 normalization itself (per PRD/TASKS T1), matching how every source-project consumer normalizes independently (`normalize(x, norm="l2")` after any UMAP step, cosine-via-Euclidean trick).

## 2. Modules to port

| PRD §3 asset | Source file(s) | Commit / state | Port notes |
|---|---|---|---|
| Mahalanobis (per-class, shared), min-cosine, kNN-density scorers | `evaluation/h1/scorers.py`, `evaluation/h1/reference_stats.py` | `59ec2472` / `99420dfd` (both committed, HEAD) | Batch/vectorized over a fixed reference pool, not per-concept. FR-4 requires refactoring into `Scorer.score(z, concept)` — this is a redesign, not a copy. |
| **knn_vmf composed scorer** | `evaluation/continual/paradigms/knn_vmf.py` (kNN-gallery gate + vMF mixture) + `evaluation/continual/paradigms/vmf_dpmm.py` (vMF math: Banerjee κ estimator, A&S 9.7.7 log-Bessel asymptotic for D=1024 underflow) | `7ef929c1` / `47d7412f` (committed, HEAD) | This is a **streaming DP-mixture paradigm**, not a stateless scorer function — it owns gallery growth and online vMF component fitting. Extract the scoring math (κ estimation, log-vMF-density, kNN gate) as library functions; do not port the DP-mixture online-fitting control flow, which F-PCMC's `Concept`/`ConceptStore` replaces. |
| UMAP+HDBSCAN clustering module | `evaluation/continual/clustering.py` (streaming: OOD-buffer sweep, Jaccard cluster dedup across `min_cluster_sizes`, promotion-criteria check) — `evaluation/h2/clustering.py` is the batch/H2 twin (near-identical `reduce_umap`+`HDBSCAN` core, no dedup/promotion) | `55a9ede4` / `3aeeab75` (both committed, HEAD) | Port `evaluation/continual/clustering.py`'s UMAP+HDBSCAN core (`reduce_umap` + sweep over `min_cluster_size ∈ {10,15,20,25,30}` + Jaccard dedup) into `fpcmc/residual.py` per PRD §5.6. Both wrappers already set `UMAP(..., random_state=config.random_seed)` — determinism precondition already satisfied (§4 below). `HDBSCAN` is `sklearn.cluster.HDBSCAN` (not the standalone `hdbscan` package) in both. |
| v1 streaming harness | `evaluation/continual/main.py` + `paradigms/mahalanobis_hdbscan.py` + `ind_model.py` + `stream.py` + `evaluation.py` (paradigm dispatch via `paradigms/factory.py`) | `e9b17c9e` / `1694e84b` / `54315fbb` / `b3051c55` / `2d0d7e5e` — **`main.py`, `stream.py`, `evaluation.py` have uncommitted local edits (§4)**; `ind_model.py`, `paradigms/mahalanobis_hdbscan.py`, `paradigms/base.py`, `paradigms/factory.py` are clean at HEAD | PRD calls for "retain untouched as `baselines/v1_stream.py`" — note the source is **not a single-file v1 pipeline**; the "v1 pipeline" is the `mahalanobis_hdbscan` paradigm running inside the same multi-paradigm `main.py` used by `knn_vmf`/`knn_dpmeans`/`vmf_dpmm`. Porting "untouched" means vendoring `main.py`+`stream.py`+`ind_model.py`+`clustering.py`+`evaluation.py`+`paradigms/{base,factory,mahalanobis_hdbscan}.py` together and pinning `config.paradigm: mahalanobis_hdbscan`, not lifting one file. |
| Metrics utilities (AUROC/AUPR/FPR@95, ARI/NMI/purity/completeness, rolling-window trackers) | `evaluation/continual/evaluation.py` (`compute_detection_metrics`, `compute_classification_accuracy`, `compute_cluster_quality`, `compute_discovery_clustering_metrics`, `compute_oversegmentation_stats`) + `evaluation/h2/evaluation.py` (ARI/NMI/purity via `sklearn.metrics`) + `stream.py` (`RollingWindowMetrics`, `CumulativeMetrics`) | `2d0d7e5e` (dirty) / `evaluation/h2/evaluation.py` committed / `b3051c55` (dirty) | Straightforward port into `eval/`; these are already GT-isolated at the call boundary (ground truth passed in, not looked up). |
| Batch knn_vmf pipeline (B2 baseline) | **No dedicated batch script exists.** `evaluation/continual/paradigms/knn_vmf.py` run via `main.py --paradigm knn_vmf` end-to-end (still a *stream*, not a per-checkpoint batch scorer) is the closest existing artifact and is what T6's M1 gate number (§3 below) is pinned against. | `7ef929c1` (committed) | PRD's `baselines/batch_knn_vmf.py` ("wrapper invoking the existing batch pipeline at each P1/P2 checkpoint") has **no source to reuse for the "batch" framing** — it must be newly written, reusing `knn_vmf.py`'s scoring math (kNN gate + vMF mixture fit) but restructured to score a static IND-vs-OOD pool once per checkpoint instead of consuming a live stream. Do not port `knn_vmf.py`'s stream/gallery-growth loop verbatim for this baseline. |
| Oracle ceiling (B3) | **No standalone `oracle.py` harness file.** Ground-truth-oracle behavior is a `--oracle-mode {detection,clustering,both}` CLI flag baked into `main.py` + each paradigm's `step_oracle()` method (`paradigms/base.py` defines the `Paradigm` protocol including `step_oracle`). | mixed (`base.py` clean `7501a516`; `main.py` dirty) | "Existing oracle harness" in PRD §7.4/T14 is this flag-driven mechanism, not a separate script. Port `--oracle-mode` handling from `main.py` + each paradigm's `step_oracle`, or reimplement the same ground-truth-substitution contract directly against F-PCMC's `ConceptStore`. |

## 3. Reference numbers (for `tests/reference_numbers.yaml`)

See `tests/reference_numbers.yaml` for the pinned values and full provenance (fresh seed-42 reproduction run, this session, 2026-07-10). Headline pointers:
- T6 M1 gate: batch/streaming knn_vmf detection AUROC (all-OOD / near-OOD / far-OOD).
- T14 v1 regression pin: mahalanobis_hdbscan detection AUROC, overall classification accuracy, promoted-cluster count, end-of-stream median purity, residual buffer size.

## 4. Determinism preconditions (relevant to NFR-3 / T0 rng.py / T11 byte-determinism)

- Both existing UMAP wrappers (`evaluation/continual/clustering.py:67-73`, `evaluation/h2/clustering.py:71-77`) already pass `random_state=` (seeded from `config.random_seed`) — the byte-determinism precondition F-PCMC's `test_byte_determinism` needs is **already satisfied** by the code being ported; no fix required, just preserve the pattern (`fpcmc/residual.py`'s UMAP call must also pass `random_state`).
- Setting `random_state` forces UMAP into single-threaded mode (`numba` disables its parallel path when a seed is fixed) — this is a **known runtime cost**, not a bug. Budget NFR-1 accordingly: the source project's full-stream run (13,326 examples, periodic UMAP+HDBSCAN sweeps) took materially longer per-step during clustering sweeps than the ~1,700 ex/s pure-Mahalanobis rate (README claims ~86 ex/s blended; this session's fresh seed-42 run measured highly variable 3–75 ex/s instantaneous throughput depending on sweep activity, ~10-12 min wall-clock total on this machine for `mahalanobis_hdbscan`, full 13,326-example stream). `test_init_runtime`'s 60s budget (T6) and `test_runtime_budget` (T11) should be checked against this, not against the unseeded/multi-threaded UMAP rate.
- `sklearn.cluster.HDBSCAN` (not the standalone `hdbscan` PyPI package) is used in both existing wrappers — it takes no `random_state` (EOM cluster selection is itself deterministic given its inputs). Confirm F-PCMC imports the same `sklearn.cluster.HDBSCAN`, not the third-party `hdbscan` package, to avoid subtly different cluster-selection behavior.
- Pinned library versions at HEAD (`evaluation/continual/uv.lock`, `evaluation/h2/uv.lock`, both resolved identically): `numpy==2.4.3`, `scikit-learn==1.8.0`, `scipy==1.17.1`, `umap-learn==0.5.11`. Pin F-PCMC's `pyproject.toml` to these exact versions (not just the `>=` floors in the source `pyproject.toml`s) — the M1 gate is comparing AUROC to ±0.01, and UMAP/HDBSCAN behavior has drifted across minor versions historically.

## 5. `.pt` schema verification (T1's `test_real_pool_schemas`, run ad hoc this session)

Ran a verification pass (`torch.load` each file, check shape/dtype/NaN/Inf/label-length-alignment/class-count) against the live files at `${EMBEDDINGS_DIR}`:

| File | Shape | dtype | NaN | Inf | Label arrays aligned | Unique subclasses | Unique superclasses |
|---|---|---|---|---|---|---|---|
| `real_cifar100.pt` | (60000, 1024) | float32 | none | none | yes | 100 | 20 |
| `ind.pt` | (250, 1024) | float32 | none | none | yes | 10 | 2 |
| `novel_subclasses.pt` | (500, 1024) | float32 | none | none | yes | 6 | 3 |
| `novel_superclasses.pt` | (2576, 1024) | float32 | none | none | yes | 43 | 16 |

All counts match PRD §3 exactly (50,000 / 10,000 / 250 / 500 / 2,576 after splitting `real_cifar100.pt` by `sources`). **No mismatch found** — T1's schema gate should pass unmodified against this data.

## 6. Known gaps / risks for the coding agent

- `ind.pt` covers only 10 of the 100 CIFAR-100 subclasses (partial synthetic generation, per `msproject_misc/MEMORY.md`), all under `fish`/`small_mammals` superclasses. P1's "Synthetic IND (250)" pool is real but not representative of all 100 classes — do not assume synthetic-IND coverage when building P2 phase schedules that reference synthetic IND by class.
- `novel_superclasses.pt` covers 16 of the 9-novel-superclass-times-4(ish) design in the original README table (36 planned classes across 9 superclasses) vs. the 43-class/16-superclass actual — generation was partial/superset relative to the original plan (the README's "9 novel superclasses" figure is stale; treat the `.pt` file's actual `label_mappings` as ground truth over the README prose).
- `evaluation/continual/{config.py,config.yaml,evaluation.py,main.py,stream.py}` have **uncommitted local changes** on top of HEAD `e723f028` (adding an opt-in `stream_order` ablation; default value `random` preserves prior behavior, confirmed by diff). The reproduction run backing `tests/reference_numbers.yaml` was executed against this working-tree state, not bare HEAD — the git blob hashes recorded in `tests/reference_numbers.yaml` are `git hash-object` content hashes (valid even though uncommitted), not commit hashes, for exactly this reason.

## 7. Byte-identity pins and the v1 P1 stream (added 2026-07-13, owner-directed)

### 7.1 The four `.pt` files are now hash-pinned

Everything pinned in this project is silently conditional on these exact bytes: `tests/reference_numbers.yaml` came from a seed-42 run against them, as did the T6 M1 gate and the T14 v1 regression pin. `test_real_pool_schemas` now asserts these sha256s before it checks anything derived from them.

| file | sha256 | rows |
|---|---|---|
| `real_cifar100.pt` | `dd78fe2321995a8880b7e60acae5c5725942c17a1d521d9db00b473aca0f9fde` | 50,000 train + 10,000 test |
| `ind.pt` | `6ea73944da4ad8559210edeb9cbdad4db99562f945b33cf350041d473029a3ae` | 250 |
| `novel_subclasses.pt` | `ef545ed7ef53c4e590e2c6cc16d74c043fe0bd921b4fc37809dfe4863c284ee1` | 500 |
| `novel_superclasses.pt` | `6ea4bbf1a297601dd4ab4e495e979e1169e4396b70faf9742f7e86e2b0cedaa7` | 2,576 |

**Why this matters beyond schema drift.** The v1 P1 stream's pool counts (250 / 500 / 2,576) are **derived from these files' row counts, not asserted anywhere in the source**, and the source project's own design documents a fuller extraction (5,000 synthetic IND / 600 near / 3,600 far — see §6, and the source's `CLAUDE.md`). These files are a **partial extraction**. That is self-consistent and fine — the entire project, including every pinned number, is built on them — but a completed re-extraction would change the pools, the stream ordering, and every pinned metric **silently**. If a hash assertion ever fails: **do not re-pin it.** Re-derive `tests/reference_numbers.yaml` from the new files first. This is the CLAUDE.md stop condition "a real-data schema check fails".

### 7.2 The v1 P1 stream ordering is exactly reproducible, and archived

Investigated in the source project at pinned commit `e723f028` (read-only). The ordering **is** deterministic at seed 42 — `build_stream` (`stream.py:35-69`, blob `55af9878bd6ffb57fd6de9dfcd3ca9d3b80c24d9`; pools assembled by `load_all_data`, `data_loader.py:56-152`, blob `0860808ea1e84208c141fb5d1ec3d81811fb63c1`; sole call site `main.py:353`) constructs a fresh `np.random.default_rng(seed)` (PCG64, not legacy `RandomState`, not Python `random`, not torch) from `config.random_seed`, threaded as a parameter rather than read from a global, seeded once and called once. No unseeded shuffle, no `glob`/`listdir`, no set/dict iteration feeding the ordering. Confirmed empirically: identical content hash across three fresh processes at `PYTHONHASHSEED` 0, 1 and 12345.

Algorithm, in order: pools appended in fixed order (IND_REAL rows where `source == "cifar100_test"`, ascending; then IND_SYNTHETIC; NEAR_OOD; FAR_OOD) → `rng.permutation(10000)` over the 10,000-row `cifar100_test` pool, first 1,000 = warmup, remaining 9,000 = leftover → the 9,000 concatenated with the other three pools (12,326 total) → **one** `rng.permutation(12326)` over that concatenation → `warmup + shuffled_remainder` = 13,326. So the warmup is a 1,000/9,000 split of a single pool and is **disjoint** from the interleave; the four pools are concatenated then shuffled once, **not** shuffled per-pool and merged. Only `ind_warmup_count: 1000` is a config constant; the rest are derived (see §7.1).

`StreamItem` carries `embedding`, `true_class` (str), `true_superclass` (str), `novelty_type` — **no integer label id and no within-pool index**. Join on `true_class` strings; the per-file integer label ids never enter the stream.

**Archived (durable), because index-level equality is assertable and T12 will need it:**

```
/home/alex/data/evaluation/v1_p1_stream/stream_seed42_e723f028.npz
sha256 77783853e826fbe52a2c4864ef49d3c10132311ad96a57b4a1c8e1d49834b73f
```

Length 13,326. Keys: `pool`, `within_pool_index`, `true_class`, `true_superclass`, `phase`, `seed`, `ind_warmup_count`, `commit`. Verified on archive: counts `ind_real` 10,000 / `far_ood` 2,576 / `near_ood` 500 / `ind_synthetic` 250; first 1,000 steps all `ind_real`; `t=0` → `ind_real[8132]` "fox", `t=1` → `ind_real[8268]` "telephone", `t=2` → `ind_real[719]` "pear". For `ind_real`, `within_pool_index` indexes the 10,000-row `cifar100_test` subset in ascending row order.

**T12 consequence:** `test_p1_matches_v1` takes the **index-level equality** branch of its either/or, not the distributional fallback. Gate it on §7.1's hashes — an index-level test fails loudly if the embeddings are ever regenerated, whereas a distributional test would keep passing against a stream that had silently changed. That is an argument *for* the strict assertion.
