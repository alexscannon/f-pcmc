# lib/ — Vendored Modules from the Source Project (verbatim, read-only)

Verbatim copies of the source-project modules that PRD §3 designates for reuse
(scorers, knn_vmf composed-scorer math, UMAP+HDBSCAN wrapper, metrics
utilities). Copied 2026-07-10 for T0. See `docs/ASSETS.md` §2 for the full
module-to-port inventory and per-module porting notes.

**Policy:** these files are a frozen reference snapshot, not a live library.
They are copied byte-for-byte from the source working tree and keep their
source-repo-relative paths under `lib/` (hence two files named
`evaluation.py`). They import source-project-internal modules and are **not
expected to be importable as-is** from this repo; consuming tasks (T2 scorers,
T10 clustering, T13 metrics) extract/adapt the math into `fpcmc/` and `eval/`
with citation comments pointing back here. Do not edit these files — if an
adaptation is needed, it happens at the extraction site, never in `lib/`.

The v1 streaming-harness file set (`main.py`, `ind_model.py`,
`paradigms/{base,factory,mahalanobis_hdbscan}.py`, configs) is deliberately
**not** vendored here; T14 vendors it into `baselines/` as its own unit (see
ASSETS §2, "v1 streaming harness" row).

## Source

- Repo: `msproject_misc` (local: `/home/alex/projects/msproject_misc`, remote `origin/main`)
- HEAD at copy time: `e723f028d84846b4a244fb3a30c9456f817b16c6` (2026-07-05)
- Three files (`continual/evaluation.py`, `continual/stream.py` — and, outside
  this set, `main.py`/`config.py`/`config.yaml`) carried **uncommitted local
  edits** at copy time; the copies below capture that exact working-tree state,
  which is the state the pinned reference numbers in
  `tests/reference_numbers.yaml` were reproduced against (see that file's
  header). Hashes are `git hash-object` content (blob) hashes, valid for
  uncommitted content.

## Inventory

| File (path under `lib/` == path in source repo) | Blob hash | Source state | Role (ASSETS §2) |
|---|---|---|---|
| `evaluation/h1/scorers.py` | `59ec2472df2a7fa874b00318bdc1d46aab175b8a` | committed @ HEAD | Mahalanobis / min-cosine / kNN-density scorers (baselines; FR-4 redesign source) |
| `evaluation/h1/reference_stats.py` | `99420dfdfae42e4635021074c869b7ec6394c3c9` | committed @ HEAD | Reference-pool statistics backing the h1 scorers |
| `evaluation/continual/paradigms/knn_vmf.py` | `7ef929c11003404df4557ce8af108c339255d044` | committed @ HEAD | knn_vmf composed scorer (kNN gate + vMF mixture) — extract scoring math only |
| `evaluation/continual/paradigms/vmf_dpmm.py` | `47d7412fdd105d00faf2fe63a8465bfd81cfbc80` | committed @ HEAD | vMF math: Banerjee κ estimator, A&S 9.7.7 log-Bessel asymptotic |
| `evaluation/continual/clustering.py` | `55a9ede41837bc3e7172f91f9d1c5d794073d531` | committed @ HEAD | UMAP+HDBSCAN wrapper (`reduce_umap`, min_cluster_size sweep, Jaccard dedup) — T10 source |
| `evaluation/continual/evaluation.py` | `2d0d7e5e279483d5d8cd6892b89169c3339a8031` | **uncommitted edit** | Detection/classification/cluster-quality metrics — T13 source |
| `evaluation/continual/stream.py` | `b3051c55ca5c21fc894f8632423a4d9c81884251` | **uncommitted edit** | `RollingWindowMetrics`, `CumulativeMetrics` trackers — T13 source |
| `evaluation/h2/evaluation.py` | `f48a8b700af4e6124c5cd9caf2fd010d8eea96d4` | committed @ HEAD | ARI/NMI/purity via sklearn.metrics — T13 source |

Verification: from `lib/`, `git hash-object <file>` must reproduce the hash
column exactly (T14's `test_v1_untouched` applies the same technique to the
`baselines/` set).
