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

## `baselines/v1/` — the T14 vendored v1 streaming harness (verbatim, read-only)

Vendored 2026-07-13 for T14 (owner-approved location for this record: this
file, per the TASKS T14 `test_v1_untouched` literal). Same policy and same
source state as the `lib/` set above: byte-identical copies of
`msproject_misc/evaluation/continual/` at the working-tree state behind
`tests/reference_numbers.yaml` (HEAD `e723f028`; the files carrying
uncommitted edits at pin time were re-verified byte-identical on 2026-07-13 at
source HEAD `cd69046a` — the edits were since committed unchanged). This set
IS importable/runnable as a unit (flat top-level imports resolve when
`baselines/v1/` is the script directory); `baselines/v1_stream.py` is the only
shim and the only non-verbatim file. **Do not edit anything under
`baselines/v1/`** — `test_v1_untouched` asserts every hash below.

| File (path under `baselines/v1/` == path under `evaluation/continual/`) | Blob hash |
|---|---|
| `main.py` | `e9b17c9eb6080dd4410fa198d68ee3354d9754aa` |
| `config.py` | `b0f9acb9f96ab97b39105bbab291e4a78c32ca6c` |
| `config.yaml` | `33154bf437d5278f68d8423b697a33c9d19117e1` |
| `data_loader.py` | `0860808ea1e84208c141fb5d1ec3d81811fb63c1` |
| `stream.py` | `b3051c55ca5c21fc894f8632423a4d9c81884251` |
| `evaluation.py` | `2d0d7e5e279483d5d8cd6892b89169c3339a8031` |
| `ind_model.py` | `54315fbb27a80673adf8b5477026e3f0f1ba291a` |
| `clustering.py` | `55a9ede41837bc3e7172f91f9d1c5d794073d531` |
| `paradigms/__init__.py` | `e5897579fce65dd34a1a48abe4d37bccc4c2c264` |
| `paradigms/base.py` | `7501a5169d555fb3eef054b8d63c9f2c369efb06` |
| `paradigms/factory.py` | `da9118af6302ac0758cb4fb042113e1b5a963d8b` |
| `paradigms/mahalanobis_hdbscan.py` | `1694e84b711689f74564e06f1974e906d7de3006` |
| `paradigms/knn_dpmeans.py` | `5e9da6f94c5d7bfc1a4986f65d62bb610ba2190b` |
| `paradigms/knn_vmf.py` | `7ef929c11003404df4557ce8af108c339255d044` |
| `paradigms/vmf_dpmm.py` | `47d7412fdd105d00faf2fe63a8465bfd81cfbc80` |

Every hash matches its pin in `tests/reference_numbers.yaml`
(`t14_v1_regression_pin.git_blob_hashes` / `t6_m1_gate.git_blob_hashes`) where
one exists; `paradigms/__init__.py` (docstring-only) was not previously pinned
and is recorded here for the first time. Known machine-specific wart, kept
because the port is untouched: `main.py:44` hardcodes
`LOG_DIR = /home/alex/projects/msproject_misc/logs` (loguru file sink) and
will create that directory when run.
