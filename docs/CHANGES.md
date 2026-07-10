# CHANGES

- A cumulative history of changes to the repo.

## 2026-07-10 — T0: Repository scaffold, config system, reference-code policy (`task/T00-scaffold`, commits `39ce798`..`054ecad`)

- **Tooling**: `pyproject.toml` with runtime deps pinned to the source project's exact versions per ASSETS §4 (`numpy==2.4.3`, `scipy==1.17.1`, `scikit-learn==1.8.0`, `umap-learn==0.5.11`) plus `pyyaml==6.0.2`; dev deps `pytest==8.4.1`, `pytest-mock==3.14.1`. `pytest.ini` registers the `slow` marker (`testpaths = tests`). `.python-version` pins 3.14.3 — verified identical to the source venv that produced `tests/reference_numbers.yaml`. `uv.lock` committed.
- **Layout**: `fpcmc/`, `eval/`, `baselines/` created as packages; `eval/`/`baselines/` are placeholders for T13/T14. Top-level-sibling reading of PRD §9 confirmed by owner (PRD's nested indentation treated as a formatting artifact).
- **Config system**: `fpcmc/config.py` — frozen `FPCMCConfig` dataclass with all PRD §8 keys and defaults (nested frozen `UmapConfig`/`HdbscanConfig`); unknown keys rejected with the offending key named (any nesting level); type-checked values; `to_yaml()`/`from_yaml()` lossless round-trip. `configs/default.yaml` transcribes PRD §8 verbatim.
- **RNG**: `fpcmc/rng.py` — single `SeedSequence`-based `make_rng(seed, stream="")` Generator factory with named substreams; the sole sanctioned randomness source repo-wide.
- **lib/ vendoring**: 8 modules copied byte-identical from `msproject_misc` (HEAD `e723f028`; `continual/{evaluation,stream}.py` capture the uncommitted-edit working-tree state the reference pins were reproduced against). Every blob hash verified against ASSETS §2. Inventory + read-only policy: `lib/PROVENANCE.md`.
- **Tests (7 passing)**: `test_config_roundtrip`, `test_config_rejects_unknown_key`, `test_config_is_frozen`, `test_rng_determinism`, `test_repo_layout` (asserts the roots.env data-access deviation: no `data/embeddings/`), `test_no_reference_imports` (vendoring guard), `test_no_learning_in_fpcmc` (frozen-encoder invariant #6, seeded at T0).
- **Approved deviations**: standalone `hdbscan` PyPI package omitted from the TASKS T0 dep list (ported code uses `sklearn.cluster.HDBSCAN`, ASSETS §4); noted in `pyproject.toml`.