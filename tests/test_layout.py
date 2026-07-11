"""T0 repo-layout guard (TASKS.md Task 0): required directories/files exist.

Layout = PRD §9 with eval/, baselines/, configs/, tests/, lib/ as top-level
siblings of fpcmc/ (confirmed reading, 2026-07-10), and the documented
data-access deviation: there is NO data/embeddings/ directory — embeddings
resolve externally via roots.env (docs/ASSETS.md §1, data/README.md).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    "fpcmc",
    "eval",
    "baselines",
    "configs",
    "tests",
    "lib",
    "reference",
    "docs",
    "data",
]

REQUIRED_FILES = [
    "pyproject.toml",
    "pytest.ini",
    "roots.env.example",
    "fpcmc/__init__.py",
    "fpcmc/rng.py",
    "fpcmc/config.py",
    "configs/default.yaml",
    "configs/p2_class_split.yaml",
    "tests/reference_numbers.yaml",
    "lib/PROVENANCE.md",
    "data/README.md",
    # T1 — data layer + synthetic fixtures
    "fpcmc/data.py",
    "tests/fixtures/vmf_world.py",
    "tests/fixtures/golden_stream.py",
    "tests/fixtures/golden_stream.npz",
    "docs/PRD_frozen_encoder_pcmc.md",
    "docs/TASKS_frozen_encoder_pcmc.md",
    "docs/ASSETS.md",
    # T2 — per-concept scorers + Concept stub
    "fpcmc/scorers.py",
    "fpcmc/concepts.py",
    # T4 — per-concept adaptive thresholds
    "fpcmc/thresholds.py",
    # T6 — LTM initialization (M1 gate)
    "fpcmc/init.py",
]


def test_repo_layout():
    missing = [d for d in REQUIRED_DIRS if not (REPO_ROOT / d).is_dir()]
    missing += [f for f in REQUIRED_FILES if not (REPO_ROOT / f).is_file()]
    assert not missing, f"missing required paths: {missing}"

    # reference/ policy (TASKS T0): either populated pinned submodules for
    # STAM + PCMC, or a citation README. Check the policy artifact, not
    # submodule population, so a clone without --recurse-submodules still passes.
    gitmodules = REPO_ROOT / ".gitmodules"
    has_submodules = (
        gitmodules.is_file()
        and "reference/stam" in gitmodules.read_text()
        and "reference/pcmc" in gitmodules.read_text()
    )
    has_readme = (REPO_ROOT / "reference" / "README.md").is_file()
    assert has_submodules or has_readme, (
        "reference/ must be pinned submodules (.gitmodules) or a citation README"
    )

    # Decided data-access deviation: no local embeddings directory, ever.
    assert not (REPO_ROOT / "data" / "embeddings").exists(), (
        "data/embeddings/ must not exist — embeddings resolve via roots.env "
        "(see data/README.md and docs/ASSETS.md §1)"
    )
