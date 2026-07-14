"""T16: full-matrix execution guards (TASKS T16; PRD §7.5, NFR-1..3).

All three tests are [I] integration tests against the ARCHIVED matrix under
``${DATA_ROOT}/evaluation/f_pcmc_runs`` (produced by ``run_matrix.py``); they
skip with a clear message when roots.env or the archive is absent. Per TASKS,
``test_scorecard_generated`` must NOT fail when a research criterion fails —
the deliverable is the diagnostic scorecard plus a populated attribution
table for every failed criterion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_matrix
from eval import workbook

pytestmark = pytest.mark.slow


def _archive_root() -> Path:
    """The default out_root, or skip if this machine has no data/archive."""
    from fpcmc.data import EmbeddingsUnavailable, embeddings_available

    ok, reason = embeddings_available()
    if not ok:
        pytest.skip(reason)
    try:
        root = run_matrix.default_out_root()
    except EmbeddingsUnavailable as e:  # no DATA_ROOT in roots.env
        pytest.skip(str(e))
    if not root.is_dir():
        pytest.skip(f"matrix archive not found at {root} — run run_matrix.py first")
    return root


def _archived_summary(root: Path, system: str, protocol: str, seed: int) -> dict:
    path = root / system / f"{protocol}_seed{seed}" / "summary.json"
    if not path.is_file():
        pytest.skip(f"cell {system} x {protocol} x seed{seed} not archived at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


#: One cell per runner family (owner-approved repro scope, 2026-07-14):
#: the F-PCMC family on P1 plus an ablation on P2 (exercising the P2
#: window_W override in resolve_cell_config), B2, and B3. B1 is excluded —
#: its byte-determinism is already pinned by test_v1_regression_pin.
REPRO_CELLS = [
    ("fpcmc_default", "p1", 42),
    ("a1_global_tau", "p2", 43),
    ("b2_batch", "p1", 42),
    ("b3_oracle", "p2", 44),
]


@pytest.mark.parametrize("system,protocol,seed", REPRO_CELLS)
def test_matrix_reproducibility(tmp_path, system, protocol, seed):
    """TASKS T16: re-running a cell from its archived config reproduces its
    archived headline metrics exactly — byte-determinism carried through."""
    root = _archive_root()
    archived = _archived_summary(root, system, protocol, seed)
    cell_dir = root / system / f"{protocol}_seed{seed}"

    # "From its archived config": the archived resolved config must be
    # exactly what this code + the committed configs resolve today, so the
    # re-run below really is a replay of the archived cell.
    archived_config = (cell_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    assert archived_config == run_matrix.resolve_cell_config(
        system, protocol, seed
    ).to_yaml(), "archived resolved_config.yaml drifted from the committed configs"

    if system == "a6_resnet50":  # not in REPRO_CELLS today, but keep it honest
        pytest.skip("A6 repro requires the ResNet-50 pools")

    rerun_dir = run_matrix.run_cell(system, protocol, seed, out_root=tmp_path)
    rerun = json.loads((rerun_dir / "summary.json").read_text(encoding="utf-8"))

    # Headline metrics exactly equal (wall time is the one legitimately
    # non-deterministic field; it lives outside the metrics by design).
    a, b = dict(archived), dict(rerun)
    a.pop("wall_time_seconds", None)
    b.pop("wall_time_seconds", None)
    assert a == b, f"{system} x {protocol} x seed{seed}: re-run summary differs"

    # Byte-determinism carried through: identical event logs (FR-9.2).
    archived_log = cell_dir / "events.jsonl"
    if archived_log.is_file():
        assert (rerun_dir / "events.jsonl").read_bytes() == archived_log.read_bytes()


def test_runtime_budgets():
    """TASKS T16 / NFR-1: every supported cell is archived and within the
    30-min inclusive budget; the wall-time table itself lives in the
    workbook (report/workbook.md)."""
    root = _archive_root()
    cells = workbook.load_matrix(root)

    supported = [
        (c["system"], c["protocol"], c["seed"])
        for c in run_matrix.plan_matrix()
        if c["supported"]
    ]
    missing = [key for key in supported if key not in cells]
    assert not missing, (
        f"matrix incomplete: {len(missing)} supported cells not archived "
        f"(first few: {missing[:5]})"
    )

    table = workbook.build_runtime_table(cells)
    over = [r for r in table["cells"]
            if r["wall_time_seconds"] is None
            or r["wall_time_seconds"] >= workbook.NFR1_BUDGET_SECONDS]
    assert not over, f"cells missing timing or over the NFR-1 budget: {over}"
    assert table["all_within_budget"]


def test_scorecard_generated():
    """TASKS T16: the scorecard exists with all five §7.5 criteria, numeric
    evidence, and explicit pass/fail. A failing research criterion does NOT
    fail this test; instead the attribution table must be populated for it."""
    root = _archive_root()
    report = workbook.generate_workbook(root)

    for name in ("workbook.json", "workbook.md", "scorecard.json"):
        assert (report / name).is_file(), f"{name} missing from {report}"

    sc = json.loads((report / "scorecard.json").read_text(encoding="utf-8"))
    criteria = sc["scorecard"]["criteria"]
    assert sorted(criteria) == ["C1", "C2", "C3", "C4", "C5"], (
        "the scorecard must carry all five §7.5 criteria"
    )

    def _numeric_clauses(crit):
        return crit["clauses"] if "clauses" in crit else [crit]

    for cid, crit in criteria.items():
        assert isinstance(crit["pass"], bool), f"{cid}: pass/fail must be explicit"
        for clause in _numeric_clauses(crit):
            assert isinstance(clause["pass"], bool)
            assert clause["comparator"] in ("<", "<=", ">=")
            assert clause["values_by_seed"], f"{cid}: numeric evidence required"
            # A criterion may fail; evidence for the verdict may not be absent.
            if clause.get("note") is None:
                assert clause["mean"] is not None
                assert clause["threshold"] is not None

    # The ablation attribution table localizes any failed criterion (§7.5:
    # "the ablation table must then localize the underperforming mechanism").
    attribution = sc["attribution"]
    for cid, crit in criteria.items():
        if crit["pass"]:
            continue
        block = attribution.get(cid)
        assert block, f"{cid} failed but has no attribution block"
        populated = [
            s for s, row in block["systems"].items()
            if s != workbook.MAIN_SYSTEM and row["mean"] is not None
        ]
        assert len(populated) >= 2, (
            f"{cid} failed but its attribution table is not populated: "
            f"only {populated} have values"
        )

    # The B1 x P2 gap is a REPORTED limitation, never a silent hole.
    assert any("B1 x P2" in item for item in sc["scorecard"]["limitations"])
