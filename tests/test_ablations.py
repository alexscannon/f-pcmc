"""T15 tests — ablation flags, run-config matrix, sweep scope guard, A6 smoke
(TASKS Task 15; PRD §7.4 A1–A6, §8).

Owner-approved T15 semantics (Q&A 2026-07-13, recorded in docs/CHANGES.md T15):

  A1 global_tau    — every concept's (tau, tau_vmf) pinned to the frozen
                     FR-5.3 GlobalPrior pair; no LOO, no shrinkage, no
                     promotion recompute.
  A2 no_stm        — all STM management off: no maturity gate (every candidate
                     tier-1-eligible from birth), no capacity/LRU eviction, no
                     FR-6 residual pool; promotion = size >= theta alone.
                     "Zero STM records in the log" = zero evict records (the
                     STM-lifecycle record type; seeds still happen).
  A3 no_recurrence — FR-7 criterion 4 dropped; other three unchanged.
  A4 no_merge      — every merge pathway off (periodic FR-8 sweep,
                     on-promotion check, FR-6 residual consolidation).
  A5               — no flag: `scorer: knn_ref` / `scorer: vmf` configs; the
                     sub-scorer identity is visible per assign record (`via`).
  A6               — no flag: `encoder: resnet50`; run_matrix maps the key to
                     the ResNet50_32px embeddings directory.

The bite tests run the frozen golden stream under the frozen golden-run
config (loaded read-only; per-flag variants via in-memory dataclasses.replace
— the file is owner-frozen gate input and is never edited). Residual
clustering's UMAP/HDBSCAN core is mocked out (`ResidualClusterer._cluster`
-> []) exactly as test_hook_schedule does: none of the asserted deltas
depends on residual consolidation, and A2/A4 skip the hook anyway; this keeps
the six golden runs inside the fast-suite budget. The A3 case lowers
theta_promote to 10 on BOTH runs: the golden burst class has only 15 examples
(<= 14 post-seed matches), so at the default theta=30 it fails the size
criterion regardless of recurrence and the flag could never show a delta —
at theta=10 the off-run still blocks it (recurrence: one window < m_windows)
and the on-run promotes it, which is precisely the pathology A3 re-enables.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import yaml

import run_matrix
from fpcmc.concepts import Concept
from fpcmc.config import AblationConfig, FPCMCConfig
from fpcmc.init import initialize_ltm
from fpcmc.residual import ResidualClusterer
from fpcmc.rng import make_rng
from fpcmc.scorers import estimate_kappa
from fpcmc.stream import StreamRunner
from fpcmc.thresholds import compute_global_prior, recompute_on_promotion
from tests.fixtures.golden_stream import load_golden, make_golden_world

REPO_ROOT = Path(__file__).resolve().parents[1]

# The frozen golden-run config (gate input): loaded once, never written back.
_GOLDEN_CONFIG = FPCMCConfig.from_yaml(REPO_ROOT / "configs" / "golden_run.yaml")

# ---------------------------------------------------------- golden-run cache


_RUN_CACHE: dict = {}


def _golden_run(**overrides) -> dict:
    """One full StreamRunner pass over the frozen golden stream under the
    frozen golden-run config + in-memory overrides. Cached per override set —
    the parametrized bite cases share the flag-off baseline."""
    key = tuple(sorted((k, repr(v)) for k, v in overrides.items()))
    if key in _RUN_CACHE:
        return _RUN_CACHE[key]

    config = dataclasses.replace(_GOLDEN_CONFIG, **overrides)
    g = load_golden()
    store = initialize_ltm(g["t0_x"], g["t0_labels"], config)
    prior = compute_global_prior(store.ltm, config)
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "run.jsonl"
        with mock.patch.object(ResidualClusterer, "_cluster", return_value=[]):
            runner = StreamRunner(config, store, prior, log_path=log_path)
            runner.run(g["stream_x"])
        records = [json.loads(line) for line in log_path.read_text().splitlines()]

    run = {
        "config": config,
        "store": store,
        "prior": prior,
        "records": records,
        "labels": np.asarray(g["stream_labels"], dtype=str),
    }
    _RUN_CACHE[key] = run
    return run


def _by_type(records, rtype):
    return [r for r in records if r["type"] == rtype]


def _ids_for_class(run, cls: str) -> set[str]:
    """Concept ids that ever received an arrival of ground-truth class `cls`
    (assign or seed), read from the log against the fixture labels."""
    steps = {r["step"]: r["concept_id"] for r in run["records"] if r["type"] in ("assign", "seed")}
    return {steps[s] for s in range(len(run["labels"])) if run["labels"][s] == cls}


# ------------------------------------------------- flag bite (TASKS [U] test)


def _bite_a1_global_tau():
    on = _golden_run(ablation=AblationConfig(global_tau=True))
    off = _golden_run()

    prior = on["prior"]
    concepts = on["store"].concepts
    assert len(concepts) > 8, "premise: the run must carry more than the T0 concepts"
    assert all(c.tau == prior.tau for c in concepts), (
        "A1: every concept's tau must be the global prior exactly"
    )
    assert all(c.tau_vmf == prior.tau_vmf for c in concepts), (
        "A1: every concept's tau_vmf must be the global prior exactly"
    )
    # And promotions really happened under A1 (recompute_on_promotion held the
    # prior rather than calibrating) — otherwise the equality is weak evidence.
    assert _by_type(on["records"], "promote"), "A1 run must still promote"

    off_taus = {c.tau for c in off["store"].ltm}
    assert len(off_taus) > 1, (
        "flag off: per-concept FR-5.1 taus must differ across LTM concepts"
    )


def _bite_a2_no_stm():
    on = _golden_run(ablation=AblationConfig(no_stm=True))
    off = _golden_run()

    assert not _by_type(on["records"], "evict"), (
        "A2: zero STM-lifecycle (evict) records — capacity/LRU is off"
    )
    assert not on["store"].eviction_log
    on_tiers = {r["tier"] for r in _by_type(on["records"], "assign")}
    assert on_tiers == {1}, (
        f"A2: no maturity gate means no tier-2 traffic, got tiers {on_tiers}"
    )

    assert _by_type(off["records"], "evict"), "flag off: LRU eviction must be live"
    assert any(r["tier"] == 2 for r in _by_type(off["records"], "assign")), (
        "flag off: immature candidates must receive tier-2 traffic"
    )


def _bite_a3_no_recurrence():
    # theta=10 on BOTH runs — see the module docstring for the derivation.
    on = _golden_run(theta_promote=10, ablation=AblationConfig(no_recurrence=True))
    off = _golden_run(theta_promote=10)

    on_burst = _ids_for_class(on, "burst_00")
    on_promoted = {r["concept_id"] for r in _by_type(on["records"], "promote")}
    assert on_burst & on_promoted, (
        f"A3: the one-shot burst class must promote once recurrence is dropped "
        f"(the pathology returns); burst={sorted(on_burst)} promoted={sorted(on_promoted)}"
    )

    off_burst = _ids_for_class(off, "burst_00")
    off_promoted = {r["concept_id"] for r in _by_type(off["records"], "promote")}
    assert not (off_burst & off_promoted), (
        "flag off: recurrence must keep blocking the burst class at the same theta"
    )
    assert off_promoted, "flag off at theta=10 must still promote recurring novelty"


def _promoted_fragment(x: np.ndarray, concept_id: str, config: FPCMCConfig, *,
                       match_count: int, windows: set[int]) -> Concept:
    """A manually promoted same-class fragment (the T9 'force via manual
    promotion' pattern from tests/test_merge.py)."""
    centroid = x.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    c = Concept(
        concept_id=concept_id,
        centroid=centroid,
        ref_set=np.array(x, dtype=np.float64),
        tau=0.30,
        kappa=estimate_kappa(x),
        tau_vmf=0.0,
        status="STM",
        provenance="seeded",
        match_count=match_count,
        match_windows=set(windows),
        rng=make_rng(15, f"t15/reservoir/{concept_id}"),
    )
    c.status = "LTM"
    c.provenance = "promoted"
    recompute_on_promotion(c, config)
    return c


def _bite_a4_no_merge(tmp_path: Path):
    """TASKS A4 delta: fragmentation index > 1 achievable on a crafted split.

    Two manually promoted fragments of the golden novel class ride through a
    short StreamRunner pass with T_merge=2. Flag off: the FR-8.3 LTM<->LTM
    sweep merges them (fragmentation back to 1). Flag on: zero merge records,
    both fragments alive — 2 promoted concepts for 1 class, fragmentation 2.
    The test drives StreamRunner because that is where the flag gates the
    merge pathways (the wiring under test), not MergeSweeper internals.
    """
    world = make_golden_world()
    known = world.known_names[0]
    stream_x = world.sample_class(known, 5, stream="t15/a4/stream")

    results = {}
    for flag_on in (False, True):
        config = dataclasses.replace(
            _GOLDEN_CONFIG, T_merge=2, ablation=AblationConfig(no_merge=flag_on)
        )
        t0 = world.t0_pool(50)
        store = initialize_ltm(t0.x, t0.labels, config)
        prior = compute_global_prior(store.ltm, config)
        a = _promoted_fragment(
            world.sample_class("novel_00", 36, stream="t15/a4/frag_a"),
            "stm_0000", config, match_count=40, windows={1, 2, 3},
        )
        b = _promoted_fragment(
            world.sample_class("novel_00", 36, stream="t15/a4/frag_b"),
            "stm_0001", config, match_count=31, windows={2, 3, 4},
        )
        store.register(a)
        store.register(b)

        log_path = tmp_path / f"a4_{'on' if flag_on else 'off'}.jsonl"
        runner = StreamRunner(config, store, prior, log_path=log_path)
        runner.run(stream_x)
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        results[flag_on] = (store, _by_type(records, "merge"))

    off_store, off_merges = results[False]
    assert [(r["kind"], r["survivor_id"], r["absorbed_id"]) for r in off_merges] == [
        ("ltm_ltm", "stm_0000", "stm_0001")
    ], "flag off: the periodic sweep must merge the promoted fragments"
    assert "stm_0001" not in off_store

    on_store, on_merges = results[True]
    assert on_merges == [], "A4: no merge records of any kind"
    fragments = [c for c in on_store.ltm if c.provenance == "promoted"]
    assert {c.concept_id for c in fragments} == {"stm_0000", "stm_0001"}, (
        "A4: both same-class fragments stay alive"
    )
    # Fragmentation index for novel_00: 2 promoted concepts / 1 class > 1.
    assert len(fragments) / 1 > 1


def _bite_a5_scorer():
    knn = _golden_run(scorer="knn_ref")
    vmf = _golden_run(scorer="vmf")
    base = _golden_run()

    knn_assigns = _by_type(knn["records"], "assign")
    assert knn_assigns and all(r["via"] == "knn_ref" and not r["fallback"] for r in knn_assigns), (
        "A5 knn_ref: every assignment identifies the knn_ref sub-scorer, no fallback"
    )

    vmf_assigns = _by_type(vmf["records"], "assign")
    assert any(r["via"] == "vmf" for r in vmf_assigns), (
        "A5 vmf: vmf-scored assignments must appear"
    )
    assert all(
        r["via"] == "vmf" or (r["via"] == "knn_ref" and r["fallback"])
        for r in vmf_assigns
    ), "A5 vmf: knn_ref may appear only as the flagged FR-4.2 small-ref_set fallback"

    # Flag off (composed knn_vmf): both identities present, and knn_ref wins
    # some assignments WITHOUT the fallback flag — only the composition does that.
    base_assigns = _by_type(base["records"], "assign")
    base_vias = {r["via"] for r in base_assigns}
    assert base_vias == {"knn_ref", "vmf"}, (
        f"composed scorer: both sub-scorer identities expected, got {base_vias}"
    )
    assert any(r["via"] == "knn_ref" and not r["fallback"] for r in base_assigns)


@pytest.mark.parametrize(
    "flag", ["a1_global_tau", "a2_no_stm", "a3_no_recurrence", "a4_no_merge", "a5_scorer"]
)
def test_ablation_flags_bite(flag, tmp_path):
    """TASKS T15: per flag, run the golden stream with the flag on and off and
    assert the flag-specific behavioral delta — a flag that changes nothing is
    a wiring bug."""
    if flag == "a1_global_tau":
        _bite_a1_global_tau()
    elif flag == "a2_no_stm":
        _bite_a2_no_stm()
    elif flag == "a3_no_recurrence":
        _bite_a3_no_recurrence()
    elif flag == "a4_no_merge":
        _bite_a4_no_merge(tmp_path)
    else:
        _bite_a5_scorer()


# ------------------------------------------- config matrix (TASKS [U] test)


# Every PRD §7.4 run row -> the exact key set its config may change
# (dot-flattened). B rows and the main run deviate in nothing: protocol and
# seed are matrix axes, never config keys.
_EXPECTED_DIFFS = {
    "fpcmc_default": {},
    "b1_v1": {},
    "b2_batch": {},
    "b3_oracle": {},
    "a1_global_tau": {"ablation.global_tau": True},
    "a2_no_stm": {"ablation.no_stm": True},
    "a3_no_recurrence": {"ablation.no_recurrence": True},
    "a4_no_merge": {"ablation.no_merge": True},
    "a5_knn_ref": {"scorer": "knn_ref"},
    "a5_vmf": {"scorer": "vmf"},
    "a6_resnet50": {"encoder": "resnet50"},
}


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def test_config_matrix_complete():
    """TASKS T15: every run row of PRD §7.4 has a committed config file, and
    each differs from the default config in exactly its declared keys."""
    assert set(run_matrix.SYSTEM_CONFIGS) == set(_EXPECTED_DIFFS), (
        "run_matrix systems and the §7.4 row inventory must match one-to-one"
    )

    default = _flat(yaml.safe_load(FPCMCConfig.from_yaml(REPO_ROOT / "configs" / "default.yaml").to_yaml()))
    for system, path in run_matrix.SYSTEM_CONFIGS.items():
        assert path.is_file(), f"missing committed config for §7.4 row {system!r}: {path}"
        resolved = _flat(yaml.safe_load(FPCMCConfig.from_yaml(path).to_yaml()))
        assert resolved.keys() == default.keys()
        diff = {k: v for k, v in resolved.items() if default[k] != v}
        assert diff == _EXPECTED_DIFFS[system], (
            f"{path.name} deviates from default in {sorted(diff)} but declares "
            f"{sorted(_EXPECTED_DIFFS[system])}"
        )

    # The matrix plan covers {system x protocol x seed} with the one
    # owner-ruled unsupported cell marked, never silently dropped.
    plan = run_matrix.plan_matrix()
    assert len(plan) == len(_EXPECTED_DIFFS) * 2 * 3
    unsupported = [c for c in plan if not c["supported"]]
    assert {(c["system"], c["protocol"]) for c in unsupported} == {("b1_v1", "p2")}
    with pytest.raises(run_matrix.CellUnsupported):
        run_matrix.run_cell("b1_v1", "p2", 42, out_root="/nonexistent-never-used")


# --------------------------------------------- sweep scope (TASKS [U] test)


def test_sweep_scope_guard():
    """TASKS T15: the sweep runner rejects any parameter outside the three
    PRD-sanctioned sweep keys, and pins sweeps to fpcmc_default/P1/seed 42."""
    assert run_matrix.SWEEP_PARAMS == ("stm_capacity", "theta_promote", "min_cohesion_ratio")

    for bad in ("k_ref", "n_mature", "merge_sim", "window_W", "tau_percentile_q", "seed"):
        with pytest.raises(ValueError, match="sanctioned sweep parameter"):
            run_matrix.run_sweep(bad, [1], dry_run=True)

    plan = run_matrix.run_sweep("stm_capacity", [50, 100, 200], dry_run=True)
    assert [c["value"] for c in plan] == [50, 100, 200]
    assert all(
        (c["system"], c["protocol"], c["seed"]) == ("fpcmc_default", "p1", 42)
        for c in plan
    ), "PRD §8: sweeps run on P1, seed 42, the default system only"

    for param in run_matrix.SWEEP_PARAMS:  # all three sanctioned keys accepted
        assert run_matrix.run_sweep(param, [], dry_run=True) == []


# ------------------------------------------------- A6 smoke (TASKS [I] test)


@pytest.mark.slow
def test_a6_resnet_smoke(tmp_path):
    """TASKS T15: the A6 config runs end-to-end on real ResNet-50 embeddings
    without error — no performance assertion (degraded results are the
    expected finding). Also exercises the run_matrix cell + resume path."""
    from fpcmc.data import POOL_SPECS, EmbeddingsUnavailable, embeddings_available

    ok, reason = embeddings_available()
    if not ok:
        pytest.skip(reason)
    try:
        resnet_dir = run_matrix.embeddings_dir_for_encoder("resnet50")
    except EmbeddingsUnavailable as e:
        pytest.skip(str(e))
    missing = [s.filename for s in POOL_SPECS if not (resnet_dir / s.filename).is_file()]
    if missing:
        pytest.skip(f"ResNet-50 pools missing under {resnet_dir}: {missing}")

    cell = run_matrix.run_cell("a6_resnet50", "p1", 42, out_root=tmp_path)

    summary = json.loads((cell / "summary.json").read_text())
    assert summary["cell"] == {"system": "a6_resnet50", "protocol": "p1", "seed": 42}
    records = [json.loads(line) for line in (cell / "events.jsonl").read_text().splitlines()]
    header = records[0]
    assert header["type"] == "config_header"
    assert header["config"]["encoder"] == "resnet50"
    assert header["n_steps"] == 13_326
    assert sum(1 for r in records if r["type"] in ("assign", "seed")) == 13_326, (
        "single-pass: every P1 arrival produced exactly one assign/seed record"
    )

    # Resumability: an immediate re-run reuses the finished cell untouched.
    before = (cell / "events.jsonl").stat().st_mtime_ns
    again = run_matrix.run_cell("a6_resnet50", "p1", 42, out_root=tmp_path)
    assert again == cell
    assert (cell / "events.jsonl").stat().st_mtime_ns == before
