"""T12 tests for fpcmc/protocols.py (TASKS.md Task 12; PRD §7.1).

test_p2_fixture_schedule [U]   — P2 builder on the fixture world: introduction
                                 steps match the schedule; zero occurrences of
                                 any introduced class after its phase (hard);
                                 past-class interleave fraction 0.30 ± 0.02 per
                                 phase; checkpoints at 1/4, 2/4, 3/4, 4/4 of
                                 each phase.
test_protocol_determinism [U]  — identical manifests for the same seed;
                                 disjoint shuffles across seeds {42, 43, 44}.
test_p1_matches_v1 [I]         — index-level equality against the archived v1
                                 seed-42 stream (docs/ASSETS.md §7.2): exact
                                 counts per pool, warmup all real IND test,
                                 total 13,326, and per-step (pool,
                                 within_pool_index, class, superclass, phase)
                                 equality.
test_p2_real_partition [I]     — the 80/20 partition equals the frozen
                                 configs/p2_class_split.yaml exactly; near
                                 phases contain exactly the 6 near classes; far
                                 phases partition the 43 far classes by
                                 superclass with none repeated.

Owner-approved T12 decisions exercised here (session 2026-07-13): phase
membership is deterministic and seed-independent (sorted-name chunking for
held-out/near; count-balanced greedy packing of far superclasses into 5
phases); phases are equal-length within each pool group using the full pools;
P1's warmup is a plain labeled prefix; the manifest is consumed eval-side
(StreamRunner untouched).
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from fpcmc.config import FPCMCConfig
from fpcmc.protocols import (
    P2ClassSplit,
    StreamManifest,
    build_p1,
    build_p2,
    load_p2_class_split,
)
from tests.fixtures.vmf_world import VMFWorld

V1_ARCHIVE_RELPATH = "evaluation/v1_p1_stream/stream_seed42_e723f028.npz"
V1_ARCHIVE_SHA256 = "77783853e826fbe52a2c4864ef49d3c10132311ad96a57b4a1c8e1d49834b73f"

# v1's StreamItem pool tags -> this repo's canonical pool names (fpcmc.data.POOL_SPECS)
V1_POOL_NAME_MAP = {
    "ind_real": "ind_test",
    "ind_synthetic": "synthetic_ind",
    "near_ood": "near_ood",
    "far_ood": "far_ood",
}


# ------------------------------------------------------------- fixture pools
@dataclass(frozen=True)
class FakePool:
    """The duck-typed pool surface build_p1/build_p2 consume."""

    x: np.ndarray
    subclass_names: np.ndarray
    superclass_names: np.ndarray


def _fake_pool(world: VMFWorld, names, n_per_class: int, stream: str, supers=None) -> FakePool:
    pool = world.make_pool(list(names), n_per_class, stream=stream)
    if supers is None:
        superclasses = np.array([f"sc_of_{c}" for c in pool.labels])
    else:
        superclasses = np.array([supers[c] for c in pool.labels])
    return FakePool(x=pool.x, subclass_names=pool.labels, superclass_names=superclasses)


def _p2_fixture():
    """A small P2 world: 4 T0 classes, 4 held-out (2 phases x 2), 2 near
    (1 phase x 2), 4 far classes in 3 superclasses (2 phases)."""
    world = VMFWorld(seed=11, k_known=14, k_novel=0, d=32)
    names = world.known_names
    t0, held_out = names[:4], names[4:8]
    near, far = names[8:10], names[10:14]
    far_supers = {far[0]: "sc_a", far[1]: "sc_a", far[2]: "sc_b", far[3]: "sc_c"}
    pools = {
        "ind_reference": _fake_pool(world, t0 + held_out, 30, "p2_ref"),
        "ind_test": _fake_pool(world, t0 + held_out, 50, "p2_test"),
        "near_ood": _fake_pool(world, near, 20, "p2_near"),
        "far_ood": _fake_pool(world, far, 15, "p2_far", supers=far_supers),
    }
    split = P2ClassSplit(t0=tuple(t0), held_out=tuple(held_out))
    return pools, split


def _p1_fixture():
    world = VMFWorld(seed=13, k_known=8, k_novel=0, d=32)
    names = world.known_names
    return {
        "ind_test": _fake_pool(world, names[:4], 40, "p1_ind"),
        "synthetic_ind": _fake_pool(world, names[:2], 10, "p1_syn"),
        "near_ood": _fake_pool(world, names[4:6], 20, "p1_near"),
        "far_ood": _fake_pool(world, names[6:8], 15, "p1_far"),
    }


def _build_p2_fixture(seed: int):
    pools, split = _p2_fixture()
    return build_p2(
        FPCMCConfig(),
        seed,
        pools=pools,
        class_split=split,
        heldout_per_phase=2,
        near_per_phase=2,
        n_far_phases=2,
    )


def _manifests_equal(a: StreamManifest, b: StreamManifest) -> bool:
    return all(
        np.array_equal(getattr(a, f), getattr(b, f))
        for f in ("pool", "within_pool_index", "true_class", "true_superclass", "phase")
    )


# ------------------------------------------------------------------ [U] tests
def test_p2_fixture_schedule():
    ps = _build_p2_fixture(seed=42)
    m = ps.manifest
    n = len(m.pool)
    pools, split = _p2_fixture()

    # Phase table: contiguous, ordered heldout -> near -> far, covers the stream.
    assert [p.group for p in ps.phases] == ["heldout", "heldout", "near", "far", "far"]
    assert ps.phases[0].start == 0 and ps.phases[-1].end == n
    for prev, nxt in zip(ps.phases, ps.phases[1:]):
        assert prev.end == nxt.start
    for p in ps.phases:
        assert np.all(m.phase[p.start : p.end] == p.name)

    # Introduction steps match the schedule; ZERO occurrences after the phase.
    for p in ps.phases:
        for cls in p.introduced_classes:
            steps = np.flatnonzero(m.true_class == cls)
            assert steps.size > 0, f"{cls} never appears"
            assert p.start <= steps.min() < p.end, f"{cls} introduced outside its phase"
            assert steps.max() < p.end, f"{cls} occurs after its phase (hard assert)"
    introduced = {c for p in ps.phases for c in p.introduced_classes}
    assert introduced == set(split.held_out) | set(pools["near_ood"].subclass_names) | set(
        pools["far_ood"].subclass_names
    )

    # T0 classes are never "introduced"; they appear only as ind_test interleave.
    t0 = set(split.t0)
    assert t0.isdisjoint(introduced)
    t0_rows = np.isin(m.true_class, sorted(t0))
    assert np.all(m.pool[t0_rows] == "ind_test")

    # Past-class interleave fraction = 0.30 +/- 0.02 per phase.
    for p in ps.phases:
        phase_classes = m.true_class[p.start : p.end]
        frac = np.isin(phase_classes, sorted(t0)).mean()
        assert abs(frac - 0.30) <= 0.02, f"{p.name}: interleave fraction {frac:.4f}"

    # Interleave rows are drawn without replacement globally.
    inter = np.flatnonzero(t0_rows)
    assert len(set(m.within_pool_index[inter])) == inter.size

    # Checkpoints at 1/4, 2/4, 3/4, 4/4 of each phase.
    expected = []
    for p in ps.phases:
        length = p.end - p.start
        expected.extend(p.start + (length * k) // 4 - 1 for k in (1, 2, 3, 4))
    assert list(ps.checkpoint_steps) == expected
    for p in ps.phases:
        assert p.end - 1 in ps.checkpoint_steps

    # Far phases: superclasses not repeated across phases (fixture-level check).
    far_phase_supers = []
    for p in ps.phases:
        if p.group != "far":
            continue
        rows = (m.phase == p.name) & (m.pool == "far_ood")
        far_phase_supers.append(set(m.true_superclass[rows]))
    assert far_phase_supers and set.union(*far_phase_supers) == {"sc_a", "sc_b", "sc_c"}
    for i, a in enumerate(far_phase_supers):
        for b in far_phase_supers[i + 1 :]:
            assert a.isdisjoint(b)

    # Embeddings line up with the manifest rows.
    assert ps.x.shape == (n, 32)
    for name, pool in pools.items():
        rows = m.pool == name
        assert np.array_equal(ps.x[rows], pool.x[m.within_pool_index[rows]])


def test_protocol_determinism():
    cfg = FPCMCConfig()

    # P1: same seed => identical; seeds {42,43,44} => pairwise different shuffles.
    pools = _p1_fixture()
    p1 = {s: build_p1(cfg, s, pools=pools, ind_warmup_count=30) for s in (42, 43, 44)}
    again = build_p1(cfg, 42, pools=pools, ind_warmup_count=30)
    assert _manifests_equal(p1[42].manifest, again.manifest)
    assert np.array_equal(p1[42].x, again.x)
    for a, b in ((42, 43), (42, 44), (43, 44)):
        assert not np.array_equal(
            p1[a].manifest.within_pool_index, p1[b].manifest.within_pool_index
        ), f"P1 seeds {a} and {b} produced the same ordering"

    # P2: same seed => identical; different seeds => different orderings but
    # IDENTICAL phase composition (membership is deterministic, seed-independent).
    p2 = {s: _build_p2_fixture(s) for s in (42, 43, 44)}
    again2 = _build_p2_fixture(42)
    assert _manifests_equal(p2[42].manifest, again2.manifest)
    assert np.array_equal(p2[42].x, again2.x)
    assert list(p2[42].checkpoint_steps) == list(again2.checkpoint_steps)
    for a, b in ((42, 43), (42, 44), (43, 44)):
        assert not _manifests_equal(p2[a].manifest, p2[b].manifest), (
            f"P2 seeds {a} and {b} produced the same manifest"
        )
    for s in (43, 44):
        assert [p.introduced_classes for p in p2[s].phases] == [
            p.introduced_classes for p in p2[42].phases
        ]
        assert [(p.start, p.end) for p in p2[s].phases] == [
            (p.start, p.end) for p in p2[42].phases
        ]


# ------------------------------------------------------------------ [I] tests
def _real_pools_or_skip():
    from fpcmc import data

    available, reason = data.embeddings_available()
    if not available:
        pytest.skip(f"real embeddings unavailable: {reason}")
    return data.load_all_pools()


def _v1_archive_or_skip() -> Path:
    from fpcmc.data import EmbeddingsUnavailable, read_roots_env

    try:
        roots = read_roots_env()
    except EmbeddingsUnavailable as e:
        pytest.skip(f"roots.env unavailable: {e}")
    data_root = roots.get("DATA_ROOT", "")
    if not data_root:
        pytest.skip("DATA_ROOT unset in roots.env")
    path = Path(data_root) / V1_ARCHIVE_RELPATH
    if not path.is_file():
        pytest.skip(f"archived v1 P1 stream not found at {path}")
    return path


@pytest.mark.slow
def test_p1_matches_v1():
    path = _v1_archive_or_skip()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == V1_ARCHIVE_SHA256, (
        f"archived v1 stream hash mismatch at {path}: {digest} — the archive is "
        "gate input; do not re-pin (docs/ASSETS.md §7)"
    )
    archive = np.load(path, allow_pickle=False)
    assert int(archive["seed"][0]) == 42 and int(archive["ind_warmup_count"][0]) == 1000

    pools = _real_pools_or_skip()
    ps = build_p1(FPCMCConfig(), 42, pools=pools)
    m = ps.manifest
    n = len(m.pool)

    # Composition: exact counts per pool; warmup only real IND test; total 13,326.
    assert n == 13_326
    counts = dict(zip(*np.unique(m.pool, return_counts=True)))
    assert counts == {"ind_test": 10_000, "synthetic_ind": 250, "near_ood": 500, "far_ood": 2_576}
    assert np.all(m.pool[:1_000] == "ind_test")
    assert np.all(m.phase[:1_000] == "warmup") and np.all(m.phase[1_000:] == "stream")
    assert [p.name for p in ps.phases] == ["warmup", "stream"]

    # Index-level equality with the archived v1 seed-42 stream.
    v1_pool = np.array([V1_POOL_NAME_MAP[p] for p in archive["pool"]])
    assert np.array_equal(m.pool, v1_pool)
    assert np.array_equal(m.within_pool_index, archive["within_pool_index"])
    assert np.array_equal(m.true_class, archive["true_class"])
    assert np.array_equal(m.true_superclass, archive["true_superclass"])
    assert np.array_equal(m.phase, archive["phase"])

    # Embeddings are the manifest-ordered pool rows (spot-check a stride).
    for t in range(0, n, 997):
        row = pools[m.pool[t]].x[m.within_pool_index[t]]
        assert np.array_equal(ps.x[t], row)


@pytest.mark.slow
def test_p2_real_partition():
    pools = _real_pools_or_skip()
    ps = build_p2(FPCMCConfig(), 42, pools=pools)
    m = ps.manifest
    split = load_p2_class_split()

    # 80/20 partition: deterministic, disjoint, covers all 100, equals the
    # frozen configs/p2_class_split.yaml lists exactly.
    all_cifar = set(pools["ind_reference"].subclass_names)
    assert len(all_cifar) == 100
    assert set(split.t0) | set(split.held_out) == all_cifar
    assert set(split.t0).isdisjoint(split.held_out)
    assert ps.t0_classes == tuple(sorted(split.t0)) and len(ps.t0_classes) == 80
    heldout_phases = [p for p in ps.phases if p.group == "heldout"]
    assert len(heldout_phases) == 4 and all(len(p.introduced_classes) == 5 for p in heldout_phases)
    assert {c for p in heldout_phases for c in p.introduced_classes} == set(split.held_out)
    assert len(split.held_out) == 20

    # Near phases contain exactly the 6 near classes (2 phases x 3).
    near_classes = set(pools["near_ood"].subclass_names)
    near_phases = [p for p in ps.phases if p.group == "near"]
    assert len(near_phases) == 2 and all(len(p.introduced_classes) == 3 for p in near_phases)
    assert {c for p in near_phases for c in p.introduced_classes} == near_classes
    assert len(near_classes) == 6

    # Far phases partition the 43 far classes by superclass, none repeated.
    far_phases = [p for p in ps.phases if p.group == "far"]
    assert len(far_phases) == 5
    far_class_lists = [p.introduced_classes for p in far_phases]
    all_far = [c for classes in far_class_lists for c in classes]
    assert len(all_far) == len(set(all_far)) == 43
    assert set(all_far) == set(pools["far_ood"].subclass_names)
    cls_to_super = dict(
        zip(pools["far_ood"].subclass_names.tolist(), pools["far_ood"].superclass_names.tolist())
    )
    phase_supers = [{cls_to_super[c] for c in classes} for classes in far_class_lists]
    assert set.union(*phase_supers) == set(pools["far_ood"].superclass_names)
    for i, a in enumerate(phase_supers):
        for b in phase_supers[i + 1 :]:
            assert a.isdisjoint(b), "a far superclass appears in two phases"

    # Every introduced class ceases after its phase, on real data too.
    for p in ps.phases:
        for cls in p.introduced_classes:
            steps = np.flatnonzero(m.true_class == cls)
            assert p.start <= steps.min() and steps.max() < p.end

    # Interleave rows are T0-class IND test examples only.
    t0_rows = np.isin(m.true_class, list(split.t0))
    assert np.all(m.pool[t0_rows] == "ind_test")
    assert 0.28 <= t0_rows.mean() <= 0.32
