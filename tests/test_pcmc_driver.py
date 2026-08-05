"""T17 Phase 2 tests — vendored-PCMC integrity, driver logic, GPU smoke.

Three layers (baselines/pcmc_sleep/PLAN.md Phase 2):

  * ``test_pcmc_vendor_untouched`` [U, fast]: every file under
    ``baselines/pcmc_sleep/vendor/`` is byte-identical to the blob hash
    recorded in lib/PROVENANCE.md, and the set matches exactly (T14
    ``test_v1_untouched`` precedent).
  * run_config [U, fast]: pure schedule/label/config logic on fake phases —
    no data, no torch, no GPU.
  * ``test_driver_smoke`` [I][slow]: tiny-budget end-to-end run on the real
    3090 (T0 pretrain, >=200 wake steps, 1 forced sleep, 1 checkpoint eval,
    artifacts + resume). Skips cleanly without the GPU env / CUDA / data.
    Runs under a hard timeout so the known upstream DataLoader-deadlock
    failure mode (HANDOFF §5) hangs the subprocess, not the suite.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from baselines.pcmc_sleep.run_config import (
    FEAT_SIZE,
    SLEEP_FREQ_SENTINEL,
    build_run_config,
    cluster_label_map,
    eval_classes_at,
    eval_label_map,
    phase_task_index,
    sleep_steps,
    snapshot_steps,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = REPO_ROOT / "baselines" / "pcmc_sleep" / "vendor"


# ---------------------------------------------------- vendored-set integrity


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def _vendor_hashes() -> dict[str, str]:
    text = (REPO_ROOT / "lib" / "PROVENANCE.md").read_text()
    marker = "## `baselines/pcmc_sleep/vendor/`"
    assert marker in text, "lib/PROVENANCE.md lost its pcmc_sleep vendor section"
    section = text[text.index(marker):]
    # Stop at the next section header, if one is ever appended after ours.
    nxt = section.find("\n## ", 1)
    if nxt != -1:
        section = section[:nxt]
    rows = re.findall(r"\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{40})`\s*\|", section)
    assert rows, "no hash rows found in the pcmc_sleep vendor provenance table"
    return dict(rows)


def test_pcmc_vendor_untouched():
    """Checksum of every vendored PCMC file matches lib/PROVENANCE.md, and
    the on-disk set is exactly the recorded set — nothing added, dropped, or
    edited (fidelity anchor: PCMC runs as published)."""
    recorded = _vendor_hashes()
    on_disk = {
        str(p.relative_to(VENDOR_DIR)): p
        for p in sorted(VENDOR_DIR.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    assert set(on_disk) == set(recorded), (
        "vendor/ file set diverged from lib/PROVENANCE.md: "
        f"unrecorded={sorted(set(on_disk) - set(recorded))}, "
        f"missing={sorted(set(recorded) - set(on_disk))}"
    )
    mismatched = {
        rel: (got, recorded[rel])
        for rel, path in on_disk.items()
        if (got := _git_blob_sha1(path)) != recorded[rel]
    }
    assert not mismatched, f"vendored PCMC files modified: {mismatched}"


# ------------------------------------------------------- run_config [U] logic


@dataclass(frozen=True)
class FakePhase:
    name: str
    group: str
    start: int
    end: int
    introduced_classes: tuple = field(default_factory=tuple)


#: Miniature P2 shape: 2 heldout, 1 near, 1 far phase, contiguous steps.
PHASES = (
    FakePhase("heldout_00", "heldout", 0, 100, ("h0", "h1")),
    FakePhase("heldout_01", "heldout", 100, 250, ("h2",)),
    FakePhase("near_00", "near", 250, 280, ("n0", "n1")),
    FakePhase("far_00", "far", 280, 340, ("f0",)),
)
T0 = ("a0", "a1", "a2")


def test_sleep_schedule_mid_phase():
    """Q2: exactly one sleep per phase, at its midpoint, in order."""
    steps = sleep_steps(PHASES)
    assert steps == [50, 175, 265, 310]
    for step, phase in zip(steps, PHASES):
        assert phase.start <= step < phase.end
    # Non-contiguous or empty phases are construction errors, not schedules.
    with pytest.raises(ValueError):
        sleep_steps((FakePhase("x", "near", 0, 10), FakePhase("y", "near", 20, 30)))
    with pytest.raises(ValueError):
        sleep_steps((FakePhase("x", "near", 5, 5),))


def test_snapshot_steps_phase_end():
    """Q4: snapshots at phase-end steps (their last checkpoints)."""
    assert snapshot_steps(PHASES) == [99, 249, 279, 339]


def test_phase_task_index():
    assert phase_task_index(PHASES) == {
        "heldout_00": 1, "heldout_01": 2, "near_00": 3, "far_00": 4,
    }


def test_eval_label_map_cifar_only_contiguous():
    """Q3(a): labels 0..C-1 over T0 order then held-out introduction order;
    synthetic classes excluded. Contiguity is load-bearing for their
    supervise()/classify() label indexing."""
    mapping = eval_label_map(T0, PHASES)
    assert mapping == {"a0": 0, "a1": 1, "a2": 2, "h0": 3, "h1": 4, "h2": 5}
    assert sorted(mapping.values()) == list(range(len(mapping)))
    for synth in ("n0", "n1", "f0"):
        assert synth not in mapping


def test_cluster_label_map_extends_eval_map():
    mapping = cluster_label_map(T0, PHASES)
    eval_map = eval_label_map(T0, PHASES)
    assert {k: mapping[k] for k in eval_map} == eval_map
    assert mapping["n0"] == 6 and mapping["n1"] == 7 and mapping["f0"] == 8
    assert sorted(mapping.values()) == list(range(len(mapping)))


def test_eval_classes_at_task():
    """Class visibility grows with their task index; current task included
    (their eval_loaders(t) include the in-progress task's classes)."""
    assert eval_classes_at(T0, PHASES, 0) == (["a0", "a1", "a2"], [])
    assert eval_classes_at(T0, PHASES, 1) == (["a0", "a1", "a2", "h0", "h1"], [])
    cifar, synth = eval_classes_at(T0, PHASES, 3)
    assert cifar == ["a0", "a1", "a2", "h0", "h1", "h2"]
    assert synth == ["n0", "n1"]
    cifar, synth = eval_classes_at(T0, PHASES, 4)
    assert synth == ["n0", "n1", "f0"]


def test_run_config_paper_faithful():
    """The §5/§7 hard facts, pinned: pretrained=False, both epoch knobs
    equal, M <= theta, their-120 geometry, CIFAR mean/std, per-arch
    feat_size, in-process loading."""
    for arch in ("resnet18", "resnet50"):
        cfg = build_run_config(arch, 42)
        assert cfg["model"]["pretrained"] is False
        layer = cfg["model"]["layers"]["layer0"]
        assert cfg["model"]["init_epochs"] == layer["init_epochs"]
        assert layer["M"] <= layer["theta"]
        assert (cfg["dataset"]["img_size"], layer["patch_size"], layer["stride"]) \
            == (120, 60, 30)
        assert cfg["dataset"]["mean"] == [0.5071, 0.4865, 0.4409]
        assert cfg["dataset"]["std"] == [0.2673, 0.2564, 0.2762]
        assert layer["feat_size"] == FEAT_SIZE[arch]
        assert cfg["model"]["n_workers"] == 0
        assert cfg["dataset"]["stream_bs"] == 1
        assert "smoke" not in cfg
    assert FEAT_SIZE["resnet50"] == 2048  # their 512 default would break RN50


def test_run_config_smoke_budget():
    """Smoke config: recorded tiny budget that still satisfies upstream hard
    invariants — init_memory's fixed 2,000-patch draw and the
    len(trainloader) // init_epochs >= 1 pretrain constraint."""
    cfg = build_run_config("resnet18", 42, smoke=True)
    smoke = cfg["smoke"]
    layer = cfg["model"]["layers"]["layer0"]
    epochs, bs = cfg["model"]["init_epochs"], layer["pretrain_bs"]
    n_batches_total = (smoke["t0_images"] * epochs) // bs  # drop_last=True
    assert n_batches_total // epochs >= 1
    patches_per_image = ((120 - layer["patch_size"]) // layer["stride"] + 1) ** 2
    consumed = (n_batches_total // epochs + 1) * bs * patches_per_image
    assert consumed >= 2000, "init_memory's hardcoded sample would crash"
    assert layer["init_clusters"] <= consumed
    assert smoke["max_stream_steps"] >= 200
    assert len(smoke["sleep_steps"]) >= 1
    assert len(smoke["checkpoint_steps"]) >= 1
    assert max(smoke["sleep_steps"] + smoke["checkpoint_steps"]) \
        < smoke["max_stream_steps"]


def test_run_config_clust_size():
    """Q6 (Phase 3 owner decision, PLAN.md): the clustering eval subsample —
    a strict subset of the test set, pinned at 25/class, and always > 2:
    their SpectralClustering requires n_clusters (= 2 x classes) <
    n_samples (= clust_size x classes)."""
    from baselines.pcmc_sleep.run_config import CLUST_SIZE, RHO

    cfg = build_run_config("resnet18", 42)
    assert cfg["dataset"]["clust_size"] == CLUST_SIZE == 25
    assert 2 < cfg["dataset"]["clust_size"] <= cfg["dataset"]["test_size"]
    assert cfg["model"]["layers"]["layer0"]["rho"] == RHO
    smoke = build_run_config("resnet18", 42, smoke=True)
    assert 2 < smoke["dataset"]["clust_size"] < smoke["dataset"]["test_size"]


def test_run_config_rejects_off_matrix_cells():
    with pytest.raises(ValueError):
        build_run_config("resnet34", 42)
    with pytest.raises(ValueError):
        build_run_config("resnet18", 7)


def test_sleep_steer_arithmetic():
    """The driver steers their trigger (step == sleep_start, OR
    (step - sleep_start) % sleep_freq == 0 — python modulo, so it CAN fire
    for step < sleep_start at congruent points). With the sentinel freq and
    next-target starts, the trigger fires exactly on the schedule."""
    schedule = [50, 175]
    fired = []
    pending = list(schedule)
    for stream_index in range(300):
        target = (pending[0] + 1) if pending else SLEEP_FREQ_SENTINEL
        step = stream_index + 1  # their Layer.step at trigger time
        if (step == target) or (step - target) % SLEEP_FREQ_SENTINEL == 0:
            fired.append(stream_index)
        if pending and stream_index == pending[0]:
            pending.pop(0)
    assert fired == schedule


# --------------------------------------------- run_pcmc_matrix [U] enumeration


def test_matrix_enumeration_order_and_naming():
    """Phase 4 (owner Q11): 12 cells = {rn18,rn50} x {sleep,nosleep} x
    {42,43,44}, arch outer then seed then SLEEP-before-NOSLEEP so the shared
    T0 cache is warm for the nosleep cell of each (arch, seed). Manifest keys
    match launch.cell_dir; selectors are unique."""
    from baselines.pcmc_sleep import launch, run_pcmc_matrix as rm

    cells = rm.enumerate_cells()
    assert len(cells) == 12
    # 3 axes, each fully covered.
    assert {c.arch for c in cells} == {"resnet18", "resnet50"}
    assert {c.seed for c in cells} == {42, 43, 44}
    assert {c.variant for c in cells} == {"sleep", "nosleep"}
    assert len(cells) == len({(c.arch, c.seed, c.variant) for c in cells})

    # First/last pin the outer->inner ordering; sleep precedes nosleep.
    assert (cells[0].arch, cells[0].seed, cells[0].variant) \
        == ("resnet18", 42, "sleep")
    assert (cells[1].arch, cells[1].seed, cells[1].variant) \
        == ("resnet18", 42, "nosleep")
    assert (cells[-1].arch, cells[-1].seed, cells[-1].variant) \
        == ("resnet50", 44, "nosleep")

    # For every (arch, seed) the SLEEP cell comes before its NOSLEEP twin —
    # the load-bearing invariant for cache warming (owner Q11).
    order = {(c.arch, c.seed, c.variant): i for i, c in enumerate(cells)}
    for arch in ("resnet18", "resnet50"):
        for seed in (42, 43, 44):
            assert order[(arch, seed, "sleep")] < order[(arch, seed, "nosleep")]

    # Manifest key == on-disk cell path suffix; selectors are unique tags.
    tmp = Path("/tmp/pcmc_matrix_out")
    for c in cells:
        assert c.cell_dir(tmp) == launch.cell_dir(tmp, c.arch, c.sleep_on, c.seed)
        assert c.key == f"pcmc_{c.arch}_{c.variant}/p2_seed{c.seed}"
    assert len({c.selector for c in cells}) == 12


def test_matrix_pretrain_cache_key_matches_driver():
    """The runner's cache path must be byte-identical to driver.py's cache_key
    for the production path, or sleep/nosleep would not share a T0 encoder
    (owner Q11). Reconstruct the driver's key from the same run_config
    constants the resolved config carries."""
    from baselines.pcmc_sleep import run_pcmc_matrix as rm
    from baselines.pcmc_sleep.run_config import IMG_SIZE, INIT_EPOCHS

    cache = Path("/tmp/cache")
    got = rm.pretrain_cache_file(cache, "resnet50", 43)
    # driver.py: f"{arch}_seed{seed}_ep{init_epochs}_img{img_size}.pkl"
    expected = cache / f"resnet50_seed43_ep{INIT_EPOCHS}_img{IMG_SIZE}.pkl"
    assert got == expected
    # sleep and nosleep of the same (arch, seed) resolve to the SAME file —
    # that collision IS the sharing (both variants pass the same cache dir).
    assert rm.pretrain_cache_file(cache, "resnet18", 42) \
        == rm.pretrain_cache_file(cache, "resnet18", 42)


def test_matrix_budget_projection():
    """A cache-miss cell is projected to pay T0 pretrain; a cache-hit reuses
    it (owner Q11) — the over-budget flag is derived from this, per cell."""
    from baselines.pcmc_sleep import run_pcmc_matrix as rm

    miss = rm.projected_hours(from_cache=False)
    hit = rm.projected_hours(from_cache=True)
    assert miss > hit
    assert hit == rm.WAKE_EVAL_PROJECTION_H
    assert miss == rm.PRETRAIN_PROJECTION_H + rm.WAKE_EVAL_PROJECTION_H


# ------------------------------------------- driver dataloader guard [U]


def test_dataloader_guard_workers_modes():
    """The driver's DataLoader guard (deadlock mitigation, HANDOFF_PHASE2 §5;
    --workers owner-authorized 2026-07-17): workers=0 forces in-process
    loading and strips worker-only kwargs; workers>0 forces
    persistent_workers=False so upstream's early iterator abandonment
    (pcmc_layer.py:341-343) tears workers down instead of deadlocking.
    NOTE: importing the driver rebinds torch.utils.data.DataLoader for this
    process — nothing else in the CPU suite constructs DataLoaders."""
    import torch

    from baselines.pcmc_sleep import driver

    ds = torch.utils.data.TensorDataset(torch.zeros(8, 3))
    # Default: in-process, worker-only kwargs stripped (their pretrain passes
    # persistent_workers=True which raises with num_workers=0 if kept).
    loader = torch.utils.data.DataLoader(
        ds, batch_size=4, num_workers=6, persistent_workers=True,
        prefetch_factor=2,
    )
    assert loader.num_workers == 0
    try:
        driver._FORCED_WORKERS = 8
        # Big training loaders (>= _WORKERS_MIN_BS) get the worker pool...
        loader = torch.utils.data.DataLoader(
            ds, batch_size=256, num_workers=0, persistent_workers=True,
            prefetch_factor=2,
        )
        assert loader.num_workers == 8
        assert loader.persistent_workers is False
        # ...but their many tiny eval loaders (bs=1) stay in-process — a
        # worker pool per eval loader is a fork storm (observed 2026-07-17).
        loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=6)
        assert loader.num_workers == 0
    finally:
        driver._FORCED_WORKERS = 0


# ----------------------------------------------------------- [I] GPU smoke


@pytest.mark.slow
def test_driver_smoke(tmp_path):
    """Tiny-budget end-to-end driver run on the 3090 (HANDOFF §9): reduced T0
    pretrain, >=200 wake steps, one forced sleep, one checkpoint eval,
    persisted artifacts, and the resume path. Hard 45-minute timeout makes
    a DataLoader deadlock a failure, not a hang."""
    from baselines.pcmc_sleep import launch
    from fpcmc.data import embeddings_available

    ok, reason = embeddings_available()
    if not ok:
        pytest.skip(reason)
    ok, reason = launch.gpu_env_available()
    if not ok:
        pytest.skip(reason)
    from baselines.pcmc_sleep.stream_mirror import _data_root

    root = _data_root()
    for sub in ("cifar100/cifar-100-python", "ms_cifar100_genai_novel_32x32"):
        if not (root / sub).is_dir():
            pytest.skip(f"raw image tree missing: {root / sub}")

    cell = launch.run_cell(
        "resnet18", 42, out_root=tmp_path, smoke=True, timeout=2700
    )

    resolved = yaml.safe_load((cell / "resolved_config.yaml").read_text())
    assert resolved["model"]["pretrained"] is False
    assert resolved["schedule"]["sleep_steps"] == [150]

    summary = json.loads((cell / "summary.json").read_text())
    assert summary["cell"]["smoke"] is True
    assert summary["n_stream_steps"] >= 200
    assert summary["sleep_steps_executed"] == [150]
    assert summary["final_ltm_size"] > 0

    t0_record = json.loads((cell / "checkpoints" / "t0.json").read_text())
    assert t0_record["task"] == 0 and 0 <= t0_record["class_acc"] <= 100
    step = resolved["schedule"]["checkpoint_steps"][0]
    record = json.loads(
        (cell / "checkpoints" / f"step_{step:05d}.json").read_text()
    )
    assert record["step"] == step
    assert 0 <= record["class_acc"] <= 100
    assert 0 <= record["clust_acc"] <= 100
    # Smoke stays inside heldout_00 (task 1): 80 T0 + 5 introduced classes.
    assert len(record["class_pc_acc"]) == 85
    assert (cell / "final_state.pt").is_file()

    # Resume: unchanged config + existing summary => skip (cheap, must not
    # retrain). The driver prints the skip and leaves artifacts untouched.
    before = (cell / "summary.json").stat().st_mtime_ns
    launch.run_cell("resnet18", 42, out_root=tmp_path, smoke=True, timeout=300)
    assert (cell / "summary.json").stat().st_mtime_ns == before
