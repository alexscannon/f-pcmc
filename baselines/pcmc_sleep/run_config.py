"""T17 Phase 2: pure run-config / schedule / label logic for the PCMC driver.

Torch-free and importable from the repo's CPU test env — everything here is
unit-testable with fakes. The GPU driver (`driver.py`) consumes these
functions verbatim so the tested logic IS the shipped logic.

Owner decisions baked in (Phase 2 Q&A, 2026-07-14 — recorded in PLAN.md):
  Q1 geometry: their-120 (img 120, patch 60, stride 30) — Phase 0.3 spike.
  Q2 sleep schedule: one sleep at each P2 phase midpoint (driver steers their
     ``sleep_start``/``sleep_freq`` attributes; trigger code untouched).
  Q3 eval sets: classification protocol restricted to CIFAR classes (T0 80 +
     held-out 20, genuine CIFAR test-split images); synthetic near/far
     classes enter ONLY the auxiliary clustering-purity eval, from
     stream-seen rows. NOTE (documented deviation-shaped fact, not a
     deviation): P2 streams the held-out classes' ind_test rows and ~80% of
     the T0 ind_test interleave, so CIFAR test images may have been SEEN
     UNLABELED in-stream — which matches the released upstream behavior
     (streams.py sets ``cls_inds_test = cls_inds_train`` for most datasets).
  Q4 snapshots: model + centroid memories at phase ends (11/run) + final.
  Q5 provenance: vendored hashes appended to lib/PROVENANCE.md.

Phase 3 owner decision baked in (Q&A 2026-07-14, recorded in PLAN.md):
  Q6 eval sizing: classification at SUP_SIZE/TEST_SIZE (100/100) at all 44
     checkpoints; the O(N^2) clustering eval subsampled to CLUST_SIZE (25)
     per class — see the CLUST_SIZE comment for the exact recipe and the
     measured cost basis. Gates Phase 4: no matrix cell launches without it.

Paper-faithful settings (HANDOFF_PHASE2.md §5/§7): ``pretrained=False`` (the
released True silently SKIPS contrastive T0 training), both epoch knobs set
to the same value (released code mismatches them: model 300 / layer 500;
paper §3.2 says 500), M=30 <= theta=30 (promotion copies M patches of ~theta
stored), CIFAR-100 mean/std, feat_size per arch (their 512 default would
break resnet50, whose backbone emits 2048-d features).
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
CIFAR100_STD = [0.2673, 0.2564, 0.2762]

ARCHS = ("resnet18", "resnet50")
SEEDS = (42, 43, 44)

#: Backbone feature width consumed by their SimCLR head and stm/ltm buffers.
FEAT_SIZE = {"resnet18": 512, "resnet50": 2048}

#: Phase 0.3 geometry decision (their-120): 9 patches per image.
IMG_SIZE = 120
PATCH_SIZE = 60
STRIDE = 30

#: Paper §3.2 epoch count; applied to BOTH mismatched released knobs.
INIT_EPOCHS = 500
SLEEP_EPOCHS = 300

#: Their Eq. 6-7 protocol sizes: labeled (supervise) / test images per class.
SUP_SIZE = 100
TEST_SIZE = 100

#: Their vote-weight threshold (pcmc_layer.py:60 ``rho_task = params.rho``;
#: released pcmc.yaml value). Protocol constant shared by both systems: the
#: layer config below and the F-PCMC scorer read this one name.
RHO = 0.25

#: Phase 3 owner decision Q6 (2026-07-14, PLAN.md): the clustering eval runs
#: on a per-class subsample — the first CLUST_SIZE items of each class's
#: seeded test draw (CIFAR; with TEST_SIZE=100 the draw is all 100 canonical
#: test rows, so this is the 25 lowest row indices per class) and the first
#: CLUST_SIZE stream arrivals per synthetic class. Classification stays
#: SUP_SIZE/TEST_SIZE at every checkpoint. One constant, consumed by BOTH
#: systems (p2_stream.py loaders for PCMC in-run; fpcmc_scorer.py for
#: F-PCMC), so the clustering eval sets stay byte-identical. Measured basis:
#: their cluster() is quadratic python-level jaccard (1.11 us/pair) —
#: 100/class would cost 2.88 h/run in-run vs 0.19 h/run at 25.
CLUST_SIZE = 25

#: sleep_freq sentinel while steering the schedule: large enough that the
#: modulo clause of their trigger (pcmc_layer.py:757) can never fire on its
#: own inside a run, so sleeps happen exactly at the steered ``sleep_start``.
SLEEP_FREQ_SENTINEL = 10**9


def build_run_config(
    arch: str, seed: int, sleep_on: bool = True, smoke: bool = False
) -> dict:
    """The resolved run config as a plain nested dict (YAML/OmegaConf-ready).

    Mirrors the released config tree (top-level keys from config/main.yaml,
    ``model`` from config/model/pcmc.yaml, ``dataset`` ours) so the vendored
    code reads every attribute it expects. Values differing from the released
    pcmc.yaml are paper-faithful settings or P2 plumbing — each is flagged.
    """
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; known: {ARCHS}")
    if seed not in SEEDS:
        raise ValueError(f"seed {seed} not in the T17 matrix {SEEDS}")

    init_epochs = 2 if smoke else INIT_EPOCHS
    config = {
        "seed": int(seed),
        # Logging roots (paths resolve relative to CWD == the cell dir).
        "log": "run",
        "pretrain_log": "run",
        "sleep_log": "run",
        "load_pretrain": False,  # driver flips for the pretrain cache
        "load_sleep": False,
        "plot": False,
        "dataset": {
            "name": "cifar100-p2",
            "img_size": IMG_SIZE,
            "channels": 3,
            "mean": list(CIFAR100_MEAN),
            "std": list(CIFAR100_STD),
            "stream_bs": 1,  # their P2-equivalent streaming granularity
            "sup_size": 4 if smoke else SUP_SIZE,
            "test_size": 4 if smoke else TEST_SIZE,
            # Q6: clustering subsample (strict subset of the test set; the
            # smoke value 2 < 4 exercises the subsampling path end-to-end).
            "clust_size": 2 if smoke else CLUST_SIZE,
        },
        "model": {
            "name": "pcmc",
            "cluster_alg": "kmeans",
            "num_layers": 1,
            "mem_update": "no-reset",
            "update_use": 1,
            "sleep_on": bool(sleep_on),
            "encoder_type": "simclr",
            "arch": arch,
            # Both epoch knobs equal (released mismatch 300/500; paper 500).
            "init_epochs": init_epochs,
            # Placeholder; the driver steers the layer attributes per-step to
            # the phase-midpoint schedule (Q2) and records the step list.
            "sleep_start": SLEEP_FREQ_SENTINEL,
            "sleep_freq": SLEEP_FREQ_SENTINEL,
            "pretrained": False,  # paper-faithful; released True skips T0
            "pretrain_only": False,
            "n_workers": 0,  # driver also force-patches DataLoader; see §5
            "layers": {
                "layer0": {
                    "name": "L0",
                    "ch": 3,
                    "feat_size": FEAT_SIZE[arch],
                    "patch_size": PATCH_SIZE,  # Phase 0.3: their-120
                    "stride": STRIDE,
                    "pretrain_only": 0,
                    "pretrain_bs": 128 if smoke else 256,
                    "init_sample_factor": 1.0,
                    "init_epochs": init_epochs,
                    "init_clusters": 200 if smoke else 500,
                    "lr": 0.6,
                    "wd": 1e-5,
                    "sleep_epochs": 2 if smoke else SLEEP_EPOCHS,
                    "sleep_bs": 512,
                    "alpha": 0.1,
                    "ltm_alpha": 0.0,
                    "beta": 0.99,
                    "theta": 30,
                    "delta": 400,
                    "M": 30,  # M <= theta invariant (pcmc_layer.py:753)
                    "M_min": 3,
                    "init_M": 5,
                    "forgetting_factor": 3,
                    "rho": RHO,
                    "temperature": 0.5,
                    "cj": 0.8,
                    "cj_b": 0.6,
                    "cj_c": 0.6,
                    "cj_s": 0.6,
                    "cj_h": 0.2,
                    "hf": 0.5,
                    "vf": 0.0,
                    "gs": 0.2,
                    "gb": 0.2,
                    "kn": 5,
                    "sigma1": 0.1,
                    "sigma2": 2.0,
                    "rot": 0,
                    "re": 0.0,
                    "crop_min": 0.3,
                    "crop_max": 1.0,
                }
            },
        },
    }
    if smoke:
        # Tiny-budget knobs the driver honours; recorded so the resolved
        # config fully determines the run (resumability contract).
        config["smoke"] = {
            "t0_images": 1024,  # >= pretrain_bs, and >= 2000 patches consumed
            "max_stream_steps": 240,
            "sleep_steps": [150],  # forced mid-smoke sleep (schedule override)
            "checkpoint_steps": [200],  # one in-smoke eval point
        }
    return config


def _check_phases(phases: Sequence) -> None:
    prev_end = None
    for p in phases:
        if p.end <= p.start:
            raise ValueError(f"empty phase {p.name}")
        if prev_end is not None and p.start != prev_end:
            raise ValueError(f"non-contiguous phases at {p.name}")
        prev_end = p.end


def sleep_steps(phases: Sequence) -> list[int]:
    """Q2: one sleep per P2 phase, at the phase midpoint (0-based stream
    index). The driver converts to their layer-step counter (which is
    stream_index + 1 at the post-step trigger check) when steering."""
    _check_phases(phases)
    return [p.start + (p.end - p.start) // 2 for p in phases]


def phase_task_index(phases: Sequence) -> dict[str, int]:
    """Their task index per phase name: T0 is task 0, first phase task 1."""
    return {p.name: i + 1 for i, p in enumerate(phases)}


def eval_label_map(
    t0_classes: Iterable[str], phases: Sequence
) -> dict[str, int]:
    """Q3(a): contiguous eval labels over CIFAR classes ONLY — T0 classes in
    their given (sorted) order, then held-out classes in introduction order.
    Contiguity is load-bearing: their supervise() indexes score columns by
    the raw integer label (pcmc.py / pcmc_layer.py), so labels must be
    0..C-1 over the classes present in the eval set."""
    mapping = {str(c): i for i, c in enumerate(t0_classes)}
    for p in phases:
        if p.group != "heldout":
            continue
        for c in p.introduced_classes:
            if str(c) in mapping:
                raise ValueError(f"class {c} introduced twice")
            mapping[str(c)] = len(mapping)
    return mapping


def cluster_label_map(
    t0_classes: Iterable[str], phases: Sequence
) -> dict[str, int]:
    """Labels for the auxiliary synthetic-inclusive clustering eval (Q3):
    the eval map extended with near/far subclass names in introduction
    order. Used only by the extra model.cluster() call; purity handles any
    contiguous labeling."""
    mapping = eval_label_map(t0_classes, phases)
    for p in phases:
        if p.group not in ("near", "far"):
            continue
        for c in p.introduced_classes:
            if str(c) in mapping:
                raise ValueError(f"class {c} introduced twice")
            mapping[str(c)] = len(mapping)
    return mapping


def eval_classes_at(
    t0_classes: Sequence[str], phases: Sequence, task: int
) -> tuple[list[str], list[str]]:
    """(cifar_eval_classes, synthetic_cluster_classes) visible at their task
    index ``task`` (0 = after T0 pretrain, before the stream), each in label
    order — the eval loaders enumerate classes in exactly this order."""
    cifar = [str(c) for c in t0_classes]
    synth: list[str] = []
    for i, p in enumerate(phases):
        if i + 1 > task:
            break
        if p.group == "heldout":
            cifar.extend(str(c) for c in p.introduced_classes)
        else:
            synth.extend(str(c) for c in p.introduced_classes)
    return cifar, synth


def snapshot_steps(phases: Sequence) -> list[int]:
    """Q4: model/centroid snapshot points = phase-end stream indices (the
    last checkpoint of each phase; 11 per P2 run)."""
    _check_phases(phases)
    return [p.end - 1 for p in phases]
