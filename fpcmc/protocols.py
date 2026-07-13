"""Stream protocols P1 and P2 (T12; PRD §7.1).

``build_p1`` reproduces the v1 stream construction exactly (index-level, at
the same seed — verified against the archived v1 seed-42 stream, docs/ASSETS.md
§7.2); ``build_p2`` builds the phased O-UCL stream. Both emit a
``StreamManifest`` (per-index: pool, within-pool index, class, superclass,
phase) plus a phase table and checkpoint steps, wrapped in a
``ProtocolStream``.

T12 decisions (owner-approved 2026-07-13, this session; recorded in
docs/CHANGES.md T12):

  - **Manifest is consumed eval-side; StreamRunner is untouched.** The runner
    consumes a protocol through its existing surface — ``run(protocol.x)`` plus
    the ``checkpoint_steps`` constructor parameter it already has. Ground-truth
    class labels live in the manifest only and never enter the runtime loop.
  - **P1's warmup is a plain labeled prefix.** In v1 the 1,000-example warmup
    fit the IND model; in F-PCMC T0 LTM init comes from ind_reference, so the
    warmup steps are ordinary routed steps, marked phase="warmup" for eval
    stratification. No mechanism anywhere keys on it.
  - **P2 phase membership is deterministic and seed-independent.** Held-out
    and near classes are chunked into phases in sorted-name order; the far
    superclasses are greedy-packed into exactly ``N_FAR_PHASES`` phases,
    balancing example counts (largest first, ties by name, into the currently
    smallest phase). Only within-phase ordering and the interleave draw vary
    with the seed, so seed variance measures ordering noise, not schedule
    noise.
  - **"Phases of equal length" holds within each pool group, on full pools.**
    Global equal length is impossible without subsampling everything to the
    near-OOD bottleneck (~250 novel examples/phase, discarding ~92% of the
    held-out CIFAR data); instead each phase uses all examples of its classes,
    so phases are equal within the held-out group and as equal as the
    superclass packing allows within the far group.
  - **The 30% past-class interleave draws from T0-class IND-test rows only**
    (never previously-introduced held-out classes — forced by the
    zero-occurrences-after-phase requirement), without replacement across the
    whole stream.
  - **Protocol constants live here, not in FPCMCConfig.** PRD §8 fixes the
    config schema (unknown keys are errors), and none of its keys describe
    protocol composition. ``config`` is accepted per the TASKS-stated builder
    signatures but does not parameterize composition.

RNG: ``build_p1`` uses ``make_rng(seed)`` with the UNNAMED substream — verified
bit-identical to v1's ``np.random.default_rng(seed)``, which is what makes
index-level parity possible without violating the determinism rule.
``build_p2`` uses the named substream "protocol/p2".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np
import yaml

from fpcmc.config import FPCMCConfig
from fpcmc.rng import make_rng

REPO_ROOT = Path(__file__).resolve().parents[1]
P2_CLASS_SPLIT_PATH = REPO_ROOT / "configs" / "p2_class_split.yaml"

# Protocol constants (PRD §7.1 literals; see module docstring on why these are
# not FPCMCConfig keys).
IND_WARMUP_COUNT = 1000
PAST_INTERLEAVE_FRAC = 0.30
CHECKPOINTS_PER_PHASE = 4
HELDOUT_CLASSES_PER_PHASE = 5
NEAR_CLASSES_PER_PHASE = 3
N_FAR_PHASES = 5

# v1 pool assembly order (docs/ASSETS.md §7.2), in this repo's pool names.
P1_POOL_ORDER = ("ind_test", "synthetic_ind", "near_ood", "far_ood")
P2_POOLS = ("ind_reference", "ind_test", "near_ood", "far_ood")


class ProtocolError(ValueError):
    """A stream protocol cannot be built from the given pools/split."""


class PoolLike(Protocol):
    """What the builders need from a pool (fpcmc.data.Pool satisfies it)."""

    x: np.ndarray
    subclass_names: np.ndarray
    superclass_names: np.ndarray


@dataclass(frozen=True)
class StreamManifest:
    """Per-index ground truth for one built stream (eval-side consumer).

    Parallel arrays over stream steps: which pool the embedding came from,
    its row index within that pool (as loaded by fpcmc.data — for ind_test
    that is the ascending cifar100_test subset order, matching the archived
    v1 stream's contract), its class/superclass name strings, and the phase
    label of the step.
    """

    pool: np.ndarray  # (N,) <U pool name (fpcmc.data.POOL_SPECS names)
    within_pool_index: np.ndarray  # (N,) int64
    true_class: np.ndarray  # (N,) <U subclass name
    true_superclass: np.ndarray  # (N,) <U superclass name
    phase: np.ndarray  # (N,) <U phase name

    def __len__(self) -> int:
        return int(self.pool.shape[0])


@dataclass(frozen=True)
class PhaseInfo:
    """One phase: [start, end) steps and the classes it introduces.

    ``introduced_classes`` is empty for P1's warmup/stream pseudo-phases and
    never contains T0 classes (interleave rows are not introductions).
    """

    name: str
    group: str  # "warmup" | "stream" | "heldout" | "near" | "far"
    start: int
    end: int
    introduced_classes: tuple[str, ...]


@dataclass(frozen=True)
class ProtocolStream:
    """A built protocol: embeddings in stream order + manifest + schedule.

    The runner consumes ``x`` (and ``checkpoint_steps`` at construction); the
    manifest and phase table are for the eval harness only.
    """

    x: np.ndarray  # (N, D) stream-ordered embeddings
    manifest: StreamManifest
    phases: tuple[PhaseInfo, ...]
    checkpoint_steps: tuple[int, ...]
    t0_classes: tuple[str, ...]


@dataclass(frozen=True)
class P2ClassSplit:
    """The 80/20 T0/held-out subclass partition consumed by build_p2."""

    t0: tuple[str, ...]
    held_out: tuple[str, ...]


def load_p2_class_split(path: Path | None = None) -> P2ClassSplit:
    """Load the frozen, human-decided split (configs/p2_class_split.yaml).

    The list is consumed verbatim — never redrawn (CLAUDE.md source-of-truth
    #5). The file's own sanity-check intent is enforced here: the protected
    superclasses' subclasses must not appear in the held-out list.
    """
    path = P2_CLASS_SPLIT_PATH if path is None else Path(path)
    data = yaml.safe_load(path.read_text())
    held_out = tuple(data["held_out_20"])
    t0 = tuple(data["t0_80"])
    if len(held_out) != 20 or len(set(held_out)) != 20:
        raise ProtocolError(f"{path}: held_out_20 must be 20 distinct classes")
    if len(t0) != 80 or len(set(t0)) != 80:
        raise ProtocolError(f"{path}: t0_80 must be 80 distinct classes")
    if set(t0) & set(held_out):
        raise ProtocolError(f"{path}: t0_80 and held_out_20 overlap")
    protected = {
        sub for subs in data["protected_superclass_subclasses"].values() for sub in subs
    }
    leaked = protected & set(held_out)
    if leaked:
        raise ProtocolError(f"{path}: protected subclasses in held_out_20: {sorted(leaked)}")
    return P2ClassSplit(t0=t0, held_out=held_out)


# ------------------------------------------------------------------ assembly
def _gather(
    pools: Mapping[str, PoolLike],
    rows: Sequence[tuple[str, int]],
    phase_labels: Sequence[str],
) -> tuple[np.ndarray, StreamManifest]:
    """Materialize (pool, index) rows into embeddings + manifest arrays."""
    pool_names = np.array([r[0] for r in rows])
    wpi = np.array([r[1] for r in rows], dtype=np.int64)
    n = len(rows)
    d = next(iter(pools.values())).x.shape[1]
    dtype = np.result_type(*[p.x.dtype for p in pools.values()])
    x = np.empty((n, d), dtype=dtype)
    true_class = np.empty(n, dtype=object)
    true_super = np.empty(n, dtype=object)
    for name in np.unique(pool_names):
        mask = pool_names == name
        pool = pools[str(name)]
        idx = wpi[mask]
        x[mask] = pool.x[idx]
        true_class[mask] = pool.subclass_names[idx]
        true_super[mask] = pool.superclass_names[idx]
    manifest = StreamManifest(
        pool=pool_names.astype(str),
        within_pool_index=wpi,
        true_class=true_class.astype(str),
        true_superclass=true_super.astype(str),
        phase=np.asarray(phase_labels, dtype=str),
    )
    return x, manifest


def _require_pools(pools: Mapping[str, PoolLike], needed: Sequence[str], what: str) -> None:
    missing = [name for name in needed if name not in pools]
    if missing:
        raise ProtocolError(f"{what}: missing pools {missing}")


def _default_pools() -> Mapping[str, PoolLike]:
    from fpcmc.data import load_all_pools  # deferred: touches roots.env/real data

    return load_all_pools()


# ------------------------------------------------------------------------ P1
def build_p1(
    config: FPCMCConfig,
    seed: int,
    pools: Mapping[str, PoolLike] | None = None,
    *,
    ind_warmup_count: int = IND_WARMUP_COUNT,
) -> ProtocolStream:
    """P1 — the v1-compatibility stream, reproduced index-for-index.

    Exact v1 algorithm (docs/ASSETS.md §7.2): pools assembled in fixed order
    (ind_test ascending, synthetic IND, near-OOD, far-OOD) → one
    ``rng.permutation`` over the ind_test pool, first ``ind_warmup_count`` =
    warmup, rest = leftover → leftover concatenated with the other three pools
    → ONE ``rng.permutation`` over that concatenation → warmup + shuffled
    remainder. Concatenate-then-shuffle-once; the warmup is disjoint from the
    interleave. T0 = all classes (LTM init from ind_reference happens
    elsewhere; the warmup is a plain labeled prefix here).
    """
    if pools is None:
        pools = _default_pools()
    _require_pools(pools, P1_POOL_ORDER, "build_p1")
    n_ind = int(pools["ind_test"].x.shape[0])
    if not 0 < ind_warmup_count < n_ind:
        raise ProtocolError(
            f"build_p1: ind_warmup_count={ind_warmup_count} must be in (0, {n_ind})"
        )

    # Unnamed substream: bit-identical to v1's np.random.default_rng(seed).
    rng = make_rng(seed)
    perm = rng.permutation(n_ind)
    warmup = perm[:ind_warmup_count]
    leftover = perm[ind_warmup_count:]
    remainder: list[tuple[str, int]] = [("ind_test", int(i)) for i in leftover]
    for name in P1_POOL_ORDER[1:]:
        remainder.extend((name, i) for i in range(int(pools[name].x.shape[0])))
    shuffled = rng.permutation(len(remainder))
    rows = [("ind_test", int(i)) for i in warmup] + [remainder[j] for j in shuffled]

    n = len(rows)
    labels = ["warmup"] * ind_warmup_count + ["stream"] * (n - ind_warmup_count)
    x, manifest = _gather(pools, rows, labels)
    phases = (
        PhaseInfo("warmup", "warmup", 0, ind_warmup_count, ()),
        PhaseInfo("stream", "stream", ind_warmup_count, n, ()),
    )
    return ProtocolStream(
        x=x,
        manifest=manifest,
        phases=phases,
        checkpoint_steps=(),  # PRD §7.1 specifies checkpoints for P2 only
        t0_classes=tuple(sorted(set(pools["ind_test"].subclass_names.tolist()))),
    )


# ------------------------------------------------------------------------ P2
def _greedy_pack_superclasses(
    superclass_names: np.ndarray, n_phases: int
) -> list[tuple[str, ...]]:
    """Pack superclasses into exactly n_phases groups, balancing example
    counts: largest first (ties by name), each into the currently smallest
    group (ties to the lowest index). Deterministic, seed-independent."""
    names, counts = np.unique(superclass_names, return_counts=True)
    count_of = dict(zip(names.tolist(), counts.tolist()))
    if len(count_of) < n_phases:
        raise ProtocolError(
            f"cannot form {n_phases} far phases from {len(count_of)} superclasses"
        )
    ordered = sorted(count_of, key=lambda s: (-count_of[s], s))
    totals = [0] * n_phases
    groups: list[list[str]] = [[] for _ in range(n_phases)]
    for sc in ordered:
        i = min(range(n_phases), key=lambda j: (totals[j], j))
        totals[i] += count_of[sc]
        groups[i].append(sc)
    return [tuple(sorted(g)) for g in groups]


def _chunk(items: Sequence[str], size: int) -> list[tuple[str, ...]]:
    if size < 1:
        raise ProtocolError(f"chunk size must be >= 1, got {size}")
    return [tuple(items[i : i + size]) for i in range(0, len(items), size)]


def build_p2(
    config: FPCMCConfig,
    seed: int,
    pools: Mapping[str, PoolLike] | None = None,
    class_split: P2ClassSplit | None = None,
    *,
    heldout_per_phase: int = HELDOUT_CLASSES_PER_PHASE,
    near_per_phase: int = NEAR_CLASSES_PER_PHASE,
    n_far_phases: int = N_FAR_PHASES,
) -> ProtocolStream:
    """P2 — the phased O-UCL stream (the paper's primary protocol).

    T0 = the split's t0 classes (LTM init from their ind_reference rows
    happens elsewhere). Phases introduce, in order: the held-out CIFAR classes
    (sorted-name chunks of ``heldout_per_phase``, each class contributing all
    of its ind_reference + ind_test rows — "drawn from their train+test
    splits"), then the near-OOD classes (sorted-name chunks), then the far-OOD
    classes grouped by superclass into exactly ``n_far_phases`` count-balanced
    phases. Introduced classes cease after their phase. Each phase interleaves
    30% T0-class ind_test examples (drawn without replacement across the whole
    stream); ``CHECKPOINTS_PER_PHASE`` checkpoints sit at the 1/4..4/4 points
    of every phase.
    """
    if pools is None:
        pools = _default_pools()
    _require_pools(pools, P2_POOLS, "build_p2")
    split = load_p2_class_split() if class_split is None else class_split

    ref, test = pools["ind_reference"], pools["ind_test"]
    pool_classes = set(ref.subclass_names.tolist())
    declared = set(split.t0) | set(split.held_out)
    if set(split.t0) & set(split.held_out):
        raise ProtocolError("build_p2: t0 and held_out overlap")
    if declared != pool_classes:
        raise ProtocolError(
            "build_p2: class split does not partition the ind_reference classes "
            f"(missing={sorted(pool_classes - declared)}, "
            f"extra={sorted(declared - pool_classes)})"
        )

    rng = make_rng(seed, "protocol/p2")

    # Interleave supply: T0-class ind_test rows, consumed without replacement.
    t0_sorted = sorted(split.t0)
    remaining = np.flatnonzero(np.isin(test.subclass_names, t0_sorted))

    rows: list[tuple[str, int]] = []
    labels: list[str] = []
    phases: list[PhaseInfo] = []
    checkpoints: list[int] = []

    def add_phase(
        group: str, index: int, novel_rows: list[tuple[str, int]], introduced: tuple[str, ...]
    ) -> None:
        nonlocal remaining
        n_novel = len(novel_rows)
        if n_novel == 0:
            raise ProtocolError(f"build_p2: phase {group}_{index:02d} has no novel examples")
        n_inter = int(round(n_novel * PAST_INTERLEAVE_FRAC / (1.0 - PAST_INTERLEAVE_FRAC)))
        if n_inter > remaining.size:
            raise ProtocolError(
                f"build_p2: T0 ind_test interleave supply exhausted at "
                f"{group}_{index:02d} (need {n_inter}, have {remaining.size})"
            )
        picked = rng.choice(remaining.size, size=n_inter, replace=False)
        inter_rows = [("ind_test", int(i)) for i in remaining[picked]]
        remaining = np.delete(remaining, picked)

        items = novel_rows + inter_rows
        order = rng.permutation(len(items))
        name = f"{group}_{index:02d}"
        start = len(rows)
        rows.extend(items[j] for j in order)
        end = len(rows)
        labels.extend([name] * (end - start))
        length = end - start
        if length < CHECKPOINTS_PER_PHASE:
            raise ProtocolError(f"build_p2: phase {name} too short for checkpoints ({length})")
        phases.append(PhaseInfo(name, group, start, end, introduced))
        checkpoints.extend(
            start + (length * k) // CHECKPOINTS_PER_PHASE - 1
            for k in range(1, CHECKPOINTS_PER_PHASE + 1)
        )

    # Held-out CIFAR phases: all train (ind_reference) + test (ind_test) rows.
    ref_idx = {c: np.flatnonzero(ref.subclass_names == c) for c in split.held_out}
    test_idx = {c: np.flatnonzero(test.subclass_names == c) for c in split.held_out}
    empty = sorted(c for c in split.held_out if ref_idx[c].size == 0 or test_idx[c].size == 0)
    if empty:
        raise ProtocolError(f"build_p2: held-out classes without pool rows: {empty}")
    for i, chunk in enumerate(_chunk(sorted(split.held_out), heldout_per_phase)):
        novel = [
            (pool_name, int(j))
            for c in chunk
            for pool_name, idx in (("ind_reference", ref_idx[c]), ("ind_test", test_idx[c]))
            for j in idx
        ]
        add_phase("heldout", i, novel, chunk)

    # Near-OOD phases.
    near = pools["near_ood"]
    near_classes = sorted(set(near.subclass_names.tolist()))
    for i, chunk in enumerate(_chunk(near_classes, near_per_phase)):
        idx = np.flatnonzero(np.isin(near.subclass_names, chunk))
        add_phase("near", i, [("near_ood", int(j)) for j in idx], chunk)

    # Far-OOD phases: superclasses packed into exactly n_far_phases groups.
    far = pools["far_ood"]
    for i, group in enumerate(_greedy_pack_superclasses(far.superclass_names, n_far_phases)):
        idx = np.flatnonzero(np.isin(far.superclass_names, group))
        introduced = tuple(sorted(set(far.subclass_names[idx].tolist())))
        add_phase("far", i, [("far_ood", int(j)) for j in idx], introduced)

    x, manifest = _gather(pools, rows, labels)
    return ProtocolStream(
        x=x,
        manifest=manifest,
        phases=tuple(phases),
        checkpoint_steps=tuple(checkpoints),
        t0_classes=tuple(t0_sorted),
    )
