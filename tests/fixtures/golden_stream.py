"""The frozen golden stream (T1) — input data for the T11 golden gate.

One frozen configuration of the vMF world, serialized to the committed
`golden_stream.npz` (sha256 pinned below) so the golden stream is byte-stable
across machines. NEVER regenerate the .npz casually: T8 and T11 assert against
this exact data, and re-pinning the hash is an owner-approval event.

World (seed=7, D=32, TASKS T1):
  - 8 known classes ("known_00".."known_07"), kappa=150, equiangular 75 deg.
  - 3 recurring novel classes ("novel_00".."novel_02"), kappa=150.
  - 1 outlier-burst class ("burst_00"), kappa=500 (a tight knot of
    near-duplicates), appearing ONLY as a single contiguous 15-example run.
  - 25 one-off distractor outliers ("distractor_00".."distractor_24"), each a
    single isolated point appearing once, all after the burst. Approved
    deviation (owner, 2026-07-10): they are not in TASKS T1's enumerated
    composition; they guarantee STM eviction pressure so T8/T11's
    burst-LRU-eviction assertions are reachable by design under a golden-run
    config with stm_capacity <= ~25, independent of emergent
    threshold-tail behavior.

Stream (2,000 steps = 8 windows of window_W=250, matching the PRD §8 default;
T8 runs the burst scenario "through 2,000 steps"):

  window | known | novel_00 | novel_01 | novel_02 | burst | distractors
  -------+-------+----------+----------+----------+-------+------------
     0   |  250  |    -     |    -     |    -     |   -   |     -
     1   |  225  |    25    |    -     |    -     |   -   |     -
     2   |  200  |    25    |    25    |    -     |   -   |     -
     3   |  160  |    25    |    25    |    25    |  15   |     -
     4   |  193  |    -     |    25    |    25    |   -   |     7
     5   |  194  |    25    |    -     |    25    |   -   |     6
     6   |  219  |    -     |    25    |    -     |   -   |     6
     7   |  194  |    25    |    -     |    25    |   -   |     6

  - Every known class appears in every window (expanding accuracy is
    measurable throughout; known counts are split evenly across the 8 classes,
    remainder to the lowest-numbered).
  - Each novel class recurs across >= 4 distinct windows (novel_00: 5), with
    25 examples per appearance — promotion (theta=30, m_windows=3) becomes
    feasible on a class's 3rd window, and every novel class has >= 1 full
    post-promotion-feasible window left (T11's promotion-aware-routing
    assertion needs post-promotion arrivals).
  - The burst is laid down contiguously mid-window-3 (an unshuffled segment
    between two shuffled window halves).

The .npz also freezes the T0 init pool (200/known class — T11's LTM init must
be byte-stable too), a held-out IND test pool (50/known class), the true class
means/kappas, and per-step window/segment metadata.

Regenerate (owner approval only): `uv run python -m tests.fixtures.golden_stream`
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from tests.fixtures.vmf_world import Segment, VMFWorld

# ----------------------------------------------------------------- frozen spec
GOLDEN_SEED = 7
D = 32
K_KNOWN = 8
K_NOVEL = 3
N_BURST_CLASSES = 1
SEPARATION_DEG = 75.0
KAPPA_KNOWN = 150.0
KAPPA_NOVEL = 150.0
KAPPA_BURST = 500.0

WINDOW_W = 250  # matches PRD §8 default window_W
N_WINDOWS = 8
N_STEPS = N_WINDOWS * WINDOW_W  # 2,000 (T8 burst scenario length)
BURST_LEN = 15
N_DISTRACTORS = 25
T0_PER_CLASS = 200
TEST_PER_CLASS = 50

GOLDEN_NPZ_PATH = Path(__file__).resolve().parent / "golden_stream.npz"

# sha256 of the committed golden_stream.npz. Pinned once at T1; changing it
# requires owner approval (it invalidates the T8/T11 gates' input data).
GOLDEN_NPZ_SHA256 = "4c3670a0d0833f09b7ad17443e1a44126e1a3d3cc1edff6d28dd58bf1ce5c2fa"

# Per-window novel appearances and distractor allotments (see module docstring).
_NOVEL_WINDOWS = {
    "novel_00": (1, 2, 3, 5, 7),
    "novel_01": (2, 3, 4, 6),
    "novel_02": (3, 4, 5, 7),
}
_NOVEL_PER_APPEARANCE = 25
_BURST_WINDOW = 3
_DISTRACTOR_WINDOWS = {4: (0, 7), 5: (7, 13), 6: (13, 19), 7: (19, 25)}  # window -> id range


def make_golden_world() -> VMFWorld:
    return VMFWorld(
        seed=GOLDEN_SEED,
        k_known=K_KNOWN,
        k_novel=K_NOVEL,
        n_burst=N_BURST_CLASSES,
        d=D,
        separation_deg=SEPARATION_DEG,
        kappa_known=KAPPA_KNOWN,
        kappa_novel=KAPPA_NOVEL,
        kappa_burst=KAPPA_BURST,
    )


def _split_known(n: int) -> dict[str, int]:
    """Split n known-class examples evenly across the 8 known classes."""
    base, rem = divmod(n, K_KNOWN)
    return {f"known_{i:02d}": base + (1 if i < rem else 0) for i in range(K_KNOWN)}


def golden_schedule() -> list[Segment]:
    """The frozen 2,000-step schedule; every window sums to exactly WINDOW_W."""
    schedule: list[Segment] = []
    for w in range(N_WINDOWS):
        novel = {name: _NOVEL_PER_APPEARANCE for name, ws in _NOVEL_WINDOWS.items() if w in ws}
        lo, hi = _DISTRACTOR_WINDOWS.get(w, (0, 0))
        distractors = tuple(range(lo, hi))
        n_known = WINDOW_W - sum(novel.values()) - (hi - lo) - (BURST_LEN if w == _BURST_WINDOW else 0)

        if w == _BURST_WINDOW:
            # Contiguous burst mid-window, between two shuffled halves.
            first = Segment(counts={**_split_known(n_known - n_known // 2),
                                    **{k: v - v // 2 for k, v in novel.items()}})
            burst = Segment(counts={"burst_00": BURST_LEN}, shuffle=False)
            second = Segment(counts={**_split_known(n_known // 2),
                                     **{k: v // 2 for k, v in novel.items()}})
            schedule.extend([first, burst, second])
        else:
            schedule.append(Segment(counts={**_split_known(n_known), **novel},
                                    distractors=distractors))
    return schedule


def build_golden() -> dict[str, np.ndarray]:
    """All golden arrays, regenerated deterministically from GOLDEN_SEED."""
    world = make_golden_world()
    stream = world.make_stream(golden_schedule())
    assert stream.x.shape == (N_STEPS, D), stream.x.shape

    t0 = world.t0_pool(T0_PER_CLASS)
    test = world.ind_test_pool(TEST_PER_CLASS)
    names = world.true_mean_class_names()
    return {
        "stream_x": stream.x,
        "stream_labels": stream.labels,
        "stream_segment_ids": stream.segment_ids,
        "stream_window_ids": np.arange(N_STEPS, dtype=np.int64) // WINDOW_W,
        "t0_x": t0.x,
        "t0_labels": t0.labels,
        "test_x": test.x,
        "test_labels": test.labels,
        "true_means": world.true_means(),
        "true_mean_class_names": np.array(names),
        "true_kappas": np.array([world.kappa(n) for n in names]),
        "meta": np.array([GOLDEN_SEED, D, WINDOW_W, N_STEPS], dtype=np.int64),
        "separation_deg": np.array([SEPARATION_DEG]),
    }


def load_golden() -> dict[str, np.ndarray]:
    """The committed golden arrays (the artifact T8/T11 must consume)."""
    with np.load(GOLDEN_NPZ_PATH, allow_pickle=False) as z:
        return dict(z)


def _main() -> None:
    data = build_golden()
    np.savez(GOLDEN_NPZ_PATH, **data)
    digest = hashlib.sha256(GOLDEN_NPZ_PATH.read_bytes()).hexdigest()
    print(f"wrote {GOLDEN_NPZ_PATH} ({GOLDEN_NPZ_PATH.stat().st_size} bytes)")
    print(f"sha256 = {digest}")


if __name__ == "__main__":
    _main()
