"""T1 [U] test for the frozen golden stream (TASKS.md Task 1).

test_golden_stream_frozen — hash of the committed .npz matches the pinned
constant, the committed contents match a fresh regeneration from seed, and the
stream's structure matches the TASKS T1 golden spec (novel classes recurring
across >= 4 windows; one outlier-burst class in a single contiguous 15-example
run; the approved distractor outliers present as one-off singletons).
"""

import hashlib

import numpy as np

from tests.fixtures import golden_stream as gs


def test_golden_stream_frozen():
    # 1. Byte-stability: the committed artifact is what was pinned.
    blob = gs.GOLDEN_NPZ_PATH.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == gs.GOLDEN_NPZ_SHA256, (
        "committed golden_stream.npz does not match its pinned sha256 — the "
        "golden stream is frozen; regenerate ONLY with owner approval"
    )

    # 2. Content determinism: regenerating from seed reproduces the committed
    # arrays. (Compared with a tiny float tolerance, not byte equality, so a
    # different BLAS build cannot fail this leg; byte identity is leg 1's job.)
    committed = dict(np.load(gs.GOLDEN_NPZ_PATH, allow_pickle=False))
    regenerated = gs.build_golden()
    assert set(committed) == set(regenerated)
    for key, fresh in regenerated.items():
        stored = committed[key]
        assert stored.shape == np.asarray(fresh).shape, key
        if np.issubdtype(stored.dtype, np.floating):
            np.testing.assert_allclose(stored, fresh, rtol=0, atol=1e-12, err_msg=key)
        else:
            assert np.array_equal(stored, fresh), key

    # 3. Structure matches the TASKS T1 golden spec.
    labels = committed["stream_labels"]
    windows = committed["stream_window_ids"]
    assert labels.shape == (gs.N_STEPS,)
    assert committed["stream_x"].shape == (gs.N_STEPS, gs.D)
    assert gs.N_STEPS == gs.N_WINDOWS * gs.WINDOW_W

    # Every novel class recurs across >= 4 distinct windows.
    for name in [f"novel_{i:02d}" for i in range(gs.K_NOVEL)]:
        assert len(set(windows[labels == name])) >= 4, name

    # The burst class appears exactly once, as a single contiguous 15-run.
    burst_steps = np.flatnonzero(labels == "burst_00")
    assert len(burst_steps) == gs.BURST_LEN == 15
    assert np.array_equal(burst_steps, np.arange(burst_steps[0], burst_steps[0] + 15))

    # Distractors: 25 one-off outliers, each a distinct singleton label,
    # all after the burst (approved deviation: they guarantee STM eviction
    # pressure for T8/T11's burst-eviction assertions).
    distractor_mask = np.char.startswith(labels, "distractor_")
    assert distractor_mask.sum() == gs.N_DISTRACTORS == 25
    d_labels, d_counts = np.unique(labels[distractor_mask], return_counts=True)
    assert len(d_labels) == 25 and (d_counts == 1).all()
    assert np.flatnonzero(distractor_mask).min() > burst_steps.max()

    # Known classes are present in every window (expanding-accuracy support).
    for name in [f"known_{i:02d}" for i in range(gs.K_KNOWN)]:
        assert len(set(windows[labels == name])) == gs.N_WINDOWS, name

    # Init pools are frozen alongside the stream (T11's LTM init must be
    # byte-stable too) and are unit-norm like everything else.
    assert committed["t0_x"].shape == (gs.K_KNOWN * gs.T0_PER_CLASS, gs.D)
    assert committed["test_x"].shape == (gs.K_KNOWN * gs.TEST_PER_CLASS, gs.D)
    for key in ("stream_x", "t0_x", "test_x", "true_means"):
        np.testing.assert_allclose(
            np.linalg.norm(committed[key], axis=1), 1.0, atol=1e-6, err_msg=key
        )
