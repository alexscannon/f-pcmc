"""T1 [U] tests for the synthetic vMF fixture world (TASKS.md Task 1).

test_fixture_determinism — same seed => identical arrays; different seed => different.
test_fixture_separations — sampled class means match requested angular separations
within tolerance; per-class sample mean direction within 5 degrees of true mean
for n=200.
"""

import numpy as np

from tests.fixtures.vmf_world import Segment, VMFWorld


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def test_fixture_determinism():
    kwargs = dict(k_known=4, k_novel=2, n_burst=1, separation_deg=70.0)
    wa = VMFWorld(seed=11, **kwargs)
    wb = VMFWorld(seed=11, **kwargs)
    wc = VMFWorld(seed=12, **kwargs)

    # Pools: bit-identical across same-seed worlds.
    pa, pb = wa.t0_pool(50), wb.t0_pool(50)
    assert np.array_equal(pa.x, pb.x)
    assert np.array_equal(pa.labels, pb.labels)

    # Streams (shuffled + contiguous segments + distractors): bit-identical.
    schedule = [
        Segment(counts={"known_00": 10, "novel_01": 5}),
        Segment(counts={"burst_00": 7}, shuffle=False),
        Segment(counts={"known_01": 8, "known_02": 4}, distractors=(0, 1)),
    ]
    sa, sb = wa.make_stream(schedule), wb.make_stream(schedule)
    assert np.array_equal(sa.x, sb.x)
    assert np.array_equal(sa.labels, sb.labels)
    assert np.array_equal(sa.segment_ids, sb.segment_ids)

    # Different seed => different draws (means and samples).
    pc = wc.t0_pool(50)
    assert not np.array_equal(pa.x, pc.x)
    sc = wc.make_stream(schedule)
    assert not np.array_equal(sa.x, sc.x)

    # Repeated calls on the SAME world instance are also reproducible
    # (sampling is a pure function of (seed, stream label, class)).
    assert np.array_equal(wa.t0_pool(50).x, pa.x)
    assert np.array_equal(wa.make_stream(schedule).x, sa.x)


def test_fixture_separations():
    sep = 60.0
    world = VMFWorld(seed=3, k_known=5, k_novel=0, separation_deg=sep, kappa_known=200.0)

    # Construction is exact: true means sit at the requested pairwise angle.
    true_angles = world.true_pairwise_angles_deg()
    off_diag = true_angles[~np.eye(len(world.class_names), dtype=bool)]
    np.testing.assert_allclose(off_diag, sep, atol=1e-6)

    # Sampled per-class mean directions: within 5 degrees of the true mean
    # (n=200 per class, kappa=200), and their pairwise angles match the
    # requested separation within tolerance.
    pool = world.t0_pool(200)
    sample_means = {}
    for name in world.known_names:
        cls_x = pool.x[pool.labels == name]
        assert cls_x.shape[0] == 200
        m = cls_x.mean(axis=0)
        m = m / np.linalg.norm(m)
        sample_means[name] = m
        assert _angle_deg(m, world.true_mean(name)) < 5.0

    names = world.known_names
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            emp = _angle_deg(sample_means[names[i]], sample_means[names[j]])
            assert abs(emp - sep) < 5.0, (names[i], names[j], emp)
