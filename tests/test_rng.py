"""T0 determinism test for the single RNG factory (TASKS.md Task 0)."""

import numpy as np

from fpcmc.rng import make_rng


def test_rng_determinism():
    # Same seed -> identical 1,000-draw sequences.
    a = make_rng(42).random(1000)
    b = make_rng(42).random(1000)
    assert np.array_equal(a, b)

    # Different seeds differ.
    c = make_rng(43).random(1000)
    assert not np.array_equal(a, c)

    # Named substreams: reproducible, distinct from the base stream and
    # from each other (later tasks give each module its own substream).
    r1 = make_rng(42, "reservoir").random(1000)
    r2 = make_rng(42, "reservoir").random(1000)
    assert np.array_equal(r1, r2)
    assert not np.array_equal(a, r1)
    assert not np.array_equal(r1, make_rng(42, "protocol").random(1000))
