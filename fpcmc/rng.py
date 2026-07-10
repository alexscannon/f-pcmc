"""Single source of randomness for all of F-PCMC.

Every module obtains its np.random.Generator from make_rng(), seeded from the
run config. No module-level RNG state, no np.random.* legacy calls, no
unseeded library randomness anywhere (CLAUDE.md determinism rule; NFR / FR-9.2
byte-determinism depends on this).
"""

import numpy as np


def make_rng(seed: int, stream: str = "") -> np.random.Generator:
    """Deterministic generator for `seed`, optionally on a named substream.

    Distinct `stream` names yield statistically independent generators from
    the same run seed (e.g. one per module: "reservoir", "protocol"), so
    consumption order in one component can never perturb another.
    """
    entropy = [int(seed)]
    if stream:
        entropy.extend(stream.encode("utf-8"))
    return np.random.default_rng(np.random.SeedSequence(entropy))
