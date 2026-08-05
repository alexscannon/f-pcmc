"""T17 Phase 1: pixel-mirror alignment tests (baselines/pcmc_sleep/PLAN.md).

[I] integration tests — they need roots.env, the real embedding pools (the
manifest is built from them) and the raw image trees; skipped with a clear
message otherwise. The mirror must replay the embedding-space P2 stream in
pixel space EXACTLY: same manifest, and every stream index resolving to a
source image whose independently-derived label matches the manifest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

SEED = 42


@pytest.fixture(scope="module")
def mirror():
    from fpcmc.data import embeddings_available, load_all_pools
    from fpcmc.config import FPCMCConfig
    from fpcmc.protocols import build_p2
    from baselines.pcmc_sleep.stream_mirror import P2PixelMirror, _data_root

    ok, reason = embeddings_available()
    if not ok:
        pytest.skip(reason)
    root = _data_root()
    for sub in ("cifar100/cifar-100-python", "ms_cifar100_genai_novel_32x32"):
        if not (root / sub).is_dir():
            pytest.skip(f"raw image tree missing: {root / sub}")

    config = FPCMCConfig.from_yaml(Path("configs") / "fpcmc_default.yaml")
    pools = load_all_pools()
    stream = build_p2(config, SEED, pools)
    return P2PixelMirror(stream, pools)


def test_mirror_full_alignment(mirror):
    """Every stream index resolves to a source whose independently-derived
    class equals the manifest's true_class; every file source exists."""
    n = len(mirror)
    assert n == len(mirror.manifest)
    mismatches = []
    missing = []
    for i in range(n):
        kind, ref = mirror.source_ref(i)
        if kind == "file" and not Path(ref).is_file():
            missing.append((i, ref))
            continue
        if mirror.true_class_from_source(i) != str(mirror.manifest.true_class[i]):
            mismatches.append(i)
    assert not missing, f"{len(missing)} missing source files, first: {missing[:3]}"
    assert not mismatches, f"{len(mismatches)} label mismatches, first: {mismatches[:5]}"


def test_mirror_images_decode(mirror):
    """Spot-decode one image per pool: (32, 32, 3) uint8."""
    pools_seen = {}
    for i in range(len(mirror)):
        p = str(mirror.manifest.pool[i])
        if p not in pools_seen:
            pools_seen[p] = i
    for p, i in sorted(pools_seen.items()):
        img = mirror.image_array(i)
        assert img.shape == (32, 32, 3) and img.dtype == np.uint8, (p, img.shape, img.dtype)


def test_mirror_determinism(mirror):
    """Same seed rebuild ⇒ identical source sequence (the mirror adds no
    randomness of its own); different seed ⇒ same multiset, different order."""
    from fpcmc.data import load_all_pools
    from fpcmc.config import FPCMCConfig
    from fpcmc.protocols import build_p2
    from baselines.pcmc_sleep.stream_mirror import P2PixelMirror

    config = FPCMCConfig.from_yaml(Path("configs") / "fpcmc_default.yaml")
    pools = load_all_pools()
    again = P2PixelMirror(build_p2(config, SEED, pools), pools)
    idx = np.linspace(0, len(mirror) - 1, 500, dtype=int)
    refs_a = [mirror.source_ref(int(i)) for i in idx]
    refs_b = [again.source_ref(int(i)) for i in idx]
    assert refs_a == refs_b

    other = P2PixelMirror(build_p2(config, SEED + 1, pools), pools)
    all_a = [mirror.source_ref(i) for i in range(len(mirror))]
    all_o = [other.source_ref(i) for i in range(len(other))]
    assert all_a != all_o, "different seed must reorder the stream"
    # T12 as-built: the 30% past-class interleave draws ind_test rows
    # without replacement across the stream, so the CONSUMED ind_test subset
    # is seed-dependent; every other pool's composition is seed-invariant.
    pool_a = mirror.manifest.pool
    pool_o = other.manifest.pool
    keep_a = sorted(r for i, r in enumerate(all_a) if pool_a[i] != "ind_test")
    keep_o = sorted(r for i, r in enumerate(all_o) if pool_o[i] != "ind_test")
    assert keep_a == keep_o, "non-interleave composition must be seed-invariant"
    assert (pool_a == "ind_test").sum() == (pool_o == "ind_test").sum()


def test_mirror_t0(mirror):
    """T0 pretraining refs: all 80 split classes × 500 train images."""
    refs = mirror.t0_image_refs()
    assert len(refs) == 80 * 500
    classes = {cls for cls, _, _ in refs}
    assert classes == {str(c) for c in mirror.t0_classes}
    assert all(kind == "cifar" and ref.startswith("train:") for _, kind, ref in refs)
