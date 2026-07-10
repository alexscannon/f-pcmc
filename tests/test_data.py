"""T1 tests for fpcmc/data.py (TASKS.md Task 1).

test_l2_on_load [U]      — all loaded/generated embeddings unit-norm (atol 1e-6);
                           double-normalization is a no-op.
test_real_pool_schemas [I] — the five real pools: expected counts
                           (50,000 / 10,000 / 250 / 500 / 2,576), D=1024, label
                           arrays align with class maps, no NaNs. Skips with a
                           clear message when roots.env is missing/unset or the
                           resolved EMBEDDINGS_DIR files are absent (the decided
                           deviation from the literal "data/embeddings/ absent"
                           wording — see data/README.md).
"""

import numpy as np
import pytest
import torch

from fpcmc import data
from fpcmc.data import DataError, PoolSpec, l2_normalize, load_pool
from tests.fixtures.vmf_world import VMFWorld


def _write_fake_pool(tmp_path, n=20, d=8):
    """A schema-conforming .pt file whose embeddings are NOT unit-norm."""
    rng = np.random.default_rng(0)
    emb = (rng.normal(size=(n, d)) * 7.0).astype(np.float32)
    assert not np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=0.1)
    subclasses = [f"class_{i % 4}" for i in range(n)]
    superclasses = [f"super_{i % 2}" for i in range(n)]
    payload = {
        "embeddings": torch.from_numpy(emb),
        "subclass_names": subclasses,
        "superclass_names": superclasses,
        "sources": ["fake_source"] * n,
        "image_paths": [f"img_{i}.png" for i in range(n)],
        "label_mappings": {
            "subclass_to_id": {f"class_{i}": i for i in range(4)},
            "id_to_subclass": {i: f"class_{i}" for i in range(4)},
            "superclass_to_id": {f"super_{i}": i for i in range(2)},
            "id_to_superclass": {i: f"super_{i}" for i in range(2)},
        },
    }
    torch.save(payload, tmp_path / "fake.pt")
    return emb


def test_l2_on_load(tmp_path):
    raw = _write_fake_pool(tmp_path)
    spec = PoolSpec(name="fake", filename="fake.pt", source=None, expected_count=20)
    pool = load_pool(spec, embeddings_dir=tmp_path)

    # Loaded embeddings are unit-norm even though the file's are not.
    np.testing.assert_allclose(np.linalg.norm(pool.x, axis=1), 1.0, atol=1e-6)
    assert pool.x.dtype == np.float32
    # Directions preserved (normalization only rescales).
    np.testing.assert_allclose(
        pool.x, raw / np.linalg.norm(raw, axis=1, keepdims=True), atol=1e-6
    )

    # Idempotent: re-normalizing already-normalized data is a no-op.
    again = l2_normalize(pool.x)
    assert np.array_equal(again, pool.x)

    # Fixture-generated embeddings are unit-norm at generation time, and
    # normalization leaves them alone too.
    world = VMFWorld(seed=5, k_known=3, k_novel=1)
    fx = world.t0_pool(40).x
    np.testing.assert_allclose(np.linalg.norm(fx, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(l2_normalize(fx), fx, atol=1e-6)

    # Schema validation bites: a wrong expected count is an error, not a warning.
    bad = PoolSpec(name="fake", filename="fake.pt", source=None, expected_count=21)
    with pytest.raises(DataError):
        load_pool(bad, embeddings_dir=tmp_path)


# Expected pool geometry for the primary (DINOv3) embeddings: data/README.md,
# verified against live data in docs/ASSETS.md §5.
_EXPECTED = {
    "ind_reference": (50_000, 100, 20, "cifar100_train"),
    "ind_test": (10_000, 100, 20, "cifar100_test"),
    "synthetic_ind": (250, 10, 2, "genai_ind"),
    "near_ood": (500, 6, 3, "genai_novel_subclass"),
    "far_ood": (2_576, 43, 16, "genai_novel_superclass"),
}


@pytest.mark.slow
def test_real_pool_schemas():
    available, reason = data.embeddings_available()
    if not available:
        pytest.skip(reason)

    pools = data.load_all_pools()
    assert set(pools) == set(_EXPECTED)

    for name, (count, n_sub, n_super, source) in _EXPECTED.items():
        pool = pools[name]
        assert pool.x.shape == (count, 1024), name
        assert pool.x.dtype == np.float32, name
        assert not np.isnan(pool.x).any(), name
        assert not np.isinf(pool.x).any(), name
        np.testing.assert_allclose(
            np.linalg.norm(pool.x, axis=1), 1.0, atol=1e-6, err_msg=name
        )

        # Parallel label arrays align with the embedding rows.
        assert len(pool.subclass_names) == count, name
        assert len(pool.superclass_names) == count, name
        assert len(pool.sources) == count, name
        assert set(pool.sources) == {source}, name

        # Label arrays align with the (file-scoped) class maps.
        sub_map = pool.label_mappings["subclass_to_id"]
        super_map = pool.label_mappings["superclass_to_id"]
        assert set(pool.subclass_names) <= set(sub_map), name
        assert set(pool.superclass_names) <= set(super_map), name
        assert len(set(pool.subclass_names)) == n_sub, name
        assert len(set(pool.superclass_names)) == n_super, name
        # Integer ids are the mapping applied to the name arrays.
        expected_ids = np.array([sub_map[s] for s in pool.subclass_names])
        assert np.array_equal(pool.subclass_ids, expected_ids), name

    # The two real-CIFAR pools come from one file split by `sources` and share
    # the canonical 100/20 taxonomy.
    ref, tst = pools["ind_reference"], pools["ind_test"]
    assert ref.label_mappings == tst.label_mappings
    assert set(ref.subclass_names) == set(tst.subclass_names)
