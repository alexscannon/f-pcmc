"""T0 config-system tests (TASKS.md Task 0).

The expected values below are the PRD §8 defaults, transcribed literally.
If these ever disagree with configs/default.yaml, the config file is wrong,
not this test.
"""

from pathlib import Path

import pytest
import yaml

from fpcmc.config import ConfigError, FPCMCConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO_ROOT / "configs" / "default.yaml"

# PRD §8, scalar keys.
PRD_S8_DEFAULTS = {
    "encoder": "dinov3_vitl16",
    "scorer": "knn_vmf",
    "k_ref": 5,
    "n_vmf_min": 10,
    "tau_percentile_q": 95,
    "n_shrink": 10,
    "alpha_stm_ema": 0.10,
    "K_max_refset": 64,
    "stm_capacity": 100,
    "n_mature": 5,
    "theta_promote": 30,
    "min_cohesion": 0.55,
    "sep_factor": 1.0,
    "m_windows": 3,
    "window_W": 250,
    "T_cluster": 500,
    "w_residual": 500,
    "T_merge": 500,
    "merge_sim": 0.80,
    "seed": 42,
}
# PRD §8, nested blocks.
PRD_S8_UMAP = {"dim": 50, "n_neighbors": 15, "min_dist": 0.0, "metric": "cosine"}
PRD_S8_HDBSCAN = {"min_cluster_sizes": (10, 15, 20, 25, 30), "selection": "eom"}


def test_config_roundtrip(tmp_path):
    cfg = FPCMCConfig.from_yaml(DEFAULT_YAML)

    # Every PRD §8 key present with its PRD default value.
    for key, expected in PRD_S8_DEFAULTS.items():
        assert getattr(cfg, key) == expected, f"{key}: {getattr(cfg, key)!r} != PRD default {expected!r}"
    for key, expected in PRD_S8_UMAP.items():
        assert getattr(cfg.umap, key) == expected
    for key, expected in PRD_S8_HDBSCAN.items():
        assert getattr(cfg.hdbscan, key) == expected

    # Serialize -> reload -> equality (string form and file form).
    text = cfg.to_yaml()
    assert FPCMCConfig.from_yaml_text(text) == cfg
    rt_path = tmp_path / "roundtrip.yaml"
    rt_path.write_text(text)
    assert FPCMCConfig.from_yaml(rt_path) == cfg

    # Serialized form carries the complete §8 key set (run artifacts embed this).
    dumped = yaml.safe_load(text)
    assert set(dumped) == set(PRD_S8_DEFAULTS) | {"umap", "hdbscan"}
    assert set(dumped["umap"]) == set(PRD_S8_UMAP)
    assert set(dumped["hdbscan"]) == set(PRD_S8_HDBSCAN)


def test_config_rejects_unknown_key():
    base = yaml.safe_load(DEFAULT_YAML.read_text())

    typo_top = dict(base)
    typo_top["theta_promot"] = typo_top.pop("theta_promote")
    with pytest.raises(ConfigError, match="theta_promot"):
        FPCMCConfig.from_yaml_text(yaml.safe_dump(typo_top))

    typo_nested = dict(base)
    typo_nested["umap"] = dict(base["umap"], spread=1.0)
    with pytest.raises(ConfigError, match="spread"):
        FPCMCConfig.from_yaml_text(yaml.safe_dump(typo_nested))


def test_config_is_frozen():
    cfg = FPCMCConfig.from_yaml(DEFAULT_YAML)
    with pytest.raises(Exception):
        cfg.seed = 0  # type: ignore[misc]
