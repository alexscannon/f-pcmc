"""Config system: YAML -> frozen FPCMCConfig (every PRD §8 parameter).

Schema-validating: unknown keys are errors naming the offending key (TASKS T0),
values are type-checked against the dataclass annotations. to_yaml() round-trips
losslessly so every run artifact can embed its resolved config (NFR-2).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Invalid config: unknown key, wrong type, or disallowed value."""


VALID_ENCODERS = ("dinov3_vitl16", "resnet50")
VALID_SCORERS = ("knn_ref", "vmf", "knn_vmf")


@dataclass(frozen=True)
class UmapConfig:
    dim: int = 50
    n_neighbors: int = 15
    min_dist: float = 0.0
    metric: str = "cosine"


@dataclass(frozen=True)
class HdbscanConfig:
    min_cluster_sizes: tuple[int, ...] = (10, 15, 20, 25, 30)
    selection: str = "eom"


@dataclass(frozen=True)
class FPCMCConfig:
    """PRD §8 parameters, PRD defaults. Field order follows the PRD block."""

    encoder: str = "dinov3_vitl16"      # or resnet50 (A6)
    scorer: str = "knn_vmf"             # knn_ref | vmf | knn_vmf
    k_ref: int = 5
    n_vmf_min: int = 10
    tau_percentile_q: int = 95
    n_shrink: int = 10
    alpha_stm_ema: float = 0.10
    K_max_refset: int = 64
    stm_capacity: int = 100             # sweep {50, 100, 200}
    n_mature: int = 5
    theta_promote: int = 30             # sweep {20, 30, 50}
    # FR-7 criterion 2 is RELATIVE: cohesion >= this ratio x median(cohesion of
    # the T0 LTM concepts). NOT an absolute cohesion bar — the retired
    # `min_cohesion: 0.55` key meant something else entirely, so never carry an
    # old config's number across (PRD FR-7, amended 2026-07-13).
    min_cohesion_ratio: float = 0.35    # sweep {0.30, 0.35, 0.40}
    sep_factor: float = 1.0
    m_windows: int = 3
    window_W: int = 250
    T_cluster: int = 500
    w_residual: int = 500
    T_merge: int = 500
    merge_sim: float = 0.80
    umap: UmapConfig = field(default_factory=UmapConfig)
    hdbscan: HdbscanConfig = field(default_factory=HdbscanConfig)
    seed: int = 42

    def __post_init__(self) -> None:
        if self.encoder not in VALID_ENCODERS:
            raise ConfigError(f"encoder must be one of {VALID_ENCODERS}, got {self.encoder!r}")
        if self.scorer not in VALID_SCORERS:
            raise ConfigError(f"scorer must be one of {VALID_SCORERS}, got {self.scorer!r}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FPCMCConfig":
        return cls.from_yaml_text(Path(path).read_text())

    @classmethod
    def from_yaml_text(cls, text: str) -> "FPCMCConfig":
        data = yaml.safe_load(text)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")
        return _build(cls, data, context="config")

    def to_yaml(self) -> str:
        return yaml.safe_dump(_as_plain_dict(self), sort_keys=False)


_NESTED = {"umap": UmapConfig, "hdbscan": HdbscanConfig}


def _build(cls: type, data: dict[str, Any], context: str):
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"unknown key(s) in {context}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        nested_cls = _NESTED.get(name) if cls is FPCMCConfig else None
        if nested_cls is not None:
            if not isinstance(value, dict):
                raise ConfigError(f"{context}.{name} must be a mapping")
            kwargs[name] = _build(nested_cls, value, context=f"{context}.{name}")
        else:
            kwargs[name] = _coerce(value, known[name].type, f"{context}.{name}")
    return cls(**kwargs)


def _coerce(value: Any, annotation: Any, where: str) -> Any:
    ann = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))
    if ann == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} must be an int, got {value!r}")
        return value
    if ann == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where} must be a number, got {value!r}")
        return float(value)
    if ann == "str":
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string, got {value!r}")
        return value
    if ann.startswith("tuple"):
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in value
        ):
            raise ConfigError(f"{where} must be a list of ints, got {value!r}")
        return tuple(value)
    raise ConfigError(f"{where}: unsupported config field type {annotation!r}")


def _as_plain_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _as_plain_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return list(obj)
    return obj
