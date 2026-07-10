"""Embedding I/O for the five precomputed real pools (T1, PRD §3 / FR-1.2).

Data access contract (data/README.md; decision record docs/ASSETS.md §1):
embeddings are NEVER copied or symlinked into this repo. Paths resolve at load
time from `roots.env` at the repo root (gitignored; copy roots.env.example) —
`EMBEDDINGS_DIR` points at the external directory holding the four `.pt`
files. The five PRD §3 pools live in those four files; `real_cifar100.pt`
carries both IND Reference and IND Test, split by its `sources` field.

Loading behavior:
  - torch.load with mmap where the file format allows it ("memory-mapped
    where possible"); torch is used ONLY for deserialization — everything
    downstream is NumPy (frozen-encoder invariant bans torch.{nn,optim,
    autograd}, enforced by tests/test_invariants.py).
  - Embeddings are NOT unit-norm on disk (norms ~11-18, ASSETS §1). They are
    L2-normalized on load, idempotently: `l2_normalize` returns already-unit
    input unchanged (bitwise), so double normalization is a no-op.
  - Schema is validated before anything is returned (shape/dtype/NaN/Inf,
    parallel label-array alignment, label_mappings consistency, expected
    counts). A violation raises DataError — per CLAUDE.md that is a STOP
    condition, not something to paper over.
  - Each file's `label_mappings` is FILE-SCOPED: integer ids are not
    comparable across files. Join across pools on class-name strings only
    (ASSETS §1). `Pool.subclass_ids`/`superclass_ids` are provided for
    within-pool convenience and carry that same caveat.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOTS_ENV_PATH = REPO_ROOT / "roots.env"

_EPS = 1e-12


class DataError(RuntimeError):
    """A real-data schema check failed (CLAUDE.md stop condition)."""


class EmbeddingsUnavailable(RuntimeError):
    """Embeddings cannot be resolved on this machine (integration tests skip)."""


# --------------------------------------------------------------------- roots.env
def read_roots_env(path: Path | None = None) -> dict[str, str]:
    """Parse roots.env (KEY=VALUE lines, ``#`` comments, ``${VAR}`` expansion).

    ``${VAR}`` resolves against earlier keys in the same file, then the process
    environment. Raises EmbeddingsUnavailable if the file is missing.
    """
    path = ROOTS_ENV_PATH if path is None else Path(path)
    if not path.is_file():
        raise EmbeddingsUnavailable(
            f"{path} not found — copy roots.env.example to roots.env and set "
            "DATA_ROOT/EMBEDDINGS_DIR (see data/README.md)"
        )
    values: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise EmbeddingsUnavailable(f"{path}:{lineno}: expected KEY=VALUE, got {raw!r}")
        key, _, value = line.partition("=")
        value = re.sub(
            r"\$\{(\w+)\}",
            lambda m: values.get(m.group(1), os.environ.get(m.group(1), "")),
            value.strip(),
        )
        values[key.strip()] = value
    return values


def resolve_embeddings_dir(roots_env_path: Path | None = None) -> Path:
    """EMBEDDINGS_DIR from roots.env, checked to exist on disk."""
    values = read_roots_env(roots_env_path)
    raw = values.get("EMBEDDINGS_DIR", "")
    if not raw:
        raise EmbeddingsUnavailable(
            "EMBEDDINGS_DIR is unset in roots.env (see data/README.md)"
        )
    path = Path(raw)
    if not path.is_dir():
        raise EmbeddingsUnavailable(f"EMBEDDINGS_DIR={path} does not exist on this machine")
    return path


# ------------------------------------------------------------------- pool specs
@dataclass(frozen=True)
class PoolSpec:
    """One PRD §3 pool: which file, which `sources` slice, expected row count."""

    name: str
    filename: str
    source: str | None  # None = whole file; else select rows where sources == source
    expected_count: int


# The five pools / four files (data/README.md, verified in docs/ASSETS.md §5).
POOL_SPECS: tuple[PoolSpec, ...] = (
    PoolSpec("ind_reference", "real_cifar100.pt", "cifar100_train", 50_000),
    PoolSpec("ind_test", "real_cifar100.pt", "cifar100_test", 10_000),
    PoolSpec("synthetic_ind", "ind.pt", None, 250),
    PoolSpec("near_ood", "novel_subclasses.pt", None, 500),
    PoolSpec("far_ood", "novel_superclasses.pt", None, 2_576),
)

_REQUIRED_KEYS = (
    "embeddings",
    "subclass_names",
    "superclass_names",
    "sources",
    "image_paths",
    "label_mappings",
)
_REQUIRED_MAPPINGS = ("subclass_to_id", "id_to_subclass", "superclass_to_id", "id_to_superclass")


@dataclass(frozen=True)
class Pool:
    """One loaded pool. x is (N, D) float32 with unit-norm rows.

    Label ids are FILE-scoped (ASSETS §1): never compare ids across pools —
    join on the name strings.
    """

    name: str
    x: np.ndarray
    subclass_names: np.ndarray  # (N,) <U str
    superclass_names: np.ndarray  # (N,) <U str
    subclass_ids: np.ndarray  # (N,) int64, via this file's subclass_to_id
    superclass_ids: np.ndarray  # (N,) int64
    sources: np.ndarray  # (N,) <U str
    image_paths: np.ndarray  # (N,) <U str
    label_mappings: dict[str, dict]


# ---------------------------------------------------------------- normalization
def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization, idempotent.

    Already-unit input (every row norm within 1e-6 of 1) is returned unchanged
    — bitwise — so normalize(normalize(x)) == normalize(x) exactly.
    """
    x = np.asarray(x)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.allclose(norms, 1.0, atol=1e-6):
        return x
    return x / np.maximum(norms, _EPS)


# ---------------------------------------------------------------------- loading
def _torch_load(path: Path) -> dict[str, Any]:
    import torch  # deserialization only; see module docstring

    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except (RuntimeError, ValueError):
        # Legacy (non-zipfile) serialization cannot be mmapped.
        return torch.load(path, map_location="cpu", weights_only=False)


def _validate_and_build(name: str, path: Path, payload: dict[str, Any], spec: PoolSpec) -> Pool:
    where = f"pool {name!r} ({path})"
    missing = [k for k in _REQUIRED_KEYS if k not in payload]
    if missing:
        raise DataError(f"{where}: missing keys {missing}")
    mappings = payload["label_mappings"]
    bad = [k for k in _REQUIRED_MAPPINGS if k not in mappings]
    if bad:
        raise DataError(f"{where}: label_mappings missing {bad}")

    emb = payload["embeddings"]
    x = emb.numpy() if hasattr(emb, "numpy") else np.asarray(emb)
    if x.ndim != 2:
        raise DataError(f"{where}: embeddings must be 2-D, got shape {x.shape}")
    if x.dtype != np.float32:
        raise DataError(f"{where}: embeddings must be float32, got {x.dtype}")

    n = x.shape[0]
    arrays = {
        "subclass_names": np.asarray(payload["subclass_names"]),
        "superclass_names": np.asarray(payload["superclass_names"]),
        "sources": np.asarray(payload["sources"]),
        "image_paths": np.asarray(payload["image_paths"]),
    }
    misaligned = {k: len(v) for k, v in arrays.items() if len(v) != n}
    if misaligned:
        raise DataError(f"{where}: label arrays misaligned with {n} embeddings: {misaligned}")

    if spec.source is not None:
        mask = arrays["sources"] == spec.source
        x = x[mask]
        arrays = {k: v[mask] for k, v in arrays.items()}
        n = x.shape[0]

    if n != spec.expected_count:
        raise DataError(f"{where}: expected {spec.expected_count} rows, got {n}")
    if np.isnan(x).any() or np.isinf(x).any():
        raise DataError(f"{where}: embeddings contain NaN/Inf")

    sub_map: dict[str, int] = mappings["subclass_to_id"]
    super_map: dict[str, int] = mappings["superclass_to_id"]
    unknown_sub = set(arrays["subclass_names"]) - set(sub_map)
    unknown_super = set(arrays["superclass_names"]) - set(super_map)
    if unknown_sub or unknown_super:
        raise DataError(
            f"{where}: label names absent from this file's label_mappings: "
            f"subclasses={sorted(unknown_sub)}, superclasses={sorted(unknown_super)}"
        )

    return Pool(
        name=name,
        x=np.ascontiguousarray(l2_normalize(x.astype(np.float32, copy=True))),
        subclass_names=arrays["subclass_names"],
        superclass_names=arrays["superclass_names"],
        subclass_ids=np.array([sub_map[s] for s in arrays["subclass_names"]], dtype=np.int64),
        superclass_ids=np.array([super_map[s] for s in arrays["superclass_names"]], dtype=np.int64),
        sources=arrays["sources"],
        image_paths=arrays["image_paths"],
        label_mappings=mappings,
    )


def load_pool(spec: PoolSpec, embeddings_dir: Path | None = None) -> Pool:
    """Load + validate + L2-normalize one pool. embeddings_dir defaults to roots.env."""
    root = resolve_embeddings_dir() if embeddings_dir is None else Path(embeddings_dir)
    path = root / spec.filename
    if not path.is_file():
        raise EmbeddingsUnavailable(f"{path} not found under EMBEDDINGS_DIR={root}")
    return _validate_and_build(spec.name, path, _torch_load(path), spec)


def load_all_pools(embeddings_dir: Path | None = None) -> dict[str, Pool]:
    """All five pools, loading each underlying .pt file exactly once."""
    root = resolve_embeddings_dir() if embeddings_dir is None else Path(embeddings_dir)
    payloads: dict[str, dict[str, Any]] = {}
    pools: dict[str, Pool] = {}
    for spec in POOL_SPECS:
        path = root / spec.filename
        if not path.is_file():
            raise EmbeddingsUnavailable(f"{path} not found under EMBEDDINGS_DIR={root}")
        if spec.filename not in payloads:
            payloads[spec.filename] = _torch_load(path)
        pools[spec.name] = _validate_and_build(spec.name, path, payloads[spec.filename], spec)
    return pools


def embeddings_available() -> tuple[bool, str]:
    """(available?, reason) — the [I]-test skip condition from data/README.md:
    roots.env missing/unset, or the resolved files absent (never a literal
    data/embeddings/ check)."""
    try:
        root = resolve_embeddings_dir()
    except EmbeddingsUnavailable as e:
        return False, str(e)
    missing = sorted({s.filename for s in POOL_SPECS if not (root / s.filename).is_file()})
    if missing:
        return False, f"missing under EMBEDDINGS_DIR={root}: {missing}"
    return True, f"embeddings resolved at {root}"
