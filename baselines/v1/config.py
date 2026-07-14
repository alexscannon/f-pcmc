"""Continual learning evaluation pipeline configuration loader."""

from dataclasses import dataclass
from pathlib import Path

import yaml


def _load_roots() -> dict[str, str]:
    """Load all variables from nearest roots.env (with os.environ fallback).

    Returns every KEY=VALUE pair found in roots.env. Values may reference
    other variables (e.g. ``EMBEDDINGS_DIR=${DATA_ROOT}/embeddings/...``);
    these are resolved before returning.  PROJECT_ROOT and DATA_ROOT are
    required; all others are optional.
    """
    import os
    import re

    search = Path(__file__).resolve().parent
    env_vals: dict[str, str] = {}
    while search != search.parent:
        candidate = search / "roots.env"
        if candidate.is_file():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env_vals[k.strip()] = v.strip()
            break
        search = search.parent

    # Merge: roots.env wins, fall back to os.environ for any referenced key
    roots: dict[str, str] = {}
    for k, v in env_vals.items():
        roots[k] = v
    for required in ("PROJECT_ROOT", "DATA_ROOT"):
        if required not in roots:
            roots[required] = os.environ.get(required, "")

    if not roots.get("PROJECT_ROOT") or not roots.get("DATA_ROOT"):
        raise FileNotFoundError(
            "Cannot find roots.env with PROJECT_ROOT and DATA_ROOT. "
            "Copy roots.env.example to roots.env and set your paths."
        )

    # Resolve variable-in-variable references (e.g. ${DATA_ROOT} inside other values)
    _VAR_RE = re.compile(r"\$\{(\w+)\}")
    changed = True
    while changed:
        changed = False
        for k, v in roots.items():
            resolved = _VAR_RE.sub(
                lambda m: roots.get(m.group(1), m.group(0)), v
            )
            if resolved != v:
                roots[k] = resolved
                changed = True

    return roots


def _resolve_path(raw: str, roots: dict[str, str]) -> Path:
    """Expand ${VAR} placeholders in a path string using roots dict."""
    for key, val in roots.items():
        raw = raw.replace(f"${{{key}}}", val)
    return Path(raw)


@dataclass
class ContinualConfig:
    """Configuration for the continual learning evaluation pipeline."""

    # Embedding paths
    embeddings_dir: Path
    cifar100_mapping: Path
    real_cifar100_file: str
    synthetic_ind_file: str
    novel_subclasses_file: str
    novel_superclasses_file: str

    # Output
    output_dir: Path

    # IND model
    regularization_epsilon: float
    threshold_percentile: int

    # Stream
    random_seed: int
    ind_warmup_count: int  # 0 = disabled (fully shuffled)
    # Post-warmup remainder ordering: "random" | "sequential" | "clustered"
    stream_order: str
    stream_cluster_size: list[int]  # [lo, hi] mini-block range for "clustered"

    # Oracle decomposition: "none" | "detection" | "clustering" | "both"
    oracle_mode: str

    # Paradigm selector: "mahalanobis_hdbscan" (default) | "knn_dpmeans" | "vmf_dpmm"
    paradigm: str

    # Clustering
    cluster_interval: int
    min_ood_for_clustering: int
    min_cluster_sizes: list[int]
    jaccard_dedup_threshold: float

    # UMAP preprocessing (None = disabled, uses raw 1024d embeddings)
    umap_n_components: int | None
    umap_n_neighbors: int
    umap_min_dist: float

    # Promotion
    min_promote_size: int
    min_intra_cosine_sim: float
    min_soft_prob: float

    # kNN-DPmeans paradigm (only used when paradigm == knn_dpmeans)
    knn_k: int
    knn_distance: str           # "kth" | "mean_k"
    dpmeans_max_iters: int
    dpmeans_convergence_tol: float

    # vMF-DPMM paradigm (only used when paradigm == vmf_dpmm)
    vmf_alpha_calibration_target: int
    vmf_alpha_search_lo: float
    vmf_alpha_search_hi: float
    vmf_alpha_search_max_iters: int
    vmf_kappa_init: float
    vmf_kappa_min: float
    vmf_kappa_max: float
    vmf_kappa_promote_min: float
    vmf_prune_singletons: bool
    vmf_prune_interval: int
    vmf_prune_age: int
    # Candidate score that populates the primary `score` field/column. One of
    # score_current (default) | score_logratio | score_entropy | score_density.
    vmf_primary_score: str

    # knn_vmf composed paradigm (only used when paradigm == knn_vmf). Inherits
    # knn_k/knn_distance for detection and vmf_kappa_*/min_promote_size for the
    # unnamed-vMF clustering; only the novel alpha target is paradigm-specific.
    knn_vmf_alpha_target_novel: int

    # Evaluation
    rolling_window: int
    csv_flush_interval: int
    tsne_sample_size: int
    tsne_perplexity: int

    @classmethod
    def from_yaml(cls, path: Path) -> "ContinualConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        roots = _load_roots()

        oracle_mode = raw.get("oracle_mode", "none")
        valid_oracle_modes = {"none", "detection", "clustering", "both"}
        if oracle_mode not in valid_oracle_modes:
            raise ValueError(
                f"oracle_mode must be one of {sorted(valid_oracle_modes)}, "
                f"got {oracle_mode!r}"
            )

        paradigm = raw.get("paradigm", "mahalanobis_hdbscan")
        valid_paradigms = {"mahalanobis_hdbscan", "knn_dpmeans", "vmf_dpmm", "knn_vmf"}
        if paradigm not in valid_paradigms:
            raise ValueError(
                f"paradigm must be one of {sorted(valid_paradigms)}, "
                f"got {paradigm!r}"
            )

        stream_order = raw.get("stream_order", "random")
        valid_stream_orders = {"random", "sequential", "clustered"}
        if stream_order not in valid_stream_orders:
            raise ValueError(
                f"stream_order must be one of {sorted(valid_stream_orders)}, "
                f"got {stream_order!r}"
            )

        stream_cluster_size = raw.get("stream_cluster_size", [2, 4])
        if (
            len(stream_cluster_size) != 2
            or stream_cluster_size[0] < 1
            or stream_cluster_size[1] < stream_cluster_size[0]
        ):
            raise ValueError(
                "stream_cluster_size must be [lo, hi] with 1 <= lo <= hi, "
                f"got {stream_cluster_size!r}"
            )

        return cls(
            embeddings_dir=_resolve_path(raw["embeddings_dir"], roots),
            cifar100_mapping=_resolve_path(raw["cifar100_mapping"], roots),
            real_cifar100_file=raw["real_cifar100_file"],
            synthetic_ind_file=raw["synthetic_ind_file"],
            novel_subclasses_file=raw["novel_subclasses_file"],
            novel_superclasses_file=raw["novel_superclasses_file"],
            output_dir=_resolve_path(raw["output_dir"], roots),
            regularization_epsilon=raw.get("regularization_epsilon", 1e-6),
            threshold_percentile=raw.get("threshold_percentile", 95),
            random_seed=raw.get("random_seed", 42),
            ind_warmup_count=raw.get("ind_warmup_count", 0),
            stream_order=stream_order,
            stream_cluster_size=stream_cluster_size,
            oracle_mode=oracle_mode,
            paradigm=paradigm,
            cluster_interval=raw.get("cluster_interval", 50),
            min_ood_for_clustering=raw.get("min_ood_for_clustering", 30),
            min_cluster_sizes=raw.get("min_cluster_sizes", [10, 15, 20, 25, 30]),
            jaccard_dedup_threshold=raw.get("jaccard_dedup_threshold", 0.80),
            umap_n_components=raw.get("umap_n_components", None),
            umap_n_neighbors=raw.get("umap_n_neighbors", 15),
            umap_min_dist=raw.get("umap_min_dist", 0.0),
            min_promote_size=raw.get("min_promote_size", 10),
            min_intra_cosine_sim=raw.get("min_intra_cosine_sim", 0.70),
            min_soft_prob=raw.get("min_soft_prob", 0.60),
            knn_k=raw.get("knn", {}).get("k", 20),
            knn_distance=raw.get("knn", {}).get("distance", "kth"),
            dpmeans_max_iters=raw.get("dpmeans", {}).get("max_iters", 20),
            dpmeans_convergence_tol=raw.get("dpmeans", {}).get("convergence_tol", 1e-4),
            vmf_alpha_calibration_target=raw.get("vmf_dpmm", {}).get(
                "alpha_calibration_target", 100
            ),
            vmf_alpha_search_lo=float(raw.get("vmf_dpmm", {}).get("alpha_search_range", [1e-6, 100.0])[0]),
            vmf_alpha_search_hi=float(raw.get("vmf_dpmm", {}).get("alpha_search_range", [1e-6, 100.0])[1]),
            vmf_alpha_search_max_iters=raw.get("vmf_dpmm", {}).get("alpha_search_max_iters", 20),
            vmf_kappa_init=float(raw.get("vmf_dpmm", {}).get("kappa_init", 100.0)),
            vmf_kappa_min=float(raw.get("vmf_dpmm", {}).get("kappa_min", 1.0)),
            vmf_kappa_max=float(raw.get("vmf_dpmm", {}).get("kappa_max", 1.0e6)),
            vmf_kappa_promote_min=float(raw.get("vmf_dpmm", {}).get("kappa_promote_min", 50.0)),
            vmf_prune_singletons=raw.get("vmf_dpmm", {}).get("prune_singletons", True),
            vmf_prune_interval=raw.get("vmf_dpmm", {}).get("prune_interval", 500),
            vmf_prune_age=raw.get("vmf_dpmm", {}).get("prune_age", 1000),
            vmf_primary_score=raw.get("vmf_dpmm", {}).get("primary_score", "score_current"),
            knn_vmf_alpha_target_novel=raw.get("knn_vmf", {}).get(
                "alpha_calibration_target_novel", 50
            ),
            rolling_window=raw.get("rolling_window", 500),
            csv_flush_interval=raw.get("csv_flush_interval", 1000),
            tsne_sample_size=raw.get("tsne_sample_size", 5000),
            tsne_perplexity=raw.get("tsne_perplexity", 30),
        )
