"""Orchestrate H2 clustering evaluation across conditions, sections, and parameters."""

import logging
from dataclasses import asdict
from enum import Enum

import numpy as np

from clustering import (
    ClusteringResult,
    analyze_contamination,
    compute_clustering_metrics,
    run_hdbscan,
)
from config import H2TestConfig
from consistency import compute_consistency_analysis
from data_loader import EmbeddingPool, H2PoolRole
from reference_stats import ReferenceStatistics
from scoring import PipelineClassification, classify_pipeline

logger = logging.getLogger("h2_clustering")

import warnings
warnings.filterwarnings("ignore", message="n_jobs value .* overridden to 1")

class EvalCondition(str, Enum):
    """H2 evaluation conditions."""

    ORACLE = "oracle"       # All OOD examples, regardless of H1 classification
    PIPELINE = "pipeline"   # Only examples classified as OOD by Mahalanobis at TPR target


class Section(str, Enum):
    """OOD sections."""

    SECTION1 = "section1"   # Novel subclasses of existing superclasses
    SECTION2 = "section2"   # Novel subclasses of novel superclasses


def _clustering_result_to_dict(result: ClusteringResult) -> dict:
    """Convert ClusteringResult to dict, excluding the labels array."""
    d = asdict(result)
    del d["labels"]
    return d


def _collect_per_image_rows(
    labels: np.ndarray,
    ground_truth: np.ndarray,
    superclasses: np.ndarray,
    section: str,
    condition: str,
    min_cluster_size: int,
    is_ind_contaminant: np.ndarray | None = None,
) -> list[dict]:
    """Build per-image label rows for CSV export."""
    rows = []
    for i in range(len(labels)):
        rows.append({
            "section": section,
            "condition": condition,
            "min_cluster_size": min_cluster_size,
            "image_idx": i,
            "subclass": str(ground_truth[i]),
            "superclass": str(superclasses[i]),
            "cluster_label": int(labels[i]),
            "is_noise": labels[i] == -1,
            "is_ind_contaminant": bool(is_ind_contaminant[i]) if is_ind_contaminant is not None else False,
        })
    return rows


def _run_oracle_condition(
    ood_pool: EmbeddingPool,
    min_cluster_sizes: list[int],
    section_name: str,
    umap_n_components: int | None = None,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.0,
    random_seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Run HDBSCAN on all OOD examples (oracle condition).

    Returns (metrics_list, per_image_rows).
    """
    results = []
    all_per_image_rows: list[dict] = []
    gt_classes = ood_pool.df["subclass"].values
    superclasses = ood_pool.df["superclass"].values

    for mcs in min_cluster_sizes:
        logger.info(f"  Oracle: min_cluster_size={mcs}")
        labels = run_hdbscan(
            ood_pool.embeddings, mcs,
            umap_n_components=umap_n_components,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            random_state=random_seed,
        )
        metrics = compute_clustering_metrics(labels, gt_classes)

        results.append({
            "min_cluster_size": mcs,
            **_clustering_result_to_dict(metrics),
        })

        all_per_image_rows.extend(_collect_per_image_rows(
            labels, gt_classes, superclasses,
            section_name, EvalCondition.ORACLE.value, mcs,
        ))

    return results, all_per_image_rows


def _run_pipeline_condition(
    ood_pool: EmbeddingPool,
    ind_test_pool: EmbeddingPool,
    pipeline_class: PipelineClassification,
    min_cluster_sizes: list[int],
    section_name: str,
    umap_n_components: int | None = None,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.0,
    random_seed: int = 42,
) -> tuple[list[dict], dict, list[dict]]:
    """Run HDBSCAN on H1-filtered examples (pipeline condition).

    Combines OOD true positives with IND false positives, runs HDBSCAN,
    and tracks contamination.

    Returns (metrics_list, pipeline_analysis_dict, per_image_rows).
    """
    # OOD examples classified as OOD by H1
    ood_predicted_ood_mask = pipeline_class.ood_predicted_ood
    ood_surviving = ood_pool.embeddings[ood_predicted_ood_mask]
    ood_surviving_gt = ood_pool.df["subclass"].values[ood_predicted_ood_mask]
    ood_surviving_super = ood_pool.df["superclass"].values[ood_predicted_ood_mask]

    # IND examples incorrectly classified as OOD (contaminants)
    ind_predicted_ood_mask = pipeline_class.ind_predicted_ood
    ind_contaminants = ind_test_pool.embeddings[ind_predicted_ood_mask]
    ind_contaminant_sub = ind_test_pool.df["subclass"].values[ind_predicted_ood_mask]
    ind_contaminant_super = ind_test_pool.df["superclass"].values[ind_predicted_ood_mask]

    n_ood_surviving = len(ood_surviving)
    n_ind_contaminants = len(ind_contaminants)

    logger.info(
        f"  Pipeline: {n_ood_surviving} OOD surviving + "
        f"{n_ind_contaminants} IND contaminants = "
        f"{n_ood_surviving + n_ind_contaminants} total"
    )

    # Combine for clustering
    combined_embeddings = np.concatenate([ood_surviving, ind_contaminants], axis=0)
    is_ind_contaminant = np.concatenate([
        np.zeros(n_ood_surviving, dtype=bool),
        np.ones(n_ind_contaminants, dtype=bool),
    ])
    combined_gt = np.concatenate([ood_surviving_gt, ind_contaminant_sub])
    combined_super = np.concatenate([ood_surviving_super, ind_contaminant_super])

    # Original per-class counts before H1 filtering
    ood_original_counts = dict(ood_pool.df["subclass"].value_counts())

    results = []
    contamination_summaries: dict[str, dict] = {}
    all_per_image_rows: list[dict] = []

    for mcs in min_cluster_sizes:
        logger.info(f"  Pipeline: min_cluster_size={mcs}")
        labels = run_hdbscan(
            combined_embeddings, mcs,
            umap_n_components=umap_n_components,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            random_state=random_seed,
        )

        # Compute metrics on OOD points only (exclude IND contaminants from ARI/NMI)
        ood_labels = labels[:n_ood_surviving]
        metrics = compute_clustering_metrics(ood_labels, ood_surviving_gt)

        # Contamination analysis on full set
        contamination = analyze_contamination(
            labels, is_ind_contaminant, ood_surviving_gt, ood_original_counts,
        )

        results.append({
            "min_cluster_size": mcs,
            **_clustering_result_to_dict(metrics),
            "n_ood_input": n_ood_surviving,
            "n_ind_contaminants": n_ind_contaminants,
            "n_total_input": n_ood_surviving + n_ind_contaminants,
        })

        contamination_summaries[str(mcs)] = {
            "ind_in_clusters": contamination.ind_in_clusters,
            "ind_in_noise": contamination.ind_in_noise,
            "ind_total": contamination.ind_total,
            "ind_per_cluster": {str(k): v for k, v in contamination.ind_per_cluster.items()},
        }

        all_per_image_rows.extend(_collect_per_image_rows(
            labels, combined_gt, combined_super,
            section_name, EvalCondition.PIPELINE.value, mcs,
            is_ind_contaminant=is_ind_contaminant,
        ))

    # OOD class depletion (same regardless of min_cluster_size)
    ood_depletion: dict[str, dict] = {}
    for class_name in sorted(ood_original_counts.keys()):
        original = int(ood_original_counts[class_name])
        after = int((ood_surviving_gt == class_name).sum())
        ood_depletion[class_name] = {
            "original": original,
            "after_h1": after,
            "lost": original - after,
            "lost_fraction": round((original - after) / original, 4) if original > 0 else 0.0,
        }

    pipeline_analysis = {
        "contamination_by_mcs": contamination_summaries,
        "ood_class_depletion": ood_depletion,
        "threshold": pipeline_class.threshold,
        "achieved_tpr": pipeline_class.achieved_tpr,
        "achieved_fpr": pipeline_class.achieved_fpr,
    }

    return results, pipeline_analysis, all_per_image_rows


def run_evaluation(
    pools: dict[H2PoolRole, EmbeddingPool],
    ref_stats: ReferenceStatistics,
    config: H2TestConfig,
) -> dict:
    """Run full H2 evaluation: 2 sections x 2 conditions x N min_cluster_sizes.

    Returns structured dict ready for JSON serialization.
    """
    section_pools = {
        Section.SECTION1: pools[H2PoolRole.OOD_SECTION1],
        Section.SECTION2: pools[H2PoolRole.OOD_SECTION2],
    }

    all_results: dict[str, dict] = {}
    all_pipeline_analysis: dict[str, dict] = {}
    all_consistency: dict[str, dict] = {}
    all_per_image_rows: list[dict] = []

    for section, ood_pool in section_pools.items():
        logger.info(
            f"=== {section.value} ({len(ood_pool.df)} examples, "
            f"{ood_pool.df['subclass'].nunique()} classes) ==="
        )

        # --- Oracle condition ---
        logger.info(f"Running oracle condition for {section.value}")
        oracle_results, oracle_rows = _run_oracle_condition(
            ood_pool, config.min_cluster_sizes, section.value,
            umap_n_components=config.umap_n_components,
            umap_n_neighbors=config.umap_n_neighbors,
            umap_min_dist=config.umap_min_dist,
            random_seed=config.random_seed,
        )

        # --- Pipeline condition ---
        logger.info(f"Running pipeline condition for {section.value}")
        pipeline_class = classify_pipeline(
            pools[H2PoolRole.IND_TEST].embeddings,
            ood_pool.embeddings,
            ref_stats,
            config.pipeline_tpr_target,
        )
        pipeline_results, pipeline_analysis, pipeline_rows = _run_pipeline_condition(
            ood_pool, pools[H2PoolRole.IND_TEST],
            pipeline_class, config.min_cluster_sizes,
            section.value,
            umap_n_components=config.umap_n_components,
            umap_n_neighbors=config.umap_n_neighbors,
            umap_min_dist=config.umap_min_dist,
            random_seed=config.random_seed,
        )

        all_results[section.value] = {
            EvalCondition.ORACLE.value: oracle_results,
            EvalCondition.PIPELINE.value: pipeline_results,
        }
        all_pipeline_analysis[section.value] = pipeline_analysis
        all_per_image_rows.extend(oracle_rows)
        all_per_image_rows.extend(pipeline_rows)

        # --- Prompt consistency control ---
        logger.info(f"Running prompt-consistency analysis for {section.value}")
        consistency = compute_consistency_analysis(
            ood_pool.embeddings,
            ood_pool.df["subclass"].values,
            pools[H2PoolRole.IND_REFERENCE].embeddings,
            pools[H2PoolRole.IND_REFERENCE].df["subclass"].values,
            config.random_seed,
        )
        all_consistency[section.value] = {
            "mean_ratio": consistency.mean_ratio,
            "median_ratio": consistency.median_ratio,
            "std_ratio": consistency.std_ratio,
            "per_class": consistency.per_class,
        }

    # Metadata (data-driven, not hardcoded)
    metadata = {
        "min_cluster_sizes": config.min_cluster_sizes,
        "umap_n_components": config.umap_n_components,
        "umap_n_neighbors": config.umap_n_neighbors,
        "umap_min_dist": config.umap_min_dist,
        "pipeline_tpr_target": config.pipeline_tpr_target,
        "distance_metric": config.distance_metric,
        "regularization_epsilon": config.regularization_epsilon,
        "random_seed": config.random_seed,
        "pool_sizes": {role.value: len(pools[role].df) for role in H2PoolRole},
        "embedding_dim": int(pools[H2PoolRole.IND_TEST].embeddings.shape[1]),
        "section_details": {
            section.value: {
                "n_examples": len(pool.df),
                "n_classes": pool.df["subclass"].nunique(),
                "classes": sorted(pool.df["subclass"].unique().tolist()),
            }
            for section, pool in section_pools.items()
        },
    }

    return {
        "metadata": metadata,
        "clustering_results": all_results,
        "pipeline_analysis": all_pipeline_analysis,
        "prompt_consistency": all_consistency,
        "per_image_rows": all_per_image_rows,
    }
