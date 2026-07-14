"""Post-stream evaluation: detection metrics, cluster quality, and visualization."""

import collections
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    roc_auc_score,
    roc_curve,
)

from clustering import ClusteringEvent
from config import ContinualConfig
from data_loader import NoveltyType, StreamItem
from stream import StepRecord


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------

def _stratified_detection_metrics(
    scores: np.ndarray, novelty_types: list[str]
) -> dict:
    """All/near/far-OOD detection metrics from a score array aligned to records."""
    ind_types = {NoveltyType.IND_REAL.value, NoveltyType.IND_SYNTHETIC.value}
    gt_is_ood = np.array([nt not in ind_types for nt in novelty_types], dtype=np.int32)
    ind_scores = scores[gt_is_ood == 0]

    results = {}

    # All OOD
    ood_mask = gt_is_ood == 1
    if ood_mask.sum() > 0:
        results["all_ood"] = _ood_metrics(ind_scores, scores[ood_mask])

    # Near-OOD only
    near_mask = np.array([nt == NoveltyType.NEAR_OOD.value for nt in novelty_types])
    if near_mask.sum() > 0:
        results["near_ood"] = _ood_metrics(ind_scores, scores[near_mask])

    # Far-OOD only
    far_mask = np.array([nt == NoveltyType.FAR_OOD.value for nt in novelty_types])
    if far_mask.sum() > 0:
        results["far_ood"] = _ood_metrics(ind_scores, scores[far_mask])

    return results


def compute_detection_metrics(records: list[StepRecord]) -> dict:
    """Compute AUROC, AUPR, FPR@95TPR for IND/OOD detection.

    Stratified by: all OOD, near-OOD only, far-OOD only. Uses the primary
    ``score`` field.
    """
    scores = np.array([r.score for r in records])
    novelty_types = [r.novelty_type for r in records]
    return _stratified_detection_metrics(scores, novelty_types)


def compute_detection_metrics_per_score(records: list[StepRecord]) -> dict:
    """Per-score-variant detection metrics, keyed by score-column name.

    Returns ``{}`` when no record carries ``extras`` — so paradigms that don't
    log auxiliary scores leave ``results_summary.json`` byte-identical. When
    extras are present, returns ``{"score": {...}, "<extra>": {...}, ...}``: the
    primary ``score`` column plus each auxiliary candidate score, every value a
    stratified all/near/far block.
    """
    score_names: list[str] = []
    for r in records:
        for k in r.extras:
            if k not in score_names:
                score_names.append(k)
    if not score_names:
        return {}

    novelty_types = [r.novelty_type for r in records]
    out = {
        "score": _stratified_detection_metrics(
            np.array([r.score for r in records]), novelty_types
        )
    }
    for name in score_names:
        scores = np.array([r.extras.get(name, np.nan) for r in records])
        out[name] = _stratified_detection_metrics(scores, novelty_types)
    return out


def _ood_metrics(ind_scores: np.ndarray, ood_scores: np.ndarray) -> dict:
    """Compute OOD detection metrics from IND and OOD score arrays."""
    labels = np.concatenate([
        np.zeros(len(ind_scores), dtype=np.int32),
        np.ones(len(ood_scores), dtype=np.int32),
    ])
    all_scores = np.concatenate([ind_scores, ood_scores])

    auroc = float(roc_auc_score(labels, all_scores))
    aupr_ood = float(average_precision_score(labels, all_scores))
    aupr_ind = float(average_precision_score(1 - labels, -all_scores))

    fpr, tpr, _ = roc_curve(labels, all_scores)
    idx = np.where(tpr >= 0.95)[0]
    fpr_at_95 = float(fpr[idx[0]]) if len(idx) > 0 else 1.0

    return {
        "auroc": auroc,
        "aupr_ood_positive": aupr_ood,
        "aupr_ind_positive": aupr_ind,
        "fpr_at_95_tpr": fpr_at_95,
        "n_ind": len(ind_scores),
        "n_ood": len(ood_scores),
    }


# ---------------------------------------------------------------------------
# Classification accuracy
# ---------------------------------------------------------------------------

def compute_classification_accuracy(
    records: list[StepRecord],
    cluster_cls_correct: int = 0,
    cluster_cls_total: int = 0,
    drain_cls_correct: int = 0,
    drain_cls_total: int = 0,
) -> dict:
    """Compute classification accuracy across all sources.

    Combines four sources:
    - IND classification: true-IND examples correctly predicted as IND
    - OOD false negatives: true-OOD examples predicted as IND (almost always wrong)
    - Cluster classification: promoted cluster members scored by purity
    - Buffer drain: residual OOD buffer force-classified at stream end
    """
    ind_types = {NoveltyType.IND_REAL.value, NoveltyType.IND_SYNTHETIC.value}

    # All examples predicted as IND (not is_ood), split by ground truth
    predicted_ind = [r for r in records if not r.is_ood]
    true_ind = [r for r in predicted_ind if r.novelty_type in ind_types]
    false_neg = [r for r in predicted_ind if r.novelty_type not in ind_types]

    ind_correct = sum(1 for r in true_ind if r.predicted_class == r.true_class)
    ind_total = len(true_ind)

    fn_correct = sum(1 for r in false_neg if r.predicted_class == r.true_class)
    fn_total = len(false_neg)

    # Overall
    total_correct = ind_correct + fn_correct + cluster_cls_correct + drain_cls_correct
    total_evaluated = ind_total + fn_total + cluster_cls_total + drain_cls_total
    overall_accuracy = total_correct / total_evaluated if total_evaluated > 0 else 0.0

    def _sub_dict(n_correct: int, n_evaluated: int) -> dict:
        return {
            "accuracy": n_correct / n_evaluated if n_evaluated > 0 else 0.0,
            "n_correct": n_correct,
            "n_evaluated": n_evaluated,
        }

    return {
        "accuracy": overall_accuracy,
        "n_correct": total_correct,
        "n_evaluated": total_evaluated,
        "ind_classification": _sub_dict(ind_correct, ind_total),
        "ood_as_ind_classification": _sub_dict(fn_correct, fn_total),
        "cluster_classification": _sub_dict(cluster_cls_correct, cluster_cls_total),
        "buffer_drain_classification": _sub_dict(drain_cls_correct, drain_cls_total),
    }


# ---------------------------------------------------------------------------
# Cluster quality
# ---------------------------------------------------------------------------

def compute_cluster_quality(
    clustering_events: list[ClusteringEvent],
    ood_buffer_stream_indices: list[int],
    ood_buffer_gt_classes: list[str],
    stream_items: list[StreamItem],
) -> dict:
    """Compute quality metrics for promoted clusters using ground-truth labels.

    Args:
        clustering_events: All clustering events with promotion info.
        ood_buffer_stream_indices: Stream indices for OOD buffer items at each
            clustering event (reconstructed from promotion records).
        ood_buffer_gt_classes: Ground-truth class names for buffer items.
        stream_items: Full stream for ground-truth lookup.
    """
    total_promoted = sum(e.n_promoted for e in clustering_events)
    if total_promoted == 0:
        return {
            "n_promoted_total": 0,
            "n_clustering_events": len(clustering_events),
            "n_promotion_events": 0,
            "events": [],
        }

    event_results = []
    for event in clustering_events:
        if event.n_promoted == 0:
            continue
        event_results.append({
            "step": event.step,
            "buffer_size": event.buffer_size,
            "n_promoted": event.n_promoted,
            "clusters": event.promoted,
        })

    # Aggregate recall: how many OOD examples ended up promoted
    near_ood_total = sum(
        1 for s in stream_items if s.novelty_type == NoveltyType.NEAR_OOD
    )
    far_ood_total = sum(
        1 for s in stream_items if s.novelty_type == NoveltyType.FAR_OOD
    )

    return {
        "n_promoted_total": total_promoted,
        "n_clustering_events": len(clustering_events),
        "n_promotion_events": len(event_results),
        "near_ood_total": near_ood_total,
        "far_ood_total": far_ood_total,
        "events": event_results,
    }


def compute_cluster_snapshots(
    cluster_initial_true_classes: dict[str, list[str]],
    cluster_post_true_classes: dict[str, list[str]],
    cluster_label_map: dict[str, str],
    cluster_promotion_steps: dict[str, int],
) -> list[dict]:
    """Compute per-cluster statistics at instantiation and end of stream.

    Returns a list of snapshot dicts (one per promoted cluster) each containing
    'instantiation' and 'end_of_stream' sub-dicts with size, purity, and
    class distribution for side-by-side comparison.
    """
    snapshots = []
    for cid in sorted(cluster_initial_true_classes.keys()):
        initial_classes = cluster_initial_true_classes[cid]
        post_classes = cluster_post_true_classes.get(cid, [])
        all_classes = initial_classes + post_classes
        majority_label = cluster_label_map[cid]

        n_initial = len(initial_classes)
        initial_correct = sum(1 for tc in initial_classes if tc == majority_label)

        n_total = len(all_classes)
        final_correct = sum(1 for tc in all_classes if tc == majority_label)

        snapshots.append({
            "id": cid,
            "majority_label": majority_label,
            "promotion_step": cluster_promotion_steps[cid],
            "instantiation": {
                "n_members": n_initial,
                "n_correct": initial_correct,
                "purity": round(initial_correct / n_initial, 4) if n_initial > 0 else 0.0,
                "class_distribution": dict(collections.Counter(initial_classes)),
            },
            "end_of_stream": {
                "n_members": n_total,
                "n_post_promotion": len(post_classes),
                "n_correct": final_correct,
                "purity": round(final_correct / n_total, 4) if n_total > 0 else 0.0,
                "class_distribution": dict(collections.Counter(all_classes)),
            },
        })
    return snapshots


# ---------------------------------------------------------------------------
# Discovery clustering quality (over-segmentation aware)
# ---------------------------------------------------------------------------

def compute_discovery_clustering_metrics(
    novelty_types: list[str],
    true_classes: list[str],
    final_clusters: list[str | None],
) -> dict:
    """ARI/NMI/homogeneity/completeness/V-measure of discovered clusters vs the
    true OOD subclass partition.

    Computed over OOD examples (all / near / far). The predicted label for each
    example is its *raw* final cluster identity (a promoted cluster id like
    ``promoted_012``, or an original IND class name if the example was absorbed
    into a known class) — NOT the resolved majority label. Using raw ids is what
    makes these metrics over-segmentation aware: splitting one true subclass
    across several promoted clusters, or re-promoting a class under multiple
    ids, lowers completeness / ARI / V-measure even when per-cluster purity is
    1.0. Examples never assigned (None) are bucketed as ``__unassigned__``.
    """
    def _metrics(keep: set[str]) -> dict:
        yt, yp = [], []
        for nt, tc, fc in zip(novelty_types, true_classes, final_clusters):
            if nt in keep:
                yt.append(tc)
                yp.append(fc if fc is not None else "__unassigned__")
        out = {
            "n": len(yt),
            "n_true_classes": len(set(yt)),
            "n_predicted_clusters": len(set(yp)),
            "ari": 0.0, "nmi": 0.0,
            "homogeneity": 0.0, "completeness": 0.0, "v_measure": 0.0,
            # fraction of these OOD examples absorbed into an original IND class
            # (i.e. never recognized as novel)
            "frac_absorbed_into_ind": 0.0,
        }
        if len(yt) < 2:
            return out
        hom, comp, vme = homogeneity_completeness_v_measure(yt, yp)
        out["ari"] = float(adjusted_rand_score(yt, yp))
        out["nmi"] = float(normalized_mutual_info_score(yt, yp))
        out["homogeneity"] = float(hom)
        out["completeness"] = float(comp)
        out["v_measure"] = float(vme)
        # `yp` holds raw predicted ids: original IND class names, promoted_* ids,
        # or __unassigned__. Absorbed-into-IND = predicted is an original IND
        # class name (the model's starting classes are all IND subclasses).
        absorbed = sum(
            1 for p in yp
            if not str(p).startswith("promoted_") and p != "__unassigned__"
        )
        out["frac_absorbed_into_ind"] = round(absorbed / len(yp), 4)
        return out

    near = {NoveltyType.NEAR_OOD.value}
    far = {NoveltyType.FAR_OOD.value}
    return {
        "all_ood": _metrics(near | far),
        "near_ood": _metrics(near),
        "far_ood": _metrics(far),
    }


def compute_oversegmentation_stats(
    cluster_snapshots: list[dict],
    novelty_types: list[str],
    true_classes: list[str],
) -> dict:
    """Coverage and over-segmentation summary for promoted clusters.

    Pairs with classification accuracy so the headline number can be read in
    context: purity / nearest-prototype accuracy is monotone in cluster count,
    so over-segmentation (duplicate-label clusters, IND contaminants) inflates
    it without reflecting genuine 1:1 class discovery.
    """
    ind_types = {NoveltyType.IND_REAL.value, NoveltyType.IND_SYNTHETIC.value}
    tc_is_ood: dict[str, bool] = {}
    for nt, tc in zip(novelty_types, true_classes):
        tc_is_ood.setdefault(tc, nt not in ind_types)
    n_true_ood = sum(1 for is_ood in tc_is_ood.values() if is_ood)

    labels = [s["majority_label"] for s in cluster_snapshots]
    label_counts = collections.Counter(labels)
    ood_labels = {l for l in labels if tc_is_ood.get(l, False)}
    n_ind_contaminant = sum(1 for l in labels if not tc_is_ood.get(l, False))
    n_duplicate = sum(1 for _, c in label_counts.items() if c > 1)
    n_promoted = len(cluster_snapshots)
    n_distinct = len(label_counts)

    return {
        "n_true_ood_classes": n_true_ood,
        "n_promoted_clusters": n_promoted,
        "n_distinct_majority_labels": n_distinct,
        "n_ood_classes_discovered": len(ood_labels),
        "ood_class_coverage": round(len(ood_labels) / n_true_ood, 4) if n_true_ood else 0.0,
        "n_ind_contaminant_clusters": n_ind_contaminant,
        "n_duplicate_label_clusters": n_duplicate,
        "clusters_per_true_ood_class": round(n_promoted / n_true_ood, 4) if n_true_ood else 0.0,
        "clusters_per_discovered_label": round(n_promoted / n_distinct, 4) if n_distinct else 0.0,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def generate_score_distribution_plot(
    records: list[StepRecord],
    threshold: float | None,
    output_dir: Path,
) -> Path:
    """Generate histogram of detection scores by novelty type.

    threshold may be None for paradigms without a scalar detection threshold
    (e.g. vMF-DPMM); in that case the threshold line is omitted.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "score_distributions.png"

    fig, ax = plt.subplots(figsize=(10, 6))

    type_labels = {
        NoveltyType.IND_REAL.value: ("IND (real)", "#2196F3"),
        NoveltyType.IND_SYNTHETIC.value: ("IND (synthetic)", "#4CAF50"),
        NoveltyType.NEAR_OOD.value: ("Near-OOD", "#FF9800"),
        NoveltyType.FAR_OOD.value: ("Far-OOD", "#F44336"),
    }

    for nt_val, (label, color) in type_labels.items():
        scores = [r.score for r in records if r.novelty_type == nt_val]
        if scores:
            ax.hist(scores, bins=100, alpha=0.5, label=label, color=color, density=True)

    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"threshold (τ={threshold:.1f})")
    ax.set_xlabel("Detection Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distributions by Novelty Type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    logger.info(f"Saved score distribution plot: {out_path}")
    return out_path


def generate_classification_timeline_plot(
    output_dir: Path,
    rolling_window: int,
    clustering_events: list[ClusteringEvent],
    ind_warmup_count: int,
) -> Path:
    """Generate line plot of classification accuracy over stream lifetime.

    Shows rolling window accuracy (primary) and cumulative accuracy (secondary),
    with vertical callouts for cluster promotions and shaded buffer drain region.
    """
    from matplotlib.ticker import MultipleLocator

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "classification_timeline.png"

    df = pd.read_csv(output_dir / "per_step.csv")
    plot_df = df[df["t"] >= ind_warmup_count].copy()

    # Remap drain t-values so they continue seamlessly from the stream end
    # without extending the x-axis beyond the actual stream length.
    stream_rows = plot_df[plot_df["phase"] != "drain"]
    drain_rows = plot_df[plot_df["phase"] == "drain"]
    stream_end_t = int(stream_rows["t"].max()) if not stream_rows.empty else 0

    fig, ax = plt.subplots(figsize=(12, 5))

    # --- Drain region shading ---
    if not drain_rows.empty:
        ax.axvspan(stream_end_t, stream_end_t + len(drain_rows),
                   alpha=0.10, color="#9E9E9E",
                   label="OOD buffer drain", zorder=0)
        ax.axvline(stream_end_t, color="#616161", linestyle="-",
                   linewidth=1.0, alpha=0.6)
        ax.annotate(
            "Buffer drain \u2192",
            xy=(stream_end_t, 0.5), xycoords=("data", "axes fraction"),
            xytext=(6, 0), textcoords="offset points",
            fontsize=8, color="#616161", ha="left", va="center",
            fontstyle="italic",
        )

    # --- Cumulative classification accuracy (secondary) ---
    ax.plot(
        plot_df["t"], plot_df["cum_cls_acc"],
        color="#6A1B9A", alpha=0.65, linewidth=1.2,
        label="Cumulative accuracy",
    )

    # --- Rolling classification accuracy (primary, bold) ---
    rolling_data = plot_df[plot_df["rolling_cls_acc"].notna()]
    ax.plot(
        rolling_data["t"], rolling_data["rolling_cls_acc"],
        color="#1976D2", alpha=1.0, linewidth=2.0,
        label=f"Rolling accuracy (w={rolling_window})",
    )

    # --- Cluster callout lines with overlap avoidance ---
    promotion_events = [
        e for e in clustering_events
        if e.n_promoted > 0 and e.step >= ind_warmup_count
    ]
    # Stagger y-positions when events are close; cycle through 3 tiers
    tier_y = [0.96, 0.90, 0.84]  # axes fraction positions
    min_step_gap = 600
    prev_step = -min_step_gap * 10
    tier_idx = 0
    for event in promotion_events:
        if event.step - prev_step < min_step_gap:
            tier_idx = (tier_idx + 1) % len(tier_y)
        else:
            tier_idx = 0
        prev_step = event.step

        label = "Cluster promotion" if event is promotion_events[0] else None
        ax.axvline(event.step, color="#E65100", linestyle="--",
                   linewidth=0.8, alpha=0.7, label=label)
        n = event.n_promoted
        ax.annotate(
            f"+{n}",
            xy=(event.step, tier_y[tier_idx]),
            xycoords=("data", "axes fraction"),
            xytext=(4, 0), textcoords="offset points",
            fontsize=7, fontweight="bold", color="#E65100",
            ha="left", va="center",
        )

    # --- Axes and labels ---
    ax.set_xlabel("Step (example number)")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("Data Stream Classification Performance", pad=12)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    logger.info(f"Saved classification timeline plot: {out_path}")
    return out_path


def generate_tsne_snapshot(
    embeddings: np.ndarray,
    novelty_types: list[str],
    promoted_centroids: np.ndarray | None,
    output_path: Path,
    config: ContinualConfig,
    title: str = "t-SNE Snapshot",
) -> None:
    """Generate a t-SNE visualization of stream embeddings.

    Uses PCA to 50 dims first, then t-SNE to 2D.
    """
    N = len(embeddings)
    if N == 0:
        return

    # Subsample if needed
    rng = np.random.default_rng(config.random_seed)
    if N > config.tsne_sample_size:
        idx = rng.choice(N, config.tsne_sample_size, replace=False)
        embeddings = embeddings[idx]
        novelty_types = [novelty_types[i] for i in idx]

    # PCA pre-reduction
    n_components = min(50, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=config.random_seed)
    reduced = pca.fit_transform(embeddings)

    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(config.tsne_perplexity, len(reduced) - 1),
        method="barnes_hut",
        random_state=config.random_seed,
        max_iter=1000,
    )
    coords = tsne.fit_transform(reduced)

    # Plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10))

    type_style = {
        NoveltyType.IND_REAL.value: ("IND (real)", "#2196F3", "o", 3, 0.2),
        NoveltyType.IND_SYNTHETIC.value: ("IND (synthetic)", "#4CAF50", "x", 10, 0.6),
        NoveltyType.NEAR_OOD.value: ("Near-OOD", "#FF9800", "D", 15, 0.8),
        NoveltyType.FAR_OOD.value: ("Far-OOD", "#F44336", "^", 15, 0.8),
    }

    for nt_val, (label, color, marker, size, alpha) in type_style.items():
        mask = [i for i, nt in enumerate(novelty_types) if nt == nt_val]
        if mask:
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=color, marker=marker, s=size, alpha=alpha, label=label,
            )

    # Overlay promoted centroids if available
    if promoted_centroids is not None and len(promoted_centroids) > 0:
        # Project centroids through same PCA (not t-SNE — approximate position)
        # Instead, mark as annotation text at center of their members
        pass  # t-SNE doesn't support out-of-sample projection cleanly

    ax.set_title(title)
    ax.legend(loc="upper right", markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(f"Saved t-SNE snapshot: {output_path}")


# ---------------------------------------------------------------------------
# Final report assembly
# ---------------------------------------------------------------------------

def build_final_report(
    detection_metrics: dict,
    classification_metrics: dict,
    cluster_quality: dict,
    clustering_events: list[ClusteringEvent],
    config: ContinualConfig,
    stream_length: int,
    threshold: float,
    n_final_classes: int,
    final_buffer_size: int,
    cluster_snapshots: list[dict] | None = None,
    discovery_clustering_metrics: dict | None = None,
    oversegmentation: dict | None = None,
    detection_metrics_per_score: dict | None = None,
) -> dict:
    """Assemble all metrics into a structured report."""
    report = {
        "metadata": {
            "oracle_mode": config.oracle_mode,
            "stream_length": stream_length,
            "random_seed": config.random_seed,
            "threshold_percentile": config.threshold_percentile,
            "threshold_tau": threshold,
            "regularization_epsilon": config.regularization_epsilon,
            "n_original_classes": 100,
            "n_final_classes": n_final_classes,
            "final_ood_buffer_size": final_buffer_size,
            "cluster_interval": config.cluster_interval,
            "min_cluster_sizes": config.min_cluster_sizes,
            "min_promote_size": config.min_promote_size,
            "min_intra_cosine_sim": config.min_intra_cosine_sim,
            "min_soft_prob": config.min_soft_prob,
        },
        "detection_metrics": detection_metrics,
        "classification_accuracy": classification_metrics,
        "cluster_quality": cluster_quality,
        "discovery_clustering_metrics": discovery_clustering_metrics or {},
        "oversegmentation": oversegmentation or {},
        "cluster_snapshots": cluster_snapshots or [],
    }
    # Only added for paradigms that log auxiliary scores; absent otherwise so
    # single-score runs (mahalanobis/knn) stay byte-identical for regression.
    if detection_metrics_per_score:
        report["detection_metrics_per_score"] = detection_metrics_per_score
    # Only recorded for non-default orderings so the i.i.d.-baseline report stays
    # byte-identical for regression (mirrors detection_metrics_per_score above).
    if config.stream_order != "random":
        report["metadata"]["stream_order"] = config.stream_order
        if config.stream_order == "clustered":
            report["metadata"]["stream_cluster_size"] = list(config.stream_cluster_size)
    return report
