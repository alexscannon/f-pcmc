"""Continual Learning Evaluation Pipeline — entry point.

Orchestrates: data loading → IND model initialization → stream processing
→ clustering/promotion → final evaluation and reporting.
"""

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

from clustering import ClusteringEvent
from config import ContinualConfig
from data_loader import NoveltyType, load_all_data, print_data_summary
from evaluation import (
    build_final_report,
    compute_classification_accuracy,
    compute_cluster_quality,
    compute_cluster_snapshots,
    compute_detection_metrics,
    compute_detection_metrics_per_score,
    compute_discovery_clustering_metrics,
    compute_oversegmentation_stats,
    generate_classification_timeline_plot,
    generate_score_distribution_plot,
    generate_tsne_snapshot,
)
from paradigms.base import WarmupData
from paradigms.factory import build_paradigm
from stream import (
    CumulativeMetrics,
    RollingWindowMetrics,
    StepLogger,
    StepRecord,
    build_stream,
)

LOG_DIR = Path("/home/alex/projects/msproject_misc/logs")


def log_results(results: dict) -> None:
    """Log formatted results summary."""
    meta = results["metadata"]
    det = results["detection_metrics"]
    cls_acc = results["classification_accuracy"]
    cluster = results["cluster_quality"]
    discovery = results.get("discovery_clustering_metrics", {})
    overseg = results.get("oversegmentation", {})

    tau = meta['threshold_tau']
    tau_str = f"{tau:.4f}" if tau is not None else "n/a"
    header = (
        f"\n{'=' * 80}\n"
        f"  Continual Learning Evaluation Results\n"
        f"{'=' * 80}\n"
        f"  Stream length:       {meta['stream_length']:,}\n"
        f"  Threshold tau:       {tau_str} "
        f"(percentile={meta['threshold_percentile']})\n"
        f"  Final IND classes:   {meta['n_final_classes']} "
        f"(original={meta['n_original_classes']}, "
        f"promoted={meta['n_final_classes'] - meta['n_original_classes']})\n"
        f"  OOD buffer residual: {meta['final_ood_buffer_size']}"
    )
    logger.info(header)

    # Detection metrics
    det_lines = [
        f"\n  {'─' * 76}",
        "  OOD Detection Metrics:",
        f"    {'Condition':>12s} {'AUROC':>8s} {'AUPR-OOD':>10s} "
        f"{'AUPR-IND':>10s} {'FPR@95':>8s} {'N_IND':>7s} {'N_OOD':>7s}",
    ]
    for condition in ["all_ood", "near_ood", "far_ood"]:
        if condition in det:
            m = det[condition]
            det_lines.append(
                f"    {condition:>12s} {m['auroc']:>8.4f} {m['aupr_ood_positive']:>10.4f} "
                f"{m['aupr_ind_positive']:>10.4f} {m['fpr_at_95_tpr']:>8.4f} "
                f"{m['n_ind']:>7,} {m['n_ood']:>7,}"
            )
    logger.info("\n".join(det_lines))

    # Classification accuracy
    ind_cls = cls_acc.get("ind_classification", {})
    fn_cls = cls_acc.get("ood_as_ind_classification", {})
    cl_cls = cls_acc.get("cluster_classification", {})
    dr_cls = cls_acc.get("buffer_drain_classification", {})
    cls_lines = [
        f"\n  {'─' * 76}",
        "  Classification Performance:",
        f"    Overall:      {cls_acc['accuracy']:.4f} "
        f"({cls_acc.get('n_correct', 0)}/{cls_acc['n_evaluated']})",
        f"    IND only:     {ind_cls.get('accuracy', 0):.4f} "
        f"({ind_cls.get('n_correct', 0)}/{ind_cls.get('n_evaluated', 0)})",
        f"    OOD as IND:   {fn_cls.get('accuracy', 0):.4f} "
        f"({fn_cls.get('n_correct', 0)}/{fn_cls.get('n_evaluated', 0)})",
        f"    Cluster:      {cl_cls.get('accuracy', 0):.4f} "
        f"({cl_cls.get('n_correct', 0)}/{cl_cls.get('n_evaluated', 0)})",
        f"    Buffer drain: {dr_cls.get('accuracy', 0):.4f} "
        f"({dr_cls.get('n_correct', 0)}/{dr_cls.get('n_evaluated', 0)})",
    ]
    logger.info("\n".join(cls_lines))

    # Clustering summary
    cl_lines = [
        f"\n  {'─' * 76}",
        "  Clustering Summary:",
        f"    Total clustering events: {cluster['n_clustering_events']}",
        f"    Promotion events:        {cluster['n_promotion_events']}",
        f"    Total clusters promoted:  {cluster['n_promoted_total']}",
    ]
    if cluster.get("events"):
        cl_lines.append("    Promoted clusters:")
        for event in cluster["events"]:
            for c in event["clusters"]:
                cl_lines.append(
                    f"      step={event['step']:>6d}  {c['id']:>14s}  "
                    f"size={c['size']:>3d}  "
                    f"intra_sim={c['intra_sim']:.3f}  "
                    f"soft_prob={c['soft_prob']:.3f}"
                )
    logger.info("\n".join(cl_lines))

    # Discovery clustering quality (over-segmentation aware) + coverage
    if discovery:
        disc_lines = [
            f"\n  {'─' * 76}",
            "  Discovery Clustering Quality (true OOD subclass vs final cluster id):",
            f"    {'condition':>10s} {'ARI':>7s} {'NMI':>7s} {'homog':>7s} "
            f"{'compl':>7s} {'V':>7s} {'#true':>6s} {'#pred':>6s}",
        ]
        for cond in ["all_ood", "near_ood", "far_ood"]:
            m = discovery.get(cond)
            if m:
                disc_lines.append(
                    f"    {cond:>10s} {m['ari']:>7.3f} {m['nmi']:>7.3f} "
                    f"{m['homogeneity']:>7.3f} {m['completeness']:>7.3f} "
                    f"{m['v_measure']:>7.3f} {m['n_true_classes']:>6d} "
                    f"{m['n_predicted_clusters']:>6d}"
                )
        if overseg:
            disc_lines.append(
                f"    coverage: {overseg.get('n_ood_classes_discovered', 0)}/"
                f"{overseg.get('n_true_ood_classes', 0)} OOD classes  |  "
                f"{overseg.get('n_promoted_clusters', 0)} clusters "
                f"({overseg.get('clusters_per_true_ood_class', 0):.2f}x), "
                f"{overseg.get('n_ind_contaminant_clusters', 0)} IND-contaminant, "
                f"{overseg.get('n_duplicate_label_clusters', 0)} duplicate-label"
            )
        logger.info("\n".join(disc_lines))

    # Cluster snapshots (instantiation vs. end-of-stream)
    snapshots = results.get("cluster_snapshots", [])
    if snapshots:
        snap_lines = [
            f"\n  {'─' * 76}",
            "  Cluster Snapshots (instantiation → end-of-stream):",
            f"    {'ID':>14s}  {'Step':>6s}  {'Label':>20s}  "
            f"{'Init Size':>9s}  {'Init Pur':>8s}  "
            f"{'Final Size':>10s}  {'Final Pur':>9s}  {'Post':>5s}",
        ]
        for s in snapshots:
            inst = s["instantiation"]
            eos = s["end_of_stream"]
            snap_lines.append(
                f"    {s['id']:>14s}  {s['promotion_step']:>6d}  "
                f"{s['majority_label']:>20s}  "
                f"{inst['n_members']:>9d}  {inst['purity']:>8.4f}  "
                f"{eos['n_members']:>10d}  {eos['purity']:>9.4f}  "
                f"{eos['n_post_promotion']:>5d}"
            )
        logger.info("\n".join(snap_lines))

    logger.info(f"\n{'=' * 80}")


def save_results(results: dict, clustering_events: list[ClusteringEvent], output_dir: Path) -> None:
    """Save results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # results_summary.json
    summary_path = output_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved {summary_path}")

    # clustering_events.json
    events_path = output_dir / "clustering_events.json"
    events_data = []
    for e in clustering_events:
        events_data.append({
            "step": e.step,
            "buffer_size": e.buffer_size,
            "n_sweep_clusters": e.n_sweep_clusters,
            "n_dedup_candidates": e.n_dedup_candidates,
            "n_promoted": e.n_promoted,
            "promoted": e.promoted,
        })
    with open(events_path, "w") as f:
        json.dump(events_data, f, indent=2)
    logger.info(f"Saved {events_path}")


def save_cluster_assignments(
    stream: list,
    final_clusters: list,
    cluster_label_map: dict[str, str],
    output_dir: Path,
) -> None:
    """Write per-example final cluster assignments to CSV.

    `final_cluster` is the raw assigned identity (original class name or a
    promoted_* id); `resolved_label` maps promoted ids to their majority label.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "final_cluster_assignments.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "stream_idx", "true_class", "true_superclass", "novelty_type",
            "final_cluster", "resolved_label",
        ])
        for i, item in enumerate(stream):
            fc = final_clusters[i]
            resolved = cluster_label_map.get(fc, fc) if fc is not None else ""
            writer.writerow([
                i, item.true_class, item.true_superclass,
                item.novelty_type.value, fc if fc is not None else "",
                resolved if resolved is not None else "",
            ])
    logger.info(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continual Learning Evaluation Pipeline"
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load data and print summary, no stream processing",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--oracle-mode",
        choices=["none", "detection", "clustering", "both"],
        default=None,
        help="Override config oracle_mode (oracle decomposition selector)",
    )
    parser.add_argument(
        "--paradigm",
        choices=["mahalanobis_hdbscan", "knn_dpmeans", "vmf_dpmm", "knn_vmf"],
        default=None,
        help="Override config paradigm selector",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Truncate the stream to the first N examples (drain still runs on "
             "the residual buffer). Dev/regression aid.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override config random_seed (stream shuffle). Dev/regression aid.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override config output_dir base (oracle suffix still appended). "
             "Dev/regression aid.",
    )
    parser.add_argument(
        "--stream-order",
        choices=["random", "sequential", "clustered"],
        default=None,
        help="Override config stream_order (post-warmup remainder ordering)",
    )
    parser.add_argument(
        "--stream-cluster-size", type=int, nargs=2, default=None,
        metavar=("LO", "HI"),
        help="Override config stream_cluster_size (clustered mode block range)",
    )
    args = parser.parse_args()

    # Configure loguru
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(LOG_DIR / "eval_continual.log", level="INFO", rotation="10 MB")

    config = ContinualConfig.from_yaml(args.config)

    # CLI overrides (dev/regression aids) applied before output-dir derivation.
    if args.paradigm is not None:
        config.paradigm = args.paradigm
    if args.seed is not None:
        config.random_seed = args.seed
    if args.stream_order is not None:
        config.stream_order = args.stream_order
    if args.stream_cluster_size is not None:
        config.stream_cluster_size = args.stream_cluster_size
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    # Route non-"random" stream orderings to a suffixed output dir so the i.i.d.
    # baseline and the ordered runs never overwrite each other. Applied before
    # the oracle suffix (e.g. continual_order_sequential_oracle_detection).
    if config.stream_order != "random":
        config.output_dir = (
            config.output_dir.parent
            / f"{config.output_dir.name}_order_{config.stream_order}"
        )

    # CLI override of oracle mode, then route outputs to a mode-specific dir so
    # the baseline and oracle runs never overwrite each other.
    if args.oracle_mode is not None:
        config.oracle_mode = args.oracle_mode
    if config.oracle_mode != "none":
        config.output_dir = (
            config.output_dir.parent
            / f"{config.output_dir.name}_oracle_{config.oracle_mode}"
        )
    logger.info(f"Oracle mode: {config.oracle_mode} (output dir: {config.output_dir})")

    # Check for existing results
    summary_path = config.output_dir / "results_summary.json"
    if summary_path.exists() and not args.force and not args.dry_run:
        logger.info(f"Results already exist at {summary_path}. Use --force to overwrite.")
        return

    # Load data
    logger.info("Loading embedding data...")
    train_data, stream_items = load_all_data(config)
    print_data_summary(train_data, stream_items)

    if args.dry_run:
        logger.info("[dry-run] Exiting before stream processing.")
        return

    # Build the paradigm (detection + discovery) and warm it up from the
    # training embeddings. class_names is the sorted unique subclass set; labels
    # index into it row-by-row (preserving order so warmup numerics are exact).
    logger.info(f"Building paradigm '{config.paradigm}' and warming up...")
    class_names = sorted(set(train_data.subclass_names.tolist()))
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    labels = np.array(
        [name_to_idx[n] for n in train_data.subclass_names], dtype=int
    )
    warmup_data = WarmupData(
        embeddings=train_data.embeddings,
        labels=labels,
        class_names=class_names,
        train_data=train_data,
    )
    paradigm = build_paradigm(config.paradigm, config, warmup_data)
    threshold_repr = (
        f"{paradigm.detection_threshold:.4f}"
        if paradigm.detection_threshold is not None
        else "n/a"
    )
    logger.info(
        f"Paradigm ready: {paradigm.n_ind_classes} classes, "
        f"threshold tau={threshold_repr}"
    )

    # Build the ordered stream (warmup prefix + ordered remainder)
    stream = build_stream(
        stream_items,
        config.random_seed,
        config.ind_warmup_count,
        order=config.stream_order,
        cluster_size=tuple(config.stream_cluster_size),
    )
    order_repr = config.stream_order
    if config.stream_order == "clustered":
        order_repr += f"{tuple(config.stream_cluster_size)}"
    if config.ind_warmup_count > 0:
        logger.info(
            f"Stream built: {len(stream):,} examples "
            f"(seed={config.random_seed}, order={order_repr}, "
            f"first {config.ind_warmup_count} IND_REAL warmup)"
        )
    else:
        logger.info(
            f"Stream built: {len(stream):,} examples "
            f"(seed={config.random_seed}, order={order_repr})"
        )

    if args.limit is not None:
        stream = stream[: args.limit]
        logger.info(f"Stream truncated to first {len(stream):,} examples (--limit)")

    # Initialize pipeline components. Paradigms may declare auxiliary score
    # columns via `extra_score_names` (empty for those that don't opt in).
    step_logger = StepLogger(
        config.output_dir,
        config.csv_flush_interval,
        extra_columns=getattr(paradigm, "extra_score_names", []),
    )
    rolling = RollingWindowMetrics(config.rolling_window)
    cumulative = CumulativeMetrics()
    all_records: list[StepRecord] = []
    clustering_events: list[ClusteringEvent] = []

    # The OOD buffer and promotion counter are owned by the paradigm. main keeps
    # the promoted centroids (one per promotion, in order) for the final t-SNE.
    promoted_centroids: list[np.ndarray] = []

    # Cluster label tracking (majority label is immutable once set)
    cluster_label_map: dict[str, str] = {}
    # Subclasses already discovered/promoted — read by the detection oracle so
    # post-promotion instances of a known class route to IND, not the buffer.
    promoted_subclasses: set[str] = set()
    cluster_cls_correct = 0
    cluster_cls_total = 0
    drain_cls_correct = 0
    drain_cls_total = 0

    # Per-cluster tracking for instantiation vs. end-of-stream snapshots
    cluster_initial_true_classes: dict[str, list[str]] = {}
    cluster_post_true_classes: dict[str, list[str]] = {}
    cluster_promotion_steps: dict[str, int] = {}

    # Per-example final cluster identity (raw id: original class name, a
    # promoted_* id, or None if never assigned). Feeds over-segmentation-aware
    # discovery clustering metrics (ARI/NMI/V-measure).
    final_clusters: list[str | None] = [None] * len(stream)

    # t-SNE snapshot tracking
    all_embeddings_for_tsne: list[np.ndarray] = []
    all_novelty_for_tsne: list[str] = []
    midpoint = len(stream) // 2

    ind_types = {NoveltyType.IND_REAL.value, NoveltyType.IND_SYNTHETIC.value}
    warmup_end = config.ind_warmup_count

    try:
        for t, item in enumerate(tqdm(stream, desc="Stream", unit="ex")):
            phase = "warmup" if t < warmup_end else "stream"

            # Detection oracle: route IND/OOD by ground truth, promotion-aware.
            # An example is IND if its true class is an original IND class OR a
            # subclass already promoted (now a known class). Det-acc metric uses
            # the same promotion-aware GT in oracle modes; static GT otherwise.
            # The paradigm consumes the resolved true_is_ood; the promotion-aware
            # bookkeeping (promoted_subclasses) stays here.
            gt_is_ind_static = item.novelty_type.value in ind_types
            if config.oracle_mode in {"detection", "both"}:
                is_known = gt_is_ind_static or (item.true_class in promoted_subclasses)
                gt_for_metric = is_known
            else:
                gt_for_metric = gt_is_ind_static

            # Step through the paradigm (detection + buffering + clustering).
            if config.oracle_mode == "none":
                result = paradigm.step(item.embedding, t)
            else:
                true_is_ood = (
                    (not is_known)
                    if config.oracle_mode in {"detection", "both"}
                    else False  # ignored by the paradigm in clustering-only mode
                )
                result = paradigm.step_oracle(
                    item.embedding, t, true_is_ood, item.true_class
                )

            # Resolve classification name from the predicted class index.
            if not result.is_ood:
                predicted_name = paradigm.class_names[result.predicted_class]
                # Resolve promoted-cluster ids to their majority label so that
                # classifications into promoted clusters are credited (consistent
                # with the drain phase). Original class names are not in the map
                # and pass through unchanged.
                predicted_class = cluster_label_map.get(predicted_name, predicted_name)
                final_clusters[t] = predicted_name
                if predicted_name in cluster_post_true_classes:
                    cluster_post_true_classes[predicted_name].append(item.true_class)
            else:
                predicted_class = ""

            # Build step record (n_ind_classes / n_ood_buffer are paradigm
            # snapshots taken before any promotion this step, matching log order).
            record = StepRecord(
                t=t,
                score=result.score,
                is_ood=result.is_ood,
                predicted_class=predicted_class,
                true_class=item.true_class,
                true_superclass=item.true_superclass,
                novelty_type=item.novelty_type.value,
                n_ind_classes=result.n_ind_classes,
                n_ood_buffer=result.n_ood_buffer,
                cum_det_acc=0.0,
                cum_cls_acc=0.0,
                phase=phase,
                rolling_cls_acc=0.0,
                extras=result.extras,
            )
            rolling.update(record, gt_for_metric)
            cumulative.update(record, gt_for_metric)
            record.cum_det_acc = cumulative.detection_accuracy
            record.cum_cls_acc = cumulative.classification_accuracy
            record.rolling_cls_acc = rolling.classification_accuracy
            step_logger.log(record)
            all_records.append(record)

            # Track for t-SNE
            all_embeddings_for_tsne.append(item.embedding)
            all_novelty_for_tsne.append(item.novelty_type.value)

            # Process any promotions emitted this step (GT bookkeeping + metric
            # injection stay here; the paradigm has already applied the mechanics
            # internally). Promotions arrive in evaluate_promotion order.
            if result.cluster_event is not None:
                promoted_info = []
                for promo in result.promotions_this_step:
                    cid = promo.cid
                    promoted_centroids.append(promo.centroid)

                    member_stream_indices = promo.member_stream_indices
                    for si in member_stream_indices:
                        final_clusters[si] = cid
                    member_true_classes = [
                        stream[si].true_class for si in member_stream_indices
                    ]
                    majority_label = (
                        collections.Counter(member_true_classes).most_common(1)[0][0]
                    )
                    cluster_label_map[cid] = majority_label
                    promoted_subclasses.add(majority_label)
                    cluster_initial_true_classes[cid] = list(member_true_classes)
                    cluster_post_true_classes[cid] = []
                    cluster_promotion_steps[cid] = t

                    # Score cluster purity and update classification metrics
                    n_cls_correct = sum(
                        1 for tc in member_true_classes if tc == majority_label
                    )
                    n_cls_total = len(member_true_classes)
                    purity = n_cls_correct / n_cls_total

                    cumulative.update_classification_batch(n_cls_correct, n_cls_total)
                    for tc in member_true_classes:
                        rolling.update_classification_only(tc == majority_label)
                    cluster_cls_correct += n_cls_correct
                    cluster_cls_total += n_cls_total

                    promoted_info.append({
                        "id": cid,
                        "size": promo.n_members,
                        "intra_sim": round(promo.intra_cosine_sim, 4),
                        "soft_prob": round(promo.mean_soft_prob, 4),
                        "mcs": promo.min_cluster_size_used,
                        "majority_label": majority_label,
                        "purity": round(purity, 4),
                    })

                ce = result.cluster_event
                event = ClusteringEvent(
                    step=t,
                    buffer_size=ce.buffer_size,
                    n_sweep_clusters=ce.sweep_counts,
                    n_dedup_candidates=ce.n_dedup_candidates,
                    n_promoted=len(result.promotions_this_step),
                    promoted=promoted_info,
                )
                clustering_events.append(event)

                logger.info(
                    f"Step {t}: Clustering triggered — "
                    f"buffer={ce.buffer_size}, candidates={ce.n_dedup_candidates}, "
                    f"promoted={len(result.promotions_this_step)}, "
                    f"buffer_after={paradigm.ood_buffer_size}"
                )

            # Periodic status
            if t > 0 and t % 2000 == 0:
                logger.info(
                    f"Step {t}: det_acc={rolling.detection_accuracy:.3f}, "
                    f"cls_acc={rolling.classification_accuracy:.3f}, "
                    f"n_classes={paradigm.n_ind_classes}, "
                    f"ood_buffer={paradigm.ood_buffer_size}"
                )

            # t-SNE at midpoint
            if t == midpoint:
                logger.info(f"Generating midpoint t-SNE snapshot (step {t})...")
                generate_tsne_snapshot(
                    embeddings=np.stack(all_embeddings_for_tsne),
                    novelty_types=list(all_novelty_for_tsne),
                    promoted_centroids=None,
                    output_path=config.output_dir / "plots" / "tsne_midpoint.png",
                    config=config,
                    title=f"t-SNE at Midpoint (step {t})",
                )

        logger.info(
            f"Stream complete: {len(all_records):,} steps processed, "
            f"{paradigm.n_ind_classes} final classes, "
            f"{paradigm.ood_buffer_size} items in residual buffer"
        )

        # --- Drain OOD buffer: force-classify remaining examples ---
        drain_assignments = paradigm.drain()  # [(stream_idx, predicted_class_idx)]
        if drain_assignments:
            drain_total = len(drain_assignments)
            logger.info(f"Draining OOD buffer: {drain_total} examples...")
            drain_start_t = len(stream)

            for drain_idx, (stream_idx, nearest_idx) in enumerate(drain_assignments):
                t_drain = drain_start_t + drain_idx
                predicted_name = paradigm.class_names[nearest_idx]
                predicted_label = cluster_label_map.get(predicted_name, predicted_name)
                final_clusters[stream_idx] = predicted_name
                true_class = stream[stream_idx].true_class
                true_superclass = stream[stream_idx].true_superclass
                novelty_type = stream[stream_idx].novelty_type.value
                if predicted_name in cluster_post_true_classes:
                    cluster_post_true_classes[predicted_name].append(true_class)

                is_correct = int(predicted_label == true_class)
                drain_cls_correct += is_correct
                drain_cls_total += 1

                cumulative.update_classification_batch(is_correct, 1)
                rolling.update_classification_only(bool(is_correct))

                record = StepRecord(
                    t=t_drain,
                    score=0.0,
                    is_ood=False,
                    predicted_class=predicted_label,
                    true_class=true_class,
                    true_superclass=true_superclass,
                    novelty_type=novelty_type,
                    n_ind_classes=paradigm.n_ind_classes,
                    n_ood_buffer=drain_total - drain_idx - 1,
                    cum_det_acc=cumulative.detection_accuracy,
                    cum_cls_acc=cumulative.classification_accuracy,
                    phase="drain",
                    rolling_cls_acc=rolling.classification_accuracy,
                )
                step_logger.log(record)
                all_records.append(record)

            logger.info(
                f"Buffer drain complete: {drain_cls_correct}/{drain_cls_total} correct "
                f"({drain_cls_correct / drain_cls_total:.4f})"
            )

    finally:
        step_logger.close()

    # --- Final evaluation ---
    logger.info("Computing final evaluation metrics...")

    # Exclude drain records from detection metrics (score=0.0, not real detections)
    stream_records = [r for r in all_records if r.phase != "drain"]
    detection_metrics = compute_detection_metrics(stream_records)
    # Per-score-variant detection metrics ({} unless the paradigm logs extras).
    detection_metrics_per_score = compute_detection_metrics_per_score(stream_records)
    classification_metrics = compute_classification_accuracy(
        stream_records,
        cluster_cls_correct=cluster_cls_correct,
        cluster_cls_total=cluster_cls_total,
        drain_cls_correct=drain_cls_correct,
        drain_cls_total=drain_cls_total,
    )

    # Cluster quality (over the paradigm's residual buffer at end-of-stream)
    residual_buffer_indices = paradigm.ood_buffer_stream_indices
    ood_buffer_gt = [stream[i].true_class for i in residual_buffer_indices]
    cluster_quality = compute_cluster_quality(
        clustering_events,
        residual_buffer_indices,
        ood_buffer_gt,
        stream,
    )

    # Cluster snapshots (instantiation vs. end-of-stream)
    cluster_snapshots = compute_cluster_snapshots(
        cluster_initial_true_classes,
        cluster_post_true_classes,
        cluster_label_map,
        cluster_promotion_steps,
    )

    # Over-segmentation-aware discovery clustering metrics + coverage stats
    novelty_all = [s.novelty_type.value for s in stream]
    trueclass_all = [s.true_class for s in stream]
    n_unassigned = sum(1 for fc in final_clusters if fc is None)
    if n_unassigned:
        logger.warning(f"{n_unassigned} examples have no final cluster assignment")
    discovery_clustering_metrics = compute_discovery_clustering_metrics(
        novelty_all, trueclass_all, final_clusters
    )
    oversegmentation = compute_oversegmentation_stats(
        cluster_snapshots, novelty_all, trueclass_all
    )

    # Build and save report
    results = build_final_report(
        detection_metrics=detection_metrics,
        classification_metrics=classification_metrics,
        cluster_quality=cluster_quality,
        clustering_events=clustering_events,
        config=config,
        stream_length=len(stream),
        threshold=paradigm.detection_threshold,
        n_final_classes=paradigm.n_ind_classes,
        final_buffer_size=paradigm.ood_buffer_size,
        cluster_snapshots=cluster_snapshots,
        discovery_clustering_metrics=discovery_clustering_metrics,
        oversegmentation=oversegmentation,
        detection_metrics_per_score=detection_metrics_per_score,
    )

    log_results(results)
    save_results(results, clustering_events, config.output_dir)

    # Per-image final cluster assignments (one row per stream example)
    save_cluster_assignments(
        stream, final_clusters, cluster_label_map, config.output_dir
    )

    # Generate plots
    logger.info("Generating visualizations...")
    generate_score_distribution_plot(
        all_records, paradigm.detection_threshold, config.output_dir
    )
    generate_classification_timeline_plot(
        output_dir=config.output_dir,
        rolling_window=config.rolling_window,
        clustering_events=clustering_events,
        ind_warmup_count=config.ind_warmup_count,
    )

    logger.info("Generating final t-SNE snapshot...")
    generate_tsne_snapshot(
        embeddings=np.stack(all_embeddings_for_tsne),
        novelty_types=list(all_novelty_for_tsne),
        promoted_centroids=np.stack(promoted_centroids) if promoted_centroids else None,
        output_path=config.output_dir / "plots" / "tsne_final.png",
        config=config,
        title=f"t-SNE at End (step {len(stream)})",
    )

    logger.info(f"All outputs saved to {config.output_dir}")


if __name__ == "__main__":
    main()
