"""HDBSCAN sweep, Jaccard deduplication, and cluster promotion evaluation.

The non-oracle path (``run_sweep`` / ``_deduplicate_clusters``) never accesses
ground-truth labels — it operates solely on embedding geometry from the OOD
buffer. The oracle path (``form_oracle_clusters``) intentionally uses true
subclass labels to form perfectly pure clusters; see its docstring.
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from umap import UMAP

from config import ContinualConfig

import warnings
warnings.filterwarnings("ignore", message="n_jobs value .* overridden to 1")

@dataclass
class ClusterCandidate:
    """A deduplicated cluster candidate from the HDBSCAN sweep."""

    member_indices: list[int]       # indices into the OOD buffer
    centroid_raw: np.ndarray        # (D,) mean of raw (unnormalized) member embeddings
    n_members: int
    intra_cosine_sim: float         # mean pairwise cosine similarity
    mean_soft_prob: float           # mean HDBSCAN membership probability
    min_cluster_size_used: int


@dataclass
class ClusteringEvent:
    """Record of a single clustering trigger."""

    step: int
    buffer_size: int
    n_sweep_clusters: dict[int, int]    # mcs -> n_clusters found
    n_dedup_candidates: int
    n_promoted: int
    promoted: list[dict]                # [{id, size, intra_sim, soft_prob}, ...]


def run_sweep(
    buffer_embeddings: np.ndarray,
    raw_embeddings: np.ndarray,
    config: ContinualConfig,
) -> list[ClusterCandidate]:
    """Run HDBSCAN sweep across min_cluster_size values and deduplicate.

    Args:
        buffer_embeddings: (N, D) embeddings in the OOD buffer (raw space).
        raw_embeddings: Same as buffer_embeddings (kept for clarity; used for
            centroid and intra-cosine computation in raw space).
        config: Pipeline configuration.

    Returns:
        Deduplicated cluster candidates sorted by n_members descending.
    """
    N = buffer_embeddings.shape[0]

    # Optional UMAP preprocessing to mitigate curse of dimensionality
    if config.umap_n_components is not None and N > config.umap_n_components:
        reducer = UMAP(
            n_components=config.umap_n_components,
            n_neighbors=min(config.umap_n_neighbors, N - 1),
            min_dist=config.umap_min_dist,
            metric="cosine",
            random_state=config.random_seed,
        )
        clusterable = normalize(reducer.fit_transform(buffer_embeddings), norm="l2")
    else:
        clusterable = normalize(buffer_embeddings, norm="l2")

    # Collect all clusters across sweep values
    all_clusters: list[tuple[set[int], int, np.ndarray]] = []  # (member_set, mcs, probs)

    sweep_counts: dict[int, int] = {}

    for mcs in config.min_cluster_sizes:
        if N < mcs:
            sweep_counts[mcs] = 0
            continue

        clusterer = HDBSCAN(
            min_cluster_size=mcs,
            metric="euclidean",
            store_centers="centroid",
            copy=True,
        )
        clusterer.fit(clusterable)
        labels = clusterer.labels_
        probs = clusterer.probabilities_

        unique_labels = set(labels)
        unique_labels.discard(-1)
        sweep_counts[mcs] = len(unique_labels)

        for label in unique_labels:
            member_mask = labels == label
            member_indices = set(np.where(member_mask)[0])
            member_probs = probs[member_mask]
            all_clusters.append((member_indices, mcs, member_probs))

    if not all_clusters:
        return []

    # Deduplicate via Jaccard overlap
    candidates = _deduplicate_clusters(
        all_clusters, raw_embeddings, config.jaccard_dedup_threshold
    )

    # Log sweep summary
    for mcs, count in sorted(sweep_counts.items()):
        logger.debug(f"  mcs={mcs}: {count} clusters")
    logger.debug(f"  After dedup: {len(candidates)} unique candidates")

    return candidates


def _deduplicate_clusters(
    all_clusters: list[tuple[set[int], int, np.ndarray]],
    raw_embeddings: np.ndarray,
    jaccard_threshold: float,
) -> list[ClusterCandidate]:
    """Deduplicate clusters from different mcs values via Jaccard overlap.

    For each group of duplicates, keep the version with highest mean soft prob.
    """
    n = len(all_clusters)
    if n == 0:
        return []

    # Build adjacency: which clusters are duplicates of each other
    merged = list(range(n))  # union-find parent

    def find(x):
        while merged[x] != x:
            merged[x] = merged[merged[x]]
            x = merged[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            merged[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            set_i, _, _ = all_clusters[i]
            set_j, _, _ = all_clusters[j]
            intersection = len(set_i & set_j)
            union_size = len(set_i | set_j)
            if union_size > 0 and intersection / union_size >= jaccard_threshold:
                union(i, j)

    # Group by root
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # For each group, pick the version with highest mean soft prob
    candidates: list[ClusterCandidate] = []
    for group_indices in groups.values():
        best_idx = max(
            group_indices,
            key=lambda i: float(np.mean(all_clusters[i][2])),
        )
        member_set, mcs, probs = all_clusters[best_idx]
        member_list = sorted(member_set)
        member_embs = raw_embeddings[member_list]

        # Compute intra-class cosine similarity
        intra_sim = _mean_pairwise_cosine(member_embs)
        mean_prob = float(np.mean(probs))

        centroid = member_embs.mean(axis=0)

        candidates.append(ClusterCandidate(
            member_indices=member_list,
            centroid_raw=centroid,
            n_members=len(member_list),
            intra_cosine_sim=intra_sim,
            mean_soft_prob=mean_prob,
            min_cluster_size_used=mcs,
        ))

    # Sort by size descending
    candidates.sort(key=lambda c: c.n_members, reverse=True)
    return candidates


def evaluate_promotion(
    candidates: list[ClusterCandidate],
    config: ContinualConfig,
) -> list[ClusterCandidate]:
    """Filter cluster candidates by promotion criteria."""
    promoted = []
    for c in candidates:
        if (
            c.n_members >= config.min_promote_size
            and c.intra_cosine_sim >= config.min_intra_cosine_sim
            and c.mean_soft_prob >= config.min_soft_prob
        ):
            promoted.append(c)
    return promoted


def form_oracle_clusters(
    buffer_embeddings: np.ndarray,
    buffer_true_classes: list[str],
    config: ContinualConfig,
) -> list[ClusterCandidate]:
    """Form 100%-pure cluster candidates by grouping the buffer on true subclass.

    This is the clustering oracle. Unlike ``run_sweep``, it intentionally uses
    ground-truth subclass labels: every candidate contains members of strictly
    one subclass. HDBSCAN/UMAP are bypassed entirely (the oracle is concerned
    only with cluster *formation*, not with the promotion trigger / scan launch).

    The geometric/confidence quality gates are set to their maximum
    (``intra_cosine_sim = 1.0``, ``mean_soft_prob = 1.0``) because the oracle
    guarantees a perfectly coherent cluster. Consequently, when
    ``evaluate_promotion`` is applied, only ``n_members >= min_promote_size``
    remains an effective gate.

    Args:
        buffer_embeddings: (N, D) raw embeddings in the OOD buffer.
        buffer_true_classes: length-N true subclass label per buffer item.
        config: Pipeline configuration (unused here; kept for signature parity).

    Returns:
        Disjoint, perfectly pure cluster candidates sorted by n_members
        descending. No Jaccard dedup is needed — groups are disjoint by
        construction.
    """
    groups: dict[str, list[int]] = {}
    for idx, label in enumerate(buffer_true_classes):
        groups.setdefault(label, []).append(idx)

    candidates: list[ClusterCandidate] = []
    for member_list in groups.values():
        member_indices = sorted(member_list)
        member_embs = buffer_embeddings[member_indices]
        centroid = member_embs.mean(axis=0)
        candidates.append(ClusterCandidate(
            member_indices=member_indices,
            centroid_raw=centroid,
            n_members=len(member_indices),
            intra_cosine_sim=1.0,
            mean_soft_prob=1.0,
            min_cluster_size_used=0,  # sentinel: oracle (no HDBSCAN sweep)
        ))

    candidates.sort(key=lambda c: c.n_members, reverse=True)
    return candidates


def _mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    """Compute mean pairwise cosine similarity for a set of embeddings."""
    if len(embeddings) < 2:
        return 1.0
    sim_matrix = cosine_similarity(embeddings)
    # Extract upper triangle (excluding diagonal)
    n = len(embeddings)
    triu_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[triu_indices]
    return float(np.mean(pairwise_sims))
