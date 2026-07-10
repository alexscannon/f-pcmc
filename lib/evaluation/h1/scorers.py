"""Novelty scoring functions for OOD detection.

All scorers follow the same convention:
    - Input: query embeddings (Q, D) + pre-computed reference statistics
    - Output: (Q,) array of novelty scores
    - Higher score = more likely OOD
"""

import logging

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from reference_stats import ReferenceStatistics

logger = logging.getLogger("h1_ood")


def score_min_cosine_distance(
    query_embeddings: np.ndarray,
    ref_stats: ReferenceStatistics,
) -> np.ndarray:
    """Minimum cosine distance to nearest class centroid.

    For each query, compute cosine distance to all 100 IND class centroids
    and return the minimum. IND images should have small minimum distances;
    OOD images should have large minimum distances.
    """
    # (Q, C) distance matrix
    dists = pairwise_distances(
        query_embeddings, ref_stats.class_centroids, metric="cosine"
    )
    return dists.min(axis=1).astype(np.float32)


def score_mahalanobis(
    query_embeddings: np.ndarray,
    ref_stats: ReferenceStatistics,
    use_shared: bool = False,
) -> np.ndarray:
    """Minimum Mahalanobis distance to nearest class distribution.

    For each query, compute Mahalanobis distance to each of the 100 class
    distributions and return the minimum. Uses per-class or shared covariance.

    Mahalanobis distance: (x - mu)^T Sigma^{-1} (x - mu)
    """
    Q = query_embeddings.shape[0]
    n_classes = len(ref_stats.class_names)
    queries = query_embeddings.astype(np.float32)

    # (Q, C) matrix of Mahalanobis distances
    all_dists = np.full((Q, n_classes), np.inf, dtype=np.float32)

    for c_idx, class_name in enumerate(ref_stats.class_names):
        centroid = ref_stats.class_centroids[c_idx]  # (D,)
        diff = queries - centroid  # (Q, D)

        if use_shared:
            cov_inv = ref_stats.shared_covariance_inverse
        else:
            cov_inv = ref_stats.class_covariance_inverses[class_name]

        # (Q, D) @ (D, D) -> (Q, D), then element-wise multiply and sum
        left = diff @ cov_inv  # (Q, D)
        mahal_sq = np.sum(left * diff, axis=1)  # (Q,)

        # Clamp negative values from numerical noise
        mahal_sq = np.maximum(mahal_sq, 0.0)
        all_dists[:, c_idx] = mahal_sq

    return all_dists.min(axis=1)


def score_knn_density(
    query_embeddings: np.ndarray,
    ref_stats: ReferenceStatistics,
    k: int,
    metric: str = "cosine",
) -> np.ndarray:
    """Average distance to k nearest IND reference neighbors.

    Builds a brute-force kNN index from the 50k IND reference embeddings,
    queries each evaluation image, and returns the mean distance to its
    k nearest neighbors. IND images have nearby neighbors (low score);
    OOD images don't (high score).
    """
    knn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    knn.fit(ref_stats.reference_embeddings)

    distances, _ = knn.kneighbors(query_embeddings)  # (Q, k)
    return distances.mean(axis=1).astype(np.float32)
