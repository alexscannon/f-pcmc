"""Compute IND reference statistics for novelty scoring."""

import logging
from dataclasses import dataclass

import numpy as np

from config import H1TestConfig
from data_loader import EmbeddingPool

logger = logging.getLogger("h1_ood")


@dataclass
class ReferenceStatistics:
    """Pre-computed IND reference statistics for novelty scoring.

    All statistics are computed from the IND reference pool (CIFAR-100 train
    split, 50k images, 100 subclasses).
    """

    # Per-class centroids: (C, D), row i = mean embedding of class i
    class_centroids: np.ndarray

    # Class label ordering (maps row index to subclass name)
    class_names: list[str]

    # Global centroid: (D,) mean of all IND reference embeddings
    global_centroid: np.ndarray

    # Per-class inverse covariance: class_name -> (D, D) regularized Sigma^{-1}
    class_covariance_inverses: dict[str, np.ndarray]

    # Shared (pooled) inverse covariance: (D, D) — mean of per-class covariances, inverted
    shared_covariance_inverse: np.ndarray

    # Full reference embeddings for kNN scorer
    reference_embeddings: np.ndarray   # (N_ref, D)

    # Integer class IDs matching reference_embeddings rows
    reference_labels: np.ndarray       # (N_ref,) int


def compute_reference_statistics(
    pool: EmbeddingPool,
    config: H1TestConfig,
) -> ReferenceStatistics:
    """Compute centroids, covariance matrices, and inverses from IND reference pool.

    Uses Tikhonov regularization (epsilon * I) to ensure invertibility of
    per-class covariance matrices, which are rank-deficient when D > N_class.
    Shared covariance is the mean of per-class covariances (not the covariance
    of the entire pool, which would be inflated by between-class variance).
    """
    embeddings = pool.embeddings  # (N, D)
    subclasses = pool.df["subclass"].values
    unique_classes = sorted(pool.df["subclass"].unique())
    n_classes = len(unique_classes)
    D = embeddings.shape[1]

    logger.info(
        f"Computing reference statistics: {n_classes} classes, "
        f"D={D}, epsilon={config.regularization_epsilon}"
    )

    # Map class names to integer IDs
    class_to_id = {name: i for i, name in enumerate(unique_classes)}
    reference_labels = np.array([class_to_id[s] for s in subclasses], dtype=np.int32)

    # Per-class centroids
    class_centroids = np.zeros((n_classes, D), dtype=np.float64)
    class_covariance_inverses: dict[str, np.ndarray] = {}
    per_class_covariances: list[np.ndarray] = []

    reg_matrix = config.regularization_epsilon * np.eye(D, dtype=np.float64)

    for class_name in unique_classes:
        class_id = class_to_id[class_name]
        mask = subclasses == class_name
        class_embs = embeddings[mask].astype(np.float64)  # (N_c, D)
        n_c = class_embs.shape[0]

        # Centroid
        centroid = class_embs.mean(axis=0)
        class_centroids[class_id] = centroid

        # Per-class covariance with regularization
        centered = class_embs - centroid  # (N_c, D)
        cov = (centered.T @ centered) / max(n_c - 1, 1)  # (D, D)
        cov_reg = cov + reg_matrix
        per_class_covariances.append(cov)

        # Invert
        class_covariance_inverses[class_name] = np.linalg.inv(cov_reg).astype(np.float32)

        if n_c < 10:
            logger.warning(
                f"Class '{class_name}' has only {n_c} samples — "
                f"covariance estimate may be unreliable"
            )

    # Global centroid
    global_centroid = embeddings.astype(np.float64).mean(axis=0)

    # Shared covariance: mean of per-class covariances (Lee et al. 2018)
    shared_cov = np.mean(per_class_covariances, axis=0)
    shared_cov_reg = shared_cov + reg_matrix
    shared_covariance_inverse = np.linalg.inv(shared_cov_reg).astype(np.float32)

    logger.info(
        f"Reference statistics computed: {n_classes} centroids, "
        f"{n_classes} per-class covariance inverses, 1 shared covariance inverse"
    )

    return ReferenceStatistics(
        class_centroids=class_centroids.astype(np.float32),
        class_names=unique_classes,
        global_centroid=global_centroid.astype(np.float32),
        class_covariance_inverses=class_covariance_inverses,
        shared_covariance_inverse=shared_covariance_inverse,
        reference_embeddings=embeddings,
        reference_labels=reference_labels,
    )
