"""IND model: Mahalanobis-based detection and nearest-prototype classification.

This module never accesses ground-truth labels. It maintains per-class means,
a shared precision matrix, and a fixed threshold for IND/OOD decisions.
"""

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from config import ContinualConfig
from data_loader import TrainData


@dataclass
class INDModel:
    """Mutable IND class model with Mahalanobis scoring.

    class_means and class_names grow when clusters are promoted.
    precision_matrix and threshold are FIXED after initialization.
    """

    class_means: np.ndarray         # (C, D) float64
    class_names: list[str]          # length C
    precision_matrix: np.ndarray    # (D, D) float64 — FIXED
    threshold: float                # tau — FIXED
    n_original_classes: int = field(init=False)

    def __post_init__(self):
        self.n_original_classes = len(self.class_names)


def initialize_ind_model(
    train_data: TrainData,
    config: ContinualConfig,
) -> INDModel:
    """Fit IND model from CIFAR-100 training embeddings.

    Computes per-class means, shared covariance (mean of per-class covariances,
    Lee et al. 2018), precision matrix, and Mahalanobis threshold at the
    configured percentile of training scores.
    """
    embeddings = train_data.embeddings.astype(np.float64)
    subclasses = train_data.subclass_names
    unique_classes = sorted(set(subclasses))
    n_classes = len(unique_classes)
    D = embeddings.shape[1]

    logger.info(
        f"Initializing IND model: {n_classes} classes, D={D}, "
        f"epsilon={config.regularization_epsilon}"
    )

    # Per-class means and covariances
    class_means = np.zeros((n_classes, D), dtype=np.float64)
    per_class_covariances: list[np.ndarray] = []
    reg_matrix = config.regularization_epsilon * np.eye(D, dtype=np.float64)

    for i, class_name in enumerate(unique_classes):
        mask = subclasses == class_name
        class_embs = embeddings[mask]
        n_c = class_embs.shape[0]

        centroid = class_embs.mean(axis=0)
        class_means[i] = centroid

        centered = class_embs - centroid
        cov = (centered.T @ centered) / max(n_c - 1, 1)
        per_class_covariances.append(cov)

    # Shared covariance: mean of per-class covariances (Lee et al. 2018)
    shared_cov = np.mean(per_class_covariances, axis=0)
    shared_cov_reg = shared_cov + reg_matrix
    precision_matrix = np.linalg.inv(shared_cov_reg)

    logger.info(f"Computed {n_classes} class centroids and shared precision matrix")

    # Score all training embeddings to set threshold
    logger.info("Scoring training embeddings for threshold calibration...")
    train_scores = _score_batch(embeddings, class_means, precision_matrix)
    threshold = float(np.percentile(train_scores, config.threshold_percentile))

    logger.info(
        f"Threshold tau = {threshold:.4f} "
        f"(percentile={config.threshold_percentile}, "
        f"score range=[{train_scores.min():.2f}, {train_scores.max():.2f}])"
    )

    return INDModel(
        class_means=class_means,
        class_names=list(unique_classes),
        precision_matrix=precision_matrix,
        threshold=threshold,
    )


def score(model: INDModel, z: np.ndarray) -> tuple[float, bool, int]:
    """Score a single embedding against the IND model.

    Returns:
        mahal_score: Minimum Mahalanobis distance to any class mean.
        is_ood: True if score >= threshold.
        nearest_class_idx: Index of the closest class in model.class_means.
    """
    z64 = z.astype(np.float64)
    diff = z64 - model.class_means  # (C, D)
    # Mahalanobis: (z - mu)^T Sigma^{-1} (z - mu) for each class
    # Vectorized: sum((diff @ precision) * diff, axis=1)
    transformed = diff @ model.precision_matrix  # (C, D)
    scores = np.sum(transformed * diff, axis=1)  # (C,)

    nearest_idx = int(np.argmin(scores))
    min_score = float(scores[nearest_idx])
    is_ood = min_score >= model.threshold

    return min_score, is_ood, nearest_idx


def add_promoted_class(
    model: INDModel,
    centroid: np.ndarray,
    class_name: str,
) -> None:
    """Add a promoted cluster as a new IND class.

    Appends the centroid to class_means and the name to class_names.
    Does NOT update precision_matrix or threshold.
    """
    centroid_64 = centroid.astype(np.float64).reshape(1, -1)
    model.class_means = np.vstack([model.class_means, centroid_64])
    model.class_names.append(class_name)


def _score_batch(
    embeddings: np.ndarray,
    class_means: np.ndarray,
    precision_matrix: np.ndarray,
) -> np.ndarray:
    """Compute min-over-classes Mahalanobis scores for a batch of embeddings.

    Args:
        embeddings: (N, D) float64
        class_means: (C, D) float64
        precision_matrix: (D, D) float64

    Returns:
        scores: (N,) float64 — minimum Mahalanobis distance per embedding.
    """
    N = embeddings.shape[0]
    C = class_means.shape[0]
    min_scores = np.full(N, np.inf, dtype=np.float64)

    # Process one class at a time to avoid (N, C, D) memory explosion
    for c in range(C):
        diff = embeddings - class_means[c]  # (N, D)
        transformed = diff @ precision_matrix  # (N, D)
        scores_c = np.sum(transformed * diff, axis=1)  # (N,)
        np.minimum(min_scores, scores_c, out=min_scores)

    return min_scores
