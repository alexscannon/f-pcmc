"""Load embedding .pt files and build stream items for continual learning evaluation."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from config import ContinualConfig


class NoveltyType(str, Enum):
    """Ground-truth novelty type for post-hoc evaluation only."""

    IND_REAL = "ind_real"
    IND_SYNTHETIC = "ind_synthetic"
    NEAR_OOD = "near_ood"
    FAR_OOD = "far_ood"


@dataclass
class TrainData:
    """CIFAR-100 training embeddings for IND model initialization."""

    embeddings: np.ndarray      # (N, D) float32
    subclass_names: np.ndarray  # (N,) string array


@dataclass
class StreamItem:
    """One example in the evaluation stream.

    Ground-truth fields (true_class, true_superclass, novelty_type) are
    for post-hoc evaluation only — never accessed by pipeline logic.
    """

    embedding: np.ndarray       # (D,) float32
    true_class: str
    true_superclass: str
    novelty_type: NoveltyType


def _load_pt_file(path: Path) -> dict:
    """Load a .pt embedding file and return its contents as numpy."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "embeddings": data["embeddings"].numpy(),
        "subclass_names": list(data["subclass_names"]),
        "superclass_names": list(data["superclass_names"]),
        "sources": list(data["sources"]),
    }


def load_all_data(config: ContinualConfig) -> tuple[TrainData, list[StreamItem]]:
    """Load all embedding files and return train data + unshuffled stream items.

    Returns:
        train_data: CIFAR-100 train split (50k) for IND model initialization.
        stream_items: All non-train examples as StreamItems (unshuffled).
    """
    stream_items: list[StreamItem] = []

    # --- Load and split real CIFAR-100 ---
    real_path = config.embeddings_dir / config.real_cifar100_file
    if not real_path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {real_path}\n"
            f"Run embedding extraction first (see embeddings/README.md)"
        )

    real_data = _load_pt_file(real_path)
    sources = np.array(real_data["sources"])
    train_mask = sources == "cifar100_train"
    test_mask = sources == "cifar100_test"

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            f"Expected sources 'cifar100_train' and 'cifar100_test' in {real_path.name}, "
            f"found: {set(real_data['sources'])}"
        )

    # Train data for initialization
    train_idx = np.where(train_mask)[0]
    train_data = TrainData(
        embeddings=real_data["embeddings"][train_idx],
        subclass_names=np.array([real_data["subclass_names"][i] for i in train_idx]),
    )
    logger.info(f"Loaded {len(train_idx):,} train embeddings from {real_path.name}")

    # Test data → stream
    test_idx = np.where(test_mask)[0]
    for i in test_idx:
        stream_items.append(StreamItem(
            embedding=real_data["embeddings"][i],
            true_class=real_data["subclass_names"][i],
            true_superclass=real_data["superclass_names"][i],
            novelty_type=NoveltyType.IND_REAL,
        ))
    logger.info(f"Loaded {len(test_idx):,} test embeddings from {real_path.name}")

    # --- Load synthetic IND ---
    ind_path = config.embeddings_dir / config.synthetic_ind_file
    if not ind_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {ind_path}")

    ind_data = _load_pt_file(ind_path)
    n_ind = ind_data["embeddings"].shape[0]
    for i in range(n_ind):
        stream_items.append(StreamItem(
            embedding=ind_data["embeddings"][i],
            true_class=ind_data["subclass_names"][i],
            true_superclass=ind_data["superclass_names"][i],
            novelty_type=NoveltyType.IND_SYNTHETIC,
        ))
    logger.info(f"Loaded {n_ind:,} synthetic IND embeddings from {ind_path.name}")

    # --- Load near-OOD (novel subclasses of existing superclasses) ---
    near_path = config.embeddings_dir / config.novel_subclasses_file
    if not near_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {near_path}")

    near_data = _load_pt_file(near_path)
    n_near = near_data["embeddings"].shape[0]
    for i in range(n_near):
        stream_items.append(StreamItem(
            embedding=near_data["embeddings"][i],
            true_class=near_data["subclass_names"][i],
            true_superclass=near_data["superclass_names"][i],
            novelty_type=NoveltyType.NEAR_OOD,
        ))
    logger.info(f"Loaded {n_near:,} near-OOD embeddings from {near_path.name}")

    # --- Load far-OOD (novel subclasses of novel superclasses) ---
    far_path = config.embeddings_dir / config.novel_superclasses_file
    if not far_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {far_path}")

    far_data = _load_pt_file(far_path)
    n_far = far_data["embeddings"].shape[0]
    for i in range(n_far):
        stream_items.append(StreamItem(
            embedding=far_data["embeddings"][i],
            true_class=far_data["subclass_names"][i],
            true_superclass=far_data["superclass_names"][i],
            novelty_type=NoveltyType.FAR_OOD,
        ))
    logger.info(f"Loaded {n_far:,} far-OOD embeddings from {far_path.name}")

    return train_data, stream_items


def print_data_summary(train_data: TrainData, stream_items: list[StreamItem]) -> None:
    """Print a summary of loaded data pools."""
    dim = train_data.embeddings.shape[1]
    n_train_classes = len(np.unique(train_data.subclass_names))

    counts = {}
    classes = {}
    for nt in NoveltyType:
        items = [s for s in stream_items if s.novelty_type == nt]
        counts[nt] = len(items)
        classes[nt] = len({s.true_class for s in items})

    logger.info(
        f"\n{'=' * 60}\n"
        f"  Continual Learning Evaluation: Data Summary\n"
        f"{'=' * 60}\n"
        f"  Embedding dim:       {dim}\n"
        f"  Train pool:          {len(train_data.embeddings):>7,} ({n_train_classes} classes)\n"
        f"  Stream total:        {len(stream_items):>7,}\n"
        f"\n"
        f"  Stream composition:\n"
        f"    IND real:          {counts[NoveltyType.IND_REAL]:>7,} ({classes[NoveltyType.IND_REAL]} classes)\n"
        f"    IND synthetic:     {counts[NoveltyType.IND_SYNTHETIC]:>7,} ({classes[NoveltyType.IND_SYNTHETIC]} classes)\n"
        f"    Near-OOD:          {counts[NoveltyType.NEAR_OOD]:>7,} ({classes[NoveltyType.NEAR_OOD]} classes)\n"
        f"    Far-OOD:           {counts[NoveltyType.FAR_OOD]:>7,} ({classes[NoveltyType.FAR_OOD]} classes)\n"
        f"{'=' * 60}"
    )
