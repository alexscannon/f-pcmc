"""Stream construction, per-step logging, and rolling window metrics."""

import collections
import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from data_loader import NoveltyType, StreamItem


@dataclass
class StepRecord:
    """Metrics for a single stream step."""

    t: int
    score: float
    is_ood: bool
    predicted_class: str        # class name if IND, "" if OOD
    true_class: str
    true_superclass: str
    novelty_type: str           # NoveltyType.value
    n_ind_classes: int
    n_ood_buffer: int
    cum_det_acc: float          # cumulative detection accuracy through step t
    cum_cls_acc: float          # cumulative classification accuracy through step t
    phase: str                  # "warmup", "stream", or "drain"
    rolling_cls_acc: float      # rolling window classification accuracy at step t
    # Paradigm-specific auxiliary scores (e.g. vMF candidate detection scores).
    # Empty for paradigms that don't opt in; logged as extra CSV columns.
    extras: dict[str, float] = field(default_factory=dict)


def _split_warmup(
    items: list[StreamItem], rng: np.random.Generator, ind_warmup_count: int,
) -> tuple[list[StreamItem], list[StreamItem]]:
    """Take an IND_REAL warmup prefix; return ``(warmup, remainder)``.

    ``remainder`` is ``leftover IND_REAL + all non-IND_REAL items`` in the exact
    construction order of the original ``build_stream`` (this is the byte-identity
    anchor for the ``random`` path: it performs exactly one
    ``rng.permutation(len(ind_real))`` draw before returning).
    """
    ind_real = [it for it in items if it.novelty_type == NoveltyType.IND_REAL]
    rest = [it for it in items if it.novelty_type != NoveltyType.IND_REAL]

    if len(ind_real) < ind_warmup_count:
        raise ValueError(
            f"ind_warmup_count={ind_warmup_count} but only "
            f"{len(ind_real)} IND_REAL items available"
        )

    ind_perm = rng.permutation(len(ind_real))
    warmup = [ind_real[i] for i in ind_perm[:ind_warmup_count]]
    leftover_ind = [ind_real[i] for i in ind_perm[ind_warmup_count:]]
    remainder = leftover_ind + rest
    return warmup, remainder


def _order_remainder(
    remainder: list[StreamItem],
    rng: np.random.Generator,
    order: str,
    cluster_size: tuple[int, int],
) -> list[StreamItem]:
    """Reorder the post-warmup remainder by grouping same-subclass items.

    Groups items by ``true_class`` (real + synthetic of a subclass share a
    group), splits each group into contiguous chunks, then shuffles the chunk
    order and concatenates:

    - ``sequential``: one chunk per class → contiguous per-class blocks
      (A,A,A,B,B,B,C,C,C).
    - ``clustered``: variable small blocks of size drawn from ``cluster_size``
      → dispersed mini-clusters (A,B,B,C,A,A,B,C,C,C,A,A,C).

    "Some degree of separation" (clustered) is *statistical*: shuffling chunks
    disperses same-class blocks, but two same-class chunks may occasionally land
    adjacent (matching the C,C,C run in the motivating example). A strict
    "never adjacent" interleave is an explicit non-goal.
    """
    groups: dict[str, list[StreamItem]] = collections.defaultdict(list)
    for it in remainder:
        groups[it.true_class].append(it)

    lo, hi = int(cluster_size[0]), int(cluster_size[1])

    chunks: list[list[StreamItem]] = []
    # Sorted class order → reproducible given the seed.
    for cls in sorted(groups):
        members = groups[cls]
        perm = rng.permutation(len(members))
        shuffled = [members[i] for i in perm]
        if order == "sequential":
            chunks.append(shuffled)
        else:  # "clustered"
            i = 0
            while i < len(shuffled):
                size = max(1, int(rng.integers(lo, hi + 1)))
                chunks.append(shuffled[i:i + size])
                i += size

    chunk_perm = rng.permutation(len(chunks))
    ordered: list[StreamItem] = []
    for ci in chunk_perm:
        ordered.extend(chunks[ci])
    return ordered


def build_stream(
    items: list[StreamItem],
    seed: int,
    ind_warmup_count: int = 0,
    order: str = "random",
    cluster_size: tuple[int, int] = (2, 4),
) -> list[StreamItem]:
    """Order stream items with a fixed seed for reproducibility.

    The IND_REAL warmup prefix (first ``ind_warmup_count`` IND_REAL items,
    shuffled) is preserved for all orderings; ``order`` controls only how the
    post-warmup remainder is arranged:

    - ``random`` (default): remainder fully shuffled — the i.i.d. baseline.
      This path is byte-for-byte identical to the pre-ordering pipeline.
    - ``sequential``: remainder grouped into contiguous per-subclass blocks.
    - ``clustered``: remainder grouped into small dispersed same-subclass blocks
      of size drawn from ``cluster_size``.
    """
    rng = np.random.default_rng(seed)

    if order == "random":
        if ind_warmup_count <= 0:
            indices = rng.permutation(len(items))
            return [items[i] for i in indices]

        warmup, remainder = _split_warmup(items, rng, ind_warmup_count)
        rem_perm = rng.permutation(len(remainder))
        shuffled_remainder = [remainder[i] for i in rem_perm]
        return warmup + shuffled_remainder

    # New orderings: "sequential" | "clustered".
    if ind_warmup_count > 0:
        warmup, remainder = _split_warmup(items, rng, ind_warmup_count)
    else:
        warmup, remainder = [], list(items)

    return warmup + _order_remainder(remainder, rng, order, cluster_size)


class StepLogger:
    """Buffered CSV logger for per-step metrics."""

    HEADER = [
        "t", "score", "is_ood", "predicted_class",
        "true_class", "true_superclass", "novelty_type",
        "n_ind_classes", "n_ood_buffer",
        "cum_det_acc", "cum_cls_acc",
        "phase", "rolling_cls_acc",
    ]

    def __init__(
        self, output_dir: Path, flush_interval: int, extra_columns: list[str] = (),
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        self._path = output_dir / "per_step.csv"
        self._file = open(self._path, "w", newline="")
        self._writer = csv.writer(self._file)
        # Extra columns are appended after the fixed schema; when empty the
        # header (and every row) is byte-identical to the pre-extras format.
        self._extra_columns = list(extra_columns)
        self._writer.writerow(self.HEADER + self._extra_columns)
        self._buffer: list[list] = []
        self._flush_interval = flush_interval

    def log(self, record: StepRecord) -> None:
        row = [
            record.t,
            f"{record.score:.6f}",
            int(record.is_ood),
            record.predicted_class,
            record.true_class,
            record.true_superclass,
            record.novelty_type,
            record.n_ind_classes,
            record.n_ood_buffer,
            f"{record.cum_det_acc:.6f}",
            f"{record.cum_cls_acc:.6f}",
            record.phase,
            f"{record.rolling_cls_acc:.6f}",
        ]
        for col in self._extra_columns:
            v = record.extras.get(col)
            row.append(f"{v:.6f}" if v is not None else "")
        self._buffer.append(row)
        if len(self._buffer) >= self._flush_interval:
            self._flush()

    def _flush(self) -> None:
        if self._buffer:
            self._writer.writerows(self._buffer)
            self._file.flush()
            self._buffer.clear()

    def close(self) -> None:
        self._flush()
        self._file.close()


class RollingWindowMetrics:
    """Rolling window metrics for detection and classification accuracy."""

    def __init__(self, window_size: int):
        self._detection_correct: collections.deque[bool] = collections.deque(maxlen=window_size)
        self._class_correct: collections.deque[bool] = collections.deque(maxlen=window_size)

    def update(self, record: StepRecord, gt_is_ind: bool) -> None:
        correct_detection = record.is_ood != gt_is_ind
        self._detection_correct.append(correct_detection)
        if not record.is_ood:
            self._class_correct.append(record.predicted_class == record.true_class)

    def update_classification_only(self, correct: bool) -> None:
        """Push a single classification result (e.g. from cluster purity)."""
        self._class_correct.append(correct)

    @property
    def detection_accuracy(self) -> float:
        if not self._detection_correct:
            return 0.0
        return sum(self._detection_correct) / len(self._detection_correct)

    @property
    def classification_accuracy(self) -> float:
        if not self._class_correct:
            return 0.0
        return sum(self._class_correct) / len(self._class_correct)


class CumulativeMetrics:
    """Cumulative (all-time) detection and classification accuracy."""

    def __init__(self) -> None:
        self._det_correct = 0
        self._det_total = 0
        self._cls_correct = 0
        self._cls_total = 0

    def update(self, record: StepRecord, gt_is_ind: bool) -> None:
        self._det_total += 1
        if record.is_ood != gt_is_ind:
            self._det_correct += 1
        if not record.is_ood:
            self._cls_total += 1
            if record.predicted_class == record.true_class:
                self._cls_correct += 1

    def update_classification_batch(self, n_correct: int, n_total: int) -> None:
        """Bulk-update classification counters (e.g. cluster purity, buffer drain)."""
        self._cls_correct += n_correct
        self._cls_total += n_total

    @property
    def detection_accuracy(self) -> float:
        if self._det_total == 0:
            return 0.0
        return self._det_correct / self._det_total

    @property
    def classification_accuracy(self) -> float:
        if self._cls_total == 0:
            return 0.0
        return self._cls_correct / self._cls_total
