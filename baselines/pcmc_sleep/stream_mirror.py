"""T17 Phase 1: pixel-space mirror of the P2 stream (baselines/pcmc_sleep/PLAN.md).

The sleep-retrained PCMC baseline consumes raw images, while P2 is defined
over precomputed embeddings. This module replays the EXACT embedding-space
stream in pixel space: the manifest comes from the same `fpcmc.protocols
.build_p2` call (same frozen class split, same seed), and each stream index
is resolved to its source image through the pool rows' `image_paths`
provenance — the alignment proven row-by-row in Phase 0.1 (0 mismatches over
all 63,326 rows; see PLAN.md).

Torch-free on the hot path (pickle + numpy; PIL only in `image_pil`), so the
repo's CPU test env can verify alignment without the GPU environment.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from fpcmc.data import Pool, read_roots_env
from fpcmc.protocols import ProtocolStream

#: DATA_ROOT-relative roots of the raw image trees per pool (ASSETS §1 /
#: Phase 0.1). CIFAR pools resolve through the canonical python pickles.
_SYNTH_ROOTS = {
    "synthetic_ind": "ms_cifar100_genai_ind_32x32",
    "near_ood": "ms_cifar100_genai_novel_32x32/novel_subclasses",
    "far_ood": "ms_cifar100_genai_novel_32x32/novel_superclasses",
}
_CIFAR_DIR = "cifar100/cifar-100-python"


def _data_root() -> Path:
    env = read_roots_env()
    root = env.get("DATA_ROOT")
    if not root:
        raise FileNotFoundError("roots.env does not define DATA_ROOT")
    return Path(root)


@dataclass(frozen=True)
class _CifarSplit:
    images: np.ndarray  # (N, 32, 32, 3) uint8, canonical pickle order
    fine_names: np.ndarray  # (N,) <U str


class P2PixelMirror:
    """Image-space view of one built P2 stream.

    Constructed from the SAME `ProtocolStream` the embedding run uses (so
    identity of order/composition holds by construction, not by re-derivation)
    plus the loaded pools, whose `image_paths` map each (pool,
    within_pool_index) row to its source image.
    """

    def __init__(self, stream: ProtocolStream, pools: dict[str, Pool],
                 data_root: str | Path | None = None):
        self._stream = stream
        self._pools = dict(pools)
        self._root = Path(data_root) if data_root is not None else _data_root()

    # ------------------------------------------------------------- identity

    @property
    def manifest(self):
        return self._stream.manifest

    @property
    def t0_classes(self):
        return self._stream.t0_classes

    @property
    def checkpoint_steps(self):
        return self._stream.checkpoint_steps

    def __len__(self) -> int:
        return len(self._stream.manifest)

    # ------------------------------------------------------------ resolution

    @cached_property
    def _cifar(self) -> dict[str, _CifarSplit]:
        meta = pickle.load(
            open(self._root / _CIFAR_DIR / "meta", "rb"), encoding="bytes")
        names = np.array([n.decode() for n in meta[b"fine_label_names"]])
        out = {}
        for split in ("train", "test"):
            raw = pickle.load(
                open(self._root / _CIFAR_DIR / split, "rb"), encoding="bytes")
            images = raw[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            out[split] = _CifarSplit(
                images=np.ascontiguousarray(images),
                fine_names=names[np.asarray(raw[b"fine_labels"])],
            )
        return out

    def source_ref(self, i: int) -> tuple[str, str]:
        """(kind, ref) for stream index i: ("cifar", "train:00042") or
        ("file", absolute path). Cheap — no image decoding."""
        pool_name = str(self.manifest.pool[i])
        wpi = int(self.manifest.within_pool_index[i])
        image_path = str(self._pools[pool_name].image_paths[wpi])
        if pool_name in ("ind_reference", "ind_test"):
            # e.g. "cifar100_train_00037" (format verified in Phase 0.1)
            _, split, idx = image_path.rsplit("_", 2)
            return "cifar", f"{split}:{idx}"
        return "file", str(self._root / _SYNTH_ROOTS[pool_name] / image_path)

    def image_array(self, i: int) -> np.ndarray:
        """(H, W, 3) uint8 image for stream index i."""
        kind, ref = self.source_ref(i)
        if kind == "cifar":
            split, idx = ref.split(":")
            return self._cifar[split].images[int(idx)]
        from PIL import Image  # lazy: only needed off the CIFAR path

        return np.asarray(Image.open(ref).convert("RGB"))

    def image_pil(self, i: int):
        from PIL import Image

        return Image.fromarray(self.image_array(i))

    def true_class_from_source(self, i: int) -> str:
        """The class as derivable from the SOURCE alone (pickle label / path
        component) — the alignment cross-check used by the tests, computed
        without touching the manifest's own true_class."""
        kind, ref = self.source_ref(i)
        if kind == "cifar":
            split, idx = ref.split(":")
            return str(self._cifar[split].fine_names[int(idx)])
        return Path(ref).parent.name

    # -------------------------------------------------------------- T0 side

    def t0_image_refs(self) -> list[tuple[str, str, str]]:
        """(class, kind, ref) for every T0 pretraining image: all
        ind_reference (cifar100_train) rows of the stream's T0 classes, in
        pool-row order — deterministic, label-free consumption downstream."""
        pool = self._pools["ind_reference"]
        t0 = set(str(c) for c in self.t0_classes)
        refs = []
        for row in range(len(pool.subclass_names)):
            cls = str(pool.subclass_names[row])
            if cls not in t0:
                continue
            _, split, idx = str(pool.image_paths[row]).rsplit("_", 2)
            refs.append((cls, "cifar", f"{split}:{idx}"))
        return refs
