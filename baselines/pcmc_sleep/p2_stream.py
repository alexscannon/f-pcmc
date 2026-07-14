"""T17 Phase 2: ``P2UPLStream`` — the stream shim their code consumes.

GPU-env-only module (imports torch + the vendored ``core.*``; the driver puts
``vendor/`` and the repo root on ``sys.path``). Satisfies exactly the
interface upstream ``main.py`` consumes from a stream (HANDOFF §4.3):
``__iter__/__next__ -> (data, label, t)``, ``pretrain_dataloader``,
``eval_loaders(t)``, ``task_bounds``, ``eval_times``, ``__len__`` — fed from
the Phase 1 ``P2PixelMirror`` so the replayed order IS the embedding-space P2
stream. Wake tensors are CPU (their layers ``.cuda()`` internally; CUDA
inputs corrupt their stored-example device bookkeeping — HANDOFF §5).

Eval sets implement the owner's Q3(a) ruling — see run_config.py's docstring.
The supervise draw is seeded per class through ``fpcmc.rng.make_rng`` and
independent of the checkpoint, so every checkpoint supervises with the same
labeled set.
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from core.stream.collate import DefaultCollateFunction
from core.stream.samplers import ExtendedSampler

from baselines.pcmc_sleep.run_config import (
    eval_classes_at,
    eval_label_map,
    cluster_label_map,
    phase_task_index,
)
from fpcmc.rng import make_rng


class MirrorRefDataset(torch.utils.data.Dataset):
    """(PIL image, int label, str fname) items resolved through the mirror.

    Item refs: ``("stream", stream_index)`` or ``("cifar", split, row)``.
    The 3-tuple item shape matches what every upstream collate indexes.
    """

    def __init__(self, mirror, refs: list[tuple], labels: list[int]):
        assert len(refs) == len(labels)
        self._mirror = mirror
        self._refs = refs
        self.labels = np.asarray(labels)

    def __len__(self) -> int:
        return len(self._refs)

    def __getitem__(self, index: int):
        ref = self._refs[index]
        if ref[0] == "stream":
            img = self._mirror.image_pil(ref[1])
            fname = f"stream_{ref[1]}"
        else:
            _, split, row = ref
            img = Image.fromarray(self._mirror.cifar_image(split, row))
            fname = f"cifar_{split}_{row}"
        return img, int(self.labels[index]), fname


class P2UPLStream:
    """Pixel-space P2 stream + eval loaders, upstream-interface compatible."""

    def __init__(self, mirror, config, *, max_steps: int | None = None):
        self.mirror = mirror
        self.config = config
        self.seed = int(config.seed)
        self.img_size = int(config.dataset.img_size)
        self.mean = list(config.dataset.mean)
        self.std = list(config.dataset.std)
        self.sup_size = int(config.dataset.sup_size)
        self.test_size = int(config.dataset.test_size)

        self.phases = mirror.phases
        self.task_index = phase_task_index(self.phases)
        self.eval_map = eval_label_map(mirror.t0_classes, self.phases)
        self.cluster_map = cluster_label_map(mirror.t0_classes, self.phases)
        self._step_task = np.array(
            [self.task_index[str(p)] for p in mirror.manifest.phase], dtype=np.int64
        )
        self._step_label = np.array(
            [self.eval_map.get(str(c), -1) for c in mirror.manifest.true_class],
            dtype=np.int64,
        )

        self.n_steps = len(mirror) if max_steps is None else min(int(max_steps), len(mirror))
        # Their UPLStream transform, verbatim semantics (streams.py:30).
        self.transform = T.Compose(
            [
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        self._index = 0
        self.pretrain_dataloader = self._build_pretrain_loader()

    # ------------------------------------------------------------- pretrain

    def _build_pretrain_loader(self):
        """T0 loader. Only ``.dataset``/``.sampler`` are consumed upstream
        (Layer.pretrain rebuilds the loader with its own patch collate;
        init_memory then iterates that rebuilt loader)."""
        refs = self.mirror.t0_image_refs()  # (class, "cifar", "train:NNNNN")
        smoke = getattr(self.config, "smoke", None)
        if smoke is not None:
            rng = make_rng(self.seed, "pcmc_sleep/smoke_t0")
            keep = rng.choice(len(refs), size=int(smoke.t0_images), replace=False)
            refs = [refs[int(i)] for i in np.sort(keep)]
        items, labels = [], []
        for cls, _, ref in refs:
            split, row = ref.split(":")
            items.append(("cifar", split, int(row)))
            labels.append(self.eval_map[str(cls)])
        dataset = MirrorRefDataset(self.mirror, items, labels)
        return torch.utils.data.DataLoader(
            dataset,
            sampler=ExtendedSampler(
                np.arange(len(dataset)), shuffle=True,
                repeats=int(self.config.model.init_epochs),
            ),
            collate_fn=DefaultCollateFunction(self.img_size, self.mean, self.std),
            batch_size=int(self.config.model.layers["layer0"].pretrain_bs),
            num_workers=0,
        )

    # ----------------------------------------------------------- wake stream

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= self.n_steps:
            raise StopIteration
        i = self._index
        self._index += 1
        data = torch.stack([self.transform(self.mirror.image_pil(i))])
        label = torch.tensor([int(self._step_label[i])])
        return data, label, int(self._step_task[i])

    def __len__(self) -> int:
        return self.n_steps

    def task_of_step(self, i: int) -> int:
        return int(self._step_task[i])

    # ------------------------------------------------------------ eval side

    def _loader(self, items: list[tuple], labels: list[int]):
        """batch_size=1, unshuffled — their eval loader shape (pcmc.py does
        ``y.item()``; streams.py builds sup/test loaders exactly like this)."""
        dataset = MirrorRefDataset(self.mirror, items, labels)
        return torch.utils.data.DataLoader(
            dataset,
            sampler=ExtendedSampler(np.arange(len(dataset)), shuffle=False),
            collate_fn=DefaultCollateFunction(self.img_size, self.mean, self.std),
            batch_size=1,
        )

    def _cifar_eval_items(self, task: int):
        """(sup_items, sup_labels, test_items, test_labels) for the CIFAR
        classes visible at ``task``, in label order (labels contiguous —
        load-bearing for their supervise/classify indexing)."""
        cifar, _ = eval_classes_at(self.mirror.t0_classes, self.phases, task)
        sup_items, sup_labels, test_items, test_labels = [], [], [], []
        for cls in cifar:
            label = self.eval_map[cls]
            train_rows = self.mirror.cifar_rows("train", cls)
            rng = make_rng(self.seed, f"pcmc_sleep/sup/{cls}")
            pick = np.sort(
                rng.choice(train_rows.size, size=self.sup_size, replace=False)
            )
            sup_items += [("cifar", "train", int(r)) for r in train_rows[pick]]
            sup_labels += [label] * self.sup_size

            test_rows = self.mirror.cifar_rows("test", cls)
            if self.test_size < test_rows.size:
                rng = make_rng(self.seed, f"pcmc_sleep/test/{cls}")
                pick = np.sort(
                    rng.choice(test_rows.size, size=self.test_size, replace=False)
                )
                test_rows = test_rows[pick]
            test_items += [("cifar", "test", int(r)) for r in test_rows]
            test_labels += [label] * len(test_rows)
        return sup_items, sup_labels, test_items, test_labels

    def eval_loaders(self, task: int):
        sup_items, sup_labels, test_items, test_labels = self._cifar_eval_items(task)
        return self._loader(sup_items, sup_labels), self._loader(test_items, test_labels)

    def cluster_loader(self, task: int):
        """Q3: auxiliary synthetic-inclusive clustering set — the CIFAR test
        set plus, per synthetic class visible at ``task``, its first
        ``test_size`` stream arrivals (stream order; stream-seen by
        definition). None while no synthetic class has been introduced."""
        _, synth = eval_classes_at(self.mirror.t0_classes, self.phases, task)
        if not synth:
            return None
        _, _, items, labels = self._cifar_eval_items(task)
        true_class = self.mirror.manifest.true_class
        for cls in synth:
            steps = np.flatnonzero(true_class == cls)[: self.test_size]
            items += [("stream", int(s)) for s in steps]
            labels += [self.cluster_map[cls]] * len(steps)
        return self._loader(items, labels)

    # --------------------------------------- upstream main.py parity extras

    def task_bounds(self, eval_freq=None):
        return [0] + [p.start for p in self.phases]

    def eval_times(self, eval_freq=None):
        return list(self.mirror.checkpoint_steps), 1
