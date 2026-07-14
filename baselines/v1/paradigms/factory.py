"""Paradigm dispatch: map ``config.paradigm`` to a constructed, warmed-up paradigm."""

from __future__ import annotations

from config import ContinualConfig
from paradigms.base import Paradigm, WarmupData


def build_paradigm(
    name: str, config: ContinualConfig, warmup_data: WarmupData
) -> Paradigm:
    """Instantiate the named paradigm and run its warmup.

    Raises ValueError for unknown names. New paradigms (knn_dpmeans, vmf_dpmm)
    are wired here as they are implemented.
    """
    if name == "mahalanobis_hdbscan":
        from paradigms.mahalanobis_hdbscan import MahalanobisHDBSCANParadigm

        paradigm: Paradigm = MahalanobisHDBSCANParadigm()
    elif name == "knn_dpmeans":
        from paradigms.knn_dpmeans import KNNDPMeansParadigm

        paradigm = KNNDPMeansParadigm()
    elif name == "vmf_dpmm":
        from paradigms.vmf_dpmm import VMFDPMMParadigm

        paradigm = VMFDPMMParadigm()
    elif name == "knn_vmf":
        from paradigms.knn_vmf import KNNVMFParadigm

        paradigm = KNNVMFParadigm()
    else:
        raise ValueError(
            f"Unknown paradigm: {name!r} "
            f"(expected one of: mahalanobis_hdbscan, knn_dpmeans, vmf_dpmm, knn_vmf)"
        )

    paradigm.warmup(
        warmup_data.embeddings,
        warmup_data.labels,
        warmup_data.class_names,
        config,
    )
    return paradigm
