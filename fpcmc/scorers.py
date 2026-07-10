"""Per-concept scorers: knn_ref, vmf, and the composed knn_vmf (T2; FR-4.1–4.3).

Interface contract (frozen at T2 — later tasks import, never modify):
  - ``score(z, concept) -> float`` — lower = more compatible. `z` and all
    concept geometry are assumed L2-normalized (FR-1.2; the data layer and
    fixture world guarantee it) — scorers never re-normalize.
  - ``accepts(z, concept) -> bool`` — score <= the concept's threshold for
    this scorer (FR-9 uses <=): `concept.tau` for knn_ref (and for vmf in
    its small-ref_set fallback mode), `concept.tau_vmf` for native vmf.
  - ``margin(z, concept) -> float`` — normalized assignment margin
    ``(tau - s) / |tau|``. Approved deviation (owner, 2026-07-10) from
    FR-4.3's literal ``(tau - s) / tau``: vmf scores are negative
    log-likelihoods, so tau_vmf < 0 at D=1024 and the literal formula
    inverts orientation there; dividing by |tau| preserves
    "accept <=> margin >= 0, larger = deeper acceptance" for both signs and
    equals the PRD formula whenever tau > 0 (every knn_ref case).
  - ``score_detail(z, concept) -> ScoreDetail`` — the full outcome, including
    the FR-4.2 fallback flag (the "return metadata" home) and which
    sub-scorer produced the margin (surfaced later in T5 routing records and
    the A5 ablation).
  - ``select(z, concepts) -> Selection | None`` — FR-4.3 assignment: best
    margin among accepting concepts; exact ties broken by lexicographic
    concept_id (determinism).

KnnVmfScorer (FR-4.3): accepts iff EITHER sub-scorer accepts under its own
per-concept threshold; per-concept margin = best sub-scorer margin. Its scalar
``score`` is the knn_ref sub-score (owner-approved 2026-07-10): the validated
source paradigm behind the T6 pin uses a pure-kNN detection statistic
(lib/evaluation/continual/paradigms/knn_vmf.py, `_process`), with vMF entering
acceptance/clustering only — so the composed scalar mirrors that, and
`accepts` is deliberately not derivable from `score` alone.

Scorers are stateless, deterministic pure functions of (z, concept): no RNG,
no caches, so T5 can add a vectorized batch path across concepts without
touching this per-concept contract.

Numerical core adapted (copy-with-citation per lib/PROVENANCE.md; lib/ is
read-only) from lib/evaluation/continual/paradigms/vmf_dpmm.py, blob
47d7412fdd105d00faf2fe63a8465bfd81cfbc80:
  - `estimate_kappa`: Banerjee et al. 2005 point estimate with the r_bar clip.
    The source's kappa_min/kappa_max config clips are dropped — PRD §8 defines
    no such keys; the r_bar clip alone keeps the estimate finite.
  - `log_iv_uniform`: Abramowitz & Stegun 9.7.7 uniform large-order asymptotic
    for log I_v(kappa). This is the production path at every D, as in the
    source: at D=1024 (v=511) scipy.special.ive underflows to 0 across the
    kappa range Banerjee produces, so log I_v via ive would be -inf.
  - `log_C_D`: log vMF normalizing constant, with the source's near-uniform
    kappa guard.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from fpcmc.concepts import Concept
from fpcmc.config import FPCMCConfig

_EPS = 1e-12


# --------------------------------------------------------------- vMF numerics


def estimate_kappa(ref_set: np.ndarray) -> float:
    """Banerjee et al. 2005 vMF concentration estimate from a reference set.

    FR-4.2: with r_bar = ||mean of ref embeddings||,
    kappa ~= r_bar * (D - r_bar^2) / (1 - r_bar^2).
    Adapted from lib/.../vmf_dpmm.py::estimate_kappa (r_sum/n form).
    """
    if ref_set.ndim != 2 or ref_set.shape[0] == 0:
        raise ValueError(f"ref_set must be a non-empty (K, D) array, got shape {ref_set.shape}")
    d = ref_set.shape[1]
    r_bar = float(np.linalg.norm(ref_set.mean(axis=0)))
    r_bar = min(r_bar, 1.0 - 1e-10)
    return float(r_bar * (d - r_bar * r_bar) / (1.0 - r_bar * r_bar))


def log_iv_uniform(v: float, kappa: float) -> float:
    """log I_v(kappa) via the A&S 9.7.7 large-order uniform asymptotic.

    Verbatim from lib/.../vmf_dpmm.py::log_iv_uniform — finite for large v
    across the kappa range of interest (the production path; see module
    docstring for why ive is unusable at D=1024).
    """
    z = kappa / v
    s = np.sqrt(1.0 + z * z)
    eta = s + np.log(z / (1.0 + s))
    return -0.5 * np.log(2.0 * np.pi * v) + v * eta - 0.25 * np.log(1.0 + z * z)


def log_C_D(kappa: float, D: int) -> float:
    """log normalizing constant of vMF(mu, kappa) on S^{D-1}.

    Verbatim from lib/.../vmf_dpmm.py::log_C_D:
    (D/2 - 1) log kappa - (D/2) log 2*pi - log I_{D/2-1}(kappa).
    """
    v = D / 2.0 - 1.0
    if kappa < 1e-8:
        # near-uniform; return a constant that makes density ~ uniform
        return 0.0
    return v * np.log(kappa) - (D / 2.0) * np.log(2.0 * np.pi) - log_iv_uniform(v, kappa)


# ----------------------------------------------------------------- interface


@dataclass(frozen=True)
class ScoreDetail:
    """Full outcome of scoring one embedding against one concept.

    score:    the scorer's scalar (lower = more compatible).
    accepted: score <= the applicable per-concept threshold (OR of the
              sub-scorers for knn_vmf).
    margin:   normalized margin (tau - s)/|tau| under the threshold used; for
              knn_vmf, the best sub-scorer margin.
    scorer:   name of the scorer that produced this detail.
    via:      sub-scorer whose score/threshold produced `margin` ("knn_ref"
              or "vmf"; equals `scorer` for the simple scorers).
    fallback: FR-4.2 metadata flag — True iff the vmf branch delegated to
              knn_ref because ref_set < n_vmf_min.
    """

    score: float
    accepted: bool
    margin: float
    scorer: str
    via: str
    fallback: bool = False


@dataclass(frozen=True)
class Selection:
    """FR-4.3 assignment outcome: the accepting concept with the best margin."""

    concept: Concept
    detail: ScoreDetail


def _margin(tau: float, score: float) -> float:
    """(tau - s) / |tau| (approved |tau| deviation; see module docstring)."""
    return (tau - score) / max(abs(tau), _EPS)


class Scorer(ABC):
    """Common per-concept scorer interface (FR-4). Stateless; deterministic."""

    name: str

    @abstractmethod
    def score_detail(self, z: np.ndarray, concept: Concept) -> ScoreDetail:
        """Score one L2-normalized embedding against one concept."""

    def score(self, z: np.ndarray, concept: Concept) -> float:
        return self.score_detail(z, concept).score

    def accepts(self, z: np.ndarray, concept: Concept) -> bool:
        return self.score_detail(z, concept).accepted

    def margin(self, z: np.ndarray, concept: Concept) -> float:
        return self.score_detail(z, concept).margin

    def select(self, z: np.ndarray, concepts: Iterable[Concept]) -> Selection | None:
        """Best-margin assignment among accepting concepts (FR-4.3).

        Exact margin ties break to the lexicographically smallest concept_id,
        independent of iteration order (determinism, TASKS T2).
        """
        best: Selection | None = None
        for concept in concepts:
            detail = self.score_detail(z, concept)
            if not detail.accepted:
                continue
            if (
                best is None
                or detail.margin > best.detail.margin
                or (
                    detail.margin == best.detail.margin
                    and concept.concept_id < best.concept.concept_id
                )
            ):
                best = Selection(concept=concept, detail=detail)
        return best


# ------------------------------------------------------------ knn_ref (FR-4.1)


class KnnRefScorer(Scorer):
    """Mean cosine distance to the k_ref nearest ref_set members (FR-4.1).

    k_ref clips to the ref_set size, so the scorer works from size 1 upward.
    """

    name = "knn_ref"

    def __init__(self, k_ref: int = 5) -> None:
        if k_ref < 1:
            raise ValueError(f"k_ref must be >= 1, got {k_ref}")
        self.k_ref = int(k_ref)

    def score_detail(self, z: np.ndarray, concept: Concept) -> ScoreDetail:
        dists = 1.0 - concept.ref_set @ z
        k = min(self.k_ref, dists.shape[0])
        s = float(np.mean(np.partition(dists, k - 1)[:k]))
        return ScoreDetail(
            score=s,
            accepted=s <= concept.tau,
            margin=_margin(concept.tau, s),
            scorer=self.name,
            via=self.name,
        )


# ---------------------------------------------------------------- vmf (FR-4.2)


class VmfScorer(Scorer):
    """Negative vMF log-likelihood under vMF(concept.centroid, concept.kappa).

    score = -(log C_D(kappa) + kappa * <centroid, z>), thresholded against
    concept.tau_vmf. Below n_vmf_min reference points the estimate is
    unreliable (FR-4.2): the scorer delegates wholesale to knn_ref — score on
    the knn scale, acceptance against concept.tau — and flags `fallback=True`
    in the returned metadata.
    """

    name = "vmf"

    def __init__(self, n_vmf_min: int = 10, fallback: KnnRefScorer | None = None) -> None:
        if n_vmf_min < 1:
            raise ValueError(f"n_vmf_min must be >= 1, got {n_vmf_min}")
        self.n_vmf_min = int(n_vmf_min)
        self._fallback = fallback if fallback is not None else KnnRefScorer()

    def score_detail(self, z: np.ndarray, concept: Concept) -> ScoreDetail:
        if concept.ref_set.shape[0] < self.n_vmf_min:
            delegated = self._fallback.score_detail(z, concept)
            return ScoreDetail(
                score=delegated.score,
                accepted=delegated.accepted,
                margin=delegated.margin,
                scorer=self.name,
                via=delegated.via,
                fallback=True,
            )
        kappa = concept.kappa
        if not np.isfinite(kappa):
            raise ValueError(
                f"concept {concept.concept_id!r}: kappa={kappa} is not finite but "
                f"ref_set has {concept.ref_set.shape[0]} >= n_vmf_min={self.n_vmf_min} "
                "members — the owner must maintain the cached Banerjee estimate (FR-4.2)"
            )
        d = concept.centroid.shape[0]
        s = -(log_C_D(kappa, d) + kappa * float(concept.centroid @ z))
        return ScoreDetail(
            score=s,
            accepted=s <= concept.tau_vmf,
            margin=_margin(concept.tau_vmf, s),
            scorer=self.name,
            via=self.name,
        )


# ------------------------------------------------------------ knn_vmf (FR-4.3)


class KnnVmfScorer(Scorer):
    """Composed scorer: OR-accept over knn_ref and vmf, best sub-margin wins.

    The scalar `score` is the knn_ref sub-score (see module docstring). When
    the vmf branch is in fallback mode both sub-details are knn_ref against
    concept.tau, so the margins tie exactly and `via` stays "knn_ref".
    """

    name = "knn_vmf"

    def __init__(self, knn: KnnRefScorer | None = None, vmf: VmfScorer | None = None) -> None:
        self._knn = knn if knn is not None else KnnRefScorer()
        self._vmf = vmf if vmf is not None else VmfScorer(fallback=self._knn)

    def score_detail(self, z: np.ndarray, concept: Concept) -> ScoreDetail:
        d_knn = self._knn.score_detail(z, concept)
        d_vmf = self._vmf.score_detail(z, concept)
        best = d_vmf if d_vmf.margin > d_knn.margin else d_knn
        return ScoreDetail(
            score=d_knn.score,
            accepted=d_knn.accepted or d_vmf.accepted,
            margin=best.margin,
            scorer=self.name,
            via=best.via,
            fallback=d_vmf.fallback,
        )


# -------------------------------------------------------------------- factory


def make_scorer(config: FPCMCConfig) -> Scorer:
    """Build the configured scorer (PRD §8 `scorer: knn_ref | vmf | knn_vmf`)."""
    knn = KnnRefScorer(k_ref=config.k_ref)
    if config.scorer == "knn_ref":
        return knn
    vmf = VmfScorer(n_vmf_min=config.n_vmf_min, fallback=knn)
    if config.scorer == "vmf":
        return vmf
    if config.scorer == "knn_vmf":
        return KnnVmfScorer(knn=knn, vmf=vmf)
    raise ValueError(f"unknown scorer {config.scorer!r}")  # unreachable post-config-validation
