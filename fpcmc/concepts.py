"""Concept data structure and per-concept dynamics (T2 stub completed at T3).

PRD FR-1: every known class — T0-initial and promoted-novel alike — is a
`Concept`. This module carries the dataclass plus the two per-concept
dynamics entry points:

  - ``add_observation(z, step)`` — FR-1.1 reservoir maintenance, FR-1.3
    centroid dynamics (EMA for STM, bit-frozen for LTM), and the match /
    window / LRU bookkeeping the T7+ memory machinery reads.
  - ``Concept.seed(...)`` — FR-3.2 singleton constructor for novel embeddings.

Approved deviation (owner, 2026-07-10, T2): FR-1 declares a single
`tau: float`, but FR-4.3's composed scorer accepts "under its respective
per-concept threshold" and the two sub-scorers live on incommensurable scales
(knn_ref: cosine distance in [0, 2]; vmf: negative log-likelihood,
large-magnitude and negative at D=1024). The concept therefore carries `tau`
(the knn_ref threshold — also what single-scorer configs and FR-3.2 singleton
seeding use) and `tau_vmf` (the vmf sub-scorer's threshold). T4 computes both
via leave-one-out percentiles under the respective sub-scorer, and the FR-5.3
global prior likewise becomes per-sub-scorer.

Approved T3 decisions (owner Q&A, 2026-07-10; docs/CHANGES.md T3):

  1. kappa cadence — the cached Banerjee estimate (FR-4.2; `VmfScorer` raises
     on a non-finite value once ref_set >= n_vmf_min) is recomputed on every
     observation that changes ref_set. Cost is one mean+norm over a
     (<=K_max, D) array; T4's lazy >=25% trigger governs tau recomputation
     only, never kappa.
  2. Seed semantics — the seeding embedding counts in `ref_count_seen` (it
     occupies the reservoir, so FR-1.1's K_max/ref_count_seen needs it) but
     NOT in `match_count`/`match_windows` (FR-9 treats seeding as the
     no-match branch; PCMC counts matches after creation).
     `last_matched_at = created_at = step` so FR-3.1 LRU is defined from
     birth.
  3. Plumbing — TASKS states `add_observation(z, step)` verbatim, so the
     concept itself carries its reservoir Generator (a per-concept named
     substream from fpcmc.rng.make_rng, e.g. "reservoir/{concept_id}") and
     the `window_W`/`k_max`/`alpha_ema` config scalars, all fixed at
     construction. Reservoir contents are thereby a pure function of the
     concept's own observation sequence, independent across concepts.
  4. `provenance` gains the value "seeded" for unpromoted STM candidates
     (FR-1's Literal["initial", "promoted"] names no value for them, and
     FR-8.3's never-merge-two-"initial" rule must not confuse a candidate
     with a T0 concept). T8 promotion flips "seeded" -> "promoted".

Draw discipline (load-bearing for test_reservoir_uniformity's exact
vectorized replay): while ref_set < k_max an observation consumes no
randomness (append); once full it consumes exactly one ``rng.random(2)``
pair (u, v) — replace iff ``u < k_max / ref_count_seen``, at slot
``int(v * k_max)`` (uniform: k_max divides 2**53). This is the literal
FR-1.1 rule.

A `Concept` owns its arrays: `seed` copies, the reservoir replaces rows
in place, and the EMA rebinds `centroid` — callers must not rely on
aliasing arrays they passed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

_EPS = 1e-12


@dataclass
class Concept:
    """PRD FR-1 concept record (see module docstring for approved deviations).

    Scoring geometry (consumed by the frozen fpcmc.scorers interface):
      centroid: (D,) L2-normalized mean direction.
      ref_set:  (K, D) L2-normalized reference embeddings, 1 <= K <= k_max.
      tau:      knn_ref acceptance threshold (accept iff score <= tau, FR-9).
      kappa:    cached Banerjee vMF concentration estimate from ref_set
                (FR-4.2); kept consistent with ref_set on every mutation here.
      tau_vmf:  vmf acceptance threshold; NaN is fine while the vmf branch is
                unused (knn_ref-only configs, or ref_set < n_vmf_min where
                FR-4.2 falls back to knn_ref against `tau`).

    FR-1 bookkeeping:
      status:          "STM" candidate or "LTM" accepted concept.
      ref_count_seen:  total embeddings ever absorbed (reservoir denominator);
                       defaults to len(ref_set) when constructed directly.
      match_count:     matched examples since creation (seed excluded).
      match_windows:   distinct step//window_W indices with >=1 match (FR-7.4).
      created_at / last_matched_at: stream steps (LRU on the latter, FR-3.1).
      provenance:      "initial" (T0 init) | "seeded" (STM candidate) |
                       "promoted" (passed FR-7).
      gt_majority_label: eval-only; never read by pipeline logic.
      merged_from:     absorbed concept_ids, accumulated transitively (FR-1.4
                       lineage; maintained by the T9 merge machinery).
      refset_changes_since_tau: T4 dirty counter for FR-5.1's lazy threshold
                       recomputation — counts actual ref_set mutations
                       (appends and reservoir replacements; post-fill skipped
                       draws don't count) since the taus were last computed.
                       Incremented here, read/reset by fpcmc.thresholds; it
                       governs tau/tau_vmf only, never kappa (T3 decision 1).

    Operational (fixed at construction; T5's ConceptStore fills them from
    FPCMCConfig — approved T3 decision 3):
      window_W, k_max, alpha_ema: PRD §8 scalars this concept's dynamics use.
      rng: per-concept reservoir Generator; required only once ref_set is
           full (add_observation raises a clear error otherwise).
    """

    concept_id: str
    centroid: np.ndarray
    ref_set: np.ndarray
    tau: float
    kappa: float
    tau_vmf: float = float("nan")
    status: Literal["STM", "LTM"] = "STM"
    ref_count_seen: int = 0
    match_count: int = 0
    match_windows: set[int] = field(default_factory=set)
    created_at: int = 0
    last_matched_at: int = 0
    provenance: Literal["initial", "seeded", "promoted"] = "seeded"
    gt_majority_label: Optional[str] = None
    merged_from: list[str] = field(default_factory=list)
    refset_changes_since_tau: int = 0
    window_W: int = 250
    k_max: int = 64
    alpha_ema: float = 0.10
    rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        # Directly constructed concepts (tests, T6 init) start consistent:
        # every ref_set member has been "seen".
        if self.ref_count_seen == 0:
            self.ref_count_seen = int(self.ref_set.shape[0])

    # concept_id is immutable for the lifetime of the run (FR-1.4,
    # cross-cutting invariant 4): the first assignment (dataclass __init__)
    # is the only one allowed.
    def __setattr__(self, name: str, value: object) -> None:
        if name == "concept_id" and "concept_id" in self.__dict__:
            raise AttributeError("Concept.concept_id is immutable (FR-1.4 / invariant 4)")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name == "concept_id":
            raise AttributeError("Concept.concept_id is immutable (FR-1.4 / invariant 4)")
        super().__delattr__(name)

    # ------------------------------------------------------------ constructors

    @classmethod
    def seed(
        cls,
        z: np.ndarray,
        step: int,
        tau_prior: float,
        tau_vmf_prior: float = float("nan"),
        *,
        concept_id: str,
        rng: np.random.Generator,
        window_W: int = 250,
        k_max: int = 64,
        alpha_ema: float = 0.10,
    ) -> "Concept":
        """FR-3.2 singleton: a novel embedding becomes a new STM candidate.

        centroid = z, ref_set = [z], status STM, thresholds bootstrapped from
        the global per-sub-scorer priors (FR-5.3; the pair per the PRD FR-5
        note — `tau_vmf_prior` may stay NaN in knn_ref-only configs, where the
        vmf branch never runs). Seed-count semantics per approved decision 2.
        """
        z = np.array(z, dtype=np.float64)  # owned copy
        concept = cls(
            concept_id=concept_id,
            centroid=z,
            ref_set=z[None, :].copy(),
            tau=float(tau_prior),
            kappa=_estimate_kappa(z[None, :]),
            tau_vmf=float(tau_vmf_prior),
            status="STM",
            ref_count_seen=1,
            match_count=0,
            created_at=int(step),
            last_matched_at=int(step),
            provenance="seeded",
            window_W=int(window_W),
            k_max=int(k_max),
            alpha_ema=float(alpha_ema),
            rng=rng,
        )
        return concept

    # -------------------------------------------------------------- dynamics

    def add_observation(self, z: np.ndarray, step: int) -> None:
        """Absorb one assigned embedding (FR-1.1/1.3 + match bookkeeping).

        z is assumed L2-normalized (FR-1.2; same contract as the scorers).
        """
        self.match_count += 1
        self.ref_count_seen += 1
        self.match_windows.add(int(step) // self.window_W)
        self.last_matched_at = int(step)

        # FR-1.1 reservoir (see module docstring for the exact draw discipline).
        changed = False
        if self.ref_set.shape[0] < self.k_max:
            self.ref_set = np.vstack([self.ref_set, np.asarray(z, dtype=np.float64)[None, :]])
            changed = True
        else:
            if self.rng is None:
                raise ValueError(
                    f"concept {self.concept_id!r}: ref_set is full but no reservoir "
                    "rng was provided at construction (approved T3 decision 3)"
                )
            u, v = self.rng.random(2)
            if u < self.k_max / self.ref_count_seen:
                self.ref_set[int(v * self.k_max)] = z
                changed = True

        # FR-1.3 centroid dynamics: EMA + re-normalize for STM, frozen for LTM.
        if self.status == "STM":
            c = (1.0 - self.alpha_ema) * self.centroid + self.alpha_ema * z
            self.centroid = c / max(float(np.linalg.norm(c)), _EPS)

        # Approved decision 1: kappa tracks ref_set exactly (recompute on
        # every mutation; unchanged ref_set => cache already consistent).
        # The T4 dirty counter records the same mutations for FR-5.1's lazy
        # tau recomputation (fpcmc.thresholds resets it).
        if changed:
            self.kappa = _estimate_kappa(self.ref_set)
            self.refset_changes_since_tau += 1


def _estimate_kappa(ref_set: np.ndarray) -> float:
    """fpcmc.scorers.estimate_kappa via deferred import (scorers imports us)."""
    from fpcmc.scorers import estimate_kappa

    return estimate_kappa(ref_set)
