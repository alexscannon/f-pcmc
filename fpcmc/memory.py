"""STM -> LTM promotion (T8; PRD FR-7.1-7.2, FR-5.4).

``PromotionEvaluator`` evaluates the four FR-7 criteria against mature STM
candidates and applies atomic promotion. T9 adds the MergeSweeper (FR-8) to
this module.

The four criteria (all must hold; evaluated in PRD order, never
short-circuited, so a decision always names every failing criterion):

  1. size:       match_count >= theta_promote (post-seed matches, T3
                 decision 2 — the seeding embedding never counts).
  2. cohesion:   mean pairwise cosine similarity within ref_set over DISTINCT
                 pairs (self-similarities excluded — they are identically 1
                 and would dilute the statistic) >= min_cohesion.
  3. separation: the candidate's centroid would itself be rejected by every
                 LTM concept (see decision 13 below).
  4. recurrence: matches span >= m_windows distinct window_W windows
                 (len(match_windows); windows are step // window_W, T3).

Owner-approved T8 decisions (Q&A 2026-07-11, pre-implementation; recorded in
docs/CHANGES.md T8 — numbering continues from the T7 notes in concepts.py):

  12. Cadence — per-assignment: PRD §8 defines no promotion-schedule key, and
      FR-7.1 says promotion is "immediate", so the production call site is
      ``check(concept, store, step)`` after every tier-1/2 assignment on the
      just-matched concept (runner-side, T11). ``check`` no-ops unless the
      concept is a mature STM candidate — every FR-7 statistic of a candidate
      changes only on its own matches, so per-assignment checks lose nothing
      but LTM-tau-drift edge cases. ``evaluate(store, step)`` is the full
      sweep for T11's periodic-hook machinery and tests.
  13. Separation — rejection by every LTM concept under the CONFIGURED scorer
      with sep_factor-scaled thresholds: under knn_vmf both sub-scorers must
      reject (s > sep_factor * tau each; OR-accept negated), which is exactly
      ``not accepts()`` at the sep_factor=1.0 default. Caveat, documented:
      tau_vmf < 0 at D=1024, so multiplying by sep_factor != 1 inverts the
      scaling direction on the vmf side; sep_factor defaults to 1.0 and is
      never swept (PRD §8).
  14. Sweeps are live, in ascending concept_id (allocation order, matching
      the FR-4.3 id-ordering convention): a promotion takes effect
      immediately, so a same-class fragment evaluated later in the same pass
      is blocked by separation against the freshly promoted concept — a
      structural guard for the golden gate's fragmentation-index assertion.
  15. Logging — ``evaluate``/``check`` RETURN ``PromotionDecision``s (pass/
      fail per criterion; the test-visible "log naming the failing
      criterion"); only successful promotions persist, as
      ``PromotionRecord``s on ``evaluator.promotion_log`` (mirroring
      ``store.eviction_log``). FR-7.2's eval-only fields ride along as
      placeholders: gt_majority_label is copied from the concept (None at
      runtime — invariant 2, the pipeline never sees labels) and purity is
      None, both filled post hoc by the T13 harness.

The reported separation margin is the binding (minimum) normalized rejection
margin ``(s - tau') / |tau'|`` with ``tau' = sep_factor * tau``, taken over
all LTM concepts and their applicable sub-scorers — positive iff criterion 3
passes (+inf when no LTM concept exists: vacuous separation).

Atomic promotion (FR-7.1) is three mutations plus a record, nothing else:
status -> "LTM" (which freezes the centroid via FR-1.3 and releases the STM
capacity slot via the live status views — T5/T7 as-built), provenance
"seeded" -> "promoted" (T3 decision 4), and
``fpcmc.thresholds.recompute_on_promotion`` (FR-5.4: pure FR-5.1 percentiles
for both taus — NEVER the status-sensitive ``recompute_thresholds``). Tier
membership is recomputed live at every ``route`` call, so the flip
participates in tier-1 routing on the very next example (proven by T5's
test_promoted_participates_immediately).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fpcmc.concepts import Concept, ConceptStore
from fpcmc.config import FPCMCConfig
from fpcmc.thresholds import recompute_on_promotion

#: FR-7 criteria in PRD (and evaluation) order; ``PromotionDecision.failed``
#: is always an in-order subsequence of this tuple.
CRITERIA = ("size", "cohesion", "separation", "recurrence")


@dataclass(frozen=True)
class PromotionRecord:
    """FR-7.2 promotion log record (one per successful promotion).

    size is the candidate's match_count at promotion (PRD FR-7's "Size"
    vocabulary, consistent with EvictionRecord.size); separation_margin is
    the binding rejection margin defined in the module docstring;
    gt_majority_label/purity are eval-only placeholders (decision 15).
    """

    step: int
    concept_id: str
    size: int
    cohesion: float
    separation_margin: float
    window_count: int
    gt_majority_label: Optional[str]
    purity: Optional[float]


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of evaluating one candidate against the four FR-7 criteria.

    All four criteria are always evaluated (no short-circuit); ``failed``
    names every failing criterion, in CRITERIA order — empty iff promoted.
    The measured statistics ride along for diagnostics and the event log.
    """

    concept_id: str
    step: int
    promoted: bool
    failed: tuple[str, ...]
    size: int
    cohesion: float
    separation_margin: float
    window_count: int


def cohesion(ref_set: np.ndarray) -> float:
    """FR-7 criterion 2 statistic: mean pairwise cosine similarity within the
    ref_set, over distinct pairs (self-pairs excluded; see module docstring).

    A single-member ref_set is vacuously cohesive (1.0) — unreachable for a
    mature candidate, whose ref_set holds the seed plus >= n_mature matches.
    """
    k = ref_set.shape[0]
    if k < 2:
        return 1.0
    g = ref_set @ ref_set.T
    return float((g.sum() - np.trace(g)) / (k * (k - 1)))


class PromotionEvaluator:
    """FR-7 promotion machinery over a ConceptStore (see module docstring).

    Stateful only in ``promotion_log``; construction fixes the config. The
    evaluator never touches the global prior (invariant 5 stays confined to
    the store's seeding/shrinkage paths — FR-5.4 uses the prior-free pure
    percentile hook).
    """

    def __init__(self, config: FPCMCConfig) -> None:
        self._config = config
        self.promotion_log: list[PromotionRecord] = []

    # ------------------------------------------------------------ entry points

    def check(self, concept: Concept, store: ConceptStore, step: int) -> Optional[PromotionDecision]:
        """Per-assignment cadence (decision 12): evaluate the just-matched
        concept iff it is a mature STM candidate; None otherwise (LTM and
        already-promoted concepts included — idempotence)."""
        if concept.status != "STM" or concept.match_count < self._config.n_mature:
            return None
        return self._evaluate_one(concept, store, step)

    def evaluate(self, store: ConceptStore, step: int) -> list[PromotionDecision]:
        """Full sweep over mature STM candidates, ascending concept_id, LIVE
        LTM view (decision 14): each promotion is visible to the separation
        checks of every candidate evaluated after it in the same pass."""
        candidates = sorted(
            (c for c in store.stm if c.match_count >= self._config.n_mature),
            key=lambda c: c.concept_id,
        )
        return [self._evaluate_one(c, store, step) for c in candidates]

    # -------------------------------------------------------------- evaluation

    def _evaluate_one(self, concept: Concept, store: ConceptStore, step: int) -> PromotionDecision:
        config = self._config
        coh = cohesion(concept.ref_set)
        sep = self._separation_margin(concept.centroid, store.ltm)
        checks = (
            ("size", concept.match_count >= config.theta_promote),
            ("cohesion", coh >= config.min_cohesion),
            ("separation", sep > 0.0),
            ("recurrence", len(concept.match_windows) >= config.m_windows),
        )
        failed = tuple(name for name, ok in checks if not ok)
        decision = PromotionDecision(
            concept_id=concept.concept_id,
            step=int(step),
            promoted=not failed,
            failed=failed,
            size=concept.match_count,
            cohesion=coh,
            separation_margin=sep,
            window_count=len(concept.match_windows),
        )
        if decision.promoted:
            self._promote(concept, step, coh, sep)
        return decision

    def _separation_margin(self, z: np.ndarray, ltm_concepts: list[Concept]) -> float:
        """Binding rejection margin of centroid ``z`` against every LTM
        concept (decision 13). Mirrors the frozen scorers' score expressions
        exactly (fpcmc/scorers.py; same discipline as ConceptStore's batch
        path): knn_ref = mean cosine distance to the clipped-k nearest
        ref_set members; vmf = -(log C_D(kappa) + kappa * mu.z), with the
        FR-4.2 fallback rule deciding which sub-scorers apply per concept.
        """
        # Deferred to keep import order conventional (scorers imports concepts).
        from fpcmc.scorers import _margin, log_C_D

        config = self._config
        sf = config.sep_factor
        binding = float("inf")
        for c in ltm_concepts:
            fallback = c.ref_set.shape[0] < config.n_vmf_min
            margins = []
            if config.scorer in ("knn_ref", "knn_vmf") or (config.scorer == "vmf" and fallback):
                dists = 1.0 - c.ref_set @ z
                k = min(config.k_ref, dists.shape[0])
                s = float(np.mean(np.partition(dists, k - 1)[:k]))
                margins.append(-_margin(sf * c.tau, s))
            if config.scorer in ("vmf", "knn_vmf") and not fallback:
                kappa = c.kappa
                if not np.isfinite(kappa):
                    raise ValueError(
                        f"concept {c.concept_id!r}: kappa={kappa} is not finite but "
                        f"ref_set has {c.ref_set.shape[0]} >= n_vmf_min="
                        f"{config.n_vmf_min} members — the owner must maintain the "
                        "cached Banerjee estimate (FR-4.2)"
                    )
                d = c.centroid.shape[0]
                s = -(log_C_D(kappa, d) + kappa * float(c.centroid @ z))
                margins.append(-_margin(sf * c.tau_vmf, s))
            binding = min(binding, min(margins))
        return binding

    # --------------------------------------------------------------- promotion

    def _promote(self, concept: Concept, step: int, coh: float, sep: float) -> None:
        """FR-7.1 atomic promotion (see module docstring). No store surgery:
        the status flip alone freezes the centroid, releases the STM slot,
        and enters tier-1 routing on the next call (T5/T7 live views)."""
        concept.status = "LTM"
        concept.provenance = "promoted"  # T3 decision 4
        recompute_on_promotion(concept, self._config)  # FR-5.4
        self.promotion_log.append(
            PromotionRecord(
                step=int(step),
                concept_id=concept.concept_id,
                size=concept.match_count,
                cohesion=float(coh),
                separation_margin=float(sep),
                window_count=len(concept.match_windows),
                gt_majority_label=concept.gt_majority_label,
                purity=None,
            )
        )
