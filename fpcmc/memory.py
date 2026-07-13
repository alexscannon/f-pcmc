"""STM/LTM memory management: promotion (T8; PRD FR-7.1-7.2, FR-5.4) and
duplicate-cluster merging (T9; PRD FR-8.1-8.3).

``PromotionEvaluator`` evaluates the four FR-7 criteria against mature STM
candidates and applies atomic promotion. ``MergeSweeper`` runs the periodic
FR-8 sweep (STM<->STM, STM->LTM folds, LTM<->LTM promoted pairs) plus the
on-promotion check; see its class docstring and owner decisions 16-20 below.

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

Owner-approved T9 decisions (Q&A 2026-07-11, pre-implementation; recorded in
docs/CHANGES.md T9):

  16. FR-8.1 condition-2 math — k = min(k_ref, available) throughout,
      matching every other kNN in the system. Within = the POOLED mean of
      both ref_sets' leave-one-out knn scores (the frozen
      ``fpcmc.thresholds.loo_knn_scores`` discipline: self masked to +inf);
      cross = the pooled mean over both directions (each member of A scored
      against B's full set, and vice versa). Merge iff
      cross_mean / within_mean <= MERGE_CROSS_WITHIN_MAX (1.1, the FR-8.1
      literal — deliberately not a §8 config key) AND centroid cosine
      similarity >= merge_sim.
  17. Sweep mechanics — PRD order per sweep: STM<->STM (FR-8.1), then
      STM->LTM folds (FR-8.2), then LTM<->LTM promoted pairs (FR-8.3); each
      phase iterates ascending-id pairs to a deterministic fixpoint with
      merges effective immediately (so A<-B then A<-C lands in one sweep and
      a survivor is instantly eligible for further merges).
      ``on_promotion`` runs just the LTM<->LTM phase (idempotent).
  18. Survivor bookkeeping — FR-8.1 merges (STM<->STM, LTM<->LTM) union the
      history: match_count sum, match_windows union, last_matched_at max,
      ref_count_seen sum; the survivor keeps its concept_id, created_at,
      status and provenance. An STM survivor's centroid is recomputed as the
      normalized mean of the FULL ref_set union (before any subsampling); an
      LTM survivor's centroid stays bit-frozen (FR-1.3). FR-8.2 folds move
      ref_set + ref_count_seen ONLY — a fold is not a match event, so the
      LTM concept's match statistics stay honest.
  19. Bounded union — a union larger than K_max is uniformly subsampled
      without replacement via the dedicated substream
      ``make_rng(config.seed, f"merge/{step}/{survivor}<-{absorbed}")``; the
      survivor's own reservoir Generator is never touched, keeping the
      FR-1.1 draw discipline (pinned by test_reservoir_uniformity's exact
      replay) pure.
  20. Singleton pairs — pairs with K < 2 on either side are NOT
      sweep-mergeable: the within-structure FR-8.1's ratio guards with is
      unobservable (PRD §11 names that ratio the near-OOD-collapse guard, so
      err conservative). Singleton consolidation is T10's job: its HDBSCAN
      grouping calls ``merge_pair`` directly, the clustering standing in for
      the two-condition evidence.

Merge sites replace ref_sets wholesale (outside ``add_observation``), so per
the T4 rule they recompute kappa themselves (``fpcmc.scorers.estimate_kappa``)
before recomputing taus via the status-sensitive
``fpcmc.thresholds.recompute_thresholds`` (LTM -> pure FR-5.1, STM -> FR-5.2
shrinkage; resets the dirty counter) — which is why ``MergeSweeper`` holds
the frozen ``GlobalPrior`` (the store cannot expose its own without violating
invariant 5's AST guard). Lineage: the survivor's ``merged_from`` gains the
absorbed id plus the absorbed concept's own ``merged_from`` (chained
absorption stays resolvable); ``MergeSweeper.lineage`` is the FR-1.4
store-level {survivor: [absorbed...]} view over this sweeper's merge_log.

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

from fpcmc.concepts import Concept, ConceptStore, cohesion
from fpcmc.config import FPCMCConfig
from fpcmc.rng import make_rng
from fpcmc.scorers import estimate_kappa, make_scorer
from fpcmc.thresholds import (
    GlobalPrior,
    loo_knn_scores,
    recompute_on_promotion,
    recompute_thresholds,
)

_EPS = 1e-12

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


# FR-7 criterion 2 statistic. Defined in fpcmc.concepts (a pure function of a
# ref_set) and re-exported here: the FR-9 tier-1 cohesion gate reads the same
# statistic, and the two must never disagree. Imported above; re-exported for
# the callers (and tests) that reach for `fpcmc.memory.cohesion`.
__all__ = [
    "cohesion",
    "CRITERIA",
    "PromotionEvaluator",
    "PromotionDecision",
    "PromotionRecord",
    "MergeSweeper",
    "MergeCheck",
    "MergeRecord",
]


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


# ================================================================ merging (T9)

#: FR-8.1 literal: cross-ref mean kNN distance <= 1.1x the within mean.
#: Deliberately not a PRD §8 config key (decision 16).
MERGE_CROSS_WITHIN_MAX = 1.1


@dataclass(frozen=True)
class MergeCheck:
    """FR-8.1 two-condition evidence for one candidate pair (decision 16).

    compatible = centroid_sim >= merge_sim AND cross_within_ratio <= 1.1.
    """

    centroid_sim: float
    cross_within_ratio: float
    compatible: bool


@dataclass(frozen=True)
class MergeRecord:
    """One merge/fold, as the event log will see it (mirrors EvictionRecord).

    kind: "stm_stm" (FR-8.1) | "stm_ltm" (FR-8.2 fold) | "ltm_ltm" (FR-8.3).
    cross_within_ratio is NaN for folds — FR-8.2 triggers on threshold
    acceptance, not on the two-condition rule. match counts are recorded
    post-merge for the survivor, at-absorption for the absorbed concept.
    """

    step: int
    kind: str
    survivor_id: str
    absorbed_id: str
    centroid_sim: float
    cross_within_ratio: float
    survivor_match_count: int
    absorbed_match_count: int


def _cross_knn_scores(queries: np.ndarray, ref_set: np.ndarray, k_ref: int) -> np.ndarray:
    """(N,) mean cosine distance of each query row to its min(k_ref, K)
    nearest ref_set members — the FR-4.1 expression, batched over rows."""
    dists = 1.0 - queries @ ref_set.T
    k = min(int(k_ref), ref_set.shape[0])
    return np.partition(dists, k - 1, axis=1)[:, :k].mean(axis=1)


class MergeSweeper:
    """FR-8 duplicate-cluster merging over a ConceptStore (decisions 16-20).

    ``sweep(store, step)`` runs the periodic pass in PRD order — STM<->STM,
    STM->LTM folds, LTM<->LTM promoted pairs — each phase to a deterministic
    fixpoint. ``on_promotion(store, step)`` is the FR-8 on-promotion check
    (LTM<->LTM phase only); T11's runner calls it after each promotion.
    ``merge_pair`` applies the FR-8.1 mechanics unconditionally and is the
    public seam T10's identity-preserving consolidation calls directly.

    Holds the frozen GlobalPrior because post-merge taus go through the
    status-sensitive ``recompute_thresholds`` (STM survivors shrink toward
    the prior); acceptance decisions still read only per-concept thresholds.
    """

    def __init__(self, config: FPCMCConfig, prior: GlobalPrior) -> None:
        self._config = config
        self._prior = prior
        self._scorer = make_scorer(config)
        self.merge_log: list[MergeRecord] = []

    @property
    def lineage(self) -> dict[str, list[str]]:
        """FR-1.4 store-level lineage map {survivor: [absorbed, ...]} over
        this sweeper's merges, in merge order."""
        out: dict[str, list[str]] = {}
        for rec in self.merge_log:
            out.setdefault(rec.survivor_id, []).append(rec.absorbed_id)
        return out

    # ------------------------------------------------------------ entry points

    def sweep(self, store: ConceptStore, step: int) -> None:
        """The periodic FR-8 sweep (every T_merge steps; wired by T11)."""
        self._sweep_stm_stm(store, step)
        self._sweep_stm_ltm(store, step)
        self._sweep_ltm_ltm(store, step)

    def on_promotion(self, store: ConceptStore, step: int) -> None:
        """FR-8.3 on-promotion check: newly promoted vs previously promoted.

        Runs the (idempotent) LTM<->LTM phase — restricting it to pairs
        involving the newcomer would be equivalent, since earlier pairs
        already reached fixpoint at the last sweep/promotion."""
        self._sweep_ltm_ltm(store, step)

    # ------------------------------------------------------ two-condition rule

    def check_pair(self, a: Concept, b: Concept) -> MergeCheck:
        """FR-8.1 evidence for one pair (decision 16). Pairs with K < 2 on
        either side are never sweep-compatible (decision 20): the within
        structure the ratio guards with is unobservable, and the ratio is
        reported NaN."""
        config = self._config
        sim = float(a.centroid @ b.centroid)
        if a.ref_set.shape[0] < 2 or b.ref_set.shape[0] < 2:
            return MergeCheck(centroid_sim=sim, cross_within_ratio=float("nan"), compatible=False)
        within = np.concatenate([
            loo_knn_scores(a.ref_set, config.k_ref),
            loo_knn_scores(b.ref_set, config.k_ref),
        ])
        cross = np.concatenate([
            _cross_knn_scores(a.ref_set, b.ref_set, config.k_ref),
            _cross_knn_scores(b.ref_set, a.ref_set, config.k_ref),
        ])
        ratio = float(cross.mean() / max(float(within.mean()), _EPS))
        return MergeCheck(
            centroid_sim=sim,
            cross_within_ratio=ratio,
            compatible=sim >= config.merge_sim and ratio <= MERGE_CROSS_WITHIN_MAX,
        )

    # ------------------------------------------------------------------ phases

    def _sweep_stm_stm(self, store: ConceptStore, step: int) -> None:
        """FR-8.1 phase: ascending-id STM pairs to fixpoint (decision 17)."""
        merged = True
        while merged:
            merged = False
            stm = sorted(store.stm, key=lambda c: c.concept_id)
            for i in range(len(stm)):
                for j in range(i + 1, len(stm)):
                    check = self.check_pair(stm[i], stm[j])
                    if check.compatible:
                        self.merge_pair(store, stm[i], stm[j], step, kind="stm_stm", check=check)
                        merged = True
                        break
                if merged:
                    break

    def _sweep_stm_ltm(self, store: ConceptStore, step: int) -> None:
        """FR-8.2 phase: fold every STM candidate whose centroid is ACCEPTED
        by an LTM concept (the exact complement of T8's separation criterion
        at sep_factor=1) into the best-margin accepting LTM concept — the
        frozen ``Scorer.select`` semantics, lexicographic tie-break included —
        AND whose centroid is at least ``merge_sim`` similar to that concept's.

        The similarity floor is the 2026-07-11 owner ruling (docs/CHANGES.md).
        Acceptance alone is not a safe fold trigger: a fold unions the
        candidate's ref_set into the LTM concept, so a bad fold corrupts a
        consolidated memory permanently. Worse, acceptance is measured against
        the LTM's OWN tau, which FR-5.1 recalibrates from its own ref_set — so
        a concept that absorbs one bad fold gets a looser tau, which makes it
        accept a worse candidate, which loosens tau further, *within a single
        fixpoint sweep*. Observed on the golden stream before this guard:
        ltm_006 folded in a candidate at centroid similarity 0.078, its tau
        inflated 5x (0.139 -> 0.701), its cohesion collapsed to 0.29, and it
        went on to capture 169 known-class arrivals belonging to the other
        seven classes. FR-8.1 condition 1 is the guard PRD §11 already relies
        on against near-OOD collapse; the fold path simply never applied it.
        """
        folded = True
        while folded:
            folded = False
            for cand in sorted(store.stm, key=lambda c: c.concept_id):
                selection = self._scorer.select(cand.centroid, store.ltm)
                if selection is None:
                    continue
                if not self.similar_enough(selection.concept, cand):
                    continue
                self._fold(store, selection.concept, cand, step)
                folded = True
                break

    def similar_enough(self, a: Concept, b: Concept) -> bool:
        """FR-8.1 condition 1 alone: centroid cosine similarity >= merge_sim.

        Condition 2 (the cross/within ratio) is deliberately NOT applied: it is
        unobservable for singletons (T9 decision 20: K < 2 has no
        within-structure), which is exactly why the guarded call site below
        originally ran unguarded.

        SCOPE: this is the FR-8.2 fold guard, and it is a CENTROID-vs-CENTROID
        test. `merge_sim` is calibrated for centroids-of-many and only means
        what it says at that scale — measured on the real DINOv3 pools,
        same-class 20-sample half-centroids score 0.94-0.98 and cross-class
        pairs 0.16-0.51, so 0.80 separates them cleanly. Do NOT reuse this for
        singleton admission: a same-class SINGLE embedding scores only ~0.735
        against its own class centroid (cross-class ~0.241), so a 0.80 bar
        refuses 68% of genuine same-class singletons. The residual path uses
        ``accepts_into`` instead.
        """
        sim = float(a.centroid @ b.centroid)
        return sim >= self._config.merge_sim

    def accepts_into(self, host: Concept, other: Concept) -> bool:
        """Does ``host`` accept ``other``'s centroid under the frozen scorer?

        The scale-correct admission test for merging a SINGLETON (or any small
        candidate) into a host concept — the 2026-07-11 owner ruling
        (docs/CHANGES.md). A fixed cosine bar cannot serve both scales: a
        centroid-of-many and a single embedding live at completely different
        similarity ranges, and on real data no constant separates same-class
        singletons (~0.735, min 0.497) from cross-class ones (~0.241, max
        0.583) cleanly.

        `tau` is exactly the right instrument: it IS a per-concept, adaptively
        calibrated acceptance radius for SINGLE embeddings (FR-5.1 fits it to
        the concept's own LOO score distribution), so it auto-scales to each
        class's true spread instead of imposing one global constant on classes
        whose real cohesion ranges 0.49-0.70. This is the same question routing
        asks of every arrival — "does this embedding belong to this concept?" —
        answered by the same frozen scorer.

        BUT the instrument only exists where it has been calibrated. A host
        with ref_set size 1 has no LOO distribution to fit, so FR-5.1 holds its
        tau at the global prior (T4's below-the-floor rule; T5 decision 3 seeds
        every singleton with prior.tau). Testing against that is not a
        per-concept judgement at all — it is a global constant, and a tight one:
        on the golden stream prior.tau = 0.1537, i.e. it would demand cosine
        similarity >= 0.846, STRICTER than the merge_sim bar this replaces.
        A singleton host is therefore not consulted through tau at all — it has
        no opinion to offer, only the prior's.

        So admission is the DISJUNCTION of the two scale-appropriate tests, one
        for each regime, and a pair is refused only when BOTH refuse:

          * the host has a calibrated tau (ref_set >= 2) and accepts the
            other's centroid — the singleton-scale test, which is the only one
            that works when `other` is a lone embedding (real data: same-class
            singleton-vs-centroid 0.735, which no 0.8 cosine bar admits); or
          * their centroids are >= merge_sim similar — the centroid-scale test,
            which is the only one available when the host is itself a singleton
            with nothing but the prior.

        Neither test admits the incoherent merges: the blobs this guard exists
        to prevent were singleton-into-singleton fusions of unrelated classes
        at centroid similarity 0.12-0.48 (down to -0.443), which fail the
        cosine bar, while their singleton hosts have no calibrated tau to
        appeal to. Both doors are shut for them; both are open for genuine
        consolidation at either scale.
        """
        if host.ref_set.shape[0] >= 2 and self._scorer.accepts(other.centroid, host):
            return True
        return self.similar_enough(host, other)

    def _sweep_ltm_ltm(self, store: ConceptStore, step: int) -> None:
        """FR-8.3 phase: FR-8.1 rule over provenance="promoted" pairs only —
        two "initial" concepts never merge, nor does a promoted/initial pair.
        """
        merged = True
        while merged:
            merged = False
            promoted = sorted(
                (c for c in store.ltm if c.provenance == "promoted"),
                key=lambda c: c.concept_id,
            )
            for i in range(len(promoted)):
                for j in range(i + 1, len(promoted)):
                    check = self.check_pair(promoted[i], promoted[j])
                    if check.compatible:
                        self.merge_pair(
                            store, promoted[i], promoted[j], step, kind="ltm_ltm", check=check
                        )
                        merged = True
                        break
                if merged:
                    break

    # --------------------------------------------------------------- mechanics

    def fold_pair(self, store: ConceptStore, ltm: Concept, cand: Concept, step: int) -> None:
        """Public FR-8.2 fold seam (additive, T11): re-applies one logged
        stm_ltm fold with the exact ``_fold`` mechanics — the replay module
        must reconstruct folds through T9 code, not a reimplementation."""
        self._fold(store, ltm, cand, step)

    def merge_pair(
        self,
        store: ConceptStore,
        a: Concept,
        b: Concept,
        step: int,
        *,
        kind: str = "stm_stm",
        check: Optional[MergeCheck] = None,
    ) -> Concept:
        """Apply the FR-8.1 merge mechanics unconditionally; returns the
        survivor. Public seam for T10 (decision 20): HDBSCAN-grouped immature
        candidates are merged through here, the clustering standing in for
        the two-condition check (pass check=None).

        Survivor = larger match_count, ties to the smaller concept_id.
        Bookkeeping per decision 18; bounded union per decision 19; kappa
        recomputed at the merge site before the status-sensitive tau
        recompute (T4 rule).
        """
        if b.match_count > a.match_count or (
            b.match_count == a.match_count and b.concept_id < a.concept_id
        ):
            survivor, absorbed = b, a
        else:
            survivor, absorbed = a, b

        sim = float(survivor.centroid @ absorbed.centroid) if check is None else check.centroid_sim
        union = np.vstack([survivor.ref_set, absorbed.ref_set])
        if survivor.status == "STM":
            c = union.mean(axis=0)  # full union, pre-subsample (decision 18)
            survivor.centroid = c / max(float(np.linalg.norm(c)), _EPS)
        survivor.ref_set = self._bounded_union(union, step, survivor.concept_id, absorbed.concept_id)
        survivor.ref_count_seen += absorbed.ref_count_seen
        survivor.match_count += absorbed.match_count
        survivor.match_windows |= absorbed.match_windows
        survivor.last_matched_at = max(survivor.last_matched_at, absorbed.last_matched_at)
        survivor.kappa = estimate_kappa(survivor.ref_set)
        recompute_thresholds(survivor, self._config, self._prior)
        survivor.merged_from.extend([absorbed.concept_id, *absorbed.merged_from])
        store.remove(absorbed.concept_id)
        self.merge_log.append(
            MergeRecord(
                step=int(step),
                kind=kind,
                survivor_id=survivor.concept_id,
                absorbed_id=absorbed.concept_id,
                centroid_sim=sim,
                cross_within_ratio=float("nan") if check is None else check.cross_within_ratio,
                survivor_match_count=survivor.match_count,
                absorbed_match_count=absorbed.match_count,
            )
        )
        return survivor

    def _fold(self, store: ConceptStore, ltm: Concept, cand: Concept, step: int) -> None:
        """FR-8.2 fold: ref_set + ref_count_seen move, nothing else — the
        LTM centroid stays bit-frozen and its match statistics untouched
        (decision 18); kappa/taus recomputed for the new ref_set (T4 rule;
        LTM branch = pure FR-5.1)."""
        sim = float(ltm.centroid @ cand.centroid)
        union = np.vstack([ltm.ref_set, cand.ref_set])
        ltm.ref_set = self._bounded_union(union, step, ltm.concept_id, cand.concept_id)
        ltm.ref_count_seen += cand.ref_count_seen
        ltm.kappa = estimate_kappa(ltm.ref_set)
        recompute_thresholds(ltm, self._config, self._prior)
        ltm.merged_from.extend([cand.concept_id, *cand.merged_from])
        store.remove(cand.concept_id)
        self.merge_log.append(
            MergeRecord(
                step=int(step),
                kind="stm_ltm",
                survivor_id=ltm.concept_id,
                absorbed_id=cand.concept_id,
                centroid_sim=sim,
                cross_within_ratio=float("nan"),
                survivor_match_count=ltm.match_count,
                absorbed_match_count=cand.match_count,
            )
        )

    def _bounded_union(self, union: np.ndarray, step: int, survivor_id: str, absorbed_id: str) -> np.ndarray:
        """Decision 19: cap the union at K_max via a uniform no-replacement
        subsample from the dedicated merge substream (original row order
        preserved); the survivor's reservoir Generator is never consumed."""
        k_max = self._config.K_max_refset
        if union.shape[0] <= k_max:
            return np.array(union)
        rng = make_rng(self._config.seed, f"merge/{step}/{survivor_id}<-{absorbed_id}")
        idx = np.sort(rng.choice(union.shape[0], size=k_max, replace=False))
        return np.array(union[idx])
