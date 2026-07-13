"""Concept data structure, per-concept dynamics, and the ConceptStore
routing core (T2 stub; Concept completed at T3; store added at T5).

PRD FR-1: every known class — T0-initial and promoted-novel alike — is a
`Concept`. This module carries the dataclass plus the two per-concept
dynamics entry points:

  - ``add_observation(z, step)`` — FR-1.1 reservoir maintenance, FR-1.3
    centroid dynamics (EMA for STM, bit-frozen for LTM), and the match /
    window / LRU bookkeeping the T7+ memory machinery reads.
  - ``Concept.seed(...)`` — FR-3.2 singleton constructor for novel embeddings.

T5 adds ``ConceptStore`` (LTM + STM registries, store-owned concept-id
allocation, the exact FR-9 three-tier decision cascade in ``route``) and
``RoutingResult``. T7 adds the FR-3.1 STM dynamics: capacity ``Δ``
(config.stm_capacity) enforced at the tier-3 seeding site — the only STM
growth site — by LRU eviction on ``last_matched_at`` (ties: older
``created_at`` first, then smallest ``concept_id``), each eviction logged as
an ``EvictionRecord``. FR-3.3 maturity needs no stored transition: tier
membership is already a live ``match_count >= n_mature`` predicate at route
time. Approved T7 decisions (owner Q&A, 2026-07-11; docs/CHANGES.md T7):

  10. EvictionRecord.size = match_count at eviction (PRD FR-7 criterion 1
      names match_count "Size"); age = step - created_at (lifetime). Beyond
      the TASKS-literal four fields the record carries created_at,
      last_matched_at and ref_count_seen so T11's JSONL `evict` records and
      T13's eviction-composition metric need no extra plumbing.
  11. Full LRU ties (last_matched_at AND created_at equal) evict the smallest
      concept_id; capacity is a drain-while loop before the seed lands
      (normally exactly one eviction; a store hand-built above Δ converges
      back under it). LTM is exempt regardless of staleness; mature-but-
      unpromoted STM candidates are NOT exempt (FR-3.1 exempts only LTM).

Approved T5 decisions (owner Q&A, 2026-07-11; docs/CHANGES.md T5):

  5. RoutingResult.score is the winning concept's ScoreDetail.score (under
     knn_vmf: the composed scalar = knn_ref sub-score, per the T2 decision),
     plus `via`/`fallback` metadata for the event log and the A5 ablation;
     tier 3 carries score = margin = NaN and via = None (nothing accepted).
  6. The FR-5.1 lazy threshold check (fpcmc.thresholds.maybe_recompute) runs
     after every assignment's add_observation, on the matched concept only,
     tiers 1 and 2 alike; tier-3 seeds skip it (fresh dirty counter).
  7. Every seeded singleton bootstraps tau = prior.tau AND
     tau_vmf = prior.tau_vmf — the GlobalPrior is always a full pair, so NaN
     never enters a routed concept regardless of scorer config.
  8. Concept ids are PRD-literal zero-padded ("ltm_{:03d}" / "stm_{:04d}"),
     store-owned monotone counters, never reused; the allocator raises on
     overflow rather than widening, preserving lexicographic == numeric
     ordering (the FR-4.3 margin tie-break makes id order behavior-relevant;
     "ltm_" < "stm_" resolves exact cross-status ties LTM-first).
  9. The batch scoring path computes per-concept ``ref_set @ z`` — the exact
     frozen-scorer op — and vectorizes composition/selection over concept
     arrays. TASKS T5's literal "single matrix op per tier" was measured
     bitwise-incompatible with the frozen per-concept scorer math (a stacked
     GEMV's summation order depends on row position; ~half of row dots differ
     by 1 ulp at every D incl. 1024), and the test_vectorized_matches_loop
     identity guard is the binding clause. NFR-1 is met: the per-concept
     matvec is still the FLOP-dominant op with no per-row Python work.

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

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

from fpcmc.config import FPCMCConfig
from fpcmc.rng import make_rng

if TYPE_CHECKING:  # real imports are deferred: scorers/thresholds import us
    from fpcmc.scorers import ScoreDetail
    from fpcmc.thresholds import GlobalPrior

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

    # Lazy cache for the `cohesion` property. Kept out of __init__/__repr__/
    # __eq__ so it is invisible to construction and to the bitwise concept
    # comparisons the determinism tests make.
    _cohesion_cache: Optional[float] = field(
        default=None, init=False, repr=False, compare=False
    )

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
        # Wholesale ref_set replacement (T9 merges, the reservoir's vstack
        # append) invalidates the cohesion cache here, so no caller has to
        # remember to — unlike kappa, which merge sites must recompute
        # themselves. In-place row replacement bypasses __setattr__ and is
        # invalidated explicitly in add_observation.
        if name == "ref_set":
            super().__setattr__("_cohesion_cache", None)
        super().__setattr__(name, value)

    @property
    def cohesion(self) -> float:
        """FR-7 criterion-2 statistic over the current ref_set, cached.

        Read on every route by the FR-9 tier-1 cohesion gate, so it must not
        be O(K^2 * D) per step: the cache is invalidated on every ref_set
        mutation (see __setattr__ and add_observation), and a concept's ref_set
        changes only on its own match — at most one recompute per step.
        """
        if self._cohesion_cache is None:
            object.__setattr__(self, "_cohesion_cache", cohesion(self.ref_set))
        return self._cohesion_cache

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
            # In-place reservoir replacement does not go through __setattr__.
            self._cohesion_cache = None


def _estimate_kappa(ref_set: np.ndarray) -> float:
    """fpcmc.scorers.estimate_kappa via deferred import (scorers imports us)."""
    from fpcmc.scorers import estimate_kappa

    return estimate_kappa(ref_set)


def cohesion(ref_set: np.ndarray) -> float:
    """Mean pairwise cosine similarity within the ref_set, over DISTINCT pairs
    (self-pairs are trivially 1.0 and would dilute the statistic).

    This is the FR-7 criterion-2 statistic. It lives here, not in fpcmc.memory,
    because it is a pure function of a Concept's ref_set and has TWO consumers
    that must never disagree: FR-7 promotion (fpcmc.memory re-exports this
    name) and the FR-9 tier-1 cohesion gate below. One definition, no drift.

    A single-member ref_set is vacuously cohesive (1.0) — unreachable for a
    mature candidate, whose ref_set holds the seed plus >= n_mature matches.
    """
    k = ref_set.shape[0]
    if k < 2:
        return 1.0
    g = ref_set @ ref_set.T
    return float((g.sum() - np.trace(g)) / (k * (k - 1)))


# ============================================================== routing (T5)


@dataclass(frozen=True)
class RoutingResult:
    """FR-9 outcome of routing one stream embedding (TASKS T5).

    prediction: the winning concept_id at tier 1; "unknown" at tiers 2 and 3
                (FR-9.1 — "unknown" is a legitimate prediction).
    concept_id: the assigned concept (tiers 1-2) or the freshly seeded
                singleton (tier 3).
    tier:       1 = LTM ∪ mature-STM acceptance, 2 = immature-STM acceptance,
                3 = seeded new candidate (FR-3.3 routing order).
    score:      the winning concept's scorer scalar (under knn_vmf the
                composed scalar = knn_ref sub-score, T2 owner decision, so
                acceptance is not derivable from it); NaN at tier 3 (T5 owner
                decision — nothing accepted, the seed never scored itself).
    margin:     the winning normalized margin (tau - s)/|tau|; NaN at tier 3.
    via:        sub-scorer that produced the margin ("knn_ref" / "vmf"; None
                at tier 3) — event-log and A5-ablation metadata.
    fallback:   FR-4.2 fallback flag of the winning ScoreDetail.
    """

    prediction: str
    concept_id: str
    tier: int
    score: float
    margin: float
    via: Optional[str]
    fallback: bool


@dataclass(frozen=True)
class EvictionRecord:
    """FR-3.1 eviction log record (TASKS T7; owner decisions 10–11).

    The "forgetting outliers" mechanism must be measurable: size is the
    concept's match_count at eviction (PRD FR-7's "Size" vocabulary — post-
    seed matches), age its lifetime ``step - created_at``. created_at /
    last_matched_at / ref_count_seen ride along so staleness and eviction
    composition (T13) are derivable straight from the record.
    """

    concept_id: str
    size: int
    age: int
    step: int
    created_at: int
    last_matched_at: int
    ref_count_seen: int


_ID_WIDTH = {"ltm": 3, "stm": 4}  # PRD FR-1 literals: "ltm_037" / "stm_0142"
_ID_PATTERN = re.compile(r"^(ltm|stm)_(\d+)$")


class ConceptStore:
    """LTM + STM registries and the FR-9 routing core (TASKS T5).

    The store owns concept-id allocation (approved T3 decision 3 / T5
    decision 8) and holds the frozen FR-5.3 ``GlobalPrior`` pair, supplied at
    construction (T6's ``initialize_ltm`` produces it in production). It
    consumes the frozen T2 scorer interface; acceptance decisions read only
    per-concept thresholds — the prior is reachable solely from ``_seed``
    (FR-3.2 bootstrap) and ``_assign`` (FR-5.2 shrinkage target forwarded to
    ``fpcmc.thresholds.maybe_recompute``), which cross-cutting invariant 5
    asserts structurally.

    ``ltm``/``stm``/tier membership are live views computed from
    ``concept.status`` and ``match_count`` at route time — status is the
    single source of truth, so a T8 promotion (or a manual flip in tests)
    participates in tier-1 routing on the very next call (FR-5.4), and it
    simultaneously frees the promoted concept's STM capacity slot (T7 reads
    the same live view).

    T7 (FR-3.1): STM capacity ``config.stm_capacity`` is enforced in
    ``_seed`` via LRU eviction; every eviction appends an ``EvictionRecord``
    to the public ``eviction_log``. Evicted ids are never reused.

    ``vectorized`` selects the batch scoring path (default; T5 decision 9);
    ``vectorized=False`` routes through the frozen ``Scorer.select`` loop —
    the reference implementation ``test_vectorized_matches_loop`` guards the
    batch path against.
    """

    def __init__(
        self,
        config: FPCMCConfig,
        prior: "GlobalPrior",
        concepts: Iterable[Concept] = (),
        *,
        vectorized: bool = True,
    ) -> None:
        # Deferred imports: both modules import Concept from us.
        from fpcmc.scorers import make_scorer
        from fpcmc.thresholds import maybe_recompute

        self._config = config
        self._prior = prior
        self._scorer = make_scorer(config)
        self._maybe_recompute = maybe_recompute
        self._vectorized = bool(vectorized)
        self._registry: dict[str, Concept] = {}
        self._ids_ever: set[str] = set()
        self._next_id = {"ltm": 0, "stm": 0}
        self.eviction_log: list[EvictionRecord] = []
        for concept in concepts:
            self.register(concept)

    # ------------------------------------------------------------- registry

    @property
    def concepts(self) -> list[Concept]:
        """All concepts, registration order."""
        return list(self._registry.values())

    @property
    def ltm(self) -> list[Concept]:
        """Live LTM view (status is the source of truth; see class docstring)."""
        return [c for c in self._registry.values() if c.status == "LTM"]

    @property
    def stm(self) -> list[Concept]:
        """Live STM view (status is the source of truth; see class docstring)."""
        return [c for c in self._registry.values() if c.status == "STM"]

    def get(self, concept_id: str) -> Concept:
        return self._registry[concept_id]

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, concept_id: str) -> bool:
        return concept_id in self._registry

    def register(self, concept: Concept) -> None:
        """Add a concept, enforcing run-wide id uniqueness (invariant 4).

        Ids are never reused, even across future removals (T7 eviction, T9
        merges): rejection checks every id ever registered, and the
        allocation counters advance past any externally allocated id.
        """
        cid = concept.concept_id
        if cid in self._ids_ever:
            raise ValueError(
                f"concept_id {cid!r} was already used in this run — ids are "
                "never reused (FR-1.4 / invariant 4)"
            )
        m = _ID_PATTERN.match(cid)
        if m:
            kind, num = m.group(1), int(m.group(2))
            self._next_id[kind] = max(self._next_id[kind], num + 1)
        self._ids_ever.add(cid)
        self._registry[cid] = concept

    def remove(self, concept_id: str) -> Concept:
        """Remove a concept from the live registry and return it (T9 merges).

        Absorption is not forgetting: unlike ``_evict`` this writes no
        EvictionRecord — the T9 MergeSweeper logs a MergeRecord instead. The
        id stays burned in ``_ids_ever`` (invariant 4: never reused), so an
        absorbed id can never reappear in routing. KeyError if absent.
        """
        return self._registry.pop(concept_id)

    def new_concept_id(self, kind: str) -> str:
        """Allocate the next zero-padded id (T5 decision 8; never reused).

        Overflow raises instead of widening: the FR-4.3 lexicographic
        tie-break relies on lexicographic == numeric id ordering.
        """
        width = _ID_WIDTH.get(kind)
        if width is None:
            raise ValueError(f"unknown concept kind {kind!r}; expected 'ltm' or 'stm'")
        n = self._next_id[kind]
        if n >= 10**width:
            raise ValueError(
                f"{kind} id space exhausted at {10 ** width} ids — widening would "
                "break the lexicographic ordering the margin tie-break relies on"
            )
        self._next_id[kind] = n + 1
        return f"{kind}_{n:0{width}d}"

    # -------------------------------------------------------------- routing

    def route(self, z: np.ndarray, step: int) -> RoutingResult:
        """Exactly the FR-9 decision cascade (FR-3.3 routing order).

        Tier 1: LTM ∪ tier1-eligible STM acceptance → best-margin assignment.
        Tier 2: else the remaining STM acceptance → assignment, prediction
                "unknown" (they cannot claim traffic from tier 1, however
                good their margin).
        Tier 3: else seed a new STM singleton (FR-3.2), prediction "unknown".

        The two STM sets partition the STM registry (`_tier1_stm` is the sole
        predicate), so the cascade stays total.
        """
        z = np.asarray(z, dtype=np.float64)
        concepts = list(self._registry.values())

        tier1 = [c for c in concepts if c.status == "LTM" or self._tier1_stm(c)]
        hit = self._select(z, tier1)
        if hit is not None:
            concept, detail = hit
            self._assign(concept, z, step)
            return RoutingResult(
                prediction=concept.concept_id,
                concept_id=concept.concept_id,
                tier=1,
                score=detail.score,
                margin=detail.margin,
                via=detail.via,
                fallback=detail.fallback,
            )

        tier2 = [c for c in concepts if c.status == "STM" and not self._tier1_stm(c)]
        hit = self._select(z, tier2)
        if hit is not None:
            concept, detail = hit
            self._assign(concept, z, step)
            return RoutingResult(
                prediction="unknown",
                concept_id=concept.concept_id,
                tier=2,
                score=detail.score,
                margin=detail.margin,
                via=detail.via,
                fallback=detail.fallback,
            )

        concept = self._seed(z, step)
        return RoutingResult(
            prediction="unknown",
            concept_id=concept.concept_id,
            tier=3,
            score=float("nan"),
            margin=float("nan"),
            via=None,
            fallback=False,
        )

    def _tier1_stm(self, concept: Concept) -> bool:
        """May this STM candidate compete in tier 1 (FR-3.3 + cohesion gate)?

        FR-3.3 maturity (`match_count >= n_mature`) AND cohesion >=
        `min_cohesion` — the same FR-7 criterion-2 statistic and the same §8
        key promotion uses, so a candidate too incoherent to ever be promoted
        is also too incoherent to answer for a class at tier 1.

        The cohesion conjunct is the 2026-07-11 owner ruling on the golden
        gate's clause-5 red (docs/CHANGES.md). Without it, a candidate seeded
        from one known class's tau-tail reject goes on to absorb the tau-tail
        rejects of MANY classes; FR-5.1 then calibrates tau at the 95th
        percentile of that heterogeneous ref_set's own LOO scores, handing it a
        tau several times looser than any well-calibrated concept's. Its
        (tau - s)/|tau| margin therefore outbids the correct LTM concept and it
        starts winning tier-1 traffic — while being unpromotable (cohesion),
        unfoldable (FR-8.2 needs some LTM to accept its mixed centroid; none
        does) and beyond the reach of FR-6 residual clustering (mature
        candidates have left the pool). It was measured doing exactly this on
        the golden stream: cohesion 0.21, tau 3.4x the mean LTM tau, 47
        known-class arrivals stolen in the final 84 steps.

        Note this gates ROUTING only. Such a candidate keeps its identity, its
        bookkeeping and its tier-2 traffic, and stays LRU-evictable — it is
        demoted from answering, not deleted.
        """
        return (
            concept.match_count >= self._config.n_mature
            and concept.cohesion >= self._config.min_cohesion
        )

    def _assign(self, concept: Concept, z: np.ndarray, step: int) -> None:
        """Absorb an assigned embedding, then the FR-5.1 lazy threshold check.

        T5 decision 6: the check runs after every assignment's
        add_observation, on the matched concept only, tiers 1 and 2 alike —
        ref_set mutations happen nowhere else at T5, so the ratio trigger
        needs no other call site.
        """
        concept.add_observation(z, step)
        self._maybe_recompute(concept, self._config, self._prior)

    def _seed(self, z: np.ndarray, step: int) -> Concept:
        """FR-3.2: seed and register a new STM singleton for a novel embedding.

        Both thresholds bootstrap from the frozen global prior pair (T5
        decision 7); the reservoir Generator is the per-concept named
        substream (approved T3 decision 3). As the only STM growth site,
        this is also where FR-3.1 capacity is enforced (T7): the LRU victim
        is evicted before the new candidate lands.
        """
        self._evict_for_capacity(step)
        concept_id = self.new_concept_id("stm")
        concept = Concept.seed(
            z,
            step,
            self._prior.tau,
            self._prior.tau_vmf,
            concept_id=concept_id,
            rng=make_rng(self._config.seed, f"reservoir/{concept_id}"),
            window_W=self._config.window_W,
            k_max=self._config.K_max_refset,
            alpha_ema=self._config.alpha_stm_ema,
        )
        self.register(concept)
        return concept

    # ------------------------------------------------------- eviction (T7)

    def _evict_for_capacity(self, step: int) -> None:
        """FR-3.1: drain STM below capacity Δ so the next seed fits.

        Victim order: least-recently-matched first (LRU on last_matched_at),
        ties to older created_at, then to the smallest concept_id (owner
        decision 11). Normally exactly one eviction per seed; a store built
        above capacity converges back under Δ. Only the live STM view is
        eligible — LTM is exempt regardless of staleness, mature-but-
        unpromoted candidates are not.
        """
        stm = self.stm
        while stm and len(stm) >= self._config.stm_capacity:
            victim = min(stm, key=lambda c: (c.last_matched_at, c.created_at, c.concept_id))
            self._evict(victim, step)
            stm.remove(victim)

    def _evict(self, concept: Concept, step: int) -> None:
        """Remove a concept from the live registry and log the FR-3.1 record.

        The id stays burned in ``_ids_ever`` (invariant 4: never reused).
        """
        del self._registry[concept.concept_id]
        self.eviction_log.append(
            EvictionRecord(
                concept_id=concept.concept_id,
                size=concept.match_count,
                age=int(step) - concept.created_at,
                step=int(step),
                created_at=concept.created_at,
                last_matched_at=concept.last_matched_at,
                ref_count_seen=concept.ref_count_seen,
            )
        )

    # ------------------------------------------------------------ selection

    def _select(self, z: np.ndarray, concepts: list[Concept]) -> tuple[Concept, "ScoreDetail"] | None:
        """FR-4.3 best-margin assignment within one tier, or None."""
        if not concepts:
            return None
        if not self._vectorized:
            selection = self._scorer.select(z, concepts)
            return None if selection is None else (selection.concept, selection.detail)
        return self._select_batch(z, concepts)

    def _select_batch(self, z: np.ndarray, concepts: list[Concept]) -> tuple[Concept, "ScoreDetail"] | None:
        """Batch scoring across concepts, bitwise-faithful to Scorer.select.

        T5 decision 9: per-concept scores use the exact frozen-scorer
        expressions (``1.0 - ref_set @ z`` per concept — a stacked GEMV is
        NOT bitwise-equal to it); the composition (FR-4.2 fallback, FR-4.3
        OR-accept / best sub-margin) and the selection (max margin, exact
        ties to the lexicographically smallest concept_id) run over arrays.
        Guarded by test_vectorized_matches_loop against the loop path.
        """
        from fpcmc.scorers import ScoreDetail, _margin, log_C_D  # deferred; scorers imports us

        config = self._config
        kind = config.scorer
        n = len(concepts)

        # knn_ref side — mirrors KnnRefScorer.score_detail per concept.
        s_knn = np.empty(n)
        m_knn = np.empty(n)
        acc_knn = np.zeros(n, dtype=bool)
        for i, c in enumerate(concepts):
            dists = 1.0 - c.ref_set @ z
            k = min(config.k_ref, dists.shape[0])
            s = float(np.mean(np.partition(dists, k - 1)[:k]))
            s_knn[i] = s
            acc_knn[i] = s <= c.tau
            m_knn[i] = _margin(c.tau, s)

        if kind == "knn_ref":
            score, accepted, margin = s_knn, acc_knn, m_knn
            via = ["knn_ref"] * n
            fallback = np.zeros(n, dtype=bool)
        else:
            # vmf side — mirrors VmfScorer.score_detail per concept: native
            # above n_vmf_min, wholesale knn_ref delegation below (FR-4.2).
            s_vmf = np.empty(n)
            m_vmf = np.empty(n)
            acc_vmf = np.zeros(n, dtype=bool)
            fallback = np.zeros(n, dtype=bool)
            for i, c in enumerate(concepts):
                if c.ref_set.shape[0] < config.n_vmf_min:
                    fallback[i] = True
                    s_vmf[i], acc_vmf[i], m_vmf[i] = s_knn[i], acc_knn[i], m_knn[i]
                    continue
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
                s_vmf[i] = s
                acc_vmf[i] = s <= c.tau_vmf
                m_vmf[i] = _margin(c.tau_vmf, s)

            if kind == "vmf":
                score, accepted, margin = s_vmf, acc_vmf, m_vmf
                via = ["knn_ref" if fb else "vmf" for fb in fallback]
            else:  # knn_vmf — mirrors KnnVmfScorer.score_detail composition.
                accepted = acc_knn | acc_vmf
                vmf_wins = m_vmf > m_knn  # exact ties prefer the knn detail
                margin = np.where(vmf_wins, m_vmf, m_knn)
                score = s_knn  # composed scalar = knn_ref sub-score (T2 decision)
                via = ["vmf" if w else "knn_ref" for w in vmf_wins]

        candidates = np.flatnonzero(accepted)
        if candidates.size == 0:
            return None
        best_margin = margin[candidates].max()
        tied = candidates[margin[candidates] == best_margin]
        w = int(min(tied, key=lambda i: concepts[i].concept_id))
        detail = ScoreDetail(
            score=float(score[w]),
            accepted=True,
            margin=float(margin[w]),
            scorer=self._scorer.name,
            via=via[w],
            fallback=bool(fallback[w]),
        )
        return concepts[w], detail
