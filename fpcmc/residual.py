"""Residual clustering: identity-preserving consolidation (T10; PRD
FR-6.1-6.2).

Online single-linkage-to-centroid assignment under-segments slowly (PRD
section 5.6): near-OOD classes fragment into immature STM candidates whose
prior-seeded singleton thresholds never pull the class back together (the T9
finding: sweep-time STM is nearly all K=1 singletons, which the FR-8.1
two-condition rule deliberately cannot merge). This module is the assist
mechanism: a small pool of "orphaned" seed embeddings plus the immature-STM
centroids are periodically clustered with the source project's UMAP+HDBSCAN
machinery, and clusters that group >= 2 existing immature candidates drive
merges of THOSE candidates through the T9 ``MergeSweeper.merge_pair`` seam
(decision 20) - never fresh anonymous clusters, so cluster identities are
preserved and this pathway allocates no concept ids.

Owner-approved T10 decisions (Q&A 2026-07-11, pre-implementation; recorded in
docs/CHANGES.md T10 - numbering continues from the T9 notes in memory.py):

  21. Clustering core - faithful port of the source sweep (ASSETS section 2):
      optional UMAP reduction, one HDBSCAN fit per section 8
      ``min_cluster_sizes`` value, Jaccard deduplication across the sweep at
      the source literal 0.80 (``JACCARD_DEDUP_THRESHOLD`` below; not a
      section 8 key). Adapted, with citations, from the read-only
      ``lib/evaluation/continual/clustering.py`` (blob ``55a9ede4``) -
      nothing here imports lib/.
  22. Live-predicate pool (FR-6.1): what enters is the SEED embedding of a
      seeded-as-singleton candidate ("singleton" = every tier-3 seed, so a
      candidate with 1..n_mature-1 matches still enters). Entry at
      ``created_at + w_residual`` iff the candidate is alive and immature
      (match_count < n_mature); an immature candidate evicted earlier enters
      at eviction time instead (owner ruling 2026-07-11 - embedding only,
      the concept and its id stay dead); a candidate absorbed by any merge
      never enters, and leaves the pool if already in (its seed already
      lives in the survivor's ref_set union); a candidate that matures or is
      promoted after entry leaves at the next hook; an entry whose parent is
      evicted after entry stays (a dead parent can never mature).
  23. Retention: pool rows that land in any final (post-dedup) cluster are
      consumed - removed from the pool - whether or not the cluster drove a
      merge; HDBSCAN noise rows stay (FR-6.2). This keeps the pool from
      regrowing v1's monolithic terminal buffer while leaving unexplained
      mass available to later passes.
  24. Clusters drive merges only: a cluster containing >= 2 alive immature
      candidates merges them sequentially (ascending concept_id) via
      ``merge_pair(kind="residual")``; pool rows are density context only
      (they make singleton structure visible to HDBSCAN's min_cluster_size
      but never join a ref_set through this pathway); pool-only and
      single-candidate clusters are no-ops. FR-6.2: noise candidates are
      untouched and stay LRU-eligible.

Wiring contract (T11 runner):
  - ``note_seed(concept_id, z, step)`` immediately after every tier-3
    ``route`` (the seed embedding is only available there).
  - ``hook(store, step)`` once per stream step. It reconciles the pool
    lifecycle on every call (evictions are discovered through an internal
    cursor over ``store.eviction_log``; a tracked id gone from the store
    WITHOUT an eviction record was absorbed by a merge) and runs
    consolidation only when ``step % T_cluster == 0`` (step > 0) and the
    pool holds >= RESIDUAL_POOL_MIN embeddings - both FR-6.1 trigger
    conditions live here, not in the runner.

Row-layout contract (tests + ``clustering_membership``): the clustering
input matrix is the pool embeddings in entry order (entered_at, concept_id)
followed by the alive immature-STM centroids in ascending concept_id.

Determinism: UMAP receives ``random_state=config.seed`` (the source
wrappers' pattern, ASSETS section 4 - this also forces single-threaded
numba, a known cost); HDBSCAN/EOM is deterministic given its input; sweep
label iteration is sorted; Jaccard dedup keeps the highest-mean-probability
duplicate with first-wins ties; final clusters sort by size descending
(stable). Merges inherit T9's dedicated ``merge/{step}/...`` substreams, so
the survivor reservoir discipline stays pure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fpcmc.concepts import Concept, ConceptStore
from fpcmc.config import FPCMCConfig
from fpcmc.memory import MergeSweeper

#: FR-6.1 literal: consolidation runs only when the pool holds >= 30
#: embeddings. Deliberately not a PRD section 8 config key (the same pattern
#: as T9's MERGE_CROSS_WITHIN_MAX).
RESIDUAL_POOL_MIN = 30

#: Jaccard overlap at/above which two sweep clusters are duplicates - the
#: FR-6.1 "existing hyperparameters" literal, sourced from the source
#: project's resolved config (msproject_misc @ e723f028,
#: evaluation/continual/config.yaml ``jaccard_dedup_threshold: 0.80``; same
#: default in config.py; identical at HEAD and in the archived T6 config
#: snapshot). Not a section 8 key (decision 21).
JACCARD_DEDUP_THRESHOLD = 0.80

_PENDING, _POOLED, _RESOLVED = "pending", "pooled", "resolved"


@dataclass
class _TrackedSeed:
    """Lifecycle of one seeded candidate's seed embedding (decision 22)."""

    concept_id: str
    z: np.ndarray
    created_at: int
    state: str = _PENDING
    entered_at: Optional[int] = None
    evicted: bool = False


@dataclass(frozen=True)
class ResidualRunRecord:
    """One consolidation run, for diagnostics and the T11 event log."""

    step: int
    pool_size: int
    n_immature: int
    n_clusters: int
    n_merges: int
    n_pool_consumed: int


class ResidualClusterer:
    """FR-6 residual pool + periodic identity-preserving consolidation.

    Holds the T9 ``MergeSweeper`` (shared with the periodic FR-8 sweep by
    T11's runner, so the run has ONE merge log / lineage map); every merge
    this module drives goes through ``sweeper.merge_pair`` with
    ``kind="residual"`` so the event log can attribute it (A4/T13).
    """

    def __init__(self, config: FPCMCConfig, sweeper: MergeSweeper) -> None:
        self._config = config
        self.sweeper = sweeper
        self._tracked: dict[str, _TrackedSeed] = {}
        self._eviction_cursor = 0
        self.run_log: list[ResidualRunRecord] = []

    # ------------------------------------------------------------ wiring

    def note_seed(self, concept_id: str, z: np.ndarray, step: int) -> None:
        """Record a tier-3 seed's embedding (call right after the route)."""
        if concept_id in self._tracked:
            raise ValueError(f"seed for {concept_id!r} was already recorded")
        self._tracked[concept_id] = _TrackedSeed(
            concept_id=concept_id,
            z=np.array(z, dtype=np.float64),  # owned copy (FR-1.2 normalized)
            created_at=int(step),
        )

    def hook(self, store: ConceptStore, step: int) -> None:
        """Per-step hook: reconcile the pool, then consolidate on schedule.

        Both FR-6.1 trigger conditions (every T_cluster steps AND pool >=
        RESIDUAL_POOL_MIN) are enforced here.
        """
        self._reconcile(store, step)
        if (
            step > 0
            and step % self._config.T_cluster == 0
            and len(self._pool()) >= RESIDUAL_POOL_MIN
        ):
            self._consolidate(store, step)

    # -------------------------------------------------------------- views

    @property
    def pool_ids(self) -> list[str]:
        """Pooled seed ids in entry order (entered_at, concept_id)."""
        return [t.concept_id for t in self._pool()]

    def clustering_membership(self, store: ConceptStore) -> tuple[list[str], list[str]]:
        """The row-layout contract: (pool ids in entry order, alive
        immature-STM candidate ids ascending). Clustering input row i is
        pool[i] for i < len(pool), else immature[i - len(pool)]'s centroid."""
        pool = [t.concept_id for t in self._pool()]
        immature = sorted(c.concept_id for c in self._immature(store))
        return pool, immature

    def _pool(self) -> list[_TrackedSeed]:
        pooled = [t for t in self._tracked.values() if t.state == _POOLED]
        pooled.sort(key=lambda t: (t.entered_at, t.concept_id))
        return pooled

    def _immature(self, store: ConceptStore) -> list[Concept]:
        n_mature = self._config.n_mature
        return sorted(
            (c for c in store.stm if c.match_count < n_mature),
            key=lambda c: c.concept_id,
        )

    # ---------------------------------------------------------- lifecycle

    def _reconcile(self, store: ConceptStore, step: int) -> None:
        """Advance every tracked seed's lifecycle (decision 22)."""
        # 1. New evictions since the last call (owner ruling: an immature
        #    candidate evicted early contributes its seed at eviction time;
        #    EvictionRecord.size is match_count at eviction, T7 decision 10).
        log = store.eviction_log
        while self._eviction_cursor < len(log):
            record = log[self._eviction_cursor]
            self._eviction_cursor += 1
            t = self._tracked.get(record.concept_id)
            if t is None:
                continue
            t.evicted = True
            if t.state == _PENDING:
                if record.size < self._config.n_mature:
                    t.state, t.entered_at = _POOLED, int(record.step)
                else:
                    t.state = _RESOLVED  # matured singletons never enter
            # A pooled entry whose parent is evicted stays pooled.

        # 2. Parent lifecycle for everything still live.
        w_residual = self._config.w_residual
        n_mature = self._config.n_mature
        for t in self._tracked.values():
            if t.state == _RESOLVED:
                continue
            if t.concept_id not in store:
                if not t.evicted:
                    # Gone without an eviction record: absorbed by a merge -
                    # its seed lives in the survivor's union. Never enters
                    # (pending) / leaves (pooled).
                    t.state = _RESOLVED
                continue
            concept = store.get(t.concept_id)
            mature = concept.status == "LTM" or concept.match_count >= n_mature
            if mature:
                # Matured (or promoted) candidates never enter, and leave the
                # pool if already in - routing owns them now.
                t.state = _RESOLVED
            elif t.state == _PENDING and step - t.created_at >= w_residual:
                t.state, t.entered_at = _POOLED, t.created_at + w_residual

    # ------------------------------------------------------ consolidation

    def _consolidate(self, store: ConceptStore, step: int) -> None:
        """One clustering pass: cluster pool + immature centroids, merge
        grouped candidates (decision 24), consume clustered pool rows
        (decision 23)."""
        pool = self._pool()
        immature = self._immature(store)
        n_pool = len(pool)
        X = np.vstack(
            [t.z for t in pool] + [c.centroid for c in immature]
        )
        clusters = self._cluster(X)

        consumed: set[str] = set()
        n_merges = 0
        for members in clusters:
            candidate_ids = []
            for i in sorted(members):
                if i < n_pool:
                    consumed.add(pool[i].concept_id)
                else:
                    candidate_ids.append(immature[i - n_pool].concept_id)
            # A candidate absorbed by an earlier cluster this pass is gone
            # from the store; its survivor may legitimately merge again.
            alive = [store.get(cid) for cid in candidate_ids if cid in store]
            if len(alive) < 2:
                continue  # pool-only / single-candidate clusters: no-ops
            survivor = alive[0]
            for other in alive[1:]:
                survivor = self.sweeper.merge_pair(
                    store, survivor, other, step, kind="residual"
                )
                n_merges += 1

        for cid in consumed:
            self._tracked[cid].state = _RESOLVED
        self.run_log.append(
            ResidualRunRecord(
                step=int(step),
                pool_size=n_pool,
                n_immature=len(immature),
                n_clusters=len(clusters),
                n_merges=n_merges,
                n_pool_consumed=len(consumed),
            )
        )

    # ----------------------------------------------------------- clustering

    def _cluster(self, X: np.ndarray) -> list[list[int]]:
        """UMAP+HDBSCAN sweep + Jaccard dedup -> final clusters as sorted
        member-index lists, ordered by size descending (decision 21).

        Adapted with the section 8 hyperparameters from the read-only
        ``lib/evaluation/continual/clustering.py::run_sweep`` /
        ``_deduplicate_clusters`` (blob ``55a9ede4``); deviations from the
        lib code: config keys come from FPCMCConfig (umap.*, hdbscan.*, seed),
        ``cluster_selection_method`` is passed explicitly (the lib relies on
        sklearn's "eom" default - section 8 pins the same value), label
        iteration is sorted (the lib iterates a set), and no logging.
        """
        # Deferred imports: umap's numba machinery is heavy, and unit tests
        # exercise this module with _cluster mocked.
        from sklearn.cluster import HDBSCAN
        from sklearn.preprocessing import normalize

        config = self._config
        n = X.shape[0]

        # Optional UMAP preprocessing (lib clustering.py:66-76): only when
        # the input outnumbers the target dimensionality.
        if n > config.umap.dim:
            from umap import UMAP

            reducer = UMAP(
                n_components=config.umap.dim,
                n_neighbors=min(config.umap.n_neighbors, n - 1),
                min_dist=config.umap.min_dist,
                metric=config.umap.metric,
                random_state=config.seed,  # determinism (ASSETS section 4)
            )
            clusterable = normalize(reducer.fit_transform(X), norm="l2")
        else:
            clusterable = normalize(X, norm="l2")

        # Sweep (lib clustering.py:83-106): one fit per min_cluster_size.
        all_clusters: list[tuple[set[int], np.ndarray]] = []
        for mcs in config.hdbscan.min_cluster_sizes:
            if n < mcs:
                continue
            clusterer = HDBSCAN(
                min_cluster_size=int(mcs),
                metric="euclidean",
                cluster_selection_method=config.hdbscan.selection,
                copy=True,
            )
            clusterer.fit(clusterable)
            labels, probs = clusterer.labels_, clusterer.probabilities_
            for label in sorted(set(labels) - {-1}):
                mask = labels == label
                all_clusters.append((set(np.where(mask)[0]), probs[mask]))
        if not all_clusters:
            return []

        # Jaccard dedup (lib clustering.py:124-174): union-find over pairs
        # with overlap >= JACCARD_DEDUP_THRESHOLD; keep the duplicate with
        # the highest mean HDBSCAN membership probability (first wins ties).
        parent = list(range(len(all_clusters)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(len(all_clusters)):
            for j in range(i + 1, len(all_clusters)):
                set_i, set_j = all_clusters[i][0], all_clusters[j][0]
                union_size = len(set_i | set_j)
                if union_size and len(set_i & set_j) / union_size >= JACCARD_DEDUP_THRESHOLD:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj

        groups: dict[int, list[int]] = {}
        for i in range(len(all_clusters)):
            groups.setdefault(find(i), []).append(i)
        final = [
            sorted(all_clusters[max(g, key=lambda i: float(np.mean(all_clusters[i][1])))][0])
            for g in groups.values()
        ]
        final.sort(key=len, reverse=True)  # stable: ties keep discovery order
        return [[int(i) for i in members] for members in final]
