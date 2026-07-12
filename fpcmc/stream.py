"""Stream runner: the FR-9 wake loop, periodic hooks, and the JSONL event
log (T11; PRD FR-9, NFR-1-3).

``StreamRunner`` wires the as-built pipeline over one embedding stream:
routing (T5 ``ConceptStore.route``), per-assignment promotion checks (T8
decision 12), the FR-8 merge sweep + on-promotion check (T9), and FR-6
residual clustering (T10) — and writes a structured JSONL event log
sufficient to reconstruct the final store state (``fpcmc.replay``) and every
figure (NFR-3) without re-running.

T11 decisions (2026-07-11, agent recommendations under the owner's
proceed-on-recommendations instruction; recorded in docs/CHANGES.md T11):

  25. Surface: ``StreamRunner(config, store, prior, *, log_path,
      checkpoint_steps=())``. The frozen GlobalPrior rides alongside the
      store (the T9 plumbing note: the store cannot expose it without
      violating invariant 5's AST guard); ``promoter``/``sweeper``/
      ``residual`` are public attributes, and the runner shares ONE
      MergeSweeper between the periodic sweep and the residual clusterer so
      the run has a single merge log / lineage map.
  26. Wake-loop order per step t (FR-9 pseudocode + hook list order):
        route(z, t)
        -> drain new EvictionRecords (they happen inside ``_seed`` BEFORE
           the new candidate registers, so draining first keeps the log in
           mutation order for replay)
        -> emit assign/seed record; tier 3 additionally wires
           ``residual.note_seed`` (the seed embedding exists nowhere else)
        -> tier 1/2: ``promoter.check`` on the just-matched concept (T8
           decision 12 — the production cadence; there is no periodic
           promotion schedule key); on success emit the promote record and
           run ``sweeper.on_promotion`` (T9 decision 17)
        -> ``residual.hook(store, t)`` (self-schedules on T_cluster +
           pool >= 30; both trigger conditions live in the module)
        -> ``sweeper.sweep(store, t)`` when t % T_merge == 0 and t > 0
        -> checkpoint record when t is in checkpoint_steps.
      Merge records drain from the shared ``sweeper.merge_log`` after every
      mutating sub-step, so record order == mutation order.
  27. Log format: strict JSON, one record per line,
      ``json.dumps(sort_keys=True, separators=(",", ":"), allow_nan=False)``
      — non-finite floats are serialized as null (replay/eval map null back
      to NaN). Exactly the seven TASKS record types (config_header, assign,
      seed, evict, promote, merge, checkpoint); residual-driven merges are
      ordinary merge records with kind="residual"; residual pass statistics
      surface in checkpoint records (residual_pool_size) and in
      ``ResidualClusterer.run_log`` (not logged — reconstructable from the
      merge records plus determinism). The config_header embeds the resolved
      config (NFR-2), n_steps, checkpoint_steps, and LOG_SCHEMA_VERSION.
      Nothing wall-clock enters the log (FR-9.2 byte determinism).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Optional, TextIO

import numpy as np
import yaml

from fpcmc.concepts import ConceptStore, EvictionRecord, RoutingResult
from fpcmc.config import FPCMCConfig
from fpcmc.memory import MergeRecord, MergeSweeper, PromotionEvaluator, PromotionRecord
from fpcmc.residual import ResidualClusterer
from fpcmc.thresholds import GlobalPrior

LOG_SCHEMA_VERSION = 1


def _finite(x: float) -> Optional[float]:
    """Strict-JSON float: non-finite (NaN/inf) becomes null (decision 27)."""
    x = float(x)
    return x if math.isfinite(x) else None


def _dumps(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)


class StreamRunner:
    """The FR-9 single-pass wake loop over a precomputed embedding stream."""

    def __init__(
        self,
        config: FPCMCConfig,
        store: ConceptStore,
        prior: GlobalPrior,
        *,
        log_path: str | Path,
        checkpoint_steps: Iterable[int] = (),
    ) -> None:
        self.config = config
        self.store = store
        self.prior = prior
        self.promoter = PromotionEvaluator(config)
        self.sweeper = MergeSweeper(config, prior)
        self.residual = ResidualClusterer(config, self.sweeper)
        self._log_path = Path(log_path)
        self._checkpoint_steps = tuple(sorted({int(s) for s in checkpoint_steps}))
        self._evictions_logged = 0
        self._merges_logged = 0

    # ------------------------------------------------------------------ run

    def run(self, stream_x: np.ndarray) -> None:
        """Process the stream in exactly one pass (invariant 1), writing the
        event log to ``log_path``."""
        X = np.asarray(stream_x, dtype=np.float64)
        checkpoints = set(self._checkpoint_steps)
        with self._log_path.open("w", encoding="utf-8", newline="\n") as f:
            self._emit(f, {
                "type": "config_header",
                "schema": LOG_SCHEMA_VERSION,
                "config": yaml.safe_load(self.config.to_yaml()),
                "n_steps": int(X.shape[0]),
                "checkpoint_steps": list(self._checkpoint_steps),
            })
            for step in range(X.shape[0]):
                self._step(f, X[step], step, checkpoints)

    # ----------------------------------------------------------- wake loop

    def _step(self, f: TextIO, z: np.ndarray, step: int, checkpoints: set[int]) -> None:
        result = self.store.route(z, step)
        # Evictions occurred inside route._seed before the seed registered:
        # drain them first so the log stays in mutation order (decision 26).
        self._drain_evictions(f)

        if result.tier == 3:
            self._emit(f, {"type": "seed", "step": step, "concept_id": result.concept_id})
            self.residual.note_seed(result.concept_id, z, step)
        else:
            self._emit(f, self._assign_record(result, step))
            decision = self.promoter.check(
                self.store.get(result.concept_id), self.store, step
            )
            if decision is not None and decision.promoted:
                self._emit(f, self._promote_record(self.promoter.promotion_log[-1]))
                self.sweeper.on_promotion(self.store, step)  # FR-8 on-promotion check
                self._drain_merges(f)

        self.residual.hook(self.store, step)
        self._drain_merges(f)

        if step > 0 and step % self.config.T_merge == 0:
            self.sweeper.sweep(self.store, step)
            self._drain_merges(f)

        if step in checkpoints:
            self._emit(f, self._checkpoint_record(step))

    # ------------------------------------------------------------- records

    def _emit(self, f: TextIO, record: dict) -> None:
        f.write(_dumps(record) + "\n")

    def _assign_record(self, result: RoutingResult, step: int) -> dict:
        return {
            "type": "assign",
            "step": step,
            "concept_id": result.concept_id,
            "prediction": result.prediction,
            "tier": result.tier,
            "score": _finite(result.score),
            "margin": _finite(result.margin),
            "via": result.via,
            "fallback": bool(result.fallback),
        }

    def _promote_record(self, rec: PromotionRecord) -> dict:
        return {
            "type": "promote",
            "step": rec.step,
            "concept_id": rec.concept_id,
            "size": rec.size,
            "cohesion": _finite(rec.cohesion),
            "separation_margin": _finite(rec.separation_margin),
            "window_count": rec.window_count,
            "gt_majority_label": rec.gt_majority_label,  # None at runtime (invariant 2)
            "purity": rec.purity,                        # None; filled post hoc by T13
        }

    def _evict_record(self, rec: EvictionRecord) -> dict:
        return {
            "type": "evict",
            "step": rec.step,
            "concept_id": rec.concept_id,
            "size": rec.size,
            "age": rec.age,
            "created_at": rec.created_at,
            "last_matched_at": rec.last_matched_at,
            "ref_count_seen": rec.ref_count_seen,
        }

    def _merge_record(self, rec: MergeRecord) -> dict:
        return {
            "type": "merge",
            "step": rec.step,
            "kind": rec.kind,
            "survivor_id": rec.survivor_id,
            "absorbed_id": rec.absorbed_id,
            "centroid_sim": _finite(rec.centroid_sim),
            "cross_within_ratio": _finite(rec.cross_within_ratio),
            "survivor_match_count": rec.survivor_match_count,
            "absorbed_match_count": rec.absorbed_match_count,
        }

    def _checkpoint_record(self, step: int) -> dict:
        return {
            "type": "checkpoint",
            "step": step,
            "n_ltm": len(self.store.ltm),
            "n_stm": len(self.store.stm),
            "n_concepts": len(self.store),
            "n_evictions": len(self.store.eviction_log),
            "n_promotions": len(self.promoter.promotion_log),
            "n_merges": len(self.sweeper.merge_log),
            "residual_pool_size": len(self.residual.pool_ids),
        }

    def _drain_evictions(self, f: TextIO) -> None:
        log = self.store.eviction_log
        while self._evictions_logged < len(log):
            self._emit(f, self._evict_record(log[self._evictions_logged]))
            self._evictions_logged += 1

    def _drain_merges(self, f: TextIO) -> None:
        log = self.sweeper.merge_log
        while self._merges_logged < len(log):
            self._emit(f, self._merge_record(log[self._merges_logged]))
            self._merges_logged += 1
