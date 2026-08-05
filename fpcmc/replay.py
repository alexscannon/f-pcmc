"""Replay: reconstruct the final ConceptStore state from the JSONL event log
(T11; NFR-3).

Replay is EVENT APPLICATION, never re-scoring (decision 28, docs/CHANGES.md
T11): the log is authoritative for every routing/promotion/merge/eviction
decision, and this module re-applies the corresponding mutations to a fresh
initial store (built identically to the live run's — e.g. via
``fpcmc.init.initialize_ltm`` on the same T0 pool) using the same stream
embeddings. Because every random draw in the pipeline flows through named
substreams (per-concept reservoirs ``reservoir/{concept_id}``, per-merge
bounded unions ``merge/{step}/{survivor}<-{absorbed}``), the replayed
mutations are bit-reproducible: the T11 reconstruction test asserts exact
equality, strictly stronger than the TASKS atol 1e-9 tolerance.

Replay validates as it goes: an FR-8.1-style merge record whose recomputed
survivor (T9's larger-match_count rule over the replayed state) disagrees
with the logged survivor raises ``ReplayError`` — divergence means the log
and the replayed state have already drifted apart. Folds (kind="stm_ltm")
re-apply through the T9 ``MergeSweeper.fold_pair`` seam; UMAP/HDBSCAN never
runs during replay (residual-driven merges are ordinary logged merge
records with kind="residual").
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fpcmc.concepts import Concept, ConceptStore
from fpcmc.config import FPCMCConfig
from fpcmc.memory import MergeSweeper
from fpcmc.rng import make_rng
from fpcmc.thresholds import GlobalPrior, maybe_recompute, recompute_on_promotion


class ReplayError(RuntimeError):
    """The log and the replayed state disagree (or the log is malformed)."""


def read_log(path: str | Path) -> list[dict]:
    """Parse a JSONL event log into records (strict JSON per line)."""
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def replay(
    log_path: str | Path,
    stream_x: np.ndarray,
    store: ConceptStore,
    prior: GlobalPrior,
) -> ConceptStore:
    """Re-apply the event log to a fresh initial ``store`` and return it.

    ``stream_x`` must be the same embedding stream the live run consumed
    (assign/seed records reference embeddings by step index only — the log
    itself stays label- and embedding-free). ``prior`` must be the same
    frozen FR-5.3 pair the live run used (recomputable as a pure function of
    the untouched T0 concepts).
    """
    records = read_log(log_path)
    config = _parse_header(records)
    sweeper = MergeSweeper(config, prior)
    X = np.asarray(stream_x, dtype=np.float64)

    for rec in records[1:]:
        _apply_record(rec, store, sweeper, config, prior, X)
    return store


def iter_checkpoint_states(
    log_path: str | Path,
    stream_x: np.ndarray,
    store: ConceptStore,
    prior: GlobalPrior,
):
    """Yield ``(step, store)`` at the task-0 state and after every checkpoint.

    Additive T17 seam over the same event application ``replay`` performs
    (one shared ``_apply_record``, so the semantics cannot drift): first
    yields ``(-1, store)`` — the post-init, pre-stream state — then, for each
    checkpoint record in the log, ``(step, store)`` with every record up to
    and including that step applied (mutation records precede their step's
    checkpoint record in the log — decision 26 mutation order). One pass over
    the log reconstructs all checkpoint states.

    The SAME live ``store`` object is yielded each time: consumers must read
    (score) and resume iteration without mutating it or holding references to
    its internals across yields. Exhausting the iterator leaves ``store``
    equal to ``replay``'s final state.
    """
    records = read_log(log_path)
    config = _parse_header(records)
    sweeper = MergeSweeper(config, prior)
    X = np.asarray(stream_x, dtype=np.float64)

    yield -1, store
    for rec in records[1:]:
        _apply_record(rec, store, sweeper, config, prior, X)
        if rec["type"] == "checkpoint":
            yield rec["step"], store


def _parse_header(records: list[dict]) -> FPCMCConfig:
    if not records or records[0].get("type") != "config_header":
        raise ReplayError("log must start with a config_header record")
    return FPCMCConfig.from_yaml_text(yaml.safe_dump(records[0]["config"]))


def _apply_record(
    rec: dict,
    store: ConceptStore,
    sweeper: MergeSweeper,
    config: FPCMCConfig,
    prior: GlobalPrior,
    X: np.ndarray,
) -> None:
    """Apply one event record to ``store`` (checkpoint records are no-ops)."""
    rtype = rec["type"]
    if rtype == "assign":
        concept = store.get(rec["concept_id"])
        concept.add_observation(X[rec["step"]], rec["step"])
        maybe_recompute(concept, config, prior)  # mirrors ConceptStore._assign
    elif rtype == "seed":
        cid = rec["concept_id"]
        store.register(Concept.seed(  # mirrors ConceptStore._seed
            X[rec["step"]],
            rec["step"],
            prior.tau,
            prior.tau_vmf,
            concept_id=cid,
            rng=make_rng(config.seed, f"reservoir/{cid}"),
            window_W=config.window_W,
            k_max=config.K_max_refset,
            alpha_ema=config.alpha_stm_ema,
        ))
    elif rtype == "evict":
        store.remove(rec["concept_id"])
    elif rtype == "promote":
        concept = store.get(rec["concept_id"])
        concept.status = "LTM"  # mirrors PromotionEvaluator._promote
        concept.provenance = "promoted"
        recompute_on_promotion(concept, config)
    elif rtype == "merge":
        survivor = store.get(rec["survivor_id"])
        absorbed = store.get(rec["absorbed_id"])
        if rec["kind"] == "stm_ltm":
            sweeper.fold_pair(store, survivor, absorbed, rec["step"])
        else:
            out = sweeper.merge_pair(
                store, survivor, absorbed, rec["step"], kind=rec["kind"]
            )
            if out.concept_id != rec["survivor_id"]:
                raise ReplayError(
                    f"step {rec['step']}: replayed merge chose survivor "
                    f"{out.concept_id!r} but the log records "
                    f"{rec['survivor_id']!r} — state has diverged"
                )
    elif rtype == "checkpoint":
        return
    else:
        raise ReplayError(f"unknown record type {rtype!r}")
