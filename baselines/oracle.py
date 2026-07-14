"""B3 baseline: ground-truth-labeled routing ceiling (PRD §7.4; TASKS T14).

Owner-approved design (2026-07-13): a PURE ground-truth router, embedding-free
— the source project's ``--oracle-mode`` mechanism (per-paradigm
``step_oracle``, still available verbatim in the vendored ``baselines/v1/``
for v1-side oracle runs) adapted to F-PCMC's evaluation surface by emitting a
T13 schema-v2 JSONL that ``eval.harness.evaluate_run`` scores directly.

Routing law (the ceiling):

  - T0 (known) class arrival  -> assign tier 1 to that class's own concept
    (``oracle_ltm_<class>``); novelty 0.0.
  - Novel-class first arrival -> seed ``oracle_stm_<class>`` (one concept per
    class, forever — fragmentation 1.0 and purity 1.0 by construction);
    novelty 1.0.
  - Further pre-promotion arrivals -> assign tier 2, prediction "unknown"
    (FR-9.1), novelty 1.0. Post-seed matches count exactly as F-PCMC's do
    (the seed itself is not a match — T3 semantics), and the class promotes
    atomically once ``match_count >= config.theta_promote`` (FR-7 size
    criterion; the oracle needs no cohesion/separation/recurrence — ground
    truth makes them vacuous). The promote record is emitted AFTER that
    step's arrival record: an arrival at the promotion step is pre-promotion
    (FR-9 route-before-hook order, mirrored by the §7.2 "unknown" law).
  - Post-promotion arrivals -> assign tier 1 to the class concept; novelty
    0.0 (the novelty statistic is the promotion-aware perfect detector:
    1.0 while the class is unknown, 0.0 once known).

One-off outlier "classes" (golden distractors, the burst class) never reach
theta_promote and stay unpromoted candidates — the ceiling forgets outliers
for free, like FR-3.1 intends. Deterministic: a pure function of the ground
truth; no RNG anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import yaml

from eval.gt import StreamGroundTruth
from fpcmc.config import FPCMCConfig


def _record(f, record: dict) -> None:
    f.write(json.dumps(record) + "\n")


def run_oracle(
    gt: StreamGroundTruth,
    config: FPCMCConfig,
    log_path: str | Path,
    *,
    checkpoint_steps: Sequence[int] = (),
) -> Path:
    """Route the stream by ground truth; write a schema-v2 JSONL event log."""
    log_path = Path(log_path)
    theta = int(config.theta_promote)
    checkpoints = sorted({int(s) for s in checkpoint_steps})
    checkpoint_set = set(checkpoints)

    matches: dict[str, int] = {}  # novel class -> post-seed match count
    promoted: set[str] = set()
    n_ltm = len(gt.known_classes)
    n_stm = 0
    n_promotions = 0

    with log_path.open("w", encoding="utf-8", newline="\n") as f:
        _record(f, {
            "type": "config_header",
            "schema": 2,
            "config": {
                "baseline": "oracle",
                **yaml.safe_load(config.to_yaml()),
            },
            "n_steps": len(gt),
            "checkpoint_steps": checkpoints,
        })
        for step in range(len(gt)):
            cls = str(gt.true_class[step])
            if cls in gt.known_classes:
                _record(f, _assign(step, f"oracle_ltm_{cls}", tier=1, novelty=0.0))
            elif cls in promoted:
                _record(f, _assign(step, f"oracle_stm_{cls}", tier=1, novelty=0.0))
            elif cls not in matches:
                matches[cls] = 0
                n_stm += 1
                _record(f, {
                    "type": "seed",
                    "step": step,
                    "concept_id": f"oracle_stm_{cls}",
                    "novelty": 1.0,
                })
            else:
                _record(f, _assign(step, f"oracle_stm_{cls}", tier=2, novelty=1.0,
                                   prediction="unknown"))
                matches[cls] += 1
                if matches[cls] >= theta:
                    promoted.add(cls)
                    n_stm -= 1
                    n_ltm += 1
                    n_promotions += 1
                    _record(f, {
                        "type": "promote",
                        "step": step,
                        "concept_id": f"oracle_stm_{cls}",
                        "size": matches[cls],
                        "cohesion": None,
                        "separation_margin": None,
                        "window_count": None,
                        "gt_majority_label": cls,  # eval-side field; the
                        "purity": 1.0,             # oracle IS ground truth
                    })
            if step in checkpoint_set:
                _record(f, {
                    "type": "checkpoint",
                    "step": step,
                    "n_ltm": n_ltm,
                    "n_stm": n_stm,
                    "n_concepts": n_ltm + n_stm,
                    "n_evictions": 0,
                    "n_promotions": n_promotions,
                    "n_merges": 0,
                    "residual_pool_size": 0,
                    "taus": {},  # the oracle has no thresholds
                })
    return log_path


def _assign(
    step: int, concept_id: str, *, tier: int, novelty: float,
    prediction: str | None = None,
) -> dict:
    return {
        "type": "assign",
        "step": step,
        "concept_id": concept_id,
        "prediction": concept_id if prediction is None else prediction,
        "tier": tier,
        "score": 0.0,
        "margin": None,
        "via": "oracle",
        "fallback": False,
        "novelty": novelty,
    }


__all__ = ["run_oracle"]
