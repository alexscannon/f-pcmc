"""Cross-cutting invariant tests (TASKS.md "Cross-Cutting Invariant Tests").

Extended as tasks land. Present from T0:
  - invariant 7: reference-code isolation (`test_no_reference_imports`, the
    vendoring guard from Task 0);
  - invariant 6: frozen encoder — no torch autograd / optimizers / model
    forward passes anywhere under fpcmc/.
Added at T3:
  - invariant 4 (immutability half): concept_id can never be reassigned or
    deleted. The uniqueness-across-a-run half needs a ConceptStore and lands
    with T5.
Added at T5:
  - invariant 4 (uniqueness half): concept ids are unique across an entire
    run and never reused — the store rejects duplicate registration and its
    allocator never re-issues an id.
  - invariant 5: no global threshold in the decision cascade — the routing
    path's acceptance decisions read only per-concept tau/tau_vmf; the global
    prior is reachable solely from the seeding and threshold-recompute
    (shrinkage) code paths.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Per TASKS T0 / CLAUDE.md: nothing in fpcmc/, eval/, or tests/ may import
# from reference/. (lib/ and baselines/ are vendored snapshots of the source
# project and are covered by their own provenance checks, not this scan.)
ISOLATED_DIRS = ("fpcmc", "eval", "tests")


def _py_files(*dirs: str):
    for d in dirs:
        root = REPO_ROOT / d
        if root.is_dir():
            yield from sorted(root.rglob("*.py"))


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(), filename=str(path))


def _is_reference_module(name: str | None) -> bool:
    return name is not None and (name == "reference" or name.startswith("reference."))


def test_no_reference_imports():
    violations = []
    for path in _py_files(*ISOLATED_DIRS):
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_reference_module(alias.name):
                        violations.append(f"{path}:{node.lineno} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and _is_reference_module(node.module):
                    violations.append(f"{path}:{node.lineno} imports from {node.module}")
    assert not violations, "reference/ is consultation-only; never import it:\n" + "\n".join(violations)


# Frozen-encoder invariant. `import torch` alone is allowed (T1's data.py
# must torch.load the .pt files); learning machinery is not.
_FORBIDDEN_TORCH_MODULES = ("torch.nn", "torch.optim", "torch.autograd")


def _attr_chain(node: ast.Attribute) -> str:
    parts = []
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def test_no_learning_in_fpcmc():
    violations = []
    for path in _py_files("fpcmc"):
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name == m or alias.name.startswith(m + ".") for m in _FORBIDDEN_TORCH_MODULES):
                        violations.append(f"{path}:{node.lineno} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mod = node.module
                if mod == "torch":
                    banned = {a.name for a in node.names} & {"nn", "optim", "autograd"}
                    if banned:
                        violations.append(f"{path}:{node.lineno} imports torch.{{{','.join(sorted(banned))}}}")
                elif any(mod == m or mod.startswith(m + ".") for m in _FORBIDDEN_TORCH_MODULES):
                    violations.append(f"{path}:{node.lineno} imports from {mod}")
            elif isinstance(node, ast.Attribute):
                chain = _attr_chain(node)
                if any(chain == m or chain.startswith(m + ".") for m in _FORBIDDEN_TORCH_MODULES):
                    violations.append(f"{path}:{node.lineno} uses {chain}")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "backward":
                violations.append(f"{path}:{node.lineno} calls .backward()")
    assert not violations, "no learning allowed in fpcmc/ (PRD §2.2, CLAUDE.md):\n" + "\n".join(violations)


# Invariant 4 (T3+): concept-id immutability. Run-wide uniqueness is asserted
# once ConceptStore exists (T5).


def test_concept_id_immutability_invariant():
    from fpcmc.concepts import Concept

    z = np.array([1.0, 0.0])
    concept = Concept(concept_id="ltm_000", centroid=z, ref_set=z[None], tau=0.5, kappa=float("nan"))
    with pytest.raises(AttributeError):
        concept.concept_id = "ltm_001"
    with pytest.raises(AttributeError):
        del concept.concept_id
    assert concept.concept_id == "ltm_000"


# Invariant 4 (T5+), uniqueness half: concept ids are unique across the entire
# run — the store refuses duplicate registration, and its allocator never
# re-issues an id (including ids that were registered externally).


def test_concept_id_uniqueness_invariant():
    from fpcmc.concepts import Concept, ConceptStore
    from fpcmc.config import FPCMCConfig
    from fpcmc.rng import make_rng
    from fpcmc.scorers import estimate_kappa
    from fpcmc.thresholds import compute_global_prior, recompute_thresholds
    from tests.fixtures.vmf_world import VMFWorld

    config = FPCMCConfig()
    world = VMFWorld(seed=29, k_known=2, k_novel=2)
    pool = world.t0_pool(n_per_class=20)

    def _ltm(i: int, name: str) -> Concept:
        ref = np.array(pool.x[pool.labels == name])
        centroid = ref.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        return Concept(
            concept_id=f"ltm_{i:03d}",
            centroid=centroid,
            ref_set=ref,
            tau=0.5,
            kappa=estimate_kappa(ref),
            status="LTM",
            provenance="initial",
            rng=make_rng(29, f"invariants/reservoir/ltm_{i:03d}"),
        )

    concepts = [_ltm(i, name) for i, name in enumerate(world.known_names)]
    prior = compute_global_prior(concepts, config)
    for c in concepts:
        recompute_thresholds(c, config, prior)
    store = ConceptStore(config, prior, concepts)

    # An externally registered id must never be re-issued by the allocator.
    ext = np.array([world.distractor_point(90)])
    store.register(
        Concept(
            concept_id="stm_0002",
            centroid=ext[0],
            ref_set=ext,
            tau=prior.tau,
            kappa=estimate_kappa(ext),
            status="STM",
            provenance="seeded",
        )
    )

    # Duplicate registration is rejected outright.
    with pytest.raises(ValueError, match="stm_0002"):
        store.register(
            Concept(
                concept_id="stm_0002",
                centroid=ext[0],
                ref_set=ext.copy(),
                tau=prior.tau,
                kappa=estimate_kappa(ext),
            )
        )

    # A stream with heavy novelty (seeds on every distractor and early novel
    # sample) never produces a duplicate id.
    queries = np.vstack(
        [world.sample_class(n, 25, stream="invariants/uniq") for n in world.known_names]
        + [world.sample_class(n, 20, stream="invariants/uniq") for n in world.novel_names]
        + [world.distractor_point(i)[None, :] for i in range(30)]
    )
    perm = make_rng(29, "invariants/uniq/perm").permutation(len(queries))
    for step, z in enumerate(queries[perm]):
        store.route(z, step)

    ids = [c.concept_id for c in store.concepts]
    assert len(ids) == len(set(ids)), "duplicate concept_id in a single run"
    assert len(store.stm) > 1, "the stream must actually have seeded candidates"


# Invariant 5 (T5+): no global threshold in the decision cascade. The FR-9
# acceptance decisions read only per-concept thresholds (concept.tau /
# concept.tau_vmf via the frozen scorers); the global prior may be touched
# only by seeding (FR-3.2 bootstrap) and threshold recomputation (FR-5.2
# shrinkage target), never by selection. Enforced structurally: within
# ConceptStore, only __init__ (storing it), _assign (forwarding it to
# fpcmc.thresholds.maybe_recompute) and _seed may reference the prior, and
# the scorer module must not know priors exist at all.

_PRIOR_ALLOWED_STORE_METHODS = {"__init__", "_assign", "_seed"}


def _prior_references(tree: ast.AST) -> list[ast.AST]:
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and "prior" in node.id.lower():
            hits.append(node)
        elif isinstance(node, ast.Attribute) and "prior" in node.attr.lower():
            hits.append(node)
    return hits


def test_no_global_threshold_in_decision_cascade():
    concepts_tree = _parse(REPO_ROOT / "fpcmc" / "concepts.py")

    store_cls = next(
        node
        for node in ast.walk(concepts_tree)
        if isinstance(node, ast.ClassDef) and node.name == "ConceptStore"
    )
    offenders = []
    for method in store_cls.body:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if method.name in _PRIOR_ALLOWED_STORE_METHODS:
            continue
        for hit in _prior_references(method):
            offenders.append(f"ConceptStore.{method.name}:{hit.lineno}")
    assert not offenders, (
        "the decision cascade must not reach the global prior (invariant 5); "
        "prior references allowed only in "
        f"{sorted(_PRIOR_ALLOWED_STORE_METHODS)}, found:\n" + "\n".join(offenders)
    )

    # The scorers (where accept/margin decisions actually happen) must be
    # entirely prior-free: acceptance reads concept.tau / concept.tau_vmf only.
    scorer_hits = _prior_references(_parse(REPO_ROOT / "fpcmc" / "scorers.py"))
    assert not scorer_hits, (
        "fpcmc/scorers.py must not reference any prior/global threshold, found at lines: "
        + ", ".join(str(h.lineno) for h in scorer_hits)
    )
