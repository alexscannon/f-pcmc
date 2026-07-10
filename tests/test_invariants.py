"""Cross-cutting invariant tests (TASKS.md "Cross-Cutting Invariant Tests").

Extended as tasks land. Present from T0:
  - invariant 7: reference-code isolation (`test_no_reference_imports`, the
    vendoring guard from Task 0);
  - invariant 6: frozen encoder — no torch autograd / optimizers / model
    forward passes anywhere under fpcmc/.
"""

import ast
from pathlib import Path

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
