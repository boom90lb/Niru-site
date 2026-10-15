"""The spectral data names the source that wrote it; a kept copy must be the same program.

The data file records the SHA-256 of the generator that wrote it. When the
generator changes without changing its computation (a comment, a format), the
exact old bytes are kept in scripts/spectral-history/ so the data still binds
to real source. This check makes that claim narrow and explicit: the kept copy
and the current generator parse to the same Python program once docstrings are
set aside (the generator never reads its own). A change to the computation
fails here and needs a new sweep.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts" / "spectral-certificate.py"
HISTORY = ROOT / "scripts" / "spectral-history"
DATA = ROOT / "src" / "data" / "spectral-certificate.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _program(path: Path) -> str:
    """The parsed program with every docstring removed."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def test_data_source_is_the_generator_or_the_same_program() -> None:
    recorded = json.loads(DATA.read_text(encoding="utf-8"))["generator"]["sha256"]
    if _sha256(GENERATOR) == recorded:
        return
    kept = [p for p in sorted(HISTORY.glob("*.py")) if _sha256(p) == recorded]
    assert kept, f"no kept source has the recorded digest {recorded}"
    assert _program(kept[0]) == _program(GENERATOR), (
        f"{kept[0].name} and the current generator are different programs; "
        "the data needs a new sweep"
    )
