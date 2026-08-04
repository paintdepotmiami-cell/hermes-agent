"""Frozen host-owned governance violation reasons."""

from __future__ import annotations

import ast
from pathlib import Path


def test_frozen_reasons_cover_every_literal_violation_origin():
    from tools.governance_violation_reasons import GOVERNANCE_VIOLATION_REASONS

    dispatch_path = Path(__file__).parents[2] / "tools" / "governed_mcp_dispatch.py"
    tree = ast.parse(dispatch_path.read_text(encoding="utf-8"))
    emitted = set()

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_fail"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            emitted.add(node.args[0].value)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "_violation"
                    and isinstance(node.value.value, str)
                ):
                    emitted.add(node.value.value)

    assert emitted == GOVERNANCE_VIOLATION_REASONS
