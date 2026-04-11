from __future__ import annotations

import ast
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def test_src_internal_imports_are_package_qualified() -> None:
    """Prevent regressions like `from data_loader import ...` inside src modules."""
    violations: list[str] = []

    for py_file in SRC_DIR.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module
                if not module:
                    continue
                if module.startswith("src"):
                    continue
                if module in {"__future__", "typing", "dataclasses", "pathlib", "json", "argparse", "warnings"}:
                    continue
                # third-party modules are fine (pandas, matplotlib, etc.)
                if module.split(".")[0] in {"pandas", "matplotlib"}:
                    continue

                # importing sibling modules without `src.` prefix causes pytest collection errors
                local_module = SRC_DIR / f"{module.split('.')[0]}.py"
                if local_module.exists():
                    violations.append(
                        f"{py_file.name}:{node.lineno} uses non-package import `from {module} import ...`"
                    )

    assert not violations, "\n".join(violations)
