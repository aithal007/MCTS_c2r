"""
Apply a single-file patch to the Cargo workspace and run the existing CRust verifier.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..verifier import CRustVerifier


@dataclass
class VerifyOutcome:
    success: bool
    compile_ok: bool
    tests_ok: bool
    n_compile_errors: int
    diagnostics: List[Dict[str, Any]]
    test_output: str
    raw: Dict[str, Any]

    @property
    def compilable(self) -> bool:
        return self.compile_ok


def _count_error_level(diagnostics: List[Dict[str, Any]]) -> int:
    return sum(1 for d in diagnostics if (d or {}).get("level") == "error")


def verify_rust_file(
    verifier: CRustVerifier,
    rel_path: str,
    code: str,
    *,
    require_tests: bool = True,
) -> VerifyOutcome:
    action = {"file_path": rel_path, "code_content": code}
    raw = verifier.verify(action)
    stage = str(raw.get("stage", ""))
    diags: List[Dict[str, Any]] = list(raw.get("diagnostics") or [])
    compile_ok = stage in ("testing", "complete")
    n_err = _count_error_level(diags) if stage == "compilation" else 0
    tests_ok = bool(raw.get("success") and stage == "complete")
    if not require_tests and stage == "testing":
        success = True
    else:
        success = bool(raw.get("success"))

    return VerifyOutcome(
        success=success,
        compile_ok=compile_ok or stage == "complete",
        tests_ok=tests_ok,
        n_compile_errors=n_err if not compile_ok else 0,
        diagnostics=diags,
        test_output=str(raw.get("test_output", "") or raw.get("stderr", "")),
        raw=raw,
    )


def read_file(workspace: str, rel_path: str) -> str:
    p = os.path.normpath(os.path.join(workspace, rel_path))
    if not p.startswith(os.path.normpath(workspace)):
        raise ValueError("Invalid path")
    with open(p, encoding="utf-8") as f:
        return f.read()
