# agents/executor.py — Helios Sandboxed Python Tool-Executor Agent
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import ast
import io
import logging
import sys
import textwrap
import threading
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

from config import cfg
from agents.base import BaseAgent

logger = logging.getLogger("helios.agents.executor")

# ── Import whitelist ──────────────────────────────────────────────────────────
# Only these top-level modules may be imported inside executed code.
_ALLOWED_IMPORTS: frozenset[str] = frozenset({
    "math", "statistics", "random", "re", "string", "datetime", "json",
    "collections", "itertools", "functools", "operator", "typing",
    "numpy", "pandas", "scipy",
})

_FORBIDDEN_BUILTINS: frozenset[str] = frozenset({
    "__import__", "open", "exec", "eval", "compile", "globals", "locals",
    "vars", "delattr", "setattr", "__loader__", "__spec__",
})


class _ImportGuard(ast.NodeVisitor):
    """AST visitor that rejects imports of non-whitelisted modules."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top not in _ALLOWED_IMPORTS:
                self.violations.append(f"import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        top = (node.module or "").split(".")[0]
        if top not in _ALLOWED_IMPORTS:
            self.violations.append(f"from {node.module} import ...")
        self.generic_visit(node)


def _validate_code(code: str) -> list[str]:
    """Parse and AST-check code; return list of violation strings."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc}"]
    guard = _ImportGuard()
    guard.visit(tree)
    return guard.violations


class _TimeoutError(Exception):
    pass


def _run_with_timeout(code: str, timeout: int) -> tuple[str, str, Exception | None]:
    """Execute code in a thread with resource timeout; return (stdout, stderr, exc)."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    exc_holder: list[Exception] = []

    _safe_globals = {
        "__builtins__": {
            k: v for k, v in __builtins__.items()  # type: ignore[union-attr]
            if k not in _FORBIDDEN_BUILTINS
        },
    }

    def _target() -> None:
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, _safe_globals)  # noqa: S102
        except Exception as e:
            exc_holder.append(e)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return "", "", _TimeoutError(f"Execution timed out after {timeout}s")

    exc = exc_holder[0] if exc_holder else None
    return stdout_buf.getvalue(), stderr_buf.getvalue(), exc


class ExecutorAgent(BaseAgent):
    """
    Sandboxed Python execution agent.

    Receives code from the planner (via state["code_to_run"]) or generates
    a short computation block from the query, runs it inside a restricted
    environment, and returns stdout/stderr + any exception.
    """

    name = "executor"

    def _run(self, state: dict[str, Any]) -> dict[str, Any]:
        plan: dict = state.get("plan", {})

        if not plan.get("requires_code", False):
            self.logger.info("Code execution skipped — plan says not required")
            return {**state, "execution_result": None}

        code: str | None = state.get("code_to_run")
        if not code:
            self.logger.info("No code_to_run in state — skipping executor")
            return {**state, "execution_result": None}

        code = textwrap.dedent(code)

        # ── Safety: validate before running ──────────────────────────────
        violations = _validate_code(code)
        if violations:
            msg = "Code validation failed: " + "; ".join(violations)
            self.logger.warning(msg)
            return {
                **state,
                "execution_result": {
                    "stdout": "",
                    "stderr": msg,
                    "error": msg,
                    "success": False,
                },
            }

        self.logger.info("Executing code (timeout=%ds):\n%s", cfg.executor_timeout_seconds, code[:200])
        stdout, stderr, exc = _run_with_timeout(code, cfg.executor_timeout_seconds)

        # Truncate output to prevent oversized state payloads
        if len(stdout) > 8000:
            stdout = stdout[:8000] + "\n[output truncated]"
        if len(stderr) > 2000:
            stderr = stderr[:2000] + "\n[stderr truncated]"

        result = {
            "stdout": stdout,
            "stderr": stderr,
            "error": str(exc) if exc else None,
            "success": exc is None,
        }
        self.logger.info("Execution done — success=%s  stdout_len=%d", result["success"], len(stdout))
        return {**state, "execution_result": result}
