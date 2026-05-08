# resilience/saga.py — Helios Saga Orchestrator
# Author: Hridam Biswas | Project: Helios
#
# Implements the Saga pattern for distributed multi-step operations.
# Each step has a forward action and a compensating action.
# On failure the completed steps are compensated in reverse order,
# preventing partial-write orphans (e.g. a file in MinIO with no DB record).

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable


logger = logging.getLogger("helios.resilience.saga")


@dataclass
class _SagaStep:
    name: str
    action: Callable[..., Any]
    compensate: Callable[..., Any]
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    comp_args: tuple[Any, ...] = field(default_factory=tuple)
    comp_kwargs: dict[str, Any] = field(default_factory=dict)


class SagaExecutionError(Exception):
    def __init__(self, step: str, cause: Exception, compensated: bool) -> None:
        self.step = step
        self.cause = cause
        self.compensated = compensated
        super().__init__(
            f"Saga failed at step '{step}': {cause}"
            + (" (compensated)" if compensated else " (compensation also failed)")
        )


class Saga:
    """
    Sequential async saga with automatic compensation on failure.

    Forward actions and compensators may be plain callables or coroutines.

    Usage::

        results = await (
            Saga("document-ingest")
            .step("upload",   upload_fn,   delete_fn,   args=(key, data))
            .step("index",    index_fn,    deindex_fn,  args=(ids, vecs))
            .step("persist",  db_insert,   db_delete)
            .execute()
        )
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._steps: list[_SagaStep] = []

    def step(
        self,
        name: str,
        action: Callable[..., Any],
        compensate: Callable[..., Any],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        comp_args: tuple[Any, ...] = (),
        comp_kwargs: dict[str, Any] | None = None,
    ) -> "Saga":
        self._steps.append(
            _SagaStep(
                name=name,
                action=action,
                compensate=compensate,
                args=args,
                kwargs=kwargs or {},
                comp_args=comp_args,
                comp_kwargs=comp_kwargs or {},
            )
        )
        return self

    async def execute(self) -> list[Any]:
        """Run all steps; compensate in reverse on the first failure."""
        results: list[Any] = []
        completed: list[_SagaStep] = []

        for step in self._steps:
            try:
                if asyncio.iscoroutinefunction(step.action):
                    result = await step.action(*step.args, **step.kwargs)
                else:
                    result = step.action(*step.args, **step.kwargs)
                results.append(result)
                completed.append(step)
                logger.debug("Saga '%s' step '%s' OK", self.name, step.name)
            except Exception as exc:
                logger.error(
                    "Saga '%s' step '%s' failed: %s", self.name, step.name, exc
                )
                compensated = await self._compensate(completed)
                raise SagaExecutionError(step.name, exc, compensated)

        logger.info("Saga '%s' completed (%d steps)", self.name, len(results))
        return results

    async def _compensate(self, completed: list[_SagaStep]) -> bool:
        all_ok = True
        for step in reversed(completed):
            try:
                if asyncio.iscoroutinefunction(step.compensate):
                    await step.compensate(*step.comp_args, **step.comp_kwargs)
                else:
                    step.compensate(*step.comp_args, **step.comp_kwargs)
                logger.info(
                    "Saga '%s' compensated step '%s'", self.name, step.name
                )
            except Exception as exc:
                logger.error(
                    "Saga '%s' compensation for '%s' failed: %s",
                    self.name, step.name, exc,
                )
                all_ok = False
        return all_ok
