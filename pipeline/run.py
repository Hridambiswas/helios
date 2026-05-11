# pipeline/run.py — Helios LangGraph agentic pipeline
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging
import threading
import time
from typing import Any, TypedDict, Optional

from langgraph.graph import StateGraph, END

from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.executor import ExecutorAgent
from agents.synthesizer import SynthesizerAgent
from agents.critic import CriticAgent
from observability.metrics import pipeline_latency_histogram, pipeline_requests_counter
from observability.tracing import span

_PIPELINE_TIMEOUT_SECONDS = 120
_PIPELINE_VERSION = "1.0.0"

logger = logging.getLogger("helios.pipeline.run")

# ── State schema ──────────────────────────────────────────────────────────────
# TypedDict satisfies LangGraph's StateGraph schema requirement while keeping
# type safety across node functions.  All fields are Optional because nodes
# only write the fields they produce; the rest carry over from prior nodes.
class HeliosState(TypedDict, total=False):
    query: str
    user_id: Optional[str]
    plan: Optional[dict]
    retrieved_docs: list
    web_sources: list            # DuckDuckGo results: [{title, url, snippet}]
    code_to_run: Optional[str]
    execution_result: Optional[dict]
    answer: Optional[str]
    cited_doc_ids: list
    follow_up_questions: list    # 2 suggested follow-up questions from synthesizer
    critic_scores: Optional[dict]
    critic_passed: Optional[bool]
    error: Optional[str]
    failed_agent: Optional[str]
    pipeline_start_ms: Optional[float]
    pipeline_version: Optional[str]


# ── Agent singletons (lazy-initialised to keep startup fast) ─────────────────
_planner: PlannerAgent | None = None
_retriever: RetrieverAgent | None = None
_executor: ExecutorAgent | None = None
_synthesizer: SynthesizerAgent | None = None
_critic: CriticAgent | None = None
_init_lock = threading.Lock()


def _ensure_agents() -> None:
    global _planner, _retriever, _executor, _synthesizer, _critic
    if _planner is not None:
        return
    with _init_lock:
        if _planner is not None:
            return
        logger.info("Initialising pipeline agents …")
        _planner = PlannerAgent()
        _retriever = RetrieverAgent()
        _executor = ExecutorAgent()
        _synthesizer = SynthesizerAgent()
        _critic = CriticAgent()
        logger.info("Pipeline agents ready")


# ── Node wrappers ─────────────────────────────────────────────────────────────

def node_planner(state: HeliosState) -> HeliosState:
    _ensure_agents()
    with span("helios.planner", {"query": state.get("query", "")[:80]}):
        return _planner.run(state)  # type: ignore


def node_retriever(state: HeliosState) -> HeliosState:
    _ensure_agents()
    with span("helios.retriever"):
        return _retriever.run(state)  # type: ignore


def node_executor(state: HeliosState) -> HeliosState:
    _ensure_agents()
    with span("helios.executor"):
        return _executor.run(state)  # type: ignore


def node_synthesizer(state: HeliosState) -> HeliosState:
    _ensure_agents()
    with span("helios.synthesizer"):
        return _synthesizer.run(state)  # type: ignore


def node_critic(state: HeliosState) -> HeliosState:
    _ensure_agents()
    with span("helios.critic"):
        return _critic.run(state)  # type: ignore


# ── Conditional routing ───────────────────────────────────────────────────────

def route_after_planner(state: HeliosState) -> str:
    """
    After planning: route to retriever if retrieval needed,
    else skip straight to executor (code-only queries).
    """
    if state.get("error"):
        return "synthesizer"  # skip retrieval/execution; let synthesizer surface the error
    plan: dict = state.get("plan") or {}
    if plan.get("requires_retrieval", True):
        return "retriever"
    if plan.get("requires_code", False):
        return "executor"
    return "synthesizer"


def route_after_retriever(state: HeliosState) -> str:
    plan: dict = state.get("plan") or {}
    if plan.get("requires_code", False):
        return "executor"
    return "synthesizer"


def route_after_critic(state: HeliosState) -> str:
    """
    Route after critic evaluation.
    Currently always ends — retry loops would require LangGraph persistence.
    Critic pass/fail is recorded in state for callers to inspect.
    """
    return END


# ── Build graph ───────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    # type: ignore comments below suppress LangGraph stub false-positives;
    # TypedDict state schemas work correctly at runtime.
    g = StateGraph(HeliosState)  # type: ignore

    g.add_node("planner",     node_planner)      # type: ignore
    g.add_node("retriever",   node_retriever)    # type: ignore
    g.add_node("executor",    node_executor)     # type: ignore
    g.add_node("synthesizer", node_synthesizer)  # type: ignore
    g.add_node("critic",      node_critic)       # type: ignore

    g.set_entry_point("planner")

    g.add_conditional_edges("planner", route_after_planner, {
        "retriever":   "retriever",
        "executor":    "executor",
        "synthesizer": "synthesizer",
    })
    g.add_conditional_edges("retriever", route_after_retriever, {
        "executor":    "executor",
        "synthesizer": "synthesizer",
    })
    g.add_edge("executor",    "synthesizer")
    g.add_edge("synthesizer", "critic")
    g.add_conditional_edges("critic", route_after_critic, {END: END})

    return g


_graph = _build_graph().compile()  # graph topology is static; only agents are lazy


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline(query: str, user_id: str | None = None, **extra) -> dict[str, Any]:
    """
    Execute the full Helios agent pipeline for a query.
    Returns the final state dict including answer and critic_scores.
    """
    initial_state: HeliosState = {  # type: ignore[typeddict-item]
        "query": query,
        "user_id": user_id,
        "plan": None,
        "retrieved_docs": [],
        "web_sources": [],
        "execution_result": None,
        "answer": None,
        "cited_doc_ids": [],
        "follow_up_questions": [],
        "critic_scores": None,
        "critic_passed": None,
        "error": None,
        "pipeline_start_ms": time.perf_counter() * 1000,
        "pipeline_version": _PIPELINE_VERSION,
        **extra,
    }

    t0 = time.perf_counter()
    with span("helios.pipeline", {"query": query[:80], "user_id": str(user_id)}):
        try:
            final_state = _graph.invoke(initial_state)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            pipeline_latency_histogram.observe(elapsed_ms)

            status = "success" if not final_state.get("error") else "failed"
            if not final_state.get("critic_passed") and status == "success":
                status = "critic_failed"
            pipeline_requests_counter.labels(status=status).inc()

            logger.info(
                "Pipeline done: status=%s elapsed=%.0fms critic_passed=%s",
                status, elapsed_ms, final_state.get("critic_passed"),
            )
            return final_state
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            pipeline_latency_histogram.observe(elapsed_ms)
            pipeline_requests_counter.labels(status="failed").inc()
            logger.exception("Pipeline crashed: %s", exc)
            return {**initial_state, "error": str(exc)}
