# graph/pipeline.py — Helios LangGraph agentic pipeline
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging
import time
from typing import Any

from langgraph.graph import StateGraph, END

from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.executor import ExecutorAgent
from agents.synthesizer import SynthesizerAgent
from agents.critic import CriticAgent
from observability.metrics import pipeline_latency_histogram, pipeline_requests_counter
from observability.tracing import span

logger = logging.getLogger("helios.graph.pipeline")

# ── State type alias ──────────────────────────────────────────────────────────
# LangGraph passes this dict between nodes; each agent reads what it needs
# and merges its output back in.
HeliosState = dict[str, Any]


# ── Agent singletons ──────────────────────────────────────────────────────────
_planner = PlannerAgent()
_retriever = RetrieverAgent()
_executor = ExecutorAgent()
_synthesizer = SynthesizerAgent()
_critic = CriticAgent()


# ── Node wrappers ─────────────────────────────────────────────────────────────

def node_planner(state: HeliosState) -> HeliosState:
    with span("helios.planner", {"query": state.get("query", "")[:80]}):
        return _planner.run(state)


def node_retriever(state: HeliosState) -> HeliosState:
    with span("helios.retriever"):
        return _retriever.run(state)


def node_executor(state: HeliosState) -> HeliosState:
    with span("helios.executor"):
        return _executor.run(state)


def node_synthesizer(state: HeliosState) -> HeliosState:
    with span("helios.synthesizer"):
        return _synthesizer.run(state)


def node_critic(state: HeliosState) -> HeliosState:
    with span("helios.critic"):
        return _critic.run(state)


# ── Conditional routing ───────────────────────────────────────────────────────

def route_after_planner(state: HeliosState) -> str:
    """
    After planning: route to retriever if retrieval needed,
    else skip straight to executor (code-only queries).
    """
    plan = state.get("plan", {})
    if plan.get("requires_retrieval", True):
        return "retriever"
    if plan.get("requires_code", False):
        return "executor"
    return "synthesizer"


def route_after_retriever(state: HeliosState) -> str:
    plan = state.get("plan", {})
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
    g = StateGraph(HeliosState)

    g.add_node("planner",     node_planner)
    g.add_node("retriever",   node_retriever)
    g.add_node("executor",    node_executor)
    g.add_node("synthesizer", node_synthesizer)
    g.add_node("critic",      node_critic)

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


_graph = _build_graph().compile()


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline(query: str, user_id: str | None = None, **extra) -> dict[str, Any]:
    """
    Execute the full Helios agent pipeline for a query.
    Returns the final state dict including answer and critic_scores.
    """
    initial_state: HeliosState = {
        "query": query,
        "user_id": user_id,
        "plan": None,
        "retrieved_docs": [],
        "execution_result": None,
        "answer": None,
        "critic_scores": None,
        "critic_passed": None,
        "error": None,
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
