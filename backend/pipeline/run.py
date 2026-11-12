# pipeline/run.py — Helios LangGraph agentic pipeline
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict, Optional

from langgraph.graph import StateGraph, END

from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.executor import ExecutorAgent
from agents.answerer import AnswererAgent
from agents.debate_critics import AdvocateCriticAgent, SkepticCriticAgent
from agents.judge import JudgeAgent
from agents.verifier import VerifierAgent
from observability.metrics import pipeline_latency_histogram, pipeline_requests_counter
from observability.tracing import span

_PIPELINE_TIMEOUT_SECONDS = 180
_PIPELINE_VERSION = "1.3.0-debate"

logger = logging.getLogger("helios.pipeline.run")

# ── State schema ──────────────────────────────────────────────────────────────
# TypedDict satisfies LangGraph's StateGraph schema requirement while keeping
# type safety across node functions.  All fields are Optional because nodes
# only write the fields they produce; the rest carry over from prior nodes.
class HeliosState(TypedDict, total=False):
    query: str
    user_id: Optional[str]
    conversation_history: list   # [{"role": "user"|"assistant", "content": str}]
    plan: Optional[dict]
    retrieved_docs: list
    web_sources: list            # DuckDuckGo results: [{title, url, snippet}]
    code_to_run: Optional[str]
    execution_result: Optional[dict]

    # ── Debate pipeline (013) ──────────────────────────────────────────────────
    draft_answer: Optional[str]           # written by Answerer (Gemini)
    answerer_provider: Optional[str]
    answerer_model: Optional[str]
    advocate_critique: Optional[dict]     # JSON from AdvocateCriticAgent
    skeptic_critique: Optional[dict]      # JSON from SkepticCriticAgent
    answer: Optional[str]                 # final answer the user sees (Judge)
    final_scores: Optional[dict]          # groundedness/faithfulness/completeness/overall
    judge_verdict: Optional[str]          # one-line summary of what changed vs draft
    judge_sources: Optional[list]         # [{index, title, url}] actually cited
    judge_provider: Optional[str]
    judge_model: Optional[str]
    judge_fallback: Optional[bool]
    cited_doc_ids: list
    cited_web_indices: list
    follow_up_questions: list

    # Verifier stays as optional final sanity-check (Gemini).
    verifier_scores: Optional[dict]
    verifier_passed: Optional[bool]

    _token_callback: Optional[Callable[[str], None]]  # set by websocket; not persisted
    _stage_callback: Optional[Callable[[str, dict], None]]  # WS stage events
    error: Optional[str]
    failed_agent: Optional[str]
    pipeline_start_ms: Optional[float]
    pipeline_version: Optional[str]


# ── Agent singletons (lazy-initialised to keep startup fast) ─────────────────
_planner: PlannerAgent | None = None
_retriever: RetrieverAgent | None = None
_executor: ExecutorAgent | None = None
_answerer: AnswererAgent | None = None
_advocate: AdvocateCriticAgent | None = None
_skeptic: SkepticCriticAgent | None = None
_judge: JudgeAgent | None = None
_verifier: VerifierAgent | None = None
_init_lock = threading.Lock()


def _ensure_agents() -> None:
    global _planner, _retriever, _executor, _answerer
    global _advocate, _skeptic, _judge, _verifier
    if _planner is not None:
        return
    with _init_lock:
        if _planner is not None:
            return
        logger.info("Initialising debate pipeline agents …")
        _planner = PlannerAgent()
        _retriever = RetrieverAgent()
        _executor = ExecutorAgent()
        _answerer = AnswererAgent()
        _advocate = AdvocateCriticAgent()
        _skeptic = SkepticCriticAgent()
        _judge = JudgeAgent()
        from config import cfg as _cfg
        if _cfg.verifier_enabled:
            _verifier = VerifierAgent()
        logger.info(
            "Debate pipeline agents ready (verifier=%s)",
            "on" if _verifier else "off",
        )


def _emit_stage(state: HeliosState, stage: str, extra: dict | None = None) -> None:
    """Fire the WebSocket stage callback if present. Always safe to call."""
    cb = state.get("_stage_callback")
    if cb is None:
        return
    try:
        cb(stage, extra or {})
    except Exception as exc:  # pragma: no cover
        logger.warning("stage callback for %s raised %s", stage, exc)


# ── Node wrappers ─────────────────────────────────────────────────────────────

def node_planner(state: HeliosState) -> HeliosState:
    _ensure_agents()
    _emit_stage(state, "planning")
    with span("helios.planner", {"query": state.get("query", "")[:80]}):
        return _planner.run(state)  # type: ignore


def node_researcher(state: HeliosState) -> HeliosState:
    """DuckDuckGo web search + optional hybrid local retrieval."""
    _ensure_agents()
    _emit_stage(state, "researching")
    with span("helios.researcher"):
        return _retriever.run(state)  # type: ignore


def node_executor(state: HeliosState) -> HeliosState:
    _ensure_agents()
    with span("helios.executor"):
        return _executor.run(state)  # type: ignore


def node_answerer(state: HeliosState) -> HeliosState:
    _ensure_agents()
    _emit_stage(state, "drafting")
    with span("helios.answerer"):
        return _answerer.run(state)  # type: ignore


def node_debate(state: HeliosState) -> HeliosState:
    """Run advocate + skeptic in parallel threads and merge their outputs."""
    _ensure_agents()
    _emit_stage(state, "advocating")
    with span("helios.debate"):
        # Each critic is isolated by a bulkhead inside BaseAgent.run(), so
        # running them in a 2-thread pool is safe.
        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(_advocate.run, state)   # type: ignore
            fs = pool.submit(_skeptic.run, state)    # type: ignore
            _emit_stage(state, "skeptic")
            adv_state = fa.result()
            skp_state = fs.result()
        merged: dict[str, Any] = {**state}
        merged["advocate_critique"] = adv_state.get("advocate_critique")
        merged["skeptic_critique"] = skp_state.get("skeptic_critique")
        # Surface either agent's error without blocking the pipeline.
        for src in (adv_state, skp_state):
            if src.get("error") and not merged.get("error"):
                merged["error"] = src["error"]
                merged["failed_agent"] = src.get("failed_agent")
        return merged  # type: ignore


def node_judge(state: HeliosState) -> HeliosState:
    _ensure_agents()
    _emit_stage(state, "judging")
    with span("helios.judge"):
        return _judge.run(state)  # type: ignore


def node_verifier(state: HeliosState) -> HeliosState:
    _ensure_agents()
    if _verifier is None:  # verifier disabled via config
        _emit_stage(state, "done")
        return {**state, "verifier_scores": None, "verifier_passed": None}  # type: ignore[typeddict-item]
    _emit_stage(state, "verifying")
    with span("helios.verifier"):
        result = _verifier.run(state)  # type: ignore
    _emit_stage(state, "done")
    return result


# ── Conditional routing ───────────────────────────────────────────────────────

def route_after_planner(state: HeliosState) -> str:
    """
    After planning: always go through the researcher (which now also does web
    search). The researcher handles the no-local-docs case internally.
    """
    if state.get("error"):
        return "judge"  # surface the error via the fallback payload
    return "researcher"


def route_after_researcher(state: HeliosState) -> str:
    plan: dict = state.get("plan") or {}
    if plan.get("requires_code", False):
        return "executor"
    return "answerer"


# ── Build graph ───────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    # type: ignore comments suppress LangGraph stub false-positives;
    # TypedDict state schemas work correctly at runtime.
    g = StateGraph(HeliosState)  # type: ignore

    g.add_node("planner",    node_planner)     # type: ignore
    g.add_node("researcher", node_researcher)  # type: ignore
    g.add_node("executor",   node_executor)    # type: ignore
    g.add_node("answerer",   node_answerer)    # type: ignore
    g.add_node("debate",     node_debate)      # type: ignore (advocate ‖ skeptic)
    g.add_node("judge",      node_judge)       # type: ignore
    g.add_node("verifier",   node_verifier)    # type: ignore

    g.set_entry_point("planner")

    g.add_conditional_edges("planner", route_after_planner, {
        "researcher": "researcher",
        "judge":      "judge",
    })
    g.add_conditional_edges("researcher", route_after_researcher, {
        "executor": "executor",
        "answerer": "answerer",
    })
    g.add_edge("executor", "answerer")
    g.add_edge("answerer", "debate")
    g.add_edge("debate",   "judge")
    g.add_edge("judge",    "verifier")
    g.add_edge("verifier", END)

    return g


_graph = _build_graph().compile()  # graph topology is static; only agents are lazy


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline(
    query: str,
    user_id: str | None = None,
    conversation_history: list[dict] | None = None,
    token_callback: Callable[[str], None] | None = None,
    stage_callback: Callable[[str, dict], None] | None = None,
    **extra,
) -> dict[str, Any]:
    """
    Execute the Helios debate pipeline for a query.

    Returns the final state dict with:
      - answer           : final markdown answer the user sees (judge-written)
      - final_scores     : {groundedness, faithfulness, completeness, overall}
      - judge_verdict    : one-line summary of what changed vs the draft
      - judge_sources    : [{index, title, url}] actually cited
      - draft_answer     : the answerer's draft, for the UI's Debate expander
      - advocate_critique / skeptic_critique : JSON payloads
      - verifier_scores  : optional Gemini sanity check (None if disabled)
    """
    initial_state: HeliosState = {  # type: ignore[typeddict-item]
        "query": query,
        "user_id": user_id,
        "conversation_history": conversation_history or [],
        "_token_callback": token_callback,
        "_stage_callback": stage_callback,
        "plan": None,
        "retrieved_docs": [],
        "web_sources": [],
        "execution_result": None,
        "draft_answer": None,
        "advocate_critique": None,
        "skeptic_critique": None,
        "answer": None,
        "final_scores": None,
        "judge_verdict": None,
        "judge_sources": [],
        "cited_doc_ids": [],
        "cited_web_indices": [],
        "follow_up_questions": [],
        "verifier_scores": None,
        "verifier_passed": None,
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
            if final_state.get("verifier_passed") is False and status == "success":
                status = "verifier_failed"
            pipeline_requests_counter.labels(status=status).inc()

            logger.info(
                "Debate pipeline done: status=%s elapsed=%.0fms "
                "answerer=%s advocate=%s skeptic=%s judge=%s verifier_passed=%s",
                status, elapsed_ms,
                final_state.get("answerer_model"),
                (final_state.get("advocate_critique") or {}).get("model"),
                (final_state.get("skeptic_critique") or {}).get("model"),
                final_state.get("judge_model"),
                final_state.get("verifier_passed"),
            )
            return final_state
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            pipeline_latency_histogram.observe(elapsed_ms)
            pipeline_requests_counter.labels(status="failed").inc()
            logger.exception("Pipeline crashed: %s", exc)
            return {**initial_state, "error": str(exc)}
