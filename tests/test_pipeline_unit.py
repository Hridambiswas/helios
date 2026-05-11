"""Unit tests for pipeline state routing logic."""
import pytest
from pipeline.run import route_after_planner, route_after_retriever, route_after_critic
from langgraph.graph import END


class TestRouteAfterPlanner:
    def test_routes_to_retriever_when_retrieval_needed(self):
        state = {"plan": {"requires_retrieval": True, "requires_code": False}}
        assert route_after_planner(state) == "retriever"

    def test_routes_to_executor_when_no_retrieval_but_code(self):
        state = {"plan": {"requires_retrieval": False, "requires_code": True}}
        assert route_after_planner(state) == "executor"

    def test_routes_to_synthesizer_when_direct_answer(self):
        state = {"plan": {"requires_retrieval": False, "requires_code": False}}
        assert route_after_planner(state) == "synthesizer"

    def test_routes_to_synthesizer_on_error(self):
        state = {"error": "something broke", "plan": {"requires_retrieval": True}}
        assert route_after_planner(state) == "synthesizer"

    def test_defaults_to_retriever_when_no_plan(self):
        assert route_after_planner({}) == "retriever"


class TestRouteAfterRetriever:
    def test_routes_to_executor_when_code_needed(self):
        state = {"plan": {"requires_code": True}}
        assert route_after_retriever(state) == "executor"

    def test_routes_to_synthesizer_when_no_code(self):
        state = {"plan": {"requires_code": False}}
        assert route_after_retriever(state) == "synthesizer"


class TestRouteAfterCritic:
    def test_always_routes_to_end(self):
        assert route_after_critic({}) == END
        assert route_after_critic({"critic_passed": True}) == END
        assert route_after_critic({"critic_passed": False}) == END
