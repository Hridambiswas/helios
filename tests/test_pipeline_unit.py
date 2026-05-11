"""Unit tests for pipeline routing logic — no agents instantiated."""
import pytest
from pipeline.run import route_after_planner, route_after_retriever


class TestRouteAfterPlanner:
    def test_error_state_routes_to_synthesizer(self):
        state = {"error": "something went wrong", "plan": {}}
        assert route_after_planner(state) == "synthesizer"

    def test_requires_retrieval_true_routes_to_retriever(self):
        state = {"plan": {"requires_retrieval": True}}
        assert route_after_planner(state) == "retriever"

    def test_requires_retrieval_false_code_true_routes_to_executor(self):
        state = {"plan": {"requires_retrieval": False, "requires_code": True}}
        assert route_after_planner(state) == "executor"

    def test_no_retrieval_no_code_routes_to_synthesizer(self):
        state = {"plan": {"requires_retrieval": False, "requires_code": False}}
        assert route_after_planner(state) == "synthesizer"

    def test_empty_plan_defaults_to_retriever(self):
        # requires_retrieval defaults to True
        state = {"plan": {}}
        assert route_after_planner(state) == "retriever"

    def test_missing_plan_defaults_to_retriever(self):
        state = {}
        assert route_after_planner(state) == "retriever"


class TestRouteAfterRetriever:
    def test_requires_code_routes_to_executor(self):
        state = {"plan": {"requires_code": True}}
        assert route_after_retriever(state) == "executor"

    def test_no_code_routes_to_synthesizer(self):
        state = {"plan": {"requires_code": False}}
        assert route_after_retriever(state) == "synthesizer"

    def test_missing_plan_routes_to_synthesizer(self):
        state = {}
        assert route_after_retriever(state) == "synthesizer"
