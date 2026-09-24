# tests/test_agents_package.py — Agent package export contract
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations


def test_verifier_agent_exported():
    """VerifierAgent must be reachable via the package alias."""
    from agents import VerifierAgent
    assert VerifierAgent.__name__ == "VerifierAgent"


def test_all_six_agents_in_package_all():
    from agents import __all__
    for expected in (
        "PlannerAgent", "RetrieverAgent", "ExecutorAgent",
        "SynthesizerAgent", "CriticAgent", "VerifierAgent",
    ):
        assert expected in __all__
