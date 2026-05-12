# eval/scorers.py — Helios eval harness scoring functions
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class DimensionScore:
    groundedness: float
    faithfulness: float
    completeness: float
    overall: float
    passed: bool
    keyword_hit_rate: float
    hallucination_detected: bool


def _contains(text: str, keywords: list[str]) -> list[bool]:
    text_lower = text.lower()
    return [kw.lower() in text_lower for kw in keywords]


def score_keyword_coverage(answer: str, expected: list[str], forbidden: list[str]) -> tuple[float, bool]:
    """
    Keyword-based proxy for completeness and hallucination.
    Returns (hit_rate [0,1], hallucination_detected).
    """
    if not expected:
        hit_rate = 1.0
    else:
        hits = _contains(answer, expected)
        hit_rate = sum(hits) / len(hits)

    hallucination = any(_contains(answer, forbidden)) if forbidden else False
    return hit_rate, hallucination


def score_critic_output(
    critic_scores: dict[str, Any] | None,
    min_groundedness: float,
    min_faithfulness: float,
    min_completeness: float,
) -> tuple[float, float, float, float]:
    """Extract numeric scores from critic agent output, with floor defaults."""
    if not critic_scores:
        return 0.0, 0.0, 0.0, 0.0
    g = float(critic_scores.get("groundedness", 0.0))
    f = float(critic_scores.get("faithfulness", 0.0))
    c = float(critic_scores.get("completeness", 0.0))
    overall = (g + f + c) / 3
    return g, f, c, overall


def score_answer(
    answer: str,
    critic_scores: dict | None,
    expected_keywords: list[str],
    forbidden_keywords: list[str],
    min_groundedness: float = 0.6,
    min_faithfulness: float = 0.6,
    min_completeness: float = 0.5,
) -> DimensionScore:
    """
    Aggregate scoring combining LLM critic scores with keyword-based checks.
    """
    keyword_hit_rate, hallucination = score_keyword_coverage(
        answer, expected_keywords, forbidden_keywords
    )

    g, f, c, overall = score_critic_output(
        critic_scores, min_groundedness, min_faithfulness, min_completeness
    )

    # Penalise on hallucination detection
    if hallucination:
        g *= 0.5
        f *= 0.5
        overall = (g + f + c) / 3

    passed = (
        g >= min_groundedness
        and f >= min_faithfulness
        and c >= min_completeness
        and keyword_hit_rate >= 0.5
        and not hallucination
    )

    return DimensionScore(
        groundedness=round(g, 3),
        faithfulness=round(f, 3),
        completeness=round(c, 3),
        overall=round(overall, 3),
        passed=passed,
        keyword_hit_rate=round(keyword_hit_rate, 3),
        hallucination_detected=hallucination,
    )


def batch_score(results: list[dict]) -> dict:
    """
    Aggregate statistics across all evaluated questions.
    results: list of {score: DimensionScore, question_id, type, latency_ms}
    """
    scores = [r["score"] for r in results]
    n = len(scores)
    if n == 0:
        return {}

    return {
        "n": n,
        "pass_rate": sum(s.passed for s in scores) / n,
        "mean_groundedness": sum(s.groundedness for s in scores) / n,
        "mean_faithfulness": sum(s.faithfulness for s in scores) / n,
        "mean_completeness": sum(s.completeness for s in scores) / n,
        "mean_overall": sum(s.overall for s in scores) / n,
        "mean_keyword_hit_rate": sum(s.keyword_hit_rate for s in scores) / n,
        "hallucination_count": sum(s.hallucination_detected for s in scores),
        "mean_latency_ms": sum(r.get("latency_ms", 0) for r in results) / n,
    }
