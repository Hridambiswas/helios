# eval/harness.py — Helios 30-question evaluation harness runner
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.questions import EVAL_QUESTIONS, EvalQuestion
from eval.scorers import score_answer, batch_score, DimensionScore
from graph.pipeline import run_pipeline

logger = logging.getLogger("helios.eval.harness")

REPORTS_DIR = Path(__file__).parent / "reports"


def run_single(question: EvalQuestion) -> dict[str, Any]:
    """Run pipeline on one question and compute scores."""
    t0 = time.perf_counter()
    state = run_pipeline(question.query)
    latency_ms = (time.perf_counter() - t0) * 1000

    answer = state.get("answer") or ""
    critic_scores = state.get("critic_scores")

    score: DimensionScore = score_answer(
        answer=answer,
        critic_scores=critic_scores,
        expected_keywords=question.expected_keywords,
        forbidden_keywords=question.expected_not_keywords,
        min_groundedness=question.min_groundedness,
        min_faithfulness=question.min_faithfulness,
        min_completeness=question.min_completeness,
    )

    return {
        "question_id": question.id,
        "type": question.type,
        "query": question.query,
        "answer": answer,
        "score": score,
        "latency_ms": round(latency_ms, 1),
        "error": state.get("error"),
    }


def run_harness(
    question_ids: list[int] | None = None,
    export_json: bool = True,
    export_csv: bool = True,
) -> dict[str, Any]:
    """
    Run all (or a subset of) eval questions and produce a full report.

    Args:
        question_ids: subset to run; None = all 30
        export_json: write reports/eval_<timestamp>.json
        export_csv:  write reports/eval_<timestamp>.csv
    """
    questions = EVAL_QUESTIONS
    if question_ids:
        questions = [q for q in EVAL_QUESTIONS if q.id in question_ids]

    logger.info("Starting eval harness: %d questions", len(questions))
    results: list[dict] = []

    for i, q in enumerate(questions, 1):
        logger.info("[%d/%d] Q%d (%s): %.60s...", i, len(questions), q.id, q.type, q.query)
        try:
            result = run_single(q)
        except Exception as exc:
            logger.exception("Q%d crashed: %s", q.id, exc)
            result = {
                "question_id": q.id, "type": q.type, "query": q.query,
                "answer": "", "score": None, "latency_ms": 0, "error": str(exc),
            }
        results.append(result)

        s = result.get("score")
        if s:
            status = "PASS" if s.passed else "FAIL"
            logger.info(
                "  Q%d %s — G=%.2f F=%.2f C=%.2f kw=%.2f lat=%.0fms",
                q.id, status, s.groundedness, s.faithfulness,
                s.completeness, s.keyword_hit_rate, result["latency_ms"],
            )

    stats = batch_score([r for r in results if r.get("score")])

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_questions": len(questions),
        "summary": stats,
        "results": [
            {
                "id": r["question_id"], "type": r["type"], "query": r["query"],
                "latency_ms": r["latency_ms"],
                "score": {
                    "groundedness": r["score"].groundedness,
                    "faithfulness": r["score"].faithfulness,
                    "completeness": r["score"].completeness,
                    "overall": r["score"].overall,
                    "passed": r["score"].passed,
                    "keyword_hit_rate": r["score"].keyword_hit_rate,
                    "hallucination": r["score"].hallucination_detected,
                } if r.get("score") else None,
                "answer_preview": r["answer"][:200] if r.get("answer") else "",
                "error": r.get("error"),
            }
            for r in results
        ],
    }

    if export_json or export_csv:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        if export_json:
            json_path = REPORTS_DIR / f"eval_{ts}.json"
            json_path.write_text(json.dumps(report, indent=2))
            logger.info("Report → %s", json_path)

        if export_csv:
            csv_path = REPORTS_DIR / f"eval_{ts}.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "id", "type", "query", "latency_ms",
                    "groundedness", "faithfulness", "completeness",
                    "overall", "passed", "keyword_hit_rate", "hallucination",
                ])
                writer.writeheader()
                for row in report["results"]:
                    sc = row["score"] or {}
                    writer.writerow({
                        "id": row["id"], "type": row["type"],
                        "query": row["query"][:80], "latency_ms": row["latency_ms"],
                        **{k: sc.get(k, "") for k in [
                            "groundedness", "faithfulness", "completeness",
                            "overall", "passed", "keyword_hit_rate", "hallucination",
                        ]},
                    })
            logger.info("CSV report → %s", csv_path)

    _print_summary(stats, len(questions))
    return report


def _print_summary(stats: dict, n: int) -> None:
    print("\n" + "=" * 60)
    print(f"  Helios Eval Harness — {n} questions")
    print("=" * 60)
    print(f"  Pass rate           : {stats.get('pass_rate', 0)*100:.1f}%")
    print(f"  Mean groundedness   : {stats.get('mean_groundedness', 0):.3f}")
    print(f"  Mean faithfulness   : {stats.get('mean_faithfulness', 0):.3f}")
    print(f"  Mean completeness   : {stats.get('mean_completeness', 0):.3f}")
    print(f"  Mean overall        : {stats.get('mean_overall', 0):.3f}")
    print(f"  Mean kw hit rate    : {stats.get('mean_keyword_hit_rate', 0):.3f}")
    print(f"  Hallucinations      : {stats.get('hallucination_count', 0)}")
    print(f"  Mean latency        : {stats.get('mean_latency_ms', 0):.0f}ms")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", type=int, help="Question IDs to run (default: all)")
    args = parser.parse_args()
    run_harness(question_ids=args.ids)
