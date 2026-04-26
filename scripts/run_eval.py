#!/usr/bin/env python3
# scripts/run_eval.py — CLI wrapper for the Helios eval harness
# Author: Hridam Biswas | Project: Helios
"""
Usage:
    python scripts/run_eval.py                  # all 30 questions
    python scripts/run_eval.py --ids 1 5 10     # specific questions
    python scripts/run_eval.py --type factual   # filter by type
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.questions import EVAL_QUESTIONS, QuestionType
from eval.harness import run_harness


def main():
    parser = argparse.ArgumentParser(description="Run Helios eval harness")
    parser.add_argument("--ids", nargs="*", type=int, help="Question IDs (default: all)")
    parser.add_argument(
        "--type",
        choices=["factual", "analytical", "code", "multi_step", "no_context"],
        help="Filter by question type",
    )
    parser.add_argument("--no-json", action="store_true", help="Skip JSON report")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV report")
    args = parser.parse_args()

    ids = args.ids
    if args.type and not ids:
        ids = [q.id for q in EVAL_QUESTIONS if q.type == args.type]

    run_harness(
        question_ids=ids,
        export_json=not args.no_json,
        export_csv=not args.no_csv,
    )


if __name__ == "__main__":
    main()
