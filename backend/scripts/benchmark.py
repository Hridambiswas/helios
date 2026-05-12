#!/usr/bin/env python3
# scripts/benchmark.py — Latency benchmark for individual Helios agents
# Author: Hridam Biswas | Project: Helios
"""
Runs a fixed set of test queries through individual agents and reports
mean/p95/p99 latency without needing a live database or Redis.

Usage:
    python scripts/benchmark.py [--n 10] [--agent planner|retriever|executor|all]
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

TEST_QUERIES = [
    "What is CARLE and how does differential encoding work?",
    "Compute the compression ratio if K_t=100, H=900, W=1600, C=4",
    "Explain the difference between dense and sparse retrieval",
    "What datasets were used to evaluate CARLE?",
    "Write Python code to implement BM25 search",
]


def _time_fn(fn, n: int) -> list[float]:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def benchmark_executor(n: int) -> None:
    from agents.executor import ExecutorAgent
    agent = ExecutorAgent()
    code = "import math\nresult = [math.sqrt(i) for i in range(1000)]\nprint(f'Done: {len(result)}')"
    times = _time_fn(
        lambda: agent.run({"query": "bench", "plan": {"requires_code": True}, "code_to_run": code}),
        n,
    )
    _print_stats("executor (sqrt loop x1000)", times)


def _print_stats(name: str, times: list[float]) -> None:
    s = sorted(times)
    p95 = s[int(len(s) * 0.95)]
    p99 = s[int(len(s) * 0.99)]
    print(f"\n  {name}")
    print(f"    mean  : {statistics.mean(times):.1f}ms")
    print(f"    median: {statistics.median(times):.1f}ms")
    print(f"    p95   : {p95:.1f}ms")
    print(f"    p99   : {p99:.1f}ms")
    print(f"    min   : {min(times):.1f}ms  max: {max(times):.1f}ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Iterations per test")
    parser.add_argument("--agent", default="all", choices=["executor", "all"])
    args = parser.parse_args()

    print(f"Helios Agent Benchmark — {args.n} iterations per agent")
    print("=" * 55)

    if args.agent in ("executor", "all"):
        benchmark_executor(args.n)

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()
