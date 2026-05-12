# eval/calibration.py — Confidence calibration check for critic scores
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import statistics
from dataclasses import dataclass
from typing import Any


@dataclass
class CalibrationReport:
    n: int
    mean_confidence: float       # critic overall score
    pass_rate: float             # fraction that passed
    overconfidence: float        # mean(confidence - actual correctness)
    ece: float                   # Expected Calibration Error (simplified)
    well_calibrated: bool        # ECE < 0.1 threshold


def compute_calibration(results: list[dict[str, Any]]) -> CalibrationReport:
    """
    Simplified calibration check: compare critic overall score (confidence)
    against keyword-based correctness (proxy for ground truth).

    A well-calibrated critic should have overall ≈ keyword_hit_rate.
    """
    confidences = []
    correctnesses = []

    for r in results:
        score = r.get("score")
        if score is None:
            continue
        confidences.append(score.overall)
        correctnesses.append(score.keyword_hit_rate)

    if not confidences:
        return CalibrationReport(0, 0, 0, 0, 0, False)

    n = len(confidences)
    mean_conf = statistics.mean(confidences)
    pass_rate = sum(1 for r in results if r.get("score") and r["score"].passed) / n

    # Overconfidence: critic score higher than keyword hit rate on average
    overconfidence = statistics.mean(c - k for c, k in zip(confidences, correctnesses))

    # ECE: bucket into 10 bins by confidence, compute |avg_conf - avg_correct| per bucket
    n_bins = 10
    bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for conf, corr in zip(confidences, correctnesses):
        bucket = min(int(conf * n_bins), n_bins - 1)
        bins[bucket].append((conf, corr))

    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = statistics.mean(c for c, _ in bucket)
        avg_corr = statistics.mean(k for _, k in bucket)
        ece += (len(bucket) / n) * abs(avg_conf - avg_corr)

    return CalibrationReport(
        n=n,
        mean_confidence=round(mean_conf, 3),
        pass_rate=round(pass_rate, 3),
        overconfidence=round(overconfidence, 3),
        ece=round(ece, 3),
        well_calibrated=ece < 0.1,
    )
