# eval/questions.py — Helios 30-question evaluation harness question bank
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

QuestionType = Literal["factual", "analytical", "code", "multi_step", "no_context"]


@dataclass
class EvalQuestion:
    id: int
    query: str
    type: QuestionType
    expected_keywords: list[str]           # must appear in answer (case-insensitive)
    expected_not_keywords: list[str]       # must NOT appear (hallucination guard)
    min_groundedness: float = 0.6
    min_faithfulness: float = 0.6
    min_completeness: float = 0.5


EVAL_QUESTIONS: list[EvalQuestion] = [
    # ── Factual (retrieval-dependent) ─────────────────────────────────────────
    EvalQuestion(
        id=1, type="factual",
        query="What is CARLE and how does it compress semantic maps?",
        expected_keywords=["class-aware", "run-length", "differential", "triplet"],
        expected_not_keywords=["pixel reconstruction", "motion estimation"],
    ),
    EvalQuestion(
        id=2, type="factual",
        query="What compression ratio does CARLE achieve on nuScenes Mini?",
        expected_keywords=["265", "60", "ms"],
        expected_not_keywords=[],
    ),
    EvalQuestion(
        id=3, type="factual",
        query="What datasets were used to evaluate CARLE?",
        expected_keywords=["nuScenes", "Cityscapes", "field-test"],
        expected_not_keywords=[],
    ),
    EvalQuestion(
        id=4, type="factual",
        query="What is the 3GPP NR-V2X latency constraint for sidelink?",
        expected_keywords=["100", "ms"],
        expected_not_keywords=[],
    ),
    EvalQuestion(
        id=5, type="factual",
        query="What segmentation backbone achieves the highest compression ratio in CARLE?",
        expected_keywords=["YOLO11n", "265"],
        expected_not_keywords=[],
    ),
    EvalQuestion(
        id=6, type="factual",
        query="How does the SemanticWorldMap receiver work?",
        expected_keywords=["persistent", "update", "triplet", "no decoder"],
        expected_not_keywords=[],
    ),
    EvalQuestion(
        id=7, type="factual",
        query="What is the null symbol used in CARLE encoding?",
        expected_keywords=["0", "null", "unchanged"],
        expected_not_keywords=[],
    ),
    EvalQuestion(
        id=8, type="factual",
        query="What is the mIoU achieved by CARLE on all three datasets?",
        expected_keywords=["1.00", "lossless"],
        expected_not_keywords=[],
    ),
    EvalQuestion(
        id=9, type="factual",
        query="What are the four CARLE semantic classes?",
        expected_keywords=["road", "vehicle", "pedestrian", "other"],
        expected_not_keywords=[],
    ),
    EvalQuestion(
        id=10, type="factual",
        query="What is the raw bandwidth of a single 900x1600 camera at 8 bits per pixel?",
        expected_keywords=["11520", "Kb"],
        expected_not_keywords=[],
    ),

    # ── Analytical ────────────────────────────────────────────────────────────
    EvalQuestion(
        id=11, type="analytical",
        query="Why do lighter segmentation models produce higher compression ratios in CARLE?",
        expected_keywords=["smooth", "homogeneous", "boundaries", "runs"],
        expected_not_keywords=[],
    ),
    EvalQuestion(
        id=12, type="analytical",
        query="Why does Delta Encoding alone achieve CR=1.0 in the ablation study?",
        expected_keywords=["pixel domain", "residual", "dense", "label structure"],
        expected_not_keywords=[],
    ),
    EvalQuestion(
        id=13, type="analytical",
        query="Why do rear-facing cameras have higher CR than front-facing cameras?",
        expected_keywords=["approaching", "objects", "lane", "transitions"],
        expected_not_keywords=[],
    ),
    EvalQuestion(
        id=14, type="analytical",
        query="How does temporal sparsity drive CARLE's compression efficiency?",
        expected_keywords=["unchanged", "sparse", "delta", "K_t"],
        expected_not_keywords=[],
    ),
    EvalQuestion(
        id=15, type="analytical",
        query="Compare CARLE to DVC in terms of CR, latency, and mIoU.",
        expected_keywords=["265", "27", "1.00", "0.87", "decoder"],
        expected_not_keywords=[],
    ),

    # ── Code ──────────────────────────────────────────────────────────────────
    EvalQuestion(
        id=16, type="code",
        query="Write Python code to compute the CARLE compression ratio given K_t, H, W, and C.",
        expected_keywords=["def", "log2", "K_t", "return"],
        expected_not_keywords=[],
    ),
    EvalQuestion(
        id=17, type="code",
        query="Write a Python function implementing the CARLE differential operator.",
        expected_keywords=["np.where", "cur", "prev", "NULL"],
        expected_not_keywords=[],
    ),
    EvalQuestion(
        id=18, type="code",
        query="Write Python code to encode a flat label array into CARLE triplets.",
        expected_keywords=["triplet", "start", "length", "class"],
        expected_not_keywords=[],
    ),
    EvalQuestion(
        id=19, type="code",
        query="Write a BM25 search function that returns top-k scored results.",
        expected_keywords=["BM25", "score", "top_k", "return"],
        expected_not_keywords=[],
    ),
    EvalQuestion(
        id=20, type="code",
        query="Implement a simple JWT token creation function using python-jose.",
        expected_keywords=["jwt.encode", "exp", "sub", "secret"],
        expected_not_keywords=[],
    ),

    # ── Multi-step ────────────────────────────────────────────────────────────
    EvalQuestion(
        id=21, type="multi_step",
        query="Calculate the bandwidth savings of CARLE vs raw masks for 6 cameras at 265x CR.",
        expected_keywords=["258", "69120", "99"],
        expected_not_keywords=[],
    ),
    EvalQuestion(
        id=22, type="multi_step",
        query="What percentage of field-test frames satisfy the 3GPP 100ms constraint?",
        expected_keywords=["99", "%"],
        expected_not_keywords=[],
    ),
    EvalQuestion(
        id=23, type="multi_step",
        query="If Δpix = 0.007%, what is the expected CR given the theoretical model?",
        expected_keywords=["179346", "inverse", "proportional"],
        expected_not_keywords=[],
    ),
    EvalQuestion(
        id=24, type="multi_step",
        query="Describe the end-to-end flow from camera frame to CARLE triplet transmission.",
        expected_keywords=["segmentation", "differential", "run-length", "triplet", "receiver"],
        expected_not_keywords=[],
    ),
    EvalQuestion(
        id=25, type="multi_step",
        query="How would you extend CARLE to support multi-vehicle collaborative perception?",
        expected_keywords=["merge", "map", "vehicle", "update"],
        expected_not_keywords=[],
    ),

    # ── No context (direct knowledge) ────────────────────────────────────────
    EvalQuestion(
        id=26, type="no_context",
        query="What is LangGraph and how does it differ from LangChain?",
        expected_keywords=["graph", "state", "nodes", "edges"],
        expected_not_keywords=[],
        min_groundedness=0.3,   # relaxed — no retrieval context expected
    ),
    EvalQuestion(
        id=27, type="no_context",
        query="What is the difference between dense and sparse retrieval?",
        expected_keywords=["embedding", "BM25", "semantic", "keyword"],
        expected_not_keywords=[],
        min_groundedness=0.3,
    ),
    EvalQuestion(
        id=28, type="no_context",
        query="What is OpenTelemetry and what problem does it solve?",
        expected_keywords=["tracing", "observability", "spans", "distributed"],
        expected_not_keywords=[],
        min_groundedness=0.3,
    ),
    EvalQuestion(
        id=29, type="no_context",
        query="Explain the purpose of Celery in a distributed Python system.",
        expected_keywords=["task queue", "worker", "broker", "async"],
        expected_not_keywords=[],
        min_groundedness=0.3,
    ),
    EvalQuestion(
        id=30, type="no_context",
        query="What is the difference between mIoU and pixel accuracy?",
        expected_keywords=["intersection", "union", "class", "average"],
        expected_not_keywords=[],
        min_groundedness=0.3,
    ),
]
