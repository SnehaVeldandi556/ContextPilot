"""Rerank node: Cross-encoder scoring using BAAI/bge-reranker-base."""

import time
from typing import Dict, Any
from contextpilot.pipeline.state import OptimizationState
from contextpilot.models.reranker import get_reranker, detect_multihop


def rerank_node(state: OptimizationState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    question = state.get("question", "")
    passages = state.get("passages", [])
    stage_latencies = dict(state.get("stage_latencies", {}))

    # Detect multi-hop query
    is_multi_hop = detect_multihop(question)

    reranker = get_reranker()
    scored = reranker.rerank_passages(question, passages)

    lat = (time.perf_counter() - t0) * 1000.0
    stage_latencies["rerank"] = round(lat, 2)

    return {
        "scored_passages": scored,
        "is_multi_hop": is_multi_hop,
        "stage_latencies": stage_latencies
    }
