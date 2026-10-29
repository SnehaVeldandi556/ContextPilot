"""Sentence prune node: Sentence-level pruning for aggressive mode."""

import time
from typing import Dict, Any
from contextpilot.pipeline.state import OptimizationState
from contextpilot.models.reranker import get_reranker


def sentence_prune_node(state: OptimizationState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    mode = state.get("mode", "guarded")
    kept = list(state.get("kept", []))
    stage_latencies = dict(state.get("stage_latencies", {}))

    if mode == "aggressive" and kept:
        query = state.get("question", "")
        thresh = state.get("sentence_prune_threshold", 0.20)
        reranker = get_reranker()
        kept = reranker.prune_sentences(query, kept, threshold=thresh)

    lat = (time.perf_counter() - t0) * 1000.0
    stage_latencies["sentence_prune"] = round(lat, 2)

    return {
        "kept": kept,
        "stage_latencies": stage_latencies
    }
