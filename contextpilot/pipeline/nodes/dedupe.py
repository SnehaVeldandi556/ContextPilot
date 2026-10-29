"""Dedupe node: Near-duplicate removal via BAAI/bge-small-en-v1.5 embeddings."""

import time
from typing import Dict, Any
from contextpilot.pipeline.state import OptimizationState
from contextpilot.models.embedder import get_embedder


def dedupe_node(state: OptimizationState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    scored_passages = state.get("scored_passages", [])
    threshold = state.get("dedupe_threshold", 0.92)
    dropped = list(state.get("dropped", []))
    stage_latencies = dict(state.get("stage_latencies", {}))

    embedder = get_embedder()
    deduped, dedupe_dropped = embedder.deduplicate(scored_passages, threshold=threshold)
    dropped.extend(dedupe_dropped)

    lat = (time.perf_counter() - t0) * 1000.0
    stage_latencies["dedupe"] = round(lat, 2)

    return {
        "scored_passages": deduped,
        "dropped": dropped,
        "stage_latencies": stage_latencies
    }
