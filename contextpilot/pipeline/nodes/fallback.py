"""Fallback node: Restores original passages when evidence guard fails and retries are exhausted."""

import time
from typing import Dict, Any
from contextpilot.pipeline.state import OptimizationState


def fallback_node(state: OptimizationState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    passages = state.get("passages", [])
    stage_latencies = dict(state.get("stage_latencies", {}))

    lat = (time.perf_counter() - t0) * 1000.0
    stage_latencies["fallback"] = round(lat, 2)

    return {
        "kept": list(passages),
        "dropped": [],
        "decision": "fallback",
        "fallback_used": True,
        "stage_latencies": stage_latencies
    }
