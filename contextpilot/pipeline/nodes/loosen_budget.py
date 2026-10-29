"""Loosen budget node: Expands token budget on guard failure for retry loop."""

import time
from typing import Dict, Any
from contextpilot.pipeline.state import OptimizationState


def loosen_budget_node(state: OptimizationState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    curr_budget = state.get("token_budget", 2000)
    retry_count = state.get("retry_count", 0)
    stage_latencies = dict(state.get("stage_latencies", {}))

    new_budget = int(curr_budget * 1.5)
    lat = (time.perf_counter() - t0) * 1000.0
    stage_latencies["loosen_budget"] = round(lat, 2)

    return {
        "token_budget": new_budget,
        "retry_count": retry_count + 1,
        "stage_latencies": stage_latencies
    }
