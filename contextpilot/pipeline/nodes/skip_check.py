"""Skip check node: Determines if context is small enough to bypass optimization."""

import time
from typing import Dict, Any
from contextpilot.pipeline.state import OptimizationState
from contextpilot.tokens.counter import default_token_counter


def skip_check_node(state: OptimizationState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    passages = state.get("passages", [])
    skip_threshold = state.get("skip_threshold_tokens", 2000)

    # Calculate tokens
    from contextpilot.llm.providers import RAG_SYSTEM_PROMPT
    passage_tokens = default_token_counter.count_passages(passages)
    msg_tokens = default_token_counter.count_messages(state.get("request_messages", []))
    sys_tokens = default_token_counter.count_text(RAG_SYSTEM_PROMPT) + 3
    total_tokens = passage_tokens + msg_tokens + sys_tokens

    lat = (time.perf_counter() - t0) * 1000.0
    stage_latencies = dict(state.get("stage_latencies", {}))
    stage_latencies["skip_check"] = round(lat, 2)

    # If no passages and no retrieval, or context is under the skip threshold
    has_retrieval = bool(state.get("retrieval"))
    if not has_retrieval and (not passages or total_tokens < skip_threshold):
        return {
            "tokens_original": total_tokens,
            "tokens_sent": total_tokens,
            "reduction_pct": 0.0,
            "decision": "skipped",
            "evidence_check": "skipped",
            "fallback_used": False,
            "cache_order_applied": False,
            "kept": list(passages),
            "dropped": [],
            "stage_latencies": stage_latencies
        }

    return {
        "tokens_original": total_tokens,
        "tokens_sent": total_tokens,
        "stage_latencies": stage_latencies
    }
