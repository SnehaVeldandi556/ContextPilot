"""Log and report node: Final metrics compilation and report structure."""

import time
from typing import Dict, Any, List
from contextpilot.pipeline.state import OptimizationState
from contextpilot.tokens.counter import default_token_counter


def log_and_report_node(state: OptimizationState) -> Dict[str, Any]:
    decision = state.get("decision", "optimized")
    mode = state.get("mode", "guarded")
    kept = state.get("kept", [])
    dropped = state.get("dropped", [])
    stage_latencies = dict(state.get("stage_latencies", {}))

    # Calculate token counts
    from contextpilot.llm.providers import RAG_SYSTEM_PROMPT
    passages = state.get("passages", [])
    p_orig_tokens = default_token_counter.count_passages(passages)
    m_tokens = default_token_counter.count_messages(state.get("request_messages", []))
    sys_tokens = default_token_counter.count_text(RAG_SYSTEM_PROMPT) + 3
    tokens_orig = p_orig_tokens + m_tokens + sys_tokens

    final_msgs = state.get("final_prompt_messages", [])
    if final_msgs:
        tokens_sent = default_token_counter.count_messages(final_msgs)
    else:
        k_tokens = default_token_counter.count_passages(kept)
        tokens_sent = k_tokens + m_tokens + sys_tokens

    if decision in ("skipped", "fallback"):
        reduction_pct = 0.0
    else:
        reduction_pct = max(0.0, round(((tokens_orig - tokens_sent) / max(1, tokens_orig)) * 100.0, 2))

    # Overhead latency: optimization stages excluding LLM execution
    opt_stages = ["rerank", "dedupe", "select_budget", "sentence_prune", "evidence_guard", "loosen_budget", "cache_order"]
    overhead_ms = round(sum(stage_latencies.get(k, 0.0) for k in opt_stages), 2)

    # Format kept and dropped lists
    kept_report = [
        {"id": str(p.get("id", "")), "rerank_score": float(p.get("rerank_score", 1.0))}
        for p in kept
    ]
    dropped_report = [
        {"id": str(d.get("id", "")), "reason": str(d.get("reason", "unknown"))}
        for d in dropped
    ]

    report: Dict[str, Any] = {
        "decision": decision,
        "tokens_original": tokens_orig,
        "tokens_sent": tokens_sent,
        "reduction_pct": reduction_pct,
        "latency_overhead_ms": overhead_ms,
        "kept": kept_report,
        "dropped": dropped_report,
        "evidence_check": state.get("evidence_check", "passed"),
        "fallback_used": state.get("fallback_used", False),
        "cache_order_applied": state.get("cache_order_applied", False),
    }

    if mode == "shadow":
        shadow_data = state.get("shadow_result", {"answer_match": 1.0, "risk_flag": False})
        report["shadow"] = shadow_data

    return {
        "tokens_original": tokens_orig,
        "tokens_sent": tokens_sent,
        "reduction_pct": reduction_pct,
        "report": report,
        "stage_latencies": stage_latencies
    }
