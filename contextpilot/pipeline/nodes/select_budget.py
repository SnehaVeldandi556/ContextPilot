"""Select budget node: Token budget selection and score drop-off cutoff."""

import time
from typing import Dict, Any
from contextpilot.pipeline.state import OptimizationState
from contextpilot.tokens.counter import default_token_counter


def select_budget_node(state: OptimizationState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    scored_passages = state.get("scored_passages", [])
    token_budget = state.get("token_budget", 2000)
    drop_ratio = state.get("score_drop_ratio", 0.35)
    is_multi_hop = state.get("is_multi_hop", False)
    dropped = list(state.get("dropped", []))
    stage_latencies = dict(state.get("stage_latencies", {}))

    # Multi-hop question: expand budget by 50% and ensure at least 2 passages kept
    effective_budget = int(token_budget * 1.5) if is_multi_hop else token_budget
    min_passages = 2 if is_multi_hop else 1

    kept = []
    current_tokens = 0
    prev_score = None

    for idx, (score, p) in enumerate(scored_passages):
        p_tokens = default_token_counter.count_text(p.get("text", ""))

        # Score drop-off check
        sharp_drop = False
        if prev_score is not None and len(kept) >= min_passages:
            if prev_score > 0 and (prev_score - score) / max(0.01, prev_score) > drop_ratio:
                sharp_drop = True

        if sharp_drop:
            dropped.append({
                "id": p.get("id", str(idx)),
                "reason": f"low relevance ({score:.3f})",
                "rerank_score": score
            })
            continue

        if current_tokens + p_tokens <= effective_budget or len(kept) < min_passages:
            kept.append({**p, "rerank_score": round(float(score), 4)})
            current_tokens += p_tokens
            prev_score = score
        else:
            dropped.append({
                "id": p.get("id", str(idx)),
                "reason": "over budget",
                "rerank_score": score
            })

    lat = (time.perf_counter() - t0) * 1000.0
    stage_latencies["select_budget"] = round(lat, 2)

    return {
        "kept": kept,
        "dropped": dropped,
        "stage_latencies": stage_latencies
    }
