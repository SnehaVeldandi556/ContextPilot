"""Cache order node: Formats final prompt for LLM provider KV-cache reuse."""

import hashlib
import time
from typing import Dict, Any, List
from contextpilot.pipeline.state import OptimizationState
from contextpilot.llm.providers import RAG_SYSTEM_PROMPT


def cache_order_node(state: OptimizationState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    kept = state.get("kept", [])
    question = state.get("question", "")
    request_messages = state.get("request_messages", [])
    stage_latencies = dict(state.get("stage_latencies", {}))

    # Deterministic stable ordering of passages by ID or content hash
    def passage_sort_key(p: Dict[str, Any]) -> str:
        pid = str(p.get("id", ""))
        if pid:
            return pid
        return hashlib.sha256(p.get("text", "").encode("utf-8")).hexdigest()

    ordered_passages = sorted(kept, key=passage_sort_key)

    # Format passages into stable text block
    formatted_passages = []
    for idx, p in enumerate(ordered_passages):
        pid = p.get("id", str(idx + 1))
        text = p.get("text", "").strip()
        formatted_passages.append(f"[{pid}] {text}")

    context_block = "\n\n".join(formatted_passages)

    # Build cache-friendly messages:
    # 1. System prompt (stable prefix)
    # 2. Stable passages context
    # 3. User question (volatile suffix)
    final_messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
    ]

    # Include any existing conversation history (except the last user turn if it was the question)
    for msg in request_messages[:-1]:
        if msg.get("role") != "system":
            final_messages.append(msg)

    # User turn with context and question
    if context_block:
        user_content = f"Reference Context:\n{context_block}\n\nQuestion: {question}"
    else:
        user_content = question

    final_messages.append({"role": "user", "content": user_content})

    lat = (time.perf_counter() - t0) * 1000.0
    stage_latencies["cache_order"] = round(lat, 2)

    return {
        "kept": ordered_passages,
        "final_prompt_messages": final_messages,
        "cache_order_applied": True,
        "stage_latencies": stage_latencies
    }
