"""Evidence guard node: NLI entailment verification of kept context."""

import time
from typing import Dict, Any
from contextpilot.pipeline.state import OptimizationState
from contextpilot.models.nli import get_nli_guard
from contextpilot.llm.providers import get_llm_provider, RAG_PROMPT_TEMPLATE
from langchain_core.messages import HumanMessage


def evidence_guard_node(state: OptimizationState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    question = state.get("question", "")
    kept = state.get("kept", [])
    mode = state.get("mode", "guarded")
    stage_latencies = dict(state.get("stage_latencies", {}))

    # Guard threshold: aggressive mode uses looser threshold
    if mode == "aggressive":
        threshold = state.get("sentence_prune_threshold", 0.30)
    else:
        threshold = state.get("guard_threshold", 0.50)

    # Obtain draft answer for verification
    draft_answer = state.get("draft_answer")
    if not draft_answer:
        # Fast draft generation using LLM provider or top context
        if kept:
            top_snippet = "\n".join([p.get("text", "") for p in kept[:2]])
            llm = get_llm_provider(model_name=state.get("model", "gpt-4o-mini"))
            prompt = f"Reference Context:\n{top_snippet}\n\nQuestion: {question}"
            try:
                res = llm.invoke([HumanMessage(content=prompt)])
                draft_answer = res.content.strip()
            except Exception:
                draft_answer = top_snippet[:150]
        else:
            draft_answer = ""

    if not draft_answer or "not have sufficient information" in draft_answer.lower():
        passed = False
        score = 0.0
    else:
        guard = get_nli_guard()
        passed, score = guard.verify_kept_context(question, kept, draft_answer, threshold=threshold)

    lat = (time.perf_counter() - t0) * 1000.0
    stage_latencies["evidence_guard"] = round(lat, 2)

    return {
        "draft_answer": draft_answer,
        "evidence_check": "passed" if passed else "failed",
        "stage_latencies": stage_latencies
    }
