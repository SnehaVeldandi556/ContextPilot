"""Call LLM node: Executes LLM generation via LangChain provider."""

import time
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from contextpilot.pipeline.state import OptimizationState
from contextpilot.llm.providers import get_llm_provider
from contextpilot.bench.judge import f1_score


def _convert_messages(msg_dicts: List[Dict[str, Any]]) -> List[BaseMessage]:
    result = []
    for m in msg_dicts:
        role = m.get("role", "user").lower()
        content = m.get("content", "")
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
        else:
            result.append(HumanMessage(content=content))
    return result


def call_llm_node(state: OptimizationState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    mode = state.get("mode", "guarded")
    model_name = state.get("model", "gpt-4o-mini")
    temperature = state.get("temperature", 0.0)
    stage_latencies = dict(state.get("stage_latencies", {}))

    llm = get_llm_provider(model_name=model_name, temperature=temperature)

    # 1. Shadow Mode Execution
    if mode == "shadow":
        # Full context prompt
        raw_passages = state.get("passages", [])
        question = state.get("question", "")
        formatted_full = "\n\n".join([f"[{p.get('id', i+1)}] {p.get('text', '')}" for i, p in enumerate(raw_passages)])
        full_messages = [
            SystemMessage(content="You are a helpful assistant. Answer based on the reference context:"),
            HumanMessage(content=f"Context:\n{formatted_full}\n\nQuestion: {question}")
        ]

        # Call LLM with full context (served response)
        full_res = llm.invoke(full_messages)
        full_answer = full_res.content if hasattr(full_res, "content") else str(full_res)

        # Call LLM with optimized context (silent evaluation)
        opt_messages = _convert_messages(state.get("final_prompt_messages", []))
        opt_res = llm.invoke(opt_messages)
        opt_answer = opt_res.content if hasattr(opt_res, "content") else str(opt_res)

        # Compute answer match
        match_score = f1_score(opt_answer, full_answer)
        match_thresh = state.get("shadow_match_threshold", 0.80)
        risk_flag = bool(match_score < match_thresh)

        shadow_info = {
            "answer_match": round(float(match_score), 4),
            "risk_flag": risk_flag
        }

        lat = (time.perf_counter() - t0) * 1000.0
        stage_latencies["call_llm"] = round(lat, 2)

        return {
            "llm_response_text": full_answer,
            "full_context_answer": full_answer,
            "shadow_result": shadow_info,
            "stage_latencies": stage_latencies
        }

    # 2. Guarded or Aggressive Mode
    prompt_msgs = _convert_messages(state.get("final_prompt_messages", []))
    res = llm.invoke(prompt_msgs)
    response_text = res.content if hasattr(res, "content") else str(res)

    lat = (time.perf_counter() - t0) * 1000.0
    stage_latencies["call_llm"] = round(lat, 2)

    return {
        "llm_response_text": response_text,
        "stage_latencies": stage_latencies
    }
