"""OpenAI-compatible /v1/chat/completions router with ContextPilot optimization."""

import time
import uuid
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, BackgroundTasks, HTTPException

from contextpilot.app.config import settings
from contextpilot.app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    UsageInfo,
    ContextPilotReport,
    KeptPassageItem,
    DroppedPassageItem,
    ShadowReport
)
from contextpilot.pipeline.graph import get_pipeline
from contextpilot.tokens.counter import default_token_counter
from contextpilot.app.db.session import async_log_request

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["Chat"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks
):
    t_start = time.perf_counter()
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Extract user question from the last user message
    question = ""
    for msg in reversed(request.messages):
        if msg.role.lower() == "user":
            question = msg.content
            break
    if not question and request.messages:
        question = request.messages[-1].content

    # Extract passages and contextpilot settings
    cp_cfg = request.contextpilot
    passages: List[Dict[str, Any]] = []
    retrieval_cfg = None

    mode = settings.CONTEXTPILOT_DEFAULT_MODE
    token_budget = settings.CONTEXTPILOT_TOKEN_BUDGET
    skip_threshold = settings.CONTEXTPILOT_SKIP_THRESHOLD_TOKENS
    dedupe_threshold = settings.CONTEXTPILOT_DEDUPE_THRESHOLD
    score_drop_ratio = settings.CONTEXTPILOT_SCORE_DROP_RATIO
    guard_threshold = settings.CONTEXTPILOT_GUARD_THRESHOLD
    prune_threshold = settings.CONTEXTPILOT_SENTENCE_PRUNE_THRESHOLD

    if cp_cfg:
        if cp_cfg.mode:
            mode = cp_cfg.mode
        if cp_cfg.token_budget is not None:
            token_budget = cp_cfg.token_budget
        if cp_cfg.skip_threshold_tokens is not None:
            skip_threshold = cp_cfg.skip_threshold_tokens
        if cp_cfg.dedupe_threshold is not None:
            dedupe_threshold = cp_cfg.dedupe_threshold
        if cp_cfg.score_drop_ratio is not None:
            score_drop_ratio = cp_cfg.score_drop_ratio
        if cp_cfg.guard_threshold is not None:
            guard_threshold = cp_cfg.guard_threshold
        if cp_cfg.sentence_prune_threshold is not None:
            prune_threshold = cp_cfg.sentence_prune_threshold
        if cp_cfg.passages:
            passages = [p.model_dump() for p in cp_cfg.passages]
        if cp_cfg.retrieval:
            retrieval_cfg = cp_cfg.retrieval.model_dump()

    raw_messages = [m.model_dump() for m in request.messages]

    # Prepare LangGraph state
    initial_state = {
        "request_messages": raw_messages,
        "question": question,
        "passages": passages,
        "retrieval": retrieval_cfg,
        "mode": mode,
        "token_budget": token_budget,
        "skip_threshold_tokens": skip_threshold,
        "model": request.model,
        "temperature": request.temperature or 0.0,
        "max_tokens": request.max_tokens,
        "dedupe_threshold": dedupe_threshold,
        "score_drop_ratio": score_drop_ratio,
        "guard_threshold": guard_threshold,
        "sentence_prune_threshold": prune_threshold,
        "retry_count": 0,
        "dropped": [],
        "stage_latencies": {},
    }

    # Execute LangGraph optimization pipeline
    try:
        pipeline = get_pipeline()
        final_state = pipeline.invoke(initial_state)
    except Exception as e:
        logger.error("ContextPilot pipeline execution error: %s", e)
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")

    total_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    response_text = final_state.get("llm_response_text", "")
    raw_report = final_state.get("report", {})

    # Compute tokens and usage
    prompt_tokens = raw_report.get("tokens_sent", 0)
    completion_tokens = default_token_counter.count_text(response_text)
    total_tokens = prompt_tokens + completion_tokens

    cost_estimate = default_token_counter.estimate_cost(
        prompt_tokens, completion_tokens, model=request.model
    )

    # Format OpenAI response & ContextPilot report
    shadow_data = None
    if "shadow" in raw_report:
        s = raw_report["shadow"]
        shadow_data = ShadowReport(
            answer_match=float(s.get("answer_match", 1.0)),
            risk_flag=bool(s.get("risk_flag", False))
        )

    cp_report = ContextPilotReport(
        decision=raw_report.get("decision", "optimized"),
        tokens_original=raw_report.get("tokens_original", prompt_tokens),
        tokens_sent=raw_report.get("tokens_sent", prompt_tokens),
        reduction_pct=raw_report.get("reduction_pct", 0.0),
        latency_overhead_ms=raw_report.get("latency_overhead_ms", 0.0),
        kept=[KeptPassageItem(**k) for k in raw_report.get("kept", [])],
        dropped=[DroppedPassageItem(**d) for d in raw_report.get("dropped", [])],
        evidence_check=raw_report.get("evidence_check", "passed"),
        fallback_used=raw_report.get("fallback_used", False),
        cache_order_applied=raw_report.get("cache_order_applied", False),
        shadow=shadow_data
    )

    response = ChatCompletionResponse(
        id=req_id,
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role="assistant", content=response_text),
                finish_reason="stop"
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        ),
        contextpilot=cp_report
    )

    # Enqueue background logging task
    background_tasks.add_task(
        async_log_request,
        request_id=req_id,
        mode=mode,
        model=request.model,
        decision=cp_report.decision,
        tokens_original=cp_report.tokens_original,
        tokens_sent=cp_report.tokens_sent,
        reduction_pct=cp_report.reduction_pct,
        cost_estimate=cost_estimate,
        latency_total_ms=total_latency_ms,
        latency_per_stage=final_state.get("stage_latencies", {}),
        evidence_check=cp_report.evidence_check,
        fallback_used=cp_report.fallback_used,
        kept_passages=raw_report.get("kept", []),
        dropped_passages=raw_report.get("dropped", []),
        shadow_result=raw_report.get("shadow"),
    )

    return response
