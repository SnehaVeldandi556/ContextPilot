"""LangGraph pipeline state definition for ContextPilot."""

from typing import TypedDict, List, Dict, Any, Optional, Tuple


class OptimizationState(TypedDict, total=False):
    # Request Inputs
    request_messages: List[Dict[str, Any]]
    question: str
    passages: List[Dict[str, Any]]
    retrieval: Optional[Dict[str, Any]]
    mode: str  # "shadow" | "guarded" | "aggressive"
    token_budget: int
    skip_threshold_tokens: int
    model: str
    temperature: float
    max_tokens: Optional[int]

    # Threshold overrides
    dedupe_threshold: float
    score_drop_ratio: float
    guard_threshold: float
    sentence_prune_threshold: float
    shadow_match_threshold: float

    # Pipeline tracking
    tokens_original: int
    tokens_sent: int
    reduction_pct: float
    token_counts: Dict[str, int]
    scored_passages: List[Tuple[float, Dict[str, Any]]]
    kept: List[Dict[str, Any]]
    dropped: List[Dict[str, Any]]
    retry_count: int
    decision: str  # "optimized" | "skipped" | "fallback"
    evidence_check: str  # "passed" | "failed" | "skipped"
    fallback_used: bool
    cache_order_applied: bool
    is_multi_hop: bool

    # Generations & Latencies
    draft_answer: Optional[str]
    full_context_answer: Optional[str]
    final_prompt_messages: List[Dict[str, Any]]
    llm_response_text: Optional[str]
    llm_usage: Dict[str, int]
    shadow_result: Optional[Dict[str, Any]]
    stage_latencies: Dict[str, float]
    report: Dict[str, Any]
