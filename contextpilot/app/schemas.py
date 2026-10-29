"""Pydantic schemas for OpenAI-compatible API and ContextPilot extensions."""

from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field


class PassageInput(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


class RetrievalConfig(BaseModel):
    collection: str = "default"
    top_k: int = 30


class ContextPilotConfig(BaseModel):
    mode: Optional[Literal["shadow", "guarded", "aggressive"]] = None
    token_budget: Optional[int] = None
    skip_threshold_tokens: Optional[int] = 2000
    dedupe_threshold: Optional[float] = None
    score_drop_ratio: Optional[float] = None
    guard_threshold: Optional[float] = None
    sentence_prune_threshold: Optional[float] = None
    passages: Optional[List[PassageInput]] = None
    retrieval: Optional[RetrievalConfig] = None


class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o-mini"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    contextpilot: Optional[ContextPilotConfig] = None


class KeptPassageItem(BaseModel):
    id: str
    rerank_score: float


class DroppedPassageItem(BaseModel):
    id: str
    reason: str


class ShadowReport(BaseModel):
    answer_match: float
    risk_flag: bool


class ContextPilotReport(BaseModel):
    decision: Literal["optimized", "skipped", "fallback"]
    tokens_original: int
    tokens_sent: int
    reduction_pct: float
    latency_overhead_ms: float
    kept: List[KeptPassageItem]
    dropped: List[DroppedPassageItem]
    evidence_check: Literal["passed", "failed", "skipped"]
    fallback_used: bool
    cache_order_applied: bool
    shadow: Optional[ShadowReport] = None


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: UsageInfo
    contextpilot: ContextPilotReport


# Administrative & Benchmarking Schemas
class RequestLogSummary(BaseModel):
    id: str
    ts: str
    mode: str
    model: str
    decision: str
    tokens_original: int
    tokens_sent: int
    reduction_pct: float
    cost_estimate: float
    latency_total_ms: float
    evidence_check: str
    fallback_used: bool
    answer_match: Optional[float] = None
    risk_flag: Optional[bool] = None


class PaginatedRequestsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[RequestLogSummary]


class StatsResponse(BaseModel):
    total_requests: int
    avg_reduction_pct: float
    estimated_cost_saved_usd: float
    fallback_rate_pct: float
    avg_latency_overhead_ms: float


class IngestRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None
    chunk_size: Optional[int] = 500
    chunk_overlap: Optional[int] = 50


class IngestResponse(BaseModel):
    collection: str
    chunks_ingested: int
    status: str = "success"


class BenchmarkRunRequest(BaseModel):
    dataset: str = "hotpotqa"
    n: int = 100
    strategies: Optional[List[str]] = None
    run_ablations: bool = True
