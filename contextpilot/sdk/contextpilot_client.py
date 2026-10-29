"""ContextPilot Python SDK client wrapping the OpenAI-compatible proxy."""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import httpx


@dataclass
class ContextPilotReportSDK:
    decision: str
    tokens_original: int
    tokens_sent: int
    reduction_pct: float
    latency_overhead_ms: float
    kept: List[Dict[str, Any]]
    dropped: List[Dict[str, Any]]
    evidence_check: str
    fallback_used: bool
    cache_order_applied: bool
    shadow: Optional[Dict[str, Any]] = None


@dataclass
class ChatResponseSDK:
    id: str
    model: str
    content: str
    usage: Dict[str, int]
    report: ContextPilotReportSDK
    raw_response: Dict[str, Any] = field(default_factory=dict)


class ContextPilotClient:
    """
    ContextPilot client for sending optimized chat completion requests.

    Integration is as simple as switching base_url:
        client = ContextPilotClient(base_url="http://localhost:8000/v1")
        res = client.chat(
            messages=[{"role": "user", "content": "What is the policy?"}],
            passages=[{"id": "p1", "text": "..."}],
            mode="guarded"
        )
    """

    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: Optional[str] = None, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "sk-contextpilot"
        self.timeout = timeout
        self._http_client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=self.timeout
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        passages: Optional[List[Dict[str, str]]] = None,
        retrieval: Optional[Dict[str, Any]] = None,
        mode: str = "guarded",
        token_budget: int = 2000,
        skip_threshold_tokens: int = 2000,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        extra_body: Optional[Dict[str, Any]] = None
    ) -> ChatResponseSDK:
        """Send an OpenAI-compatible chat completion request with ContextPilot optimization."""
        contextpilot_payload: Dict[str, Any] = {
            "mode": mode,
            "token_budget": token_budget,
            "skip_threshold_tokens": skip_threshold_tokens,
        }
        if passages:
            contextpilot_payload["passages"] = passages
        if retrieval:
            contextpilot_payload["retrieval"] = retrieval

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "contextpilot": contextpilot_payload
        }
        if extra_body:
            payload.update(extra_body)

        url = f"{self.base_url}/chat/completions"
        response = self._http_client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        content = choices[0]["message"]["content"] if choices else ""

        cp_rep_dict = data.get("contextpilot", {})
        report = ContextPilotReportSDK(
            decision=cp_rep_dict.get("decision", "optimized"),
            tokens_original=cp_rep_dict.get("tokens_original", 0),
            tokens_sent=cp_rep_dict.get("tokens_sent", 0),
            reduction_pct=cp_rep_dict.get("reduction_pct", 0.0),
            latency_overhead_ms=cp_rep_dict.get("latency_overhead_ms", 0.0),
            kept=cp_rep_dict.get("kept", []),
            dropped=cp_rep_dict.get("dropped", []),
            evidence_check=cp_rep_dict.get("evidence_check", "skipped"),
            fallback_used=cp_rep_dict.get("fallback_used", False),
            cache_order_applied=cp_rep_dict.get("cache_order_applied", False),
            shadow=cp_rep_dict.get("shadow")
        )

        return ChatResponseSDK(
            id=data.get("id", ""),
            model=data.get("model", model),
            content=content,
            usage=data.get("usage", {}),
            report=report,
            raw_response=data
        )

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        # Check base URL parent for /health
        root_url = self.base_url.rsplit("/v1", 1)[0]
        resp = httpx.get(f"{root_url}/health", timeout=5.0)
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self._http_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
