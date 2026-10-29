"""Token counting utilities using tiktoken with robust fallbacks."""

from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Price per million tokens (input / output) for common models
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "default": {"input": 0.50, "output": 1.50},
}


class TokenCounter:
    """Accurate token counter using tiktoken encoding with caching."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self._encoder = None
        self._init_encoder()

    def _init_encoder(self) -> None:
        try:
            import tiktoken
            try:
                self._encoder = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning("tiktoken not available, falling back to approximation: %s", e)
            self._encoder = None

    def count_text(self, text: str) -> int:
        """Count tokens in a plain text string."""
        if not text:
            return 0
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text, disallowed_special=()))
            except Exception:
                pass
        # Fallback: rule-of-thumb ~ 4 characters per token in English
        return max(1, (len(text) + 3) // 4)

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count tokens in OpenAI chat messages format.
        Follows OpenAI token counting specification:
        each message is wrapped in <|start|>{role/name}\n{content}<|end|>\n
        plus 3 tokens for assistant response priming.
        """
        num_tokens = 0
        for message in messages:
            num_tokens += 3  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += self.count_text(value)
                elif isinstance(value, list):
                    # Multimodal content or list of dicts
                    for item in value:
                        if isinstance(item, dict) and "text" in item:
                            num_tokens += self.count_text(item["text"])
                if key == "name":
                    num_tokens += 1
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def count_passages(self, passages: List[Dict[str, Any]]) -> int:
        """Count total tokens across a list of passages."""
        total = 0
        for p in passages:
            text = p.get("text", "") if isinstance(p, dict) else str(p)
            total += self.count_text(text)
        return total

    def estimate_cost(self, tokens_in: int, tokens_out: int = 0, model: Optional[str] = None) -> float:
        """Estimate cost in USD given input and output token counts."""
        target_model = (model or self.model_name).lower()
        pricing = None
        for k, v in sorted(MODEL_PRICING.items(), key=lambda x: len(x[0]), reverse=True):
            if k in target_model:
                pricing = v
                break
        if pricing is None:
            pricing = MODEL_PRICING["default"]

        cost_in = (tokens_in / 1_000_000.0) * pricing["input"]
        cost_out = (tokens_out / 1_000_000.0) * pricing["output"]
        return round(cost_in + cost_out, 6)


# Default global instance
default_token_counter = TokenCounter()


def count_tokens(text: str) -> int:
    return default_token_counter.count_text(text)


def count_messages_tokens(messages: List[Dict[str, Any]], model_name: str = "gpt-4o-mini") -> int:
    return TokenCounter(model_name).count_messages(messages)
