"""Unit tests for token counter and pricing estimation."""

import pytest
from contextpilot.tokens.counter import TokenCounter, count_tokens, count_messages_tokens


def test_count_empty_text():
    counter = TokenCounter()
    assert counter.count_text("") == 0


def test_count_text():
    text = "ContextPilot is an OpenAI-compatible proxy that reduces LLM tokens."
    tokens = count_tokens(text)
    assert tokens > 0
    assert 5 <= tokens <= 25


def test_count_messages():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how does token reduction work?"}
    ]
    tokens = count_messages_tokens(messages, model_name="gpt-4o-mini")
    # Message overhead: 3 per message + 3 priming = 9 + content tokens
    assert tokens > 15


def test_count_passages():
    counter = TokenCounter()
    passages = [
        {"id": "p1", "text": "This is passage one."},
        {"id": "p2", "text": "This is passage two with more words."}
    ]
    total = counter.count_passages(passages)
    assert total > 5


def test_cost_estimation():
    counter = TokenCounter(model_name="gpt-4o-mini")
    # 1,000,000 input tokens at $0.15 / M = $0.15
    cost = counter.estimate_cost(tokens_in=1_000_000, tokens_out=0)
    assert round(cost, 2) == 0.15
