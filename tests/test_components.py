"""Unit tests for individual optimization components: dedupe, score cutoff, budget, cache order, multihop."""

import pytest
from contextpilot.models.embedder import get_embedder
from contextpilot.models.reranker import get_reranker, detect_multihop
from contextpilot.pipeline.nodes.select_budget import select_budget_node
from contextpilot.pipeline.nodes.cache_order import cache_order_node


def test_dedupe_removes_near_duplicates():
    embedder = get_embedder()
    scored_passages = [
        (0.95, {"id": "p1", "text": "Employees receive 20 days of paid vacation per calendar year."}),
        (0.94, {"id": "p2", "text": "Employees receive 20 days of paid vacation per calendar year with HR approval."}),  # Near duplicate
        (0.40, {"id": "p3", "text": "The company cafeteria serves hot lunch daily from 12 PM to 2 PM."}),
    ]

    kept, dropped = embedder.deduplicate(scored_passages, threshold=0.85)
    kept_ids = [p["id"] for _, p in kept]
    dropped_ids = [d["id"] for d in dropped]

    assert "p1" in kept_ids
    assert "p2" in dropped_ids
    assert any("duplicate of p1" in d["reason"] for d in dropped)
    assert "p3" in kept_ids


def test_score_drop_off_cutoff():
    state = {
        "scored_passages": [
            (0.95, {"id": "p_top", "text": "Relevant fact directly answering the question."}),
            (0.20, {"id": "p_low1", "text": "Completely irrelevant distractor about gardening."}),
            (0.15, {"id": "p_low2", "text": "Another completely unrelated distractor snippet."}),
        ],
        "token_budget": 5000,
        "score_drop_ratio": 0.40,
        "is_multi_hop": False,
        "dropped": []
    }

    result = select_budget_node(state)
    kept_ids = [p["id"] for p in result["kept"]]
    dropped_ids = [d["id"] for d in result["dropped"]]

    assert "p_top" in kept_ids
    assert "p_low1" in dropped_ids
    assert any("low relevance" in d["reason"] for d in result["dropped"])


def test_budget_selection_respects_token_limit():
    state = {
        "scored_passages": [
            (0.9, {"id": "p1", "text": "Short passage one."}),
            (0.8, {"id": "p2", "text": "Short passage two."}),
            (0.7, {"id": "p3", "text": "Short passage three that exceeds our tiny token budget."}),
        ],
        "token_budget": 10,  # Very small budget allowing only ~1-2 passages
        "score_drop_ratio": 0.99,
        "is_multi_hop": False,
        "dropped": []
    }

    result = select_budget_node(state)
    assert len(result["kept"]) >= 1
    assert len(result["dropped"]) >= 1
    assert any("over budget" in d["reason"] for d in result["dropped"])


def test_cache_order_arranges_deterministic_prompt():
    state = {
        "question": "What is the capital of France?",
        "request_messages": [{"role": "user", "content": "What is the capital of France?"}],
        "kept": [
            {"id": "doc_z", "text": "Paris is the capital of France."},
            {"id": "doc_a", "text": "France is a country in Western Europe."}
        ],
        "stage_latencies": {}
    }

    result = cache_order_node(state)
    assert result["cache_order_applied"] is True
    messages = result["final_prompt_messages"]

    # 1. System prompt first
    assert messages[0]["role"] == "system"
    # 2. Final message is user turn containing sorted passages then volatile question
    last_msg = messages[-1]
    assert last_msg["role"] == "user"
    assert "Reference Context:" in last_msg["content"]
    assert "[doc_a]" in last_msg["content"]
    assert "[doc_z]" in last_msg["content"]
    # doc_a appears before doc_z because of sorted cache order
    assert last_msg["content"].index("[doc_a]") < last_msg["content"].index("[doc_z]")
    assert "Question: What is the capital of France?" in last_msg["content"]


def test_multi_hop_detection():
    # Comparison pattern
    assert detect_multihop("Were Scott Derrickson and Ed Wood of the same nationality?") is True
    assert detect_multihop("Which film has the director who was born earlier, Inception or Pulp Fiction?") is True
    assert detect_multihop("What government agency was headed by the author of the report on JFK?") is True

    # Simple single-hop pattern
    assert detect_multihop("Where did the 2016 Summer Olympics take place?") is False
    assert detect_multihop("What is the chemical symbol for gold?") is False
