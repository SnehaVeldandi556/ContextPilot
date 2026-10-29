"""Full scenario test suite covering all end-to-end edge cases and requirements."""

import pytest
from contextpilot.pipeline.graph import get_pipeline
from contextpilot.models.reranker import get_reranker


def test_scenario_short_prompt_skipped():
    """Scenario 1: Short prompt skipped (< skip_threshold_tokens)."""
    pipeline = get_pipeline()
    state = {
        "request_messages": [{"role": "user", "content": "What is the capital of Spain?"}],
        "question": "What is the capital of Spain?",
        "passages": [{"id": "p1", "text": "Madrid is the capital of Spain."}],
        "mode": "guarded",
        "skip_threshold_tokens": 2000,  # 2000 > prompt size (~60 tokens)
        "token_budget": 500,
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "retry_count": 0,
        "dropped": [],
        "stage_latencies": {},
    }
    result = pipeline.invoke(state)
    report = result["report"]
    assert report["decision"] == "skipped"
    assert report["reduction_pct"] == 0.0


def test_scenario_duplicates_dropped():
    """Scenario 2: Near-duplicate passages are detected and dropped."""
    pipeline = get_pipeline()
    passages = [
        {"id": "doc_main", "text": "Employees receive 25 days of paid annual leave each calendar year."},
        {"id": "doc_copy", "text": "Employees receive 25 days of paid annual leave each calendar year with supervisor approval."},
        {"id": "doc_distractor", "text": "The company cafeteria serves espresso and breakfast sandwiches starting at 8 AM."}
    ]
    state = {
        "request_messages": [{"role": "user", "content": "How many days of annual leave do employees receive?"}],
        "question": "How many days of annual leave do employees receive?",
        "passages": passages,
        "mode": "guarded",
        "skip_threshold_tokens": 10,
        "token_budget": 500,
        "dedupe_threshold": 0.80,
        "score_drop_ratio": 0.35,
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "retry_count": 0,
        "dropped": [],
        "stage_latencies": {},
    }
    result = pipeline.invoke(state)
    report = result["report"]
    dropped_reasons = {d["id"]: d["reason"] for d in report["dropped"]}

    assert "doc_copy" in dropped_reasons
    assert "duplicate of doc_main" in dropped_reasons["doc_copy"]


def test_scenario_1_relevant_plus_20_irrelevant_keeps_only_relevant():
    """Scenario 3: 1 relevant passage + 20 irrelevant distractors keeps only the relevant one."""
    pipeline = get_pipeline()
    rel_passage = {"id": "p_gold", "text": "The chemical symbol for gold is Au and its atomic number is 79."}
    irrelevant_passages = [
        {"id": f"distractor_{i+1}", "text": f"Irrelevant botanical fact #{i+1} regarding coniferous forest flora and soil moisture."}
        for i in range(20)
    ]
    all_passages = irrelevant_passages[:10] + [rel_passage] + irrelevant_passages[10:]

    state = {
        "request_messages": [{"role": "user", "content": "What is the chemical symbol for gold?"}],
        "question": "What is the chemical symbol for gold?",
        "passages": all_passages,
        "mode": "guarded",
        "skip_threshold_tokens": 10,
        "token_budget": 500,
        "score_drop_ratio": 0.40,
        "guard_threshold": 0.30,
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "retry_count": 0,
        "dropped": [],
        "stage_latencies": {},
    }
    result = pipeline.invoke(state)
    report = result["report"]
    kept_ids = [k["id"] for k in report["kept"]]

    assert "p_gold" in kept_ids
    # Vast majority of distractors must be dropped
    assert len(report["dropped"]) >= 18
    assert report["reduction_pct"] >= 50.0


def test_scenario_multi_hop_keeps_both_evidence_passages():
    """Scenario 4: Multi-hop question keeps both evidence passages."""
    pipeline = get_pipeline()
    p1 = {"id": "p_ev1", "text": "Scott Derrickson is an American filmmaker who directed Doctor Strange."}
    p2 = {"id": "p_ev2", "text": "Edward Davis Wood Jr. was an American filmmaker who directed Plan 9 from Outer Space."}
    distractors = [
        {"id": f"d_{i+1}", "text": f"General film cinema history snippet #{i+1} about lighting and camera techniques."}
        for i in range(10)
    ]
    passages = [distractors[0], p1, distractors[1], p2] + distractors[2:]

    question = "Were Scott Derrickson and Ed Wood of the same nationality?"
    state = {
        "request_messages": [{"role": "user", "content": question}],
        "question": question,
        "passages": passages,
        "mode": "guarded",
        "skip_threshold_tokens": 10,
        "token_budget": 300,
        "score_drop_ratio": 0.45,
        "guard_threshold": 0.30,
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "retry_count": 0,
        "dropped": [],
        "stage_latencies": {},
    }
    result = pipeline.invoke(state)
    report = result["report"]
    kept_ids = [k["id"] for k in report["kept"]]

    assert "p_ev1" in kept_ids
    assert "p_ev2" in kept_ids


def test_scenario_too_low_budget_triggers_fallback():
    """Scenario 5: Too-low budget triggers evidence guard failure and fallback."""
    pipeline = get_pipeline()
    passages = [
        {"id": "p1", "text": "Quantum teleportation is a technique for transferring quantum information between distant locations."},
        {"id": "p2", "text": "It was first theoretical protocol proposed by Charles Bennett and collaborators in 1993."}
    ]
    state = {
        "request_messages": [{"role": "user", "content": "Who proposed quantum teleportation in 1993?"}],
        "question": "Who proposed quantum teleportation in 1993?",
        "passages": passages,
        "mode": "guarded",
        "skip_threshold_tokens": 10,
        "token_budget": 5,  # Unusable budget
        "guard_threshold": 0.99,  # Strict threshold triggers guard failure
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "retry_count": 0,
        "dropped": [],
        "stage_latencies": {},
    }
    result = pipeline.invoke(state)
    report = result["report"]
    assert report["decision"] == "fallback"
    assert report["fallback_used"] is True


def test_scenario_no_answer_in_context_stays_not_found():
    """Scenario 6: No-answer-in-context question stays 'not found' without hallucination."""
    pipeline = get_pipeline()
    passages = [
        {"id": "p1", "text": "The speed of sound in dry air at 20 degrees Celsius is approximately 343 meters per second."},
        {"id": "p2", "text": "Sound propagates as acoustic waves through compressible media such as air, water, and solids."}
    ]
    question = "What is the average lifespan of an African elephant in the wild?"
    state = {
        "request_messages": [{"role": "user", "content": question}],
        "question": question,
        "passages": passages,
        "mode": "guarded",
        "skip_threshold_tokens": 10,
        "token_budget": 500,
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "retry_count": 0,
        "dropped": [],
        "stage_latencies": {},
    }
    result = pipeline.invoke(state)
    answer = result.get("llm_response_text", "")
    assert "not have sufficient information" in answer.lower()


def test_scenario_relevant_passage_placed_last_moved_to_position_one():
    """Scenario 7: Relevant passage placed at the very end is moved to position 1 by the reranker."""
    reranker = get_reranker()
    passages = [
        {"id": "p_irr1", "text": "Apples are cultivated worldwide and grow on small deciduous trees."},
        {"id": "p_irr2", "text": "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions."},
        {"id": "p_irr3", "text": "Trombones are musical instruments in the brass family using a telescoping slide."},
        {"id": "p_relevant", "text": "The boiling point of liquid water at standard sea level pressure is 212 degrees Fahrenheit."}
    ]
    question = "What is the boiling point of water in Fahrenheit?"
    scored = reranker.rerank_passages(question, passages)

    # Top scored passage must be p_relevant
    best_passage = scored[0][1]
    assert best_passage["id"] == "p_relevant"


def test_scenario_shadow_mode_returns_answer_match():
    """Scenario 8: Shadow mode computes answer_match float and risk_flag."""
    pipeline = get_pipeline()
    passages = [
        {"id": "p1", "text": "At sea level, pure water boils at 212 degrees Fahrenheit (100 degrees Celsius)."}
    ]
    state = {
        "request_messages": [{"role": "user", "content": "At what temperature does water boil in Fahrenheit?"}],
        "question": "At what temperature does water boil in Fahrenheit?",
        "passages": passages,
        "mode": "shadow",
        "skip_threshold_tokens": 10,
        "token_budget": 500,
        "shadow_match_threshold": 0.80,
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "retry_count": 0,
        "dropped": [],
        "stage_latencies": {},
    }
    result = pipeline.invoke(state)
    report = result["report"]
    assert "shadow" in report
    shadow = report["shadow"]
    assert isinstance(shadow["answer_match"], float)
    assert 0.0 <= shadow["answer_match"] <= 1.0
    assert isinstance(shadow["risk_flag"], bool)
