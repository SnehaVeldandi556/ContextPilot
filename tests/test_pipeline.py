"""Pipeline graph integration tests covering all LangGraph routing paths."""

import pytest
from contextpilot.pipeline.graph import get_pipeline


def test_pipeline_skipped_path():
    """Context below skip threshold is routed directly to passthrough."""
    pipeline = get_pipeline()
    state = {
        "request_messages": [{"role": "user", "content": "What is 2+2?"}],
        "question": "What is 2+2?",
        "passages": [{"id": "p1", "text": "2+2 equals 4."}],
        "mode": "guarded",
        "token_budget": 2000,
        "skip_threshold_tokens": 1000,  # 1000 > context tokens (~20 tokens)
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "retry_count": 0,
        "dropped": [],
        "stage_latencies": {},
    }

    result = pipeline.invoke(state)
    report = result.get("report", {})

    assert report.get("decision") == "skipped"
    assert report.get("fallback_used") is False
    assert report.get("evidence_check") == "skipped"
    assert report.get("reduction_pct") == 0.0


def test_pipeline_optimized_path():
    """Standard optimization path when evidence guard passes."""
    pipeline = get_pipeline()
    passages = [
        {"id": "p_rel", "text": "The 2016 Summer Olympics took place in Rio de Janeiro, Brazil."},
        {"id": "p_irr1", "text": "Bananas are rich in potassium and dietary fiber and grow in tropical climates."},
        {"id": "p_irr2", "text": "The speed of light in a vacuum is 299,792 kilometers per second in physics."},
        {"id": "p_irr3", "text": "Photosynthesis is the process used by plants to convert light energy into chemical energy."},
        {"id": "p_irr4", "text": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris."},
    ]
    state = {
        "request_messages": [{"role": "user", "content": "Where did the 2016 Summer Olympics take place?"}],
        "question": "Where did the 2016 Summer Olympics take place?",
        "passages": passages,
        "mode": "guarded",
        "token_budget": 50,
        "skip_threshold_tokens": 10,  # Force optimization
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "dedupe_threshold": 0.90,
        "score_drop_ratio": 0.35,
        "guard_threshold": 0.30,
        "retry_count": 0,
        "dropped": [],
        "stage_latencies": {},
    }

    result = pipeline.invoke(state)
    report = result.get("report", {})

    assert report.get("decision") == "optimized"
    assert report.get("evidence_check") == "passed"
    assert report.get("fallback_used") is False
    assert report.get("reduction_pct") > 0.0

    kept_ids = [k["id"] for k in report.get("kept", [])]
    assert "p_rel" in kept_ids


def test_pipeline_retry_then_pass_path(monkeypatch):
    """Guard initially fails, triggers loosen_budget retry loop, and passes on second attempt."""
    pipeline = get_pipeline()
    passages = [
        {"id": "p1", "text": "Christopher Nolan was born on 30 July 1970 in London."},
        {"id": "p2", "text": "Quentin Tarantino was born on March 27, 1963 in Knoxville, Tennessee."},
    ]

    call_count = 0
    from contextpilot.models.nli import NLIEvidenceGuard

    def mock_verify(self, question, kept, draft_answer, threshold=0.5):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return False, 0.20
        return True, 0.90

    monkeypatch.setattr(NLIEvidenceGuard, "verify_kept_context", mock_verify)

    state = {
        "request_messages": [{"role": "user", "content": "Which director was born earlier, Nolan or Tarantino?"}],
        "question": "Which director was born earlier, Nolan or Tarantino?",
        "passages": passages,
        "mode": "guarded",
        "token_budget": 200,
        "skip_threshold_tokens": 10,
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "dedupe_threshold": 0.90,
        "score_drop_ratio": 0.99,
        "guard_threshold": 0.40,
        "retry_count": 0,
        "dropped": [],
        "stage_latencies": {},
    }

    result = pipeline.invoke(state)
    report = result.get("report", {})

    assert call_count >= 2
    assert report.get("evidence_check") == "passed"
    assert result.get("retry_count", 0) >= 1
    assert report.get("decision") == "optimized"


def test_pipeline_retry_then_fallback_path():
    """When evidence guard repeatedly fails, pipeline safely falls back to full context."""
    pipeline = get_pipeline()
    passages = [
        {"id": "p1", "text": "A totally unrelated text about deep sea marine biology."},
        {"id": "p2", "text": "Another unrelated text about astrophysics and dark energy."}
    ]
    state = {
        "request_messages": [{"role": "user", "content": "What is the capital of Australia?"}],
        "question": "What is the capital of Australia?",
        "passages": passages,
        "mode": "guarded",
        "token_budget": 50,
        "skip_threshold_tokens": 10,
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "guard_threshold": 0.99,  # Impossibly high threshold forcing failure
        "retry_count": 0,
        "dropped": [],
        "stage_latencies": {},
    }

    result = pipeline.invoke(state)
    report = result.get("report", {})

    assert report.get("decision") == "fallback"
    assert report.get("fallback_used") is True
    assert report.get("evidence_check") == "failed"
