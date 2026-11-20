"""ContextPilot Live AI Knowledge Assistant Playground & Comparative Execution Engine."""

import time
import uuid
from typing import Dict, Any, List, Tuple
from contextpilot.pipeline.graph import get_pipeline
from contextpilot.tokens.counter import default_token_counter
from contextpilot.llm.providers import get_llm_provider
from contextpilot.app.db.session import async_log_request

PRESET_DOMAINS = {
    "HR & Workplace Policies": {
        "question": "Can I carry forward unused annual leave to next year, and what is the maximum limit?",
        "passages": [
            {
                "id": "hr_leave_01",
                "text": "HR Policy Section 4.2 - Annual Leave Carryover: Full-time employees are entitled to carry forward a maximum of 5 days of unused annual leave into the subsequent calendar year, subject to manager approval."
            },
            {
                "id": "hr_leave_01_dup",
                "text": "Employee Handbook Addendum (2026): Staff members may roll over up to 5 days of unused annual vacation days into the next year with prior written approval from their department head."
            },
            {
                "id": "hr_sick_02",
                "text": "HR Policy Section 7.5 - Sick Leave: Employees unable to attend work due to illness must notify their team lead before 9:00 AM. Medical certificates are required for absences exceeding two consecutive days."
            },
            {
                "id": "hr_pantry_03",
                "text": "Office Facilities Section 2.1 - Pantry & Coffee: Complimentary espresso, cold brew, and herbal teas are provided on floors 2 and 4. Snacks and oat milk are restocked every Monday morning at 8:30 AM."
            },
            {
                "id": "hr_remote_04",
                "text": "Remote Work Guidelines Section 9.3: All team members participating in flexible work arrangements must maintain active Slack presence during core collaboration hours (10:00 AM to 4:00 PM EST)."
            },
            {
                "id": "hr_parking_05",
                "text": "Building Security Section 11.0 - Parking: Basement parking permits must be renewed annually at the reception desk. Parking without a registered tag may result in municipal towing."
            }
        ]
    },
    "Cloud & DevOps Incident Response": {
        "question": "What is the procedure for handling a primary database failover in production?",
        "passages": [
            {
                "id": "devops_db_01",
                "text": "Database Runbook Section 3.1 - Disaster Recovery: If the primary PostgreSQL instance becomes unresponsive, verify replica health on the read-replica, execute 'pg_ctl promote' on the standby, and update Route53 internal DNS to the new primary."
            },
            {
                "id": "devops_db_01_dup",
                "text": "Infrastructure SOP (Postgres Failover): In an emergency database outage, promote the synchronous standby node using the cloud provider failover CLI and re-point application connection pools to the promoted endpoint."
            },
            {
                "id": "devops_cdn_02",
                "text": "Frontend Deployment: CloudFront edge caches must be purged following production releases using the invalidate cache script targeting '/*' with a 15-minute SLA."
            },
            {
                "id": "devops_k8s_03",
                "text": "Kubernetes Autoscaling: Horizontal Pod Autoscaler (HPA) triggers pod replica scaling when average pod CPU utilization exceeds 75% for 3 consecutive evaluation cycles."
            },
            {
                "id": "devops_kafka_04",
                "text": "Kafka Queue Monitoring: Consumer group lag alerts fire at priority P2 when message lag exceeds 15,000 uncommitted offsets on the billing event stream."
            }
        ]
    },
    "Legal & Privacy Compliance": {
        "question": "What is our mandatory data retention period for customer financial transaction logs?",
        "passages": [
            {
                "id": "compliance_retention_01",
                "text": "Compliance Policy Section 6.1 - Financial Records: All customer financial transaction logs, billing receipts, and payment ledger entries must be retained in encrypted cold storage for a mandatory period of 7 years."
            },
            {
                "id": "compliance_retention_01_dup",
                "text": "Data Governance Addendum: In accordance with statutory audit regulations, customer financial transaction audit trails must remain accessible in archival storage for exactly 7 years before cryptographic destruction."
            },
            {
                "id": "compliance_cookies_02",
                "text": "Website Cookie Policy: Necessary authentication cookies expire automatically upon browser session termination. Marketing analytics cookies persist for a maximum duration of 90 days."
            },
            {
                "id": "compliance_nda_03",
                "text": "Proprietary Information & NDA: Non-disclosure commitments regarding proprietary intellectual property survive employment termination for a duration of 5 years."
            },
            {
                "id": "compliance_security_04",
                "text": "Workstation Security Standards: Employees must lock workstation screens whenever stepping away from their desks. Screensaver locks must trigger after 3 minutes of user inactivity."
            }
        ]
    }
}


def run_standard_rag_baseline(
    question: str,
    passages: List[Dict[str, Any]],
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """Execute standard RAG without optimization (sends all candidate passages)."""
    t0 = time.perf_counter()
    raw_context = "\n\n".join([f"[{p['id']}] {p['text']}" for p in passages])
    system_prompt = (
        "You are an enterprise knowledge assistant. Answer the user's question accurately "
        "based strictly on the provided context passages."
    )
    user_prompt = f"Context:\n{raw_context}\n\nQuestion: {question}"

    # Count tokens
    prompt_tokens = default_token_counter.count_text(system_prompt + "\n" + user_prompt)

    # Call LLM
    llm = get_llm_provider(model_name=model, temperature=0.0)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = llm.invoke(messages)
    answer_text = response.content if hasattr(response, "content") else str(response)

    completion_tokens = default_token_counter.count_text(answer_text)
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    cost = default_token_counter.estimate_cost(prompt_tokens, completion_tokens, model=model)

    return {
        "answer": answer_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost": cost,
        "latency_ms": latency_ms,
        "passages_count": len(passages),
        "passages": passages
    }


def run_contextpilot_optimized(
    question: str,
    passages: List[Dict[str, Any]],
    mode: str = "guarded",
    token_budget: int = 500,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """Execute ContextPilot optimized RAG with reranker, dedupe, and evidence guard."""
    t0 = time.perf_counter()
    req_id = f"demo-{uuid.uuid4().hex[:8]}"

    raw_messages = [
        {"role": "user", "content": question}
    ]

    initial_state = {
        "request_messages": raw_messages,
        "question": question,
        "passages": passages,
        "retrieval": None,
        "mode": mode,
        "token_budget": token_budget,
        "skip_threshold_tokens": 100,  # Ensure demo always exercises the pipeline
        "model": model,
        "temperature": 0.0,
        "max_tokens": None,
        "dedupe_threshold": 0.90,
        "score_drop_ratio": 0.35,
        "guard_threshold": 0.45,
        "sentence_prune_threshold": 0.20,
        "retry_count": 0,
        "dropped": [],
        "stage_latencies": {},
    }

    pipeline = get_pipeline()
    final_state = pipeline.invoke(initial_state)

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    answer_text = final_state.get("llm_response_text", "")
    report = final_state.get("report", {})

    prompt_tokens = report.get("tokens_sent", 0)
    completion_tokens = default_token_counter.count_text(answer_text)
    cost = default_token_counter.estimate_cost(prompt_tokens, completion_tokens, model=model)

    # Persist log asynchronously to database for dashboard tracking
    try:
        async_log_request(
            request_id=req_id,
            mode=mode,
            model=model,
            decision=report.get("decision", "optimized"),
            tokens_original=report.get("tokens_original", prompt_tokens),
            tokens_sent=prompt_tokens,
            reduction_pct=report.get("reduction_pct", 0.0),
            cost_estimate=cost,
            latency_total_ms=latency_ms,
            latency_per_stage=report.get("latency_per_stage", {}),
            evidence_check=report.get("evidence_check", "passed"),
            fallback_used=report.get("fallback_used", False),
            kept_passages=report.get("kept", []),
            dropped_passages=report.get("dropped", [])
        )
    except Exception:
        pass

    return {
        "request_id": req_id,
        "answer": answer_text,
        "prompt_tokens": prompt_tokens,
        "original_tokens": report.get("tokens_original", prompt_tokens),
        "tokens_saved_pct": report.get("reduction_pct", 0.0),
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost": cost,
        "latency_ms": latency_ms,
        "stage_latencies": report.get("latency_per_stage", {}),
        "decision": report.get("decision", "optimized"),
        "evidence_check": report.get("evidence_check", "passed"),
        "fallback_used": report.get("fallback_used", False),
        "kept": report.get("kept", []),
        "dropped": report.get("dropped", []),
    }
