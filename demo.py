"""ContextPilot End-to-End Demo: HR Leave Policy Query with Duplicates & Irrelevant Passages."""

import json
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from contextpilot.pipeline.graph import get_pipeline
from contextpilot.tokens.counter import default_token_counter


SAMPLE_PASSAGES = [
    {
        "id": "hr_policy_leave_01",
        "text": (
            "HR Policy Section 4.2 - Annual Leave Carry Forward: Full-time employees are entitled to 20 days of paid annual leave "
            "per calendar year. Employees may carry forward up to 5 unused annual leave days into the subsequent calendar year, "
            "provided they obtain written manager approval before December 15. Carried forward days must be utilized within the "
            "first quarter (by March 31)."
        )
    },
    {
        "id": "hr_policy_leave_01_dup",
        "text": (
            "Staff Handbook Addendum - Annual Leave Carryover: Full-time staff members receive 20 days paid vacation annually. "
            "Up to 5 days of unused vacation may be rolled over into the next year with prior written approval from your department "
            "manager before Dec 15, and must be taken before March 31."
        )
    },
    {
        "id": "hr_policy_supplies_03",
        "text": (
            "HR Policy Section 2.1 - Office Supplies and Reimbursements: All office supplies, including notebooks, stationery, "
            "and laptop stands, must be requisitioned through the internal procurement portal with department head sign-off."
        )
    },
    {
        "id": "hr_policy_sick_04",
        "text": (
            "HR Policy Section 7.5 - Sick Leave and Medical Certificates: Employees unable to attend work due to illness must "
            "notify their supervisor by 9:00 AM on the day of absence. A medical practitioner certificate is required for sick "
            "leave exceeding two consecutive working days."
        )
    },
    {
        "id": "hr_policy_remote_05",
        "text": (
            "HR Policy Section 9.3 - Remote Work and Core Hours: All team members are expected to be reachable and online during "
            "core business hours between 10:00 AM and 4:00 PM in their respective regional time zones."
        )
    },
    {
        "id": "hr_policy_conduct_06",
        "text": (
            "HR Policy Section 12.0 - Code of Conduct and Cybersecurity: Employees must lock workstations when leaving their desks "
            "and adhere to enterprise multi-factor authentication protocols on all connected company devices."
        )
    }
]

QUESTION = "Can I carry forward unused annual leave to next year?"


def run_demo():
    print("=" * 80)
    print("  CONTEXTPILOT END-TO-END DEMO: HR LEAVE POLICY")
    print("=" * 80)
    print(f"\n[Question]: '{QUESTION}'\n")
    print(f"Candidate Passages Supplied: {len(SAMPLE_PASSAGES)}")
    for idx, p in enumerate(SAMPLE_PASSAGES):
        print(f"  {idx+1}. [{p['id']}] {p['text'][:85]}...")

    initial_tokens = default_token_counter.count_passages(SAMPLE_PASSAGES)
    print(f"\nInitial Raw Passage Tokens: {initial_tokens}")

    # Build pipeline state
    state = {
        "request_messages": [{"role": "user", "content": QUESTION}],
        "question": QUESTION,
        "passages": SAMPLE_PASSAGES,
        "mode": "guarded",
        "token_budget": 500,
        "skip_threshold_tokens": 100,  # Ensure demo runs full optimizer
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "dedupe_threshold": 0.88,
        "score_drop_ratio": 0.35,
        "guard_threshold": 0.50,
        "retry_count": 0,
        "dropped": [],
        "stage_latencies": {},
    }

    print("\nExecuting ContextPilot LangGraph Optimization Pipeline...")
    pipeline = get_pipeline()
    final_state = pipeline.invoke(state)

    report = final_state.get("report", {})
    answer = final_state.get("llm_response_text", "")

    print("\n" + "=" * 80)
    print("  OPTIMIZATION REPORT")
    print("=" * 80)
    print(f"Decision:               {report.get('decision')}")
    print(f"Original Tokens:        {report.get('tokens_original')}")
    print(f"Sent Tokens:            {report.get('tokens_sent')}")
    print(f"Token Reduction:        {report.get('reduction_pct'):.1f}%")
    print(f"Optimizer Overhead:     {report.get('latency_overhead_ms'):.1f} ms")
    print(f"Evidence Guard Check:   {report.get('evidence_check')}")
    print(f"Fallback Used:          {report.get('fallback_used')}")
    print(f"Cache Ordering Applied: {report.get('cache_order_applied')}")

    print("\n--- Kept Passages ---")
    for k in report.get("kept", []):
        print(f"  + [{k['id']}] (Rerank score: {k['rerank_score']:.3f})")

    print("\n--- Dropped Passages ---")
    for d in report.get("dropped", []):
        print(f"  - [{d['id']}] -> Reason: {d['reason']}")

    print("\n" + "=" * 80)
    print("  GENERATED LLM ANSWER")
    print("=" * 80)
    print(f"{answer}\n")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
