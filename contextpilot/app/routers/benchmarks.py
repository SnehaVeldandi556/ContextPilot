"""Benchmark execution and inspection router."""

import uuid
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from contextpilot.app.db.session import get_db, get_session_factory
from contextpilot.app.db.models import BenchmarkRun, BenchmarkResult, FailureCase
from contextpilot.app.schemas import BenchmarkRunRequest
from contextpilot.bench.runner import run_benchmark

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["Benchmarks"])


def _execute_and_persist_benchmark(run_id: str, dataset: str, n: int, strategies: List[str], run_ablations: bool):
    try:
        summary = run_benchmark(
            dataset_name=dataset,
            n=n,
            strategy_names=strategies,
            run_ablations=run_ablations,
            output_dir=f"benchmarks_out_{run_id}"
        )

        session_factory = get_session_factory()
        with session_factory() as db:
            run_entry = BenchmarkRun(
                id=run_id,
                dataset=dataset,
                sample_count=n,
                total_runtime_s=summary.get("total_runtime_s", 0.0)
            )
            db.add(run_entry)

            for s_name, res in summary.get("strategies", {}).items():
                b_res = BenchmarkResult(
                    run_id=run_id,
                    strategy=s_name,
                    dataset=dataset,
                    tokens_saved_pct=res.get("tokens_saved_pct", 0.0),
                    f1_retained=res.get("f1_retained_pct", 0.0),
                    em=res.get("em_mean", 0.0),
                    latency_added_ms=res.get("latency_ms", 0.0),
                    ci_low=res.get("ci_low", 0.0),
                    ci_high=res.get("ci_high", 0.0),
                )
                db.add(b_res)

            db.commit()
    except Exception as e:
        logger.error("Failed executing benchmark run %s: %s", run_id, e)


@router.post("/benchmarks/run")
def trigger_benchmark(
    request: BenchmarkRunRequest,
    background_tasks: BackgroundTasks
):
    """Trigger an asynchronous benchmark evaluation run."""
    run_id = f"bench-{uuid.uuid4().hex[:8]}"
    strategies = request.strategies or ["all"]

    background_tasks.add_task(
        _execute_and_persist_benchmark,
        run_id=run_id,
        dataset=request.dataset,
        n=request.n,
        strategies=strategies,
        run_ablations=request.run_ablations
    )

    return {
        "run_id": run_id,
        "dataset": request.dataset,
        "sample_count": request.n,
        "status": "started",
        "message": f"Benchmark evaluation {run_id} running in background."
    }


@router.get("/benchmarks/{run_id}")
def get_benchmark(run_id: str, db: Session = Depends(get_db)):
    """Retrieve results for a specific benchmark run."""
    run = db.query(BenchmarkRun).filter(BenchmarkRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Benchmark run {run_id} not found")

    results = db.query(BenchmarkResult).filter(BenchmarkResult.run_id == run_id).all()
    failures = db.query(FailureCase).filter(FailureCase.run_id == run_id).limit(20).all()

    return {
        "run_id": run.id,
        "dataset": run.dataset,
        "sample_count": run.sample_count,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "total_runtime_s": run.total_runtime_s,
        "results": [
            {
                "strategy": r.strategy,
                "tokens_saved_pct": r.tokens_saved_pct,
                "f1_retained_pct": r.f1_retained,
                "em": r.em,
                "latency_added_ms": r.latency_added_ms,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
            }
            for r in results
        ],
        "failure_cases": [
            {
                "sample_id": f.sample_id,
                "category": f.category,
                "reason": f.reason,
                "question": f.question,
                "gold_answer": f.gold_answer,
                "predicted_answer": f.predicted_answer
            }
            for f in failures
        ]
    }
