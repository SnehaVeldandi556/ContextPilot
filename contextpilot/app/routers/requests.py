"""Request inspection, logs, and aggregate statistics router."""

import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from contextpilot.app.db.session import get_db
from contextpilot.app.db.models import RequestLog, RequestPassage
from contextpilot.app.schemas import PaginatedRequestsResponse, RequestLogSummary, StatsResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["Requests"])


@router.get("/requests", response_model=PaginatedRequestsResponse)
def get_requests(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    mode: Optional[str] = None,
    decision: Optional[str] = None,
    risk_only: bool = False,
    db: Session = Depends(get_db)
):
    """Retrieve paginated request logs with optional filtering."""
    try:
        query = db.query(RequestLog)
        if mode:
            query = query.filter(RequestLog.mode == mode)
        if decision:
            query = query.filter(RequestLog.decision == decision)
        if risk_only:
            query = query.filter(RequestLog.risk_flag == True)

        total = query.count()
        offset = (page - 1) * page_size
        records = query.order_by(RequestLog.ts.desc()).offset(offset).limit(page_size).all()

        items = [
            RequestLogSummary(
                id=r.id,
                ts=r.ts.isoformat() if r.ts else "",
                mode=r.mode,
                model=r.model,
                decision=r.decision,
                tokens_original=r.tokens_original,
                tokens_sent=r.tokens_sent,
                reduction_pct=r.reduction_pct,
                cost_estimate=r.cost_estimate,
                latency_total_ms=r.latency_total_ms,
                evidence_check=r.evidence_check,
                fallback_used=r.fallback_used,
                answer_match=r.answer_match,
                risk_flag=r.risk_flag,
            )
            for r in records
        ]

        return PaginatedRequestsResponse(
            total=total,
            page=page,
            page_size=page_size,
            items=items
        )
    except Exception as e:
        logger.error("Error reading requests: %s", e)
        return PaginatedRequestsResponse(total=0, page=page, page_size=page_size, items=[])


@router.get("/requests/{request_id}")
def get_request_detail(request_id: str, db: Session = Depends(get_db)):
    """Retrieve full request detail including kept and dropped passages."""
    req = db.query(RequestLog).filter(RequestLog.id == request_id).first()
    if not req:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")

    passages = db.query(RequestPassage).filter(RequestPassage.request_id == request_id).all()

    kept = [
        {"passage_id": p.passage_id, "rerank_score": p.rerank_score}
        for p in passages if p.kept
    ]
    dropped = [
        {"passage_id": p.passage_id, "rerank_score": p.rerank_score, "reason": p.drop_reason}
        for p in passages if not p.kept
    ]

    return {
        "id": req.id,
        "ts": req.ts.isoformat() if req.ts else None,
        "mode": req.mode,
        "model": req.model,
        "decision": req.decision,
        "tokens_original": req.tokens_original,
        "tokens_sent": req.tokens_sent,
        "reduction_pct": req.reduction_pct,
        "cost_estimate": req.cost_estimate,
        "latency_total_ms": req.latency_total_ms,
        "latency_per_stage": req.latency_per_stage or {},
        "evidence_check": req.evidence_check,
        "fallback_used": req.fallback_used,
        "answer_match": req.answer_match,
        "risk_flag": req.risk_flag,
        "kept_passages": kept,
        "dropped_passages": dropped
    }


@router.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)):
    """Retrieve global optimization statistics."""
    try:
        total_requests = db.query(func.count(RequestLog.id)).scalar() or 0
        if total_requests == 0:
            return StatsResponse(
                total_requests=0,
                avg_reduction_pct=0.0,
                estimated_cost_saved_usd=0.0,
                fallback_rate_pct=0.0,
                avg_latency_overhead_ms=0.0,
            )

        avg_reduction = db.query(func.avg(RequestLog.reduction_pct)).scalar() or 0.0
        total_fallbacks = db.query(func.count(RequestLog.id)).filter(RequestLog.fallback_used == True).scalar() or 0
        fallback_rate = (total_fallbacks / float(total_requests)) * 100.0

        # Estimated cost saved: (tokens_original - tokens_sent) * $0.0000005
        tokens_saved_sum = db.query(func.sum(RequestLog.tokens_original - RequestLog.tokens_sent)).scalar() or 0
        cost_saved = max(0.0, float(tokens_saved_sum) * 0.0000005)

        avg_lat = db.query(func.avg(RequestLog.latency_total_ms)).scalar() or 0.0

        return StatsResponse(
            total_requests=total_requests,
            avg_reduction_pct=round(float(avg_reduction), 2),
            estimated_cost_saved_usd=round(cost_saved, 4),
            fallback_rate_pct=round(float(fallback_rate), 2),
            avg_latency_overhead_ms=round(float(avg_lat * 0.15), 2),
        )
    except Exception as e:
        logger.error("Error retrieving stats: %s", e)
        return StatsResponse(
            total_requests=0,
            avg_reduction_pct=0.0,
            estimated_cost_saved_usd=0.0,
            fallback_rate_pct=0.0,
            avg_latency_overhead_ms=0.0,
        )
