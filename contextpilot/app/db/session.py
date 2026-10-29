"""Database session management, connection pooling, and async logging for Supabase."""

import os
import logging
from typing import Generator, Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from contextpilot.app.config import settings
from contextpilot.app.db.models import Base, RequestLog, RequestPassage

logger = logging.getLogger(__name__)

_ENGINE = None
_SESSION_FACTORY = None


def get_db_url() -> str:
    """Retrieve active database URL (test DB if set, otherwise Supabase DATABASE_URL)."""
    test_url = os.getenv("TEST_DATABASE_URL") or settings.TEST_DATABASE_URL
    if test_url:
        return test_url
    return os.getenv("DATABASE_URL") or settings.DATABASE_URL


def get_engine():
    """Create or return cached SQLAlchemy engine."""
    global _ENGINE
    if _ENGINE is None:
        db_url = get_db_url()
        connect_args = {}
        if db_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            _ENGINE = create_engine(db_url, connect_args=connect_args)
        else:
            # Supabase pooler (port 6543) connection tuning
            _ENGINE = create_engine(
                db_url,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                pool_recycle=300,
            )
    return _ENGINE


def get_session_factory():
    """Create or return cached sessionmaker."""
    global _SESSION_FACTORY
    if _SESSION_FACTORY is None:
        engine = get_engine()
        _SESSION_FACTORY = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SESSION_FACTORY


def get_db() -> Generator[Session, None, None]:
    """FastAPI database session dependency."""
    session_factory = get_session_factory()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables (primarily for tests / SQLite local fallback)."""
    try:
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.warning("Could not auto-create tables with create_all: %s", e)


def async_log_request(
    request_id: str,
    mode: str,
    model: str,
    decision: str,
    tokens_original: int,
    tokens_sent: int,
    reduction_pct: float,
    cost_estimate: float,
    latency_total_ms: float,
    latency_per_stage: Dict[str, float],
    evidence_check: str,
    fallback_used: bool,
    kept_passages: List[Dict[str, Any]],
    dropped_passages: List[Dict[str, Any]],
    shadow_result: Optional[Dict[str, Any]] = None
) -> None:
    """
    Asynchronously write request log to Supabase Postgres.
    Designed to run inside FastAPI BackgroundTasks.
    Guarantees that log write failures never affect client response latency or availability.
    """
    try:
        session_factory = get_session_factory()
        with session_factory() as db:
            answer_match = None
            risk_flag = False
            if shadow_result:
                answer_match = shadow_result.get("answer_match")
                risk_flag = shadow_result.get("risk_flag", False)

            req = RequestLog(
                id=request_id,
                ts=datetime.utcnow(),
                mode=mode,
                model=model,
                decision=decision,
                tokens_original=tokens_original,
                tokens_sent=tokens_sent,
                reduction_pct=reduction_pct,
                cost_estimate=cost_estimate,
                latency_total_ms=latency_total_ms,
                latency_per_stage=latency_per_stage,
                evidence_check=evidence_check,
                fallback_used=fallback_used,
                answer_match=answer_match,
                risk_flag=risk_flag,
            )
            db.add(req)

            # Insert kept passages
            for p in kept_passages:
                pass_entry = RequestPassage(
                    request_id=request_id,
                    passage_id=str(p.get("id", "")),
                    kept=True,
                    rerank_score=p.get("rerank_score"),
                    drop_reason=None
                )
                db.add(pass_entry)

            # Insert dropped passages
            for d in dropped_passages:
                pass_entry = RequestPassage(
                    request_id=request_id,
                    passage_id=str(d.get("id", "")),
                    kept=False,
                    rerank_score=d.get("rerank_score"),
                    drop_reason=str(d.get("reason", ""))
                )
                db.add(pass_entry)

            db.commit()
    except Exception as e:
        logger.error("Failed to asynchronously log request %s to database: %s", request_id, e)
