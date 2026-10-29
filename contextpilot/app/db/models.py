"""SQLAlchemy database models for ContextPilot."""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text, JSON
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class RequestLog(Base):
    """Stores every incoming API request and its optimization metrics."""
    __tablename__ = "requests"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    ts = Column(DateTime, default=datetime.utcnow, index=True)
    mode = Column(String(32), nullable=False, default="guarded")
    model = Column(String(64), nullable=False)
    decision = Column(String(32), nullable=False)  # optimized, skipped, fallback
    tokens_original = Column(Integer, nullable=False, default=0)
    tokens_sent = Column(Integer, nullable=False, default=0)
    reduction_pct = Column(Float, nullable=False, default=0.0)
    cost_estimate = Column(Float, nullable=False, default=0.0)
    latency_total_ms = Column(Float, nullable=False, default=0.0)
    latency_per_stage = Column(JSON, nullable=True, default=dict)
    evidence_check = Column(String(32), nullable=False, default="skipped")
    fallback_used = Column(Boolean, nullable=False, default=False)
    answer_match = Column(Float, nullable=True)
    risk_flag = Column(Boolean, nullable=True, default=False)

    passages = relationship("RequestPassage", back_populates="request", cascade="all, delete-orphan")


class RequestPassage(Base):
    """Stores kept and dropped passages per request."""
    __tablename__ = "request_passages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), ForeignKey("requests.id", ondelete="CASCADE"), nullable=False, index=True)
    passage_id = Column(String(128), nullable=False)
    kept = Column(Boolean, nullable=False)
    rerank_score = Column(Float, nullable=True)
    drop_reason = Column(String(256), nullable=True)

    request = relationship("RequestLog", back_populates="passages")


class AppConfig(Base):
    """Runtime dynamic configurations and thresholds."""
    __tablename__ = "configs"

    key = Column(String(64), primary_key=True)
    value = Column(JSON, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class BenchmarkRun(Base):
    """Record of a benchmark evaluation run."""
    __tablename__ = "benchmark_runs"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset = Column(String(64), nullable=False)
    sample_count = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    total_runtime_s = Column(Float, nullable=False, default=0.0)

    results = relationship("BenchmarkResult", back_populates="run", cascade="all, delete-orphan")
    failure_cases = relationship("FailureCase", back_populates="run", cascade="all, delete-orphan")


class BenchmarkResult(Base):
    """Aggregate metrics per strategy for a benchmark run."""
    __tablename__ = "benchmark_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), ForeignKey("benchmark_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    strategy = Column(String(64), nullable=False)
    dataset = Column(String(64), nullable=False)
    tokens_saved_pct = Column(Float, nullable=False, default=0.0)
    f1_retained = Column(Float, nullable=False, default=0.0)
    em = Column(Float, nullable=False, default=0.0)
    latency_added_ms = Column(Float, nullable=False, default=0.0)
    ci_low = Column(Float, nullable=False, default=0.0)
    ci_high = Column(Float, nullable=False, default=0.0)

    run = relationship("BenchmarkRun", back_populates="results")


class FailureCase(Base):
    """Failure cases where optimization hurt quality or dropped evidence."""
    __tablename__ = "failure_cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), ForeignKey("benchmark_runs.id", ondelete="CASCADE"), nullable=True, index=True)
    sample_id = Column(String(64), nullable=False)
    category = Column(String(64), nullable=False)
    reason = Column(Text, nullable=False)
    question = Column(Text, nullable=False)
    gold_answer = Column(Text, nullable=False)
    predicted_answer = Column(Text, nullable=False)

    run = relationship("BenchmarkRun", back_populates="failure_cases")
