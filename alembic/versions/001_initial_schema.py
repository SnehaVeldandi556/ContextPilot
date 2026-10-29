"""001_initial_schema

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-10-30 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Enable pgvector extension on Supabase Postgres
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2. requests table
    op.create_table(
        "requests",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("ts", sa.DateTime(), nullable=False),
        sa.Column("mode", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("tokens_original", sa.Integer(), nullable=False),
        sa.Column("tokens_sent", sa.Integer(), nullable=False),
        sa.Column("reduction_pct", sa.Float(), nullable=False),
        sa.Column("cost_estimate", sa.Float(), nullable=False),
        sa.Column("latency_total_ms", sa.Float(), nullable=False),
        sa.Column("latency_per_stage", sa.JSON(), nullable=True),
        sa.Column("evidence_check", sa.String(length=32), nullable=False),
        sa.Column("fallback_used", sa.Boolean(), nullable=False),
        sa.Column("answer_match", sa.Float(), nullable=True),
        sa.Column("risk_flag", sa.Boolean(), nullable=True),
    )
    op.create_index(op.f("ix_requests_ts"), "requests", ["ts"], unique=False)

    # 3. request_passages table
    op.create_table(
        "request_passages",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("request_id", sa.String(length=64), sa.ForeignKey("requests.id", ondelete="CASCADE"), nullable=False),
        sa.Column("passage_id", sa.String(length=128), nullable=False),
        sa.Column("kept", sa.Boolean(), nullable=False),
        sa.Column("rerank_score", sa.Float(), nullable=True),
        sa.Column("drop_reason", sa.String(length=256), nullable=True),
    )
    op.create_index(op.f("ix_request_passages_request_id"), "request_passages", ["request_id"], unique=False)

    # 4. configs table
    op.create_table(
        "configs",
        sa.Column("key", sa.String(length=64), primary_key=True),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )

    # 5. benchmark_runs table
    op.create_table(
        "benchmark_runs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("dataset", sa.String(length=64), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("total_runtime_s", sa.Float(), nullable=False),
    )

    # 6. benchmark_results table
    op.create_table(
        "benchmark_results",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("run_id", sa.String(length=64), sa.ForeignKey("benchmark_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("strategy", sa.String(length=64), nullable=False),
        sa.Column("dataset", sa.String(length=64), nullable=False),
        sa.Column("tokens_saved_pct", sa.Float(), nullable=False),
        sa.Column("f1_retained", sa.Float(), nullable=False),
        sa.Column("em", sa.Float(), nullable=False),
        sa.Column("latency_added_ms", sa.Float(), nullable=False),
        sa.Column("ci_low", sa.Float(), nullable=False),
        sa.Column("ci_high", sa.Float(), nullable=False),
    )
    op.create_index(op.f("ix_benchmark_results_run_id"), "benchmark_results", ["run_id"], unique=False)

    # 7. failure_cases table
    op.create_table(
        "failure_cases",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("run_id", sa.String(length=64), sa.ForeignKey("benchmark_runs.id", ondelete="CASCADE"), nullable=True),
        sa.Column("sample_id", sa.String(length=64), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("gold_answer", sa.Text(), nullable=False),
        sa.Column("predicted_answer", sa.Text(), nullable=False),
    )
    op.create_index(op.f("ix_failure_cases_run_id"), "failure_cases", ["run_id"], unique=False)


def downgrade() -> None:
    op.drop_table("failure_cases")
    op.drop_table("benchmark_results")
    op.drop_table("benchmark_runs")
    op.drop_table("configs")
    op.drop_table("request_passages")
    op.drop_table("requests")
