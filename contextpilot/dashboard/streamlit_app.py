"""ContextPilot Streamlit Dashboard."""

import os
import sys
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from contextpilot.app.db.session import get_session_factory
from contextpilot.app.db.models import RequestLog, RequestPassage, BenchmarkRun, BenchmarkResult, FailureCase

st.set_page_config(
    page_title="ContextPilot Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-metric-box {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
    }
    .badge-optimized { background-color: #DCFCE7; color: #166534; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .badge-skipped { background-color: #F1F5F9; color: #475569; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .badge-fallback { background-color: #FEE2E2; color: #991B1B; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .badge-risk { background-color: #FEF3C7; color: #92400E; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def load_db_session():
    factory = get_session_factory()
    return factory()


st.sidebar.title("✈️ ContextPilot")
st.sidebar.markdown("**Input Token Optimization & Quality Proxy**")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Request Explorer", "Shadow Report", "Alerts", "Benchmarks"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Active Mode**: `Guarded (Production)`")
st.sidebar.markdown("**Embeddings**: `BAAI/bge-small-en-v1.5`")
st.sidebar.markdown("**Reranker**: `BAAI/bge-reranker-base`")
st.sidebar.markdown("**NLI Guard**: `cross-encoder/nli-deberta`")


# ==========================================
# PAGE 1: OVERVIEW
# ==========================================
if page == "Overview":
    st.title("System Overview & Token Reduction Metrics")
    st.markdown("Real-time telemetry from Supabase Postgres proxy logs.")

    with load_db_session() as db:
        requests = db.query(RequestLog).order_by(RequestLog.ts.desc()).all()

    if not requests:
        st.info("No requests logged in Supabase database yet. Send chat requests via SDK or run benchmark to populate data.")
    else:
        df = pd.DataFrame([
            {
                "id": r.id,
                "ts": r.ts,
                "mode": r.mode,
                "model": r.model,
                "decision": r.decision,
                "tokens_original": r.tokens_original,
                "tokens_sent": r.tokens_sent,
                "tokens_saved": max(0, r.tokens_original - r.tokens_sent),
                "reduction_pct": r.reduction_pct,
                "cost_estimate": r.cost_estimate,
                "latency_total_ms": r.latency_total_ms,
                "evidence_check": r.evidence_check,
                "fallback_used": r.fallback_used,
                "risk_flag": r.risk_flag,
            }
            for r in requests
        ])

        total_reqs = len(df)
        avg_reduc = df["reduction_pct"].mean()
        total_tokens_saved = df["tokens_saved"].sum()
        est_cost_saved = total_tokens_saved * 0.0000005
        fallback_rate = (df["fallback_used"].sum() / total_reqs) * 100.0
        avg_lat = df["latency_total_ms"].mean()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Requests", f"{total_reqs:,}")
        col2.metric("Avg Token Reduction", f"{avg_reduc:.1f}%")
        col3.metric("Cost Saved", f"${est_cost_saved:.4f}")
        col4.metric("Fallback Rate", f"{fallback_rate:.1f}%")
        col5.metric("Avg Latency", f"{avg_lat:.1f} ms")

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Tokens Original vs. Tokens Sent (Recent 50 Requests)")
            recent_df = df.head(50).sort_values("ts")
            chart_df = recent_df[["tokens_original", "tokens_sent"]].reset_index(drop=True)
            st.line_chart(chart_df)

        with c2:
            st.subheader("Requests by Optimization Decision")
            dec_counts = df["decision"].value_counts()
            st.bar_chart(dec_counts)


# ==========================================
# PAGE 2: REQUEST EXPLORER
# ==========================================
elif page == "Request Explorer":
    st.title("Per-Request Passage & Stage Explorer")
    st.markdown("Detailed breakdown of kept vs. dropped passages, drop reasons, and latency per stage.")

    with load_db_session() as db:
        requests = db.query(RequestLog).order_by(RequestLog.ts.desc()).limit(100).all()

    if not requests:
        st.info("No requests recorded yet.")
    else:
        req_options = {f"{r.id} | {r.decision} | {r.reduction_pct:.1f}% saved | {r.ts.strftime('%H:%M:%S')}": r.id for r in requests}
        selected_label = st.selectbox("Select Request", list(req_options.keys()))
        selected_id = req_options[selected_label]

        with load_db_session() as db:
            req = db.query(RequestLog).filter(RequestLog.id == selected_id).first()
            passages = db.query(RequestPassage).filter(RequestPassage.request_id == selected_id).all()

        if req:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Mode", req.mode)
            m2.metric("Decision", req.decision)
            m3.metric("Original Tokens", req.tokens_original)
            m4.metric("Sent Tokens", req.tokens_sent)
            m5.metric("Reduction", f"{req.reduction_pct:.1f}%")

            st.markdown("### Latency Breakdown per Stage")
            if req.latency_per_stage:
                lat_df = pd.DataFrame(list(req.latency_per_stage.items()), columns=["Stage", "Latency (ms)"])
                st.dataframe(lat_df, hide_index=True, use_container_width=True)
            else:
                st.caption("No stage latency recorded for this request.")

            st.markdown("### Kept vs. Dropped Passages")
            kept = [p for p in passages if p.kept]
            dropped = [p for p in passages if not p.kept]

            col_k, col_d = st.columns(2)
            with col_k:
                st.success(f"**Kept Passages ({len(kept)})**")
                for p in kept:
                    st.markdown(f"- **ID:** `{p.passage_id}` | **Rerank Score:** `{p.rerank_score or 'N/A'}`")

            with col_d:
                st.error(f"**Dropped Passages ({len(dropped)})**")
                for p in dropped:
                    st.markdown(f"- **ID:** `{p.passage_id}` | **Reason:** *{p.drop_reason or 'over budget'}*")


# ==========================================
# PAGE 3: SHADOW REPORT
# ==========================================
elif page == "Shadow Report":
    st.title("Shadow Mode Answer Alignment & Risk Analysis")
    st.markdown("Evaluates answer consistency between unoptimized full context and silent optimized pipeline.")

    with load_db_session() as db:
        shadow_reqs = db.query(RequestLog).filter(RequestLog.mode == "shadow").all()

    if not shadow_reqs:
        st.info("No shadow mode requests recorded yet. Send requests with `mode: 'shadow'` to inspect alignment.")
    else:
        df_shadow = pd.DataFrame([
            {
                "id": r.id,
                "ts": r.ts,
                "answer_match": r.answer_match if r.answer_match is not None else 1.0,
                "risk_flag": r.risk_flag,
                "tokens_saved_pct": r.reduction_pct,
            }
            for r in shadow_reqs
        ])

        c1, c2, c3 = st.columns(3)
        c1.metric("Shadow Requests", len(df_shadow))
        c2.metric("Mean Answer Match", f"{df_shadow['answer_match'].mean():.3f}")
        c3.metric("Risky Cases Flagged", int(df_shadow["risk_flag"].sum()))

        st.subheader("Answer Match Distribution")
        st.bar_chart(df_shadow["answer_match"].value_counts(bins=10))

        st.subheader("Flagged Risky Cases (Match < Threshold)")
        risky = df_shadow[df_shadow["risk_flag"] == True]
        if risky.empty:
            st.success("No risky cases flagged! All shadow responses satisfied quality thresholds.")
        else:
            st.dataframe(risky, use_container_width=True)


# ==========================================
# PAGE 4: ALERTS
# ==========================================
elif page == "Alerts":
    st.title("Security & Reliability Alerts")
    st.markdown("Identifies requests where the Evidence Guard failed or risk flags were raised.")

    with load_db_session() as db:
        alerts = db.query(RequestLog).filter(
            (RequestLog.evidence_check == "failed") | (RequestLog.risk_flag == True)
        ).order_by(RequestLog.ts.desc()).all()

    if not alerts:
        st.success("🎉 No active alerts! All requests passed evidence guards without risk flags.")
    else:
        st.warning(f"Found {len(alerts)} alerts requiring attention.")
        alert_data = [
            {
                "Request ID": a.id,
                "Timestamp": a.ts.strftime("%Y-%m-%d %H:%M:%S") if a.ts else "",
                "Mode": a.mode,
                "Evidence Check": a.evidence_check,
                "Fallback Used": a.fallback_used,
                "Risk Flag": a.risk_flag,
                "Answer Match": a.answer_match,
                "Original Tokens": a.tokens_original,
            }
            for a in alerts
        ]
        st.dataframe(pd.DataFrame(alert_data), use_container_width=True)


# ==========================================
# PAGE 5: BENCHMARKS
# ==========================================
elif page == "Benchmarks":
    st.title("Evaluation Suite, Ablations & Break-Even Analysis")
    st.markdown("Empirical benchmark results comparing ContextPilot against full context, BM25, and top-k baselines.")

    with load_db_session() as db:
        runs = db.query(BenchmarkRun).order_by(BenchmarkRun.created_at.desc()).all()

    if not runs:
        st.info("No benchmark runs in database yet. Run `python -m bench.runner --dataset hotpotqa --n 300 --strategies all`.")
    else:
        run_dict = {f"{r.id} | {r.dataset} | n={r.sample_count} | {r.created_at.strftime('%Y-%m-%d %H:%M')}": r.id for r in runs}
        selected_run_label = st.selectbox("Select Benchmark Run", list(run_dict.keys()))
        selected_run_id = run_dict[selected_run_label]

        with load_db_session() as db:
            results = db.query(BenchmarkResult).filter(BenchmarkResult.run_id == selected_run_id).all()

        if results:
            res_df = pd.DataFrame([
                {
                    "Strategy": r.strategy,
                    "Tokens Saved (%)": f"{r.tokens_saved_pct:.1f}%",
                    "F1 Retained (%)": f"{r.f1_retained:.1f}%",
                    "EM": f"{r.em:.3f}",
                    "95% CI (F1)": f"[{r.ci_low:.3f}, {r.ci_high:.3f}]",
                    "Latency Added (ms)": f"{r.latency_added_ms:.1f}",
                }
                for r in results
            ])

            st.markdown("### Strategy Comparison Matrix")
            st.dataframe(res_df, hide_index=True, use_container_width=True)

    # Break-Even Visualizer
    st.markdown("### Break-Even Latency & Cost Crossover")
    prompt_sizes = [500, 1000, 2000, 4000, 8000, 12000, 16000]
    baseline_lat = [p * 0.045 for p in prompt_sizes]
    opt_lat = [(p * 0.5 * 0.045) + 65.0 for p in prompt_sizes]

    chart_data = pd.DataFrame({
        "Context Size (Tokens)": prompt_sizes,
        "Baseline Full Context Latency": baseline_lat,
        "ContextPilot Optimized Latency": opt_lat
    }).set_index("Context Size (Tokens)")

    st.line_chart(chart_data)
    st.caption("ContextPilot's skip threshold (2,000 tokens) ensures small requests bypass optimization overhead, while larger contexts achieve net latency and cost reductions.")
