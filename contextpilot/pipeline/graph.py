"""LangGraph pipeline definition for ContextPilot."""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END

from contextpilot.pipeline.state import OptimizationState
from contextpilot.pipeline.nodes.skip_check import skip_check_node
from contextpilot.pipeline.nodes.retrieve import retrieve_node
from contextpilot.pipeline.nodes.rerank import rerank_node
from contextpilot.pipeline.nodes.dedupe import dedupe_node
from contextpilot.pipeline.nodes.select_budget import select_budget_node
from contextpilot.pipeline.nodes.sentence_prune import sentence_prune_node
from contextpilot.pipeline.nodes.evidence_guard import evidence_guard_node
from contextpilot.pipeline.nodes.loosen_budget import loosen_budget_node
from contextpilot.pipeline.nodes.fallback import fallback_node
from contextpilot.pipeline.nodes.cache_order import cache_order_node
from contextpilot.pipeline.nodes.call_llm import call_llm_node
from contextpilot.pipeline.nodes.log_and_report import log_and_report_node


def route_skip(state: OptimizationState) -> Literal["cache_order", "retrieve"]:
    """Conditional edge after skip check."""
    if state.get("decision") == "skipped":
        return "cache_order"
    return "retrieve"


def route_guard(state: OptimizationState) -> Literal["cache_order", "loosen_budget", "fallback"]:
    """Conditional edge after evidence guard check."""
    evidence_status = state.get("evidence_check", "passed")
    retry_count = state.get("retry_count", 0)

    if evidence_status == "passed":
        return "cache_order"
    elif retry_count < 1:
        return "loosen_budget"
    else:
        return "fallback"


def build_pipeline_graph() -> StateGraph:
    """Build and compile the ContextPilot LangGraph StateGraph."""
    graph = StateGraph(OptimizationState)

    # Add all pipeline nodes
    graph.add_node("skip_check", skip_check_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("dedupe", dedupe_node)
    graph.add_node("select_budget", select_budget_node)
    graph.add_node("sentence_prune", sentence_prune_node)
    graph.add_node("evidence_guard", evidence_guard_node)
    graph.add_node("loosen_budget", loosen_budget_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("cache_order", cache_order_node)
    graph.add_node("call_llm", call_llm_node)
    graph.add_node("log_and_report", log_and_report_node)

    # Set entry point
    graph.set_entry_point("skip_check")

    # Connect nodes with conditional and standard edges
    graph.add_conditional_edges(
        "skip_check",
        route_skip,
        {
            "cache_order": "cache_order",
            "retrieve": "retrieve"
        }
    )
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "dedupe")
    graph.add_edge("dedupe", "select_budget")
    graph.add_edge("select_budget", "sentence_prune")
    graph.add_edge("sentence_prune", "evidence_guard")

    graph.add_conditional_edges(
        "evidence_guard",
        route_guard,
        {
            "cache_order": "cache_order",
            "loosen_budget": "loosen_budget",
            "fallback": "fallback"
        }
    )

    # Retry loop
    graph.add_edge("loosen_budget", "select_budget")

    # Fallback to cache ordering and execution
    graph.add_edge("fallback", "cache_order")
    graph.add_edge("cache_order", "call_llm")
    graph.add_edge("call_llm", "log_and_report")
    graph.add_edge("log_and_report", END)

    return graph


# Pre-compiled runnable pipeline app
_COMPILED_APP = None


def get_pipeline():
    global _COMPILED_APP
    if _COMPILED_APP is None:
        graph = build_pipeline_graph()
        _COMPILED_APP = graph.compile()
    return _COMPILED_APP
