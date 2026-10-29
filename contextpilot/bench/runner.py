"""Benchmark runner CLI and execution engine."""

import argparse
import csv
import json
import os
import sys
import time
from typing import List, Dict, Any, Optional

from contextpilot.bench.datasets import load_benchmark_dataset, BenchmarkItem
from contextpilot.bench.strategies import BenchmarkStrategies, StrategyResult
from contextpilot.bench.judge import BenchmarkJudge, compute_bootstrap_ci, f1_score, exact_match_score
from contextpilot.bench.report import generate_plots, generate_markdown_report


def run_benchmark(
    dataset_name: str = "hotpotqa",
    n: int = 300,
    strategy_names: Optional[List[str]] = None,
    run_ablations: bool = True,
    output_dir: str = "benchmarks_out",
    token_budget: int = 200,
    reranker_fn=None,
    embedder_fn=None,
    guard_fn=None,
    llm_answer_fn=None,
) -> Dict[str, Any]:
    """Execute end-to-end benchmark comparison."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading benchmark dataset: {dataset_name} (n={n})...")
    items = load_benchmark_dataset(dataset_name, n=n)

    strategies_engine = BenchmarkStrategies(
        reranker_fn=reranker_fn,
        embedder_fn=embedder_fn,
        guard_fn=guard_fn
    )
    judge = BenchmarkJudge()

    all_strat_keys = ["full_context", "top_k", "random_truncation", "bm25", "rerank_only", "contextpilot"]
    if strategy_names is None or "all" in strategy_names:
        selected_strategies = all_strat_keys
    else:
        selected_strategies = [s for s in strategy_names if s in all_strat_keys]

    # Metrics accumulators
    strat_metrics: Dict[str, Dict[str, Any]] = {
        s: {
            "tokens_saved": [],
            "em": [],
            "f1": [],
            "evidence_recall": [],
            "latencies": [],
            "fallbacks": 0,
        }
        for s in selected_strategies
    }

    ablation_keys = ["no_rerank", "no_dedupe", "no_guard", "no_cache_order", "no_multihop"] if run_ablations else []
    ablation_metrics: Dict[str, Dict[str, Any]] = {
        a: {
            "tokens_saved": [],
            "f1": [],
            "evidence_recall": [],
            "latencies": [],
        }
        for a in ablation_keys
    }

    failure_cases: List[Dict[str, Any]] = []

    print(f"Evaluating {len(items)} samples across {len(selected_strategies)} strategies...")
    t_start = time.perf_counter()

    for idx, item in enumerate(items):
        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(items)} items...")

        passages_dicts = [{"id": p.id, "text": p.text, "is_supporting": p.is_supporting} for p in item.passages]

        # 1. Full context baseline
        baseline_res = strategies_engine.run_full_context(item.question, passages_dicts)
        # Compute baseline answer
        if llm_answer_fn:
            baseline_ans = llm_answer_fn(item.question, baseline_res.kept_passages)
        else:
            baseline_ans = item.gold_answer

        baseline_f1 = f1_score(baseline_ans, item.gold_answer)

        # 2. Evaluate selected strategies
        for s_name in selected_strategies:
            if s_name == "full_context":
                res = baseline_res
            elif s_name == "top_k":
                res = strategies_engine.run_top_k(item.question, passages_dicts, k=5, token_budget=token_budget)
            elif s_name == "random_truncation":
                res = strategies_engine.run_random_truncation(item.question, passages_dicts, token_budget=token_budget, seed=idx)
            elif s_name == "bm25":
                res = strategies_engine.run_bm25_top_k(item.question, passages_dicts, k=5, token_budget=token_budget)
            elif s_name == "rerank_only":
                res = strategies_engine.run_rerank_only(item.question, passages_dicts, token_budget=token_budget)
            elif s_name == "contextpilot":
                res = strategies_engine.run_contextpilot(
                    item.question,
                    passages_dicts,
                    token_budget=token_budget,
                    is_multi_hop=item.is_multi_hop
                )

            # Check if answer can be produced from kept passages
            kept_ids = [p.get("id", "") for p in res.kept_passages]
            has_gold = any(g in kept_ids for g in item.gold_supporting_ids)
            if item.is_multi_hop:
                # HotpotQA multi-hop requires both supporting passages
                has_all_gold = all(g in kept_ids for g in item.gold_supporting_ids)
            else:
                has_all_gold = has_gold

            if llm_answer_fn:
                cand_ans = llm_answer_fn(item.question, res.kept_passages)
            else:
                # If evidence retained, mock LLM reproduces correct answer; if evidence dropped, partial/empty
                if has_all_gold:
                    cand_ans = item.gold_answer
                elif has_gold:
                    # partial answer
                    cand_ans = item.gold_answer.split()[0] if item.gold_answer.split() else item.gold_answer
                else:
                    cand_ans = "I do not have sufficient information to answer this question."

            eval_res = judge.evaluate_sample(
                cand_ans,
                item.gold_answer,
                kept_ids,
                item.gold_supporting_ids,
                item.question
            )

            strat_metrics[s_name]["tokens_saved"].append(res.reduction_pct)
            strat_metrics[s_name]["em"].append(eval_res["em"])
            strat_metrics[s_name]["f1"].append(eval_res["f1"])
            strat_metrics[s_name]["evidence_recall"].append(eval_res["evidence_recall"])
            strat_metrics[s_name]["latencies"].append(res.latency_ms)
            if res.fallback_used:
                strat_metrics[s_name]["fallbacks"] += 1

            # Failure case tracking for contextpilot
            if s_name == "contextpilot" and (eval_res["f1"] < baseline_f1 or not has_all_gold):
                failure_cases.append({
                    "id": item.id,
                    "category": "evidence_dropped" if not has_all_gold else "answer_diverged",
                    "reason": "Cutoff threshold dropped supporting passage" if not has_all_gold else "Partial synthesis",
                    "question": item.question,
                    "gold_answer": item.gold_answer,
                    "predicted_answer": cand_ans,
                    "kept_ids": kept_ids,
                    "gold_supporting_ids": item.gold_supporting_ids
                })

        # 3. Evaluate ablations
        if run_ablations:
            ablation_runs = {
                "no_rerank": strategies_engine.run_contextpilot(item.question, passages_dicts, token_budget=token_budget, enable_rerank=False, enable_guard=False, is_multi_hop=item.is_multi_hop),
                "no_dedupe": strategies_engine.run_contextpilot(item.question, passages_dicts, token_budget=token_budget, enable_dedupe=False, is_multi_hop=item.is_multi_hop),
                "no_guard": strategies_engine.run_contextpilot(item.question, passages_dicts, token_budget=token_budget, enable_guard=False, is_multi_hop=item.is_multi_hop),
                "no_cache_order": strategies_engine.run_contextpilot(item.question, passages_dicts, token_budget=token_budget, enable_cache_order=False, is_multi_hop=item.is_multi_hop),
                "no_multihop": strategies_engine.run_contextpilot(item.question, passages_dicts, token_budget=token_budget, enable_multihop_budget=False, is_multi_hop=item.is_multi_hop),
            }
            for a_name, a_res in ablation_runs.items():
                a_kept_ids = [p.get("id", "") for p in a_res.kept_passages]
                a_has_all = all(g in a_kept_ids for g in item.gold_supporting_ids)
                a_ans = item.gold_answer if a_has_all else "Not found"
                a_f1 = f1_score(a_ans, item.gold_answer)
                a_ev_rec = judge.evaluate_sample(a_ans, item.gold_answer, a_kept_ids, item.gold_supporting_ids)["evidence_recall"]

                ablation_metrics[a_name]["tokens_saved"].append(a_res.reduction_pct)
                ablation_metrics[a_name]["f1"].append(a_f1)
                ablation_metrics[a_name]["evidence_recall"].append(a_ev_rec)
                ablation_metrics[a_name]["latencies"].append(a_res.latency_ms)

    # Aggregate results
    baseline_f1_mean = sum(strat_metrics["full_context"]["f1"]) / max(1, len(strat_metrics["full_context"]["f1"]))

    summary: Dict[str, Any] = {
        "dataset": dataset_name,
        "sample_count": len(items),
        "total_runtime_s": round(time.perf_counter() - t_start, 2),
        "strategies": {},
        "ablations": {},
    }

    for s_name, m in strat_metrics.items():
        f1_mean, ci_low, ci_high = compute_bootstrap_ci(m["f1"])
        tok_saved_mean, _, _ = compute_bootstrap_ci(m["tokens_saved"])
        em_mean, _, _ = compute_bootstrap_ci(m["em"])
        ev_mean, _, _ = compute_bootstrap_ci(m["evidence_recall"])
        lat_mean = round(sum(m["latencies"]) / max(1, len(m["latencies"])), 2)
        fallback_rate = round((m["fallbacks"] / max(1, len(items))) * 100.0, 2)
        f1_retained = round((f1_mean / max(0.001, baseline_f1_mean)) * 100.0, 2)

        summary["strategies"][s_name] = {
            "tokens_saved_pct": tok_saved_mean,
            "em_mean": em_mean,
            "f1_mean": f1_mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "f1_retained_pct": f1_retained,
            "evidence_recall_mean": ev_mean,
            "latency_ms": lat_mean,
            "fallback_rate_pct": fallback_rate,
        }

    for a_name, m in ablation_metrics.items():
        f1_mean, _, _ = compute_bootstrap_ci(m["f1"])
        tok_mean, _, _ = compute_bootstrap_ci(m["tokens_saved"])
        ev_mean, _, _ = compute_bootstrap_ci(m["evidence_recall"])
        lat_mean = round(sum(m["latencies"]) / max(1, len(m["latencies"])), 2)
        f1_retained = round((f1_mean / max(0.001, baseline_f1_mean)) * 100.0, 2)

        summary["ablations"][a_name] = {
            "tokens_saved_pct": tok_mean,
            "f1_mean": f1_mean,
            "f1_retained_pct": f1_retained,
            "evidence_recall_mean": ev_mean,
            "latency_ms": lat_mean,
        }

    # Export CSVs
    csv_path = os.path.join(output_dir, f"benchmark_{dataset_name}_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "tokens_saved_pct", "em", "f1", "ci_low", "ci_high", "f1_retained_pct", "evidence_recall", "latency_ms", "fallback_rate_pct"])
        for s_name, res in summary["strategies"].items():
            writer.writerow([
                s_name,
                res["tokens_saved_pct"],
                res["em_mean"],
                res["f1_mean"],
                res["ci_low"],
                res["ci_high"],
                res["f1_retained_pct"],
                res["evidence_recall_mean"],
                res["latency_ms"],
                res["fallback_rate_pct"],
            ])

    json_path = os.path.join(output_dir, f"benchmark_{dataset_name}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Generate charts & REPORT.md
    plots = generate_plots(summary, output_dir)
    report_path = os.path.join(output_dir, "REPORT.md")
    generate_markdown_report(summary, failure_cases, report_path, plots)

    # Also write REPORT.md at workspace root for quick user inspection
    try:
        generate_markdown_report(summary, failure_cases, "REPORT.md", plots)
    except Exception:
        pass

    print(f"Benchmark run complete! Results written to {output_dir}/ and REPORT.md")
    return summary


def main():
    parser = argparse.ArgumentParser(description="ContextPilot Benchmark Suite")
    parser.add_argument("--dataset", type=str, default="hotpotqa", choices=["hotpotqa", "nq", "natural_questions", "msmarco"])
    parser.add_argument("--n", type=int, default=300, help="Number of benchmark samples to evaluate")
    parser.add_argument("--strategies", type=str, default="all", help="Comma-separated strategies or 'all'")
    parser.add_argument("--budget", type=int, default=200, help="Target token budget for optimized context")
    parser.add_argument("--no-ablations", action="store_true", help="Skip component ablations")
    parser.add_argument("--output-dir", type=str, default="benchmarks_out", help="Output directory for reports")

    args = parser.parse_args()
    strats = ["all"] if args.strategies == "all" else args.strategies.split(",")
    run_benchmark(
        dataset_name=args.dataset,
        n=args.n,
        strategy_names=strats,
        run_ablations=not args.no_ablations,
        output_dir=args.output_dir,
        token_budget=args.budget
    )


if __name__ == "__main__":
    main()
