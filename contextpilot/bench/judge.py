"""Evaluation metrics and judge for benchmarks: EM, Token F1, Evidence Recall, LLM Judge, and Bootstrap CIs."""

import re
import string
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Exact Match (EM) metric."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common: Dict[str, int] = {}
    for tok in pred_tokens:
        if tok in gold_tokens:
            common[tok] = common.get(tok, 0) + 1

    overlap = sum(min(pred_tokens.count(tok), gold_tokens.count(tok)) for tok in set(pred_tokens) & set(gold_tokens))
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def evidence_recall_score(kept_ids: List[str], gold_supporting_ids: List[str]) -> float:
    """
    Evidence recall: fraction of gold supporting passage IDs retained in the kept context.
    Returns 1.0 if all supporting passages are kept, or the proportion kept.
    """
    if not gold_supporting_ids:
        return 1.0
    kept_set = set(kept_ids)
    hits = sum(1 for g in gold_supporting_ids if g in kept_set)
    return hits / float(len(gold_supporting_ids))


def compute_bootstrap_ci(
    values: List[float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute mean and empirical bootstrap confidence interval (ci_low, ci_high).
    Returns (mean, ci_low, ci_high).
    """
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    mean_val = float(np.mean(arr))
    if len(arr) == 1:
        return mean_val, mean_val, mean_val

    rng = np.random.RandomState(seed)
    boot_means = []
    n = len(arr)
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))

    alpha = 1.0 - confidence_level
    ci_low = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return round(mean_val, 4), round(ci_low, 4), round(ci_high, 4)


class BenchmarkJudge:
    """Evaluates question answers and context retention across benchmarks."""

    def __init__(self, llm_provider: Optional[Any] = None, sample_rate: float = 0.1):
        self.llm_provider = llm_provider
        self.sample_rate = sample_rate

    def evaluate_sample(
        self,
        prediction: str,
        gold_answer: str,
        kept_passage_ids: List[str],
        gold_supporting_ids: List[str],
        question: Optional[str] = None
    ) -> Dict[str, float]:
        """Compute all benchmark metrics for a single sample."""
        em = exact_match_score(prediction, gold_answer)
        f1 = f1_score(prediction, gold_answer)
        ev_recall = evidence_recall_score(kept_passage_ids, gold_supporting_ids)

        return {
            "em": em,
            "f1": f1,
            "evidence_recall": ev_recall,
        }

    def judge_answer_alignment(self, answer_a: str, answer_b: str) -> float:
        """
        Fast token/semantic overlap score for shadow mode comparison.
        Returns a score in [0.0, 1.0].
        """
        return f1_score(answer_a, answer_b)
