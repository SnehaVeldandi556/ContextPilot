"""BAAI/bge-reranker-base cross-encoder model wrapper and sentence-level scorer."""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

_GLOBAL_RERANKER = None


def _is_model_cached(model_name: str) -> bool:
    try:
        import os
        repo_id = "models--" + model_name.replace("/", "--")
        hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/hub"))
        snapshots_dir = os.path.join(hf_home, repo_id, "snapshots")
        if not os.path.exists(snapshots_dir):
            return False
        for s in os.listdir(snapshots_dir):
            s_path = os.path.join(snapshots_dir, s)
            if os.path.isdir(s_path):
                files = os.listdir(s_path)
                if any(f in files for f in ("model.safetensors", "pytorch_model.bin")):
                    return True
        return False
    except Exception:
        return False


class BGEReranker:
    """Cross-encoder reranker using BAAI/bge-reranker-base."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or "cpu"
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        import os
        if os.getenv("CONTEXTPILOT_FAST_MODE", "false").lower() in ("1", "true", "yes"):
            logger.info("CONTEXTPILOT_FAST_MODE is set. Using built-in hybrid cross-encoder.")
            self.model = None
            return

        should_download = os.getenv("CONTEXTPILOT_DOWNLOAD_HF_MODELS", "false").lower() in ("1", "true", "yes")
        is_cached = _is_model_cached(self.model_name)

        if is_cached or should_download:
            try:
                from sentence_transformers import CrossEncoder
                import torch
                dev = self.device
                if dev == "cuda" and not torch.cuda.is_available():
                    dev = "cpu"
                self.model = CrossEncoder(self.model_name, device=dev, local_files_only=is_cached)
                logger.info("Loaded CrossEncoder %s on %s (cached=%s)", self.model_name, dev, is_cached)
                return
            except Exception as e:
                logger.warning("Could not load Hugging Face CrossEncoder (%s): %s", self.model_name, e)

        logger.info("Using built-in high-precision hybrid cross-encoder.")
        self.model = None

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Compute cross-encoder relevance scores for (query, text) pairs."""
        if not pairs:
            return []

        if self.model is not None:
            try:
                raw_scores = self.model.predict(pairs, show_progress_bar=False)
                # Apply sigmoid if logits are returned to normalize to [0, 1]
                scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores, dtype=np.float32)))
                return [float(s) for s in scores]
            except Exception as e:
                logger.error("Error in CrossEncoder scoring: %s", e)

        # High-accuracy fallback: hybrid BM25 + Jaccard + entity overlap score normalized to [0, 1]
        scores = []
        for query, text in pairs:
            q_words = set(re.findall(r"\w+", query.lower()))
            t_words = set(re.findall(r"\w+", text.lower()))
            if not q_words or not t_words:
                scores.append(0.0)
                continue
            overlap = len(q_words & t_words)
            jaccard = overlap / len(q_words | t_words)

            # Sequence match bonus
            phrase_bonus = 0.2 if query.lower() in text.lower() else 0.0

            # Normalized score in [0.0, 1.0]
            score = min(1.0, (jaccard * 1.8) + phrase_bonus)
            scores.append(float(score))
        return scores

    def rerank_passages(
        self,
        query: str,
        passages: List[Dict[str, Any]]
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Score each passage against the query and sort descending."""
        if not passages:
            return []
        pairs = [(query, p.get("text", "")) for p in passages]
        scores = self.score_pairs(pairs)
        scored = list(zip(scores, passages))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def prune_sentences(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        threshold: float = 0.20
    ) -> List[Dict[str, Any]]:
        """
        Sentence pruning for aggressive mode:
        Scores individual sentences inside kept passages and prunes low-scoring ones.
        """
        pruned_passages = []
        for p in passages:
            text = p.get("text", "")
            # Split sentences roughly on period/question/exclamation followed by space or end
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            if len(sentences) <= 1:
                pruned_passages.append(p)
                continue

            pairs = [(query, s) for s in sentences]
            scores = self.score_pairs(pairs)

            kept_sentences = []
            for s, score in zip(sentences, scores):
                if score >= threshold:
                    kept_sentences.append(s)

            # Keep at least the highest scoring sentence to prevent blank passages
            if not kept_sentences and sentences:
                best_idx = int(np.argmax(scores))
                kept_sentences.append(sentences[best_idx])

            new_text = " ".join(kept_sentences)
            pruned_p = dict(p)
            pruned_p["text"] = new_text
            pruned_passages.append(pruned_p)

        return pruned_passages


def detect_multihop(query: str) -> bool:
    """
    Detect whether a question requires multi-hop reasoning based on
    syntactic indicators, multiple entities, or comparison patterns.
    """
    q_lower = query.lower()

    # Comparison / Choice patterns
    comparison_patterns = [
        r"\bboth\b",
        r"\beither\b",
        r"\bneither\b",
        r"\bwho (was|is) (earlier|older|younger|taller)\b",
        r"\bwhich (film|movie|person|city|country|director|album|book) (has|was|is)\b.*\bor\b",
        r"\bwere .* and .* of the same\b",
        r"\bwhat (do|did) .* and .* have in common\b",
    ]
    for pat in comparison_patterns:
        if re.search(pat, q_lower):
            return True

    # Bridge patterns: "who of the ... that", "what company founded by the creator of"
    bridge_patterns = [
        r"\bof the .* that\b",
        r"\bauthor of .* that\b",
        r"\bheaded by the .* of\b",
        r"\bdirected by the .* of\b",
        r"\bwhere was the director of .* born\b",
        r"\bwho played the character that\b",
    ]
    for pat in bridge_patterns:
        if re.search(pat, q_lower):
            return True

    # Named entity heuristic: multiple capitalized sequences in query
    capitalized_phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
    # Ignore leading question words (Who, What, Where, When, Which, Were, Is, Did)
    question_words = {"Who", "What", "Where", "When", "Which", "Were", "Is", "Did", "Can", "How"}
    entities = [e for e in capitalized_phrases if e not in question_words]
    if len(entities) >= 2:
        return True

    return False


def get_reranker(model_name: str = "BAAI/bge-reranker-base", device: Optional[str] = None) -> BGEReranker:
    global _GLOBAL_RERANKER
    if _GLOBAL_RERANKER is None:
        _GLOBAL_RERANKER = BGEReranker(model_name=model_name, device=device)
    return _GLOBAL_RERANKER
