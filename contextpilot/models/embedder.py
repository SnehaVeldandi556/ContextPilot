"""BAAI/bge-small-en-v1.5 embedding model wrapper and deduplication engine."""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

_GLOBAL_EMBEDDER = None


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


class BGEEmbedder:
    """Embedder using BAAI/bge-small-en-v1.5 (384 dimensions) with deduplication."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or "cpu"
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        import os
        if os.getenv("CONTEXTPILOT_FAST_MODE", "false").lower() in ("1", "true", "yes"):
            logger.info("CONTEXTPILOT_FAST_MODE is set. Using built-in vector embedder.")
            self.model = None
            return

        should_download = os.getenv("CONTEXTPILOT_DOWNLOAD_HF_MODELS", "false").lower() in ("1", "true", "yes")
        is_cached = _is_model_cached(self.model_name)

        if is_cached or should_download:
            try:
                from sentence_transformers import SentenceTransformer
                import torch
                dev = self.device
                if dev == "cuda" and not torch.cuda.is_available():
                    dev = "cpu"
                self.model = SentenceTransformer(self.model_name, device=dev, local_files_only=is_cached)
                logger.info("Loaded SentenceTransformer %s on %s (cached=%s)", self.model_name, dev, is_cached)
                return
            except Exception as e:
                logger.warning("Could not load Hugging Face SentenceTransformer (%s): %s", self.model_name, e)

        logger.info("Using built-in vector embedder.")
        self.model = None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Compute L2-normalized 384-dimensional embeddings for a list of texts."""
        import zlib
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                return np.array(embeddings, dtype=np.float32)
            except Exception as e:
                logger.error("Error in SentenceTransformer encoding: %s", e)

        # High-fidelity deterministic fallback: 384-dim hashed bag-of-words & char 3-grams
        vectors = []
        for text in texts:
            vec = np.zeros(384, dtype=np.float32)
            t_clean = text.lower()
            words = t_clean.split()
            for w in words:
                idx = zlib.crc32(w.encode("utf-8")) % 384
                vec[idx] += 2.0
            # Add character 3-grams
            for i in range(len(t_clean) - 2):
                shingle = t_clean[i:i+3]
                idx = zlib.crc32(shingle.encode("utf-8")) % 384
                vec[idx] += 1.0

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    def deduplicate(
        self,
        scored_passages: List[Tuple[float, Dict[str, Any]]],
        threshold: float = 0.92
    ) -> Tuple[List[Tuple[float, Dict[str, Any]]], List[Dict[str, Any]]]:
        """
        Deduplicate candidate passages based on cosine similarity.
        Keeps the higher-scored copy, records reason 'duplicate of <id>'.
        """
        if not scored_passages:
            return [], []

        # Sort descending by rerank score
        sorted_passages = sorted(scored_passages, key=lambda x: x[0], reverse=True)
        texts = [p.get("text", "") for _, p in sorted_passages]
        embeddings = self.embed_texts(texts)

        kept: List[Tuple[float, Dict[str, Any]]] = []
        kept_indices: List[int] = []
        dropped: List[Dict[str, Any]] = []

        for i, (score, p) in enumerate(sorted_passages):
            is_dup = False
            dup_target_id = ""

            # Check similarity against all already kept passages
            for k_idx in kept_indices:
                sim = float(np.dot(embeddings[i], embeddings[k_idx]))
                if sim >= threshold:
                    is_dup = True
                    dup_target_id = sorted_passages[k_idx][1].get("id", str(k_idx))
                    break

            if is_dup:
                dropped.append({
                    "id": p.get("id", str(i)),
                    "reason": f"duplicate of {dup_target_id}",
                    "rerank_score": score
                })
            else:
                kept.append((score, p))
                kept_indices.append(i)

        return kept, dropped


def get_embedder(model_name: str = "BAAI/bge-small-en-v1.5", device: Optional[str] = None) -> BGEEmbedder:
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is None:
        _GLOBAL_EMBEDDER = BGEEmbedder(model_name=model_name, device=device)
    return _GLOBAL_EMBEDDER
