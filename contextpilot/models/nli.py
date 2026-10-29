"""NLI model wrapper for the ContextPilot Evidence Guard."""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

_GLOBAL_NLI = None


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


class NLIEvidenceGuard:
    """NLI cross-encoder verifying premise entails/supports hypothesis."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or "cpu"
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        import os
        if os.getenv("CONTEXTPILOT_FAST_MODE", "false").lower() in ("1", "true", "yes"):
            logger.info("CONTEXTPILOT_FAST_MODE is set. Using built-in semantic overlap NLI guard.")
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
                logger.info("Loaded CrossEncoder NLI %s on %s (cached=%s)", self.model_name, dev, is_cached)
                return
            except Exception as e:
                logger.warning("Could not load Hugging Face CrossEncoder NLI (%s): %s", self.model_name, e)

        logger.info("Using built-in NLI evidence guard.")
        self.model = None

    def check_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Compute entailment probability that premise entails hypothesis.
        Returns a float in [0.0, 1.0].
        """
        if not premise or not hypothesis:
            return 0.0

        if self.model is not None:
            try:
                # NLI models usually output 3 logits: [contradiction, neutral, entailment]
                # or [entailment, neutral, contradiction]
                logits = self.model.predict([(premise, hypothesis)], show_progress_bar=False)[0]
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)

                # Look up model config label mapping if available
                entailment_idx = 1
                if hasattr(self.model, "config") and hasattr(self.model.config, "label2id"):
                    l2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
                    entailment_idx = l2id.get("entailment", 1)
                elif len(probs) == 3:
                    # DeBERTa-v3 NLI standard ordering: 0: contradiction, 1: entailment, 2: neutral (or 2: entailment)
                    entailment_idx = 1 if len(probs) == 3 else 0

                return float(probs[entailment_idx])
            except Exception as e:
                logger.error("Error in NLI inference: %s", e)

        # High-precision fallback: lexical and named-entity entailment check
        # An answer is entailed if its key informational entities and tokens are supported in premise
        h_tokens = [tok.lower() for tok in re.findall(r"\w+", hypothesis) if len(tok) > 2]
        if not h_tokens:
            return 1.0

        p_lower = premise.lower()
        supported_tokens = sum(1 for tok in h_tokens if tok in p_lower)
        coverage = supported_tokens / len(h_tokens)

        # Key entity check
        entities = re.findall(r"\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*\b", hypothesis)
        if entities:
            entity_supported = sum(1 for ent in entities if ent.lower() in p_lower)
            entity_ratio = entity_supported / len(entities)
            return float(0.6 * coverage + 0.4 * entity_ratio)

        return float(coverage)

    def verify_kept_context(
        self,
        question: str,
        kept_passages: List[Dict[str, Any]],
        draft_answer: str,
        threshold: float = 0.50
    ) -> Tuple[bool, float]:
        """
        Verify that kept passages provide sufficient premise to entail the draft answer.
        Returns (passed: bool, entailment_score: float).
        """
        if not kept_passages:
            return False, 0.0

        premise = " ".join([p.get("text", "") for p in kept_passages])
        hypothesis = draft_answer if draft_answer else question

        score = self.check_entailment(premise, hypothesis)
        passed = bool(score >= threshold)
        return passed, round(score, 4)


def get_nli_guard(model_name: str = "cross-encoder/nli-deberta-v3-small", device: Optional[str] = None) -> NLIEvidenceGuard:
    global _GLOBAL_NLI
    if _GLOBAL_NLI is None:
        _GLOBAL_NLI = NLIEvidenceGuard(model_name=model_name, device=device)
    return _GLOBAL_NLI
