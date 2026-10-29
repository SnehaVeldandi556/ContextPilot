"""Retrieve node: Retrieves candidate passages from Supabase PGVector via LangChain."""

import time
import logging
from typing import Dict, Any, List
from contextpilot.pipeline.state import OptimizationState

logger = logging.getLogger(__name__)


def retrieve_node(state: OptimizationState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    retrieval_cfg = state.get("retrieval")
    passages = list(state.get("passages", []))
    stage_latencies = dict(state.get("stage_latencies", {}))

    if retrieval_cfg:
        collection_name = retrieval_cfg.get("collection", "default")
        top_k = retrieval_cfg.get("top_k", 30)
        question = state.get("question", "")

        try:
            from contextpilot.app.db.vector_store import get_vector_store
            vs = get_vector_store(collection_name=collection_name)
            docs = vs.similarity_search(question, k=top_k)
            for idx, doc in enumerate(docs):
                p_id = doc.metadata.get("id", f"retrieved_{idx+1}")
                passages.append({
                    "id": str(p_id),
                    "text": doc.page_content,
                    "metadata": doc.metadata
                })
        except Exception as e:
            logger.warning("PGVector retrieval skipped or unavailable: %s", e)

    lat = (time.perf_counter() - t0) * 1000.0
    stage_latencies["retrieve"] = round(lat, 2)
    return {
        "passages": passages,
        "stage_latencies": stage_latencies
    }
