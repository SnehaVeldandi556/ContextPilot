"""LangChain PGVector vector store integration on Supabase Postgres."""

import os
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import Embeddings

from contextpilot.app.config import settings
from contextpilot.app.db.session import get_db_url
from contextpilot.models.embedder import get_embedder

logger = logging.getLogger(__name__)


class BGEEmbeddingsAdapter(Embeddings):
    """LangChain Embeddings interface adapter for BAAI/bge-small-en-v1.5."""

    def __init__(self):
        self.embedder = get_embedder()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        arr = self.embedder.embed_texts(texts)
        return arr.tolist()

    def embed_query(self, text: str) -> List[float]:
        arr = self.embedder.embed_texts([text])
        return arr[0].tolist()


_COLLECTIONS: Dict[str, Any] = {}


def get_vector_store(collection_name: str = "default") -> Any:
    """
    Get or create PGVector vector store on Supabase.
    If database is SQLite or PGVector extension not active, falls back to in-memory store.
    """
    global _COLLECTIONS
    if collection_name in _COLLECTIONS:
        return _COLLECTIONS[collection_name]

    db_url = get_db_url()
    embeddings = BGEEmbeddingsAdapter()

    if db_url.startswith("postgresql"):
        try:
            vs = PGVector(
                connection_string=db_url,
                embedding_function=embeddings,
                collection_name=collection_name,
                use_jsonb=True,
            )
            _COLLECTIONS[collection_name] = vs
            return vs
        except Exception as e:
            logger.warning("Could not initialize Supabase PGVector store (%s). Using InMemoryVectorStore: %s", collection_name, e)

    # In-memory vector store fallback
    vs = InMemoryVectorStore(embedding=embeddings)
    _COLLECTIONS[collection_name] = vs
    return vs


def ingest_documents(
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    collection_name: str = "default",
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> int:
    """
    Ingest raw texts into vector store using RecursiveCharacterTextSplitter.
    Returns count of chunks stored.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []
    for idx, text in enumerate(texts):
        meta = metadatas[idx] if (metadatas and idx < len(metadatas)) else {"source": f"doc_{idx+1}"}
        chunks = splitter.split_text(text)
        for c_idx, chunk in enumerate(chunks):
            chunk_meta = dict(meta)
            chunk_meta["chunk_id"] = f"{meta.get('source', 'doc')}_{c_idx+1}"
            docs.append(Document(page_content=chunk, metadata=chunk_meta))

    vs = get_vector_store(collection_name=collection_name)
    vs.add_documents(docs)
    return len(docs)
