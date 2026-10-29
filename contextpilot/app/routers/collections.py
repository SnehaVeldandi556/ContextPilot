"""Collections and document ingestion router for PGVector."""

import logging
from fastapi import APIRouter, HTTPException
from contextpilot.app.schemas import IngestRequest, IngestResponse
from contextpilot.app.db.vector_store import ingest_documents

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["Collections"])


@router.post("/collections/{name}/ingest", response_model=IngestResponse)
def ingest_into_collection(name: str, request: IngestRequest):
    """Ingest raw texts into a Supabase PGVector collection using LangChain text splitters."""
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty")

    try:
        count = ingest_documents(
            texts=request.texts,
            metadatas=request.metadatas,
            collection_name=name,
            chunk_size=request.chunk_size or 500,
            chunk_overlap=request.chunk_overlap or 50
        )
        return IngestResponse(
            collection=name,
            chunks_ingested=count,
            status="success"
        )
    except Exception as e:
        logger.error("Error ingesting into collection %s: %s", name, e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
