"""FastAPI application entrypoint for ContextPilot OpenAI-compatible proxy."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from contextpilot.app.config import settings
from contextpilot.app.db.session import init_db
from contextpilot.app.routers import chat, requests, benchmarks, collections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("contextpilot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ContextPilot proxy starting up...")
    init_db()
    yield
    logger.info("ContextPilot proxy shutting down...")


app = FastAPI(
    title="ContextPilot API",
    description="OpenAI-compatible proxy that reduces LLM input tokens for RAG applications while preserving quality.",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check Endpoint
@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "service": "ContextPilot",
        "version": "0.1.0",
        "device": settings.DEVICE,
        "default_mode": settings.CONTEXTPILOT_DEFAULT_MODE
    }


# Include Routers
app.include_router(chat.router)
app.include_router(requests.router)
app.include_router(benchmarks.router)
app.include_router(collections.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("contextpilot.app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
