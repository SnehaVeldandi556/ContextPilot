"""Application configuration via pydantic-settings."""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Supabase / Database URLs
    DATABASE_URL: str = "sqlite:///./contextpilot.db"
    DIRECT_URL: Optional[str] = None
    TEST_DATABASE_URL: Optional[str] = None

    # LLM Providers
    OPENAI_API_KEY: Optional[str] = "mock"
    OPENAI_MODEL: str = "gpt-4o-mini"
    SECOND_LLM_PROVIDER: str = "mock"  # "anthropic", "groq", "mock"
    ANTHROPIC_API_KEY: Optional[str] = None

    # ContextPilot Optimization Defaults
    CONTEXTPILOT_DEFAULT_MODE: str = "guarded"  # "shadow", "guarded", "aggressive"
    CONTEXTPILOT_TOKEN_BUDGET: int = 2000
    CONTEXTPILOT_SKIP_THRESHOLD_TOKENS: int = 2000
    CONTEXTPILOT_DEDUPE_THRESHOLD: float = 0.92
    CONTEXTPILOT_SCORE_DROP_RATIO: float = 0.35
    CONTEXTPILOT_GUARD_THRESHOLD: float = 0.50
    CONTEXTPILOT_AGGRESSIVE_GUARD_THRESHOLD: float = 0.30
    CONTEXTPILOT_SENTENCE_PRUNE_THRESHOLD: float = 0.20
    CONTEXTPILOT_SHADOW_SAMPLE_RATE: float = 1.0
    CONTEXTPILOT_SHADOW_MATCH_THRESHOLD: float = 0.80

    # Model names
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"
    NLI_MODEL_NAME: str = "cross-encoder/nli-deberta-v3-small"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Server settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DASHBOARD_PORT: int = 8501


settings = Settings()
