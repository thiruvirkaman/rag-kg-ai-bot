"""Configuration loader. All settings come from environment / .env file."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Data source ---
    source_url: str = Field(
        default="https://www.tn.gov.in/scheme_list.php?dep_id=Mg==",
        description="Department scheme listing URL.",
    )
    source_base: str = Field(default="https://www.tn.gov.in")
    department_name: str = Field(default="Agriculture - Farmers Welfare Department")
    department_id: str = Field(default="Mg==")

    # --- Paths ---
    data_dir: str = Field(default="data")
    raw_cache: str = Field(default="data/schemes_raw.json")
    chunks_cache: str = Field(default="data/schemes_chunks.json")

    # --- Neo4j ---
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")
    neo4j_database: str = Field(default="neo4j")

    # --- LLM (OpenAI-compatible, works with Ollama Cloud / LM Studio / OpenAI) ---
    llm_base_url: str = Field(default="https://openai.api.vikram.cloud/v1")
    llm_api_key: str = Field(default="ollama")
    llm_chat_model: str = Field(default="gpt-4o-mini")
    llm_extract_model: str = Field(default="gpt-4o-mini")
    llm_temperature: float = Field(default=0.1)
    llm_timeout: float = Field(default=60.0)

    # Embeddings (OpenAI-compatible, used for vector index in Neo4j)
    embed_base_url: str = Field(default="")
    embed_api_key: str = Field(default="")
    embed_model: str = Field(default="text-embedding-3-small")
    embed_dimensions: int = Field(default=1536)

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=6)
    retrieval_hops: int = Field(default=2)

    # --- App ---
    app_title: str = Field(default="TN Farmers Schemes - Knowledge Graph Assistant")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
