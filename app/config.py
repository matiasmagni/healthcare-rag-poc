"""Application configuration using pydantic-settings."""
import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration for the healthcare RAG application."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    base_dir: Path = PROJECT_ROOT
    index_dir_name: str = os.getenv("INDEX_DIR_NAME", "faiss_index")
    pdf_subdir: str = os.getenv("PDF_SUBDIR", "input-docs")

    chunk_size: int = Field(default=int(os.getenv("CHUNK_SIZE", "500")), gt=0)
    chunk_overlap: int = Field(default=int(os.getenv("CHUNK_OVERLAP", "50")), ge=0)
    retrieval_k: int = Field(default=int(os.getenv("RETRIEVAL_K", "15")), gt=0)

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")

    unsloth_base_url: str = os.getenv("UNSLOTH_BASE_URL", "http://localhost:8888/v1")
    unsloth_api_key: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("UNSLOTH_API_KEY", "")))
    unsloth_model: str = os.getenv("UNSLOTH_MODEL", "unsloth/gemma-3-4b-it-GGUF:BF16")
    llm_temperature: float = Field(default=float(os.getenv("LLM_TEMPERATURE", "0.0")), ge=0.0, le=2.0)
    llm_timeout: float = Field(default=float(os.getenv("LLM_TIMEOUT", "120.0")), gt=0)
    llm_max_retries: int = Field(default=int(os.getenv("LLM_MAX_RETRIES", "2")), ge=0)
    llm_enable_thinking: bool = os.getenv("LLM_ENABLE_THINKING", "false").lower() in {"true", "1", "yes", "on"}

    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        """Ensure chunk overlap is smaller than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        
        return self

    @property
    def index_path(self) -> Path:
        """Path to the persisted FAISS index directory."""
        return self.base_dir / self.index_dir_name

    @property
    def pdf_search_dirs(self) -> tuple[Path, ...]:
        """Directories to search for source PDFs, in priority order."""
        directories = [self.base_dir]
        if self.pdf_subdir:
            pdf_directory = self.base_dir / self.pdf_subdir
            if pdf_directory != self.base_dir:
                directories.append(pdf_directory)
        
        return tuple(directories)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton)."""
    return Settings()
