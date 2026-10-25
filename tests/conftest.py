"""PyTest fixtures for RAG evaluation."""
import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any

from pydantic import SecretStr
import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Migration mapping: langchain-community is sunset and archived (issue #674).
# Ragas 0.4.x still attempts legacy imports from langchain_community for VertexAI.
# Provide standalone stub modules dynamically to satisfy the legacy import without triggering
# reportMissingImports, sunset langchain-community deprecation warnings, or ModuleNotFoundError.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _chat_vertexai_mod = types.ModuleType("langchain_community.chat_models.vertexai")
    try:
        _google_vertexai = importlib.import_module("langchain_google_vertexai")
        ChatVertexAI = _google_vertexai.ChatVertexAI
    except (ImportError, AttributeError):
        class ChatVertexAI:
            pass
    _chat_vertexai_mod.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
    sys.modules["langchain_community.chat_models.vertexai"] = _chat_vertexai_mod

if "langchain_community.llms" not in sys.modules:
    _llms_mod = types.ModuleType("langchain_community.llms")
    try:
        _google_vertexai = importlib.import_module("langchain_google_vertexai")
        VertexAI = _google_vertexai.VertexAI
    except (ImportError, AttributeError):
        class VertexAI:
            pass
    _llms_mod.VertexAI = VertexAI  # type: ignore[attr-defined]
    sys.modules["langchain_community.llms"] = _llms_mod


def _get_env_str(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


@pytest.fixture(scope="session")
def llm() -> ChatOpenAI:
    """Configure local Unsloth LLM via ChatOpenAI."""
    return ChatOpenAI(
        base_url=_get_env_str("UNSLOTH_BASE_URL"),
        api_key=SecretStr(_get_env_str("UNSLOTH_API_KEY")),
        model=_get_env_str("UNSLOTH_MODEL"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
    )


@pytest.fixture(scope="session")
def evaluator_llm() -> ChatOpenAI:
    """Configure evaluator LLM via ChatOpenAI."""
    return ChatOpenAI(
        base_url=_get_env_str("UNSLOTH_BASE_URL"),
        api_key=SecretStr(_get_env_str("UNSLOTH_API_KEY")),
        model=_get_env_str("UNSLOTH_MODEL"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
    )


@pytest.fixture(scope="session")
def ragas_config() -> dict[str, Any]:
    """Configuration for Ragas evaluation."""
    return {
        "thresholds": {
            "faithfulness": float(os.getenv("RAGAS_FAITHFULNESS_THRESHOLD", "0.85")),
            "answer_relevance": float(os.getenv("RAGAS_ANSWER_RELEVANCE_THRESHOLD", "0.85")),
            "context_precision": float(os.getenv("RAGAS_CONTEXT_PRECISION_THRESHOLD", "0.0")),
            "context_recall": float(os.getenv("RAGAS_CONTEXT_RECALL_THRESHOLD", "0.0")),
        },
        "run_config": {
            "timeout": int(os.getenv("RAGAS_RUN_CONFIG_TIMEOUT", os.getenv("RAGAS_TIMEOUT", "1200"))),
            "max_workers": int(os.getenv("RAGAS_RUN_CONFIG_MAX_WORKERS", os.getenv("RAGAS_MAX_WORKERS", "1"))),
        },
        "report_path": os.getenv("RAGAS_REPORT_PATH", "tests/reports/ragas_report.html"),
    }
