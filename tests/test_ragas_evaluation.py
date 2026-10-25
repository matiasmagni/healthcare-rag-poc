"""Ragas evaluation tests for RAG pipeline."""
import json
import os
from pathlib import Path
from typing import Any, cast

import pandas as pd
from pydantic import SecretStr
import pytest
from datasets import Dataset
from dotenv import load_dotenv

from tests.generate_html_report import generate_html_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@pytest.fixture(scope="module")
def evaluation_dataset():
    """Load evaluation dataset."""
    dataset_path = Path(__file__).parent / "dataset.json"
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def _get_env_str(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def test_ragas_evaluation(evaluation_dataset, ragas_config):
    """Evaluate RAG pipeline using Ragas metrics."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics._answer_relevance import answer_relevancy
    from ragas.metrics._context_precision import context_precision
    from ragas.metrics._context_recall import context_recall
    from ragas.metrics._faithfulness import faithfulness
    from ragas.run_config import RunConfig
    
    # Configure LangChain OpenAI model pointing to local Unsloth Studio with expanded max_tokens
    openai_chat = ChatOpenAI(
        base_url=_get_env_str("UNSLOTH_BASE_URL"),
        api_key=SecretStr(_get_env_str("UNSLOTH_API_KEY")),
        model=_get_env_str("UNSLOTH_EVAL_MODEL", _get_env_str("UNSLOTH_MODEL").split(":")[0]),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        max_completion_tokens=int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "4096")),
    )
    ragas_llm = LangchainLLMWrapper(openai_chat)
    
    # Configure HuggingFace embeddings wrapped for Ragas compatibility
    hf_embeddings = HuggingFaceEmbeddings(model_name=_get_env_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
    
    # Run evaluation
    run_config = ragas_config.get("run_config", {})
    timeout = int(run_config.get("timeout", os.getenv("RAGAS_RUN_CONFIG_TIMEOUT", os.getenv("RAGAS_TIMEOUT", "1200"))))
    max_workers = int(run_config.get("max_workers", os.getenv("RAGAS_RUN_CONFIG_MAX_WORKERS", os.getenv("RAGAS_MAX_WORKERS", "1"))))
    result = evaluate(
        evaluation_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=cast(Any, ragas_llm),
        embeddings=ragas_embeddings,
        run_config=RunConfig(
            timeout=timeout,
            max_workers=max_workers,
        ),
        raise_exceptions=True,
    )
    
    # Convert to pandas DataFrame for easier access
    df: pd.DataFrame
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()  # type: ignore[attr-defined]
    elif isinstance(result, pd.DataFrame):
        df = result
    else:
        df = pd.DataFrame(result)  # type: ignore[arg-type]

    if "answer_relevance" not in df.columns and "answer_relevancy" in df.columns:
        df["answer_relevance"] = df["answer_relevancy"]
    
    # Generate HTML report
    generate_html_report(df)
    
    # Calculate averages
    thresholds = ragas_config["thresholds"]
    
    faith_avg = float(df["faithfulness"].mean())  # type: ignore
    answer_rel_avg = float(df["answer_relevance"].mean())  # type: ignore
    context_prec_avg = float(df["context_precision"].mean())  # type: ignore
    context_rec_avg = float(df["context_recall"].mean())  # type: ignore
    
    print("\n=== RAGAS Evaluation Results ===")
    print(f"Faithfulness: {faith_avg:.4f} (threshold: {thresholds['faithfulness']})")
    print(f"Answer Relevance: {answer_rel_avg:.4f} (threshold: {thresholds['answer_relevance']})")
    print(f"Context Precision: {context_prec_avg:.4f} (threshold: {thresholds['context_precision']})")
    print(f"Context Recall: {context_rec_avg:.4f} (threshold: {thresholds['context_recall']})")
    print(f"\nReport generated at: {ragas_config.get('report_path', 'tests/reports/ragas_report.html')}")
    
    # Assertions for critical metrics
    assert faith_avg >= thresholds["faithfulness"], (
        f"Faithfulness {faith_avg:.4f} below threshold {thresholds['faithfulness']}"
    )
    assert answer_rel_avg >= thresholds["answer_relevance"], (
        f"Answer Relevance {answer_rel_avg:.4f} below threshold {thresholds['answer_relevance']}"
    )