"""Factory functions for creating embeddings and chat models."""
from langchain_openai import ChatOpenAI

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import Settings


def create_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    """Create a HuggingFace embeddings instance from settings.

    Args:
        settings: Application settings.

    Returns:
        Configured HuggingFaceEmbeddings instance.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_chat_model(settings: Settings) -> ChatOpenAI:
    """Create a ChatOpenAI instance pointing to Unsloth Studio.

    Args:
        settings: Application settings.

    Returns:
        Configured ChatOpenAI instance.
    """
    return ChatOpenAI(
        base_url=settings.unsloth_base_url,
        api_key=settings.unsloth_api_key.get_secret_value(),
        model=settings.unsloth_model,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": settings.llm_enable_thinking,
            }
        },
    )
