"""API route definitions."""
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_app_settings, get_rag_service
from app.config import Settings
from app.schemas import ContextChunk, HealthResponse, QueryRequest, QueryResponse
from app.services.rag import RAGService

logger = logging.getLogger(__name__)

router = APIRouter()

RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]


@router.get("/health", response_model=HealthResponse)
def health(service: RAGServiceDep, settings: SettingsDep) -> HealthResponse:
    """Health check endpoint returning index and model configuration."""
    return HealthResponse(
        status="ok",
        indexed_chunks=service.chunk_count,
        embedding_model=settings.embedding_model,
        llm_model=settings.unsloth_model,
    )


@router.post("/query", response_model=QueryResponse)
@router.post("/endpoint", response_model=QueryResponse, include_in_schema=False)
def query(payload: QueryRequest, service: RAGServiceDep) -> QueryResponse:
    """Answer a question using the indexed documents.

    Args:
        payload: Query request containing the question.
        service: Injected RAG service instance.

    Returns:
        Answer and retrieved context chunks.

    Raises:
        HTTPException: 400 if question is blank, 502 if LLM/inference fails.
    """
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be blank.")

    try:
        result = service.answer(question)
    except Exception as exc:
        logger.exception("Failed to answer question")
        raise HTTPException(
            status_code=502,
            detail="The RAG request failed. Check the local Unsloth server and server logs.",
        ) from exc

    return QueryResponse(
        answer=result.answer,
        context=[
            ContextChunk(
                content=chunk.content,
                source=chunk.source,
                page=chunk.page,
            )
            for chunk in result.context
        ],
    )
