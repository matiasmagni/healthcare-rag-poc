"""Pydantic schemas for API requests and responses."""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        description="Question to answer from the indexed PDFs.",
    )


class ContextChunk(BaseModel):
    """A single retrieved context chunk with provenance metadata."""

    content: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    page: int | None = Field(default=None, ge=1)


class QueryResponse(BaseModel):
    """Response body for the /query endpoint."""

    answer: str
    context: list[ContextChunk]


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    indexed_chunks: int = Field(..., ge=0)
    embedding_model: str
    llm_model: str
