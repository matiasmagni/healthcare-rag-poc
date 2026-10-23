"""FastAPI dependency injection helpers."""
from fastapi import HTTPException, Request

from app.config import Settings
from app.services.rag import RAGService


def get_rag_service(request: Request) -> RAGService:
    """Retrieve the RAG service from application state.

    Raises:
        HTTPException: If the RAG pipeline has not been initialized.
    """
    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not ready.")
    
    return service


def get_app_settings(request: Request) -> Settings:
    """Retrieve application settings from application state.

    Raises:
        HTTPException: If settings have not been initialized.
    """
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(
            status_code=503, detail="Application settings are not ready."
        )
    
    return settings
