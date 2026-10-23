"""FastAPI application factory and entrypoint."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings, get_settings
from app.services.rag import build_rag_service

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str) -> None:
    """Configure application logging.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings instance (defaults to cached settings).

    Returns:
        Configured FastAPI application instance.
    """
    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """Application lifespan managing RAG service initialization."""
        logger.info("Initializing RAG service")
        application.state.rag_service = build_rag_service(app_settings)
        yield
        application.state.rag_service = None

    application = FastAPI(
        title="HIPAA RAG API",
        description="Local RAG over HIPAA guidance documents with a Unsloth-hosted LLM.",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.settings = app_settings
    application.include_router(router)
    return application


app = create_app()
