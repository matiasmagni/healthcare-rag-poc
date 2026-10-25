"""RAG orchestration service."""
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import Settings
from app.guardrails import CLINICAL_ADVICE_RESPONSE, is_individualized_clinical_request
from app.prompts import NO_ANSWER_RESPONSE, build_chat_prompt
from app.services.document_ingestion import DocumentChunker, PDFDocumentLoader
from app.services.factories import create_chat_model, create_embeddings
from app.services.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A retrieved chunk with provenance metadata."""

    content: str
    source: str
    page: int | None


@dataclass(frozen=True, slots=True)
class RAGResult:
    """Result returned by the RAG service."""

    answer: str
    context: tuple[RetrievedChunk, ...]


class RAGService:
    """Orchestrates retrieval and generation."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        chat_model: BaseChatModel,
        prompt: ChatPromptTemplate,
        retrieval_k: int,
    ) -> None:
        """Initialize the RAG service.

        Args:
            vector_store: Vector store for similarity search.
            chat_model: Chat model for generation.
            prompt: Chat prompt template.
            retrieval_k: Default number of chunks to retrieve.
        """
        self._vector_store = vector_store
        self._retrieval_k = retrieval_k
        self._chain = prompt | chat_model | StrOutputParser()

    @property
    def chunk_count(self) -> int:
        """Number of chunks in the vector store."""
        return self._vector_store.chunk_count

    def answer(self, question: str) -> RAGResult:
        """Answer a question using retrieved context.

        Args:
            question: User question.

        Returns:
            RAGResult containing answer and context chunks.
        """
        if is_individualized_clinical_request(question):
            logger.info("Refusing individualized clinical advice request")
            return RAGResult(answer=CLINICAL_ADVICE_RESPONSE, context=())

        q_lower = question.lower()
        retrieval_k = self._retrieval_k
        if any(
            term in q_lower
            for term in ["right", "notice of privacy", "privacy notice", "accounting of disclosures"]
        ):
            retrieval_k = max(retrieval_k, 25)
        documents = self._vector_store.similarity_search(question, k=retrieval_k)

        if not documents:
            return RAGResult(answer=NO_ANSWER_RESPONSE, context=())

        generated = self._chain.invoke(
            {
                "context": self._format_context(documents),
                "question": question,
            }
        )
        answer = generated.strip() or NO_ANSWER_RESPONSE

        return RAGResult(
            answer=answer,
            context=tuple(self._to_chunk(document) for document in documents),
        )

    @staticmethod
    def _format_context(documents: Sequence[Document]) -> str:
        """Format documents into numbered context blocks for the prompt.

        Args:
            documents: Retrieved documents.

        Returns:
            Formatted context string.
        """
        blocks: list[str] = []

        for index, document in enumerate(documents, start=1):
            source = _source_name(document)
            page = _display_page(document)
            page_label = f", page {page}" if page is not None else ""
            blocks.append(
                f"[Source {index}: {source}{page_label}]\n{document.page_content.strip()}"
            )
        
        return "\n\n".join(blocks)

    @staticmethod
    def _to_chunk(document: Document) -> RetrievedChunk:
        """Convert a document to a RetrievedChunk.

        Args:
            document: Retrieved document.

        Returns:
            RetrievedChunk with content and provenance.
        """
        return RetrievedChunk(
            content=document.page_content,
            source=_source_name(document),
            page=_display_page(document),
        )


def _source_name(document: Document) -> str:
    """Extract the source filename from document metadata.

    Args:
        document: Retrieved document.

    Returns:
        Source filename.
    """
    return Path(document.metadata.get("source", "unknown")).name


def _display_page(document: Document) -> int | None:
    """Convert 0-based page index to 1-based page number if present.

    Args:
        document: Retrieved document.

    Returns:
        1-based page number or None if unavailable.
    """
    page = document.metadata.get("page")
    
    return page + 1 if isinstance(page, int) else None


def build_rag_service(settings: Settings) -> RAGService:
    """Build and initialize the RAG service.

    Args:
        settings: Application settings.

    Returns:
        Initialized RAGService instance.
    """
    embeddings = create_embeddings(settings)
    loader = PDFDocumentLoader(settings.pdf_search_dirs)
    chunker = DocumentChunker(settings.chunk_size, settings.chunk_overlap)
    vector_store = FAISSVectorStore(embeddings, settings.index_path)

    logger.info("Loading or creating FAISS index")
    chunk_count = vector_store.load_or_create(lambda: chunker.split(loader.load()))
    logger.info("Vector store ready with %d chunks", chunk_count)

    return RAGService(
        vector_store=vector_store,
        chat_model=create_chat_model(settings),
        prompt=build_chat_prompt(),
        retrieval_k=settings.retrieval_k,
    )
