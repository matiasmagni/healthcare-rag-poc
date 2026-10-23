"""FAISS vector store loading, building, and searching."""
import logging
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

ChunkProvider = Callable[[], Sequence[Document]]


class FAISSVectorStore:
    """Wrapper around LangChain FAISS for local persistence and search."""

    def __init__(self, embeddings: Embeddings, index_path: Path) -> None:
        """Initialize the vector store wrapper.

        Args:
            embeddings: Embeddings model to use for indexing/search.
            index_path: Directory to persist/load the FAISS index.
        """
        self._embeddings = embeddings
        self._index_path = index_path
        self._store: FAISS | None = None

    @property
    def chunk_count(self) -> int:
        """Number of indexed chunks.

        Raises:
            RuntimeError: If the store has not been loaded or created.
        """
        if self._store is None:
            raise RuntimeError("Vector store has not been loaded or created.")
        
        return int(self._store.index.ntotal)

    def load_or_create(self, chunk_provider: ChunkProvider) -> int:
        """Load an existing index if present, otherwise build from provided chunks.

        Args:
            chunk_provider: Callable that returns document chunks to index.

        Returns:
            Number of indexed chunks.
        """
        if self._load_existing():
            return self.chunk_count
        self._build(chunk_provider())

        return self.chunk_count

    def similarity_search(self, query: str, k: int) -> list[Document]:
        """Perform similarity search against the index.

        Args:
            query: Query text.
            k: Number of results to return.

        Returns:
            List of matching documents.

        Raises:
            RuntimeError: If the store has not been loaded or created.
        """
        if self._store is None:
            raise RuntimeError("Vector store has not been loaded or created.")
        
        return self._store.similarity_search(query, k=k)

    def _load_existing(self) -> bool:
        """Attempt to load a persisted FAISS index.

        Returns:
            True if loaded successfully, False otherwise.
        """
        index_file = self._index_path / "index.faiss"

        if not index_file.is_file():
            return False

        try:
            logger.info("Loading existing FAISS index from %s", self._index_path)
            self._store = FAISS.load_local(
                str(self._index_path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            logger.exception("Failed to load FAISS index; rebuilding from PDFs.")
            shutil.rmtree(self._index_path, ignore_errors=True)
            return False
        
        return True

    def _build(self, chunks: Sequence[Document]) -> None:
        """Build and persist a new FAISS index from chunks.

        Args:
            chunks: Document chunks to index.

        Raises:
            ValueError: If no chunks are provided.
        """
        if not chunks:
            raise ValueError("Cannot build a FAISS index without document chunks.")

        logger.info("Indexing %d chunks", len(chunks))
        self._store = FAISS.from_documents(list(chunks), self._embeddings)
        self._index_path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self._index_path))
        logger.info("Saved FAISS index to %s", self._index_path)
