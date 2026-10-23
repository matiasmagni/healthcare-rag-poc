"""PDF loading and document chunking utilities."""
import logging
from collections.abc import Sequence
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class PDFDocumentLoader:
    """Load PDF files from configured directories."""

    def __init__(self, search_dirs: Sequence[Path], filenames: Sequence[str]) -> None:
        """Initialize the loader.

        Args:
            search_dirs: Directories to search for PDFs (in order).
            filenames: PDF filenames to load.
        """
        self._search_dirs = tuple(search_dirs)
        self._filenames = tuple(filenames)

    def load(self) -> list[Document]:
        """Load all configured PDFs and return LangChain Documents.

        Returns:
            List of loaded documents (one Document per page typically).
        """
        documents: list[Document] = []
        for filename in self._filenames:
            path = self._resolve_path(filename)
            logger.info("Loading PDF: %s", path)
            documents.extend(PyPDFLoader(str(path)).load())
        
        return documents

    def _resolve_path(self, filename: str) -> Path:
        """Resolve a filename to an existing file path in search directories.

        Args:
            filename: Name of the PDF file.

        Returns:
            Path to the found PDF.

        Raises:
            FileNotFoundError: If the PDF cannot be found in any search directory.
        """
        for directory in self._search_dirs:
            candidate = directory / filename

            if candidate.is_file():
                return candidate
            
        searched = ", ".join(
            str(directory / filename) for directory in self._search_dirs
        )

        raise FileNotFoundError(f"PDF not found: {filename}. Searched: {searched}")


class DocumentChunker:
    """Split documents into overlapping chunks for retrieval."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """Initialize the chunker.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between consecutive chunks.
        """
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents: Sequence[Document]) -> list[Document]:
        """Split documents into chunks.

        Args:
            documents: Input documents to split.

        Returns:
            List of chunked documents.

        Raises:
            ValueError: If no chunks are produced.
        """
        chunks = self._splitter.split_documents(list(documents))
        if not chunks:
            raise ValueError("No text chunks were produced from the PDF documents.")
        
        return chunks
