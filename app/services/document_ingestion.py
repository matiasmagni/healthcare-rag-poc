"""PDF loading and document chunking utilities."""
import logging
from collections.abc import Sequence
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class PDFDocumentLoader:
    """Load all PDF files found in configured directories."""

    def __init__(self, search_dirs: Sequence[Path]) -> None:
        """Initialize the loader.

        Args:
            search_dirs: Directories to search for PDFs (in order).
        """
        self._search_dirs = tuple(search_dirs)

    def load(self) -> list[Document]:
        """Load every PDF found in the search directories.

        Returns:
            List of loaded documents (one Document per page typically).

        Raises:
            FileNotFoundError: If no PDFs are found in any search directory.
        """
        paths = self._discover_paths()
        if not paths:
            searched = ", ".join(str(directory) for directory in self._search_dirs)
            raise FileNotFoundError(f"No PDF files found. Searched: {searched}")

        documents: list[Document] = []
        for path in paths:
            logger.info("Loading PDF: %s", path)
            documents.extend(PyPDFLoader(str(path)).load())

        return documents

    def _discover_paths(self) -> list[Path]:
        """Collect all PDF files in the search directories, without duplicates.

        Returns:
            Sorted list of PDF paths.
        """
        paths: dict[Path, Path] = {}

        for directory in self._search_dirs:
            if not directory.is_dir():
                continue

            for path in sorted(directory.glob("*.pdf")):
                if path.is_file():
                    paths.setdefault(path.resolve(), path)

        return list(paths.values())


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
