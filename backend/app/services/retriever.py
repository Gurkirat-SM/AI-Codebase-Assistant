from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from backend.app.core.config import get_settings
from backend.app.models.schemas import (
    CodeChunk,
    IndexResponse,
    IndexStatusResponse,
    QueryResponse,
    QueryResult,
    QuerySource,
)
from backend.app.services.chunker import CodeChunker
from backend.app.services.embedder import CodeEmbedder, get_embedder
from backend.app.services.scanner import RepoScanner
from backend.app.services.vector_store import FaissVectorStore


class RagService:
    """End-to-end RAG service orchestrating repository scanning, embedding, FAISS indexing, and retrieval."""

    def __init__(
        self,
        embedder: Optional[CodeEmbedder] = None,
        vector_store: Optional[FaissVectorStore] = None,
        storage_dir: Optional[str | Path] = None,
    ):
        settings = get_settings()
        self.storage_dir = Path(storage_dir or settings.INDEX_STORAGE_DIR).resolve()
        self.embedder = embedder or get_embedder()
        self.vector_store = vector_store

        # Attempt to auto-load persisted index if vector store not pre-provided
        if self.vector_store is None:
            self._init_vector_store()

    def _init_vector_store(self):
        """Initialize FAISS vector store, attempting to load from disk if an index exists."""
        self.vector_store = FaissVectorStore(dimension=self.embedder.dimension)
        if self.storage_dir.exists():
            try:
                self.vector_store.load(self.storage_dir)
            except Exception:
                # If loading fails, keep the clean empty vector store
                pass

    def index_repository(
        self,
        repo_path: str,
        persist: bool = True,
        additional_exclusions: Optional[List[str]] = None,
        chunk_size_lines: Optional[int] = None,
        chunk_overlap_lines: Optional[int] = None,
    ) -> IndexResponse:
        """Scan a repository, chunk code files, embed chunks, and build the FAISS index."""
        target_path = Path(repo_path).resolve()
        if not target_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {target_path}")
        if not target_path.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {target_path}")

        # 1. Scan repository
        scanner = RepoScanner(
            repo_path=target_path,
            additional_exclusions=additional_exclusions or [],
        )
        scan_summary = scanner.scan()

        # 2. Chunk discovered files
        chunker = CodeChunker(
            chunk_size_lines=chunk_size_lines,
            chunk_overlap_lines=chunk_overlap_lines,
        )

        all_chunks: List[CodeChunk] = []
        for file_info in scan_summary.files:
            file_chunks = chunker.chunk_file(
                file_path=file_info.absolute_path,
                relative_to=scan_summary.scanned_path,
                language=file_info.language,
            )
            all_chunks.extend(file_chunks)

        if len(all_chunks) == 0:
            return IndexResponse(
                status="warning",
                message="Repository scanned but no eligible code chunks found to index.",
                repo_path=str(target_path),
                total_files=scan_summary.total_files,
                total_chunks=0,
                embedding_dimension=self.embedder.dimension,
                model_name=self.embedder.model_name,
                persisted=False,
            )

        # 3. Generate embeddings
        embeddings = self.embedder.embed_chunks(all_chunks)

        # 4. Populate FAISS index
        self.vector_store = FaissVectorStore(dimension=self.embedder.dimension)
        self.vector_store.add_chunks(
            chunks=all_chunks,
            embeddings=embeddings,
            repo_path=str(target_path),
            model_name=self.embedder.model_name,
        )

        # 5. Optionally persist to disk
        persisted_path = None
        if persist:
            saved_dir = self.vector_store.save(self.storage_dir)
            persisted_path = str(saved_dir)

        return IndexResponse(
            status="success",
            message=f"Successfully indexed {len(all_chunks)} chunks across {scan_summary.total_files} files.",
            repo_path=str(target_path),
            total_files=scan_summary.total_files,
            total_chunks=len(all_chunks),
            embedding_dimension=self.embedder.dimension,
            model_name=self.embedder.model_name,
            persisted=persist,
            index_storage_dir=persisted_path,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        file_filter: Optional[str] = None,
    ) -> QueryResponse:
        """Perform semantic search against the indexed codebase chunks."""
        if self.vector_store is None or self.vector_store.total_chunks == 0:
            raise ValueError("No codebase has been indexed yet. Please index a repository first via /api/index.")

        query_vector = self.embedder.embed_query(query)
        results: List[QueryResult] = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
            file_filter=file_filter,
        )

        sources = [
            QuerySource(
                file_path=r.chunk.file_path,
                start_line=r.chunk.start_line,
                end_line=r.chunk.end_line,
                snippet=r.chunk.content,
                score=r.score,
                language=r.chunk.language,
            )
            for r in results
        ]

        return QueryResponse(
            query=query,
            answer="",
            sources=sources,
            total_results=len(sources),
            results=results,
        )

    def get_status(self) -> IndexStatusResponse:
        """Return status and metadata of current FAISS index."""
        is_indexed = self.vector_store is not None and self.vector_store.total_chunks > 0
        persisted_dir = str(self.storage_dir) if (self.storage_dir / "index.faiss").exists() else None

        return IndexStatusResponse(
            is_indexed=is_indexed,
            repo_path=self.vector_store.repo_path if self.vector_store else None,
            total_chunks=self.vector_store.total_chunks if self.vector_store else 0,
            embedding_dimension=self.embedder.dimension,
            model_name=self.embedder.model_name,
            persisted_dir=persisted_dir,
        )


@lru_cache
def get_rag_service() -> RagService:
    """Return cached singleton instance of RagService."""
    return RagService()

