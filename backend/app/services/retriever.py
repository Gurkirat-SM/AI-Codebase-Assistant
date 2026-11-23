import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from backend.app.core.config import get_settings
from backend.app.models.schemas import (
    CodeChunk,
    IndexResponse,
    IndexStatusResponse,
    QueryResponse,
    QueryResult,
    QuerySource,
)
from backend.app.services.bm25 import BM25Index
from backend.app.services.chunker import CodeChunker
from backend.app.services.embedder import CodeEmbedder, get_embedder
from backend.app.services.hybrid import reciprocal_rank_fusion
from backend.app.services.reranker import CodeReranker, get_reranker
from backend.app.services.scanner import RepoScanner
from backend.app.services.vector_store import FaissVectorStore


class RagService:
    """Advanced RAG service orchestrating code-aware chunking, dense vector search,
    BM25 keyword search, reciprocal rank fusion, and cross-encoder reranking.
    """

    def __init__(
        self,
        embedder: Optional[CodeEmbedder] = None,
        vector_store: Optional[FaissVectorStore] = None,
        storage_dir: Optional[str | Path] = None,
        bm25_index: Optional[BM25Index] = None,
        reranker: Optional[CodeReranker] = None,
    ):
        settings = get_settings()
        self.storage_dir = Path(storage_dir or settings.INDEX_STORAGE_DIR).resolve()
        self.embedder = embedder or get_embedder()
        self.vector_store = vector_store
        self.bm25_index = bm25_index if bm25_index is not None else BM25Index()
        self.reranker = reranker if reranker is not None else get_reranker()
        self.last_scan_summary = None

        # Attempt to auto-load persisted indexes if vector store not pre-provided
        if self.vector_store is None:
            self._init_vector_store()

    def _init_vector_store(self):
        """Initialize FAISS vector store and BM25 index, loading from disk if present."""
        self.vector_store = FaissVectorStore(dimension=self.embedder.dimension)
        if self.storage_dir.exists():
            try:
                self.vector_store.load(self.storage_dir)
            except Exception:
                pass
            try:
                self.bm25_index.load(self.storage_dir)
            except Exception:
                pass

    def index_repository(
        self,
        repo_path: str,
        persist: bool = True,
        additional_exclusions: Optional[List[str]] = None,
        chunk_size_lines: Optional[int] = None,
        chunk_overlap_lines: Optional[int] = None,
    ) -> IndexResponse:
        """Scan a repository, chunk code files using code-aware chunking,
        embed chunks, build FAISS vector index, and construct BM25 keyword index.
        """
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
        self.last_scan_summary = scan_summary

        # 2. Chunk discovered files using code-aware chunker
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

        # 3. Generate dense vector embeddings
        embeddings = self.embedder.embed_chunks(all_chunks)

        # 4. Populate FAISS vector store
        self.vector_store = FaissVectorStore(dimension=self.embedder.dimension)
        self.vector_store.add_chunks(
            chunks=all_chunks,
            embeddings=embeddings,
            repo_path=str(target_path),
            model_name=self.embedder.model_name,
        )

        # 5. Populate BM25 keyword index
        self.bm25_index = BM25Index()
        self.bm25_index.add_chunks(all_chunks)

        # 6. Optionally persist to disk
        persisted_path = None
        if persist:
            saved_dir = self.vector_store.save(self.storage_dir)
            self.bm25_index.save(self.storage_dir)
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
        enable_reranker: Optional[bool] = None,
        enable_hybrid: Optional[bool] = None,
    ) -> QueryResponse:
        """Perform code-aware hybrid search (dense vector + BM25 keyword)
        with Reciprocal Rank Fusion, query-dependent weighting, and cross-encoder reranking.
        """
        if self.vector_store is None or self.vector_store.total_chunks == 0:
            raise ValueError("No codebase has been indexed yet. Please index a repository first via /api/index.")

        settings = get_settings()
        use_hybrid = enable_hybrid if enable_hybrid is not None else settings.ENABLE_HYBRID_SEARCH
        use_reranker = enable_reranker if enable_reranker is not None else settings.ENABLE_RERANKER

        t_total_start = time.perf_counter()
        latencies: Dict[str, float] = {}

        candidate_k = max(top_k * 4, settings.RERANKER_TOP_N_CANDIDATES)

        # 1. Vector Search
        t_vec_start = time.perf_counter()
        query_vector = self.embedder.embed_query(query)
        vector_results: List[QueryResult] = self.vector_store.search(
            query_vector=query_vector,
            top_k=candidate_k,
            score_threshold=score_threshold,
            file_filter=file_filter,
        )
        latencies["vector_retrieval_sec"] = round(time.perf_counter() - t_vec_start, 4)

        # 2. BM25 Keyword Search & Targeted Symbol Retrieval
        bm25_results = []
        if use_hybrid and self.bm25_index.total_chunks > 0:
            t_bm25_start = time.perf_counter()
            bm25_results = self.bm25_index.search(
                query=query,
                top_k=candidate_k,
                file_filter=file_filter,
            )

            # Targeted retrieval: search specific code identifiers (CamelCase, snake_case)
            import re
            identifiers = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*[A-Z0-9_][A-Za-z0-9_]*\b", query)
            if identifiers:
                seen_bm25_ids = {c.chunk_id for c, _ in bm25_results}
                for ident in identifiers:
                    if len(ident) > 2 and ident.lower() not in {"the", "and", "for", "with", "from"}:
                        target_hits = self.bm25_index.search(ident, top_k=3, file_filter=file_filter)
                        for chunk, score in target_hits:
                            if chunk.chunk_id not in seen_bm25_ids:
                                bm25_results.append((chunk, score))
                                seen_bm25_ids.add(chunk.chunk_id)

            # For architecture queries: ensure we have both high-level context (docs/config)
            # and core implementation source files to verify the architecture against code
            from backend.app.services.weighting import detect_query_intent
            intent = detect_query_intent(query)
            if intent.get("documentation", 1.0) > 1.2:
                has_code_file = any(
                    not r.chunk.file_path.lower().endswith((".md", ".txt", ".rst"))
                    for r in vector_results[:5]
                )
                if not has_code_file:
                    for entry_kw in ["main", "app", "service", "pipeline"]:
                        code_hits = self.bm25_index.search(entry_kw, top_k=2, file_filter=file_filter)
                        for chunk, score in code_hits:
                            if not chunk.file_path.lower().endswith((".md", ".txt", ".rst")):
                                bm25_results.append((chunk, score * 1.1))
                                break

            latencies["bm25_retrieval_sec"] = round(time.perf_counter() - t_bm25_start, 4)

        # 3. Hybrid Fusion via Reciprocal Rank Fusion (RRF) with file-type weighting
        t_fusion_start = time.perf_counter()
        if use_hybrid and bm25_results:
            candidates = reciprocal_rank_fusion(
                vector_results=vector_results,
                bm25_results=bm25_results,
                query=query,
                top_k=candidate_k,
            )
        else:
            candidates = vector_results
        latencies["fusion_sec"] = round(time.perf_counter() - t_fusion_start, 4)

        # 4. Cross-Encoder Reranking
        t_rerank_start = time.perf_counter()
        if use_reranker and self.reranker.is_available:
            final_results = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )
        else:
            final_results = candidates[:top_k]
        latencies["reranking_sec"] = round(time.perf_counter() - t_rerank_start, 4)

        latencies["total_retrieval_sec"] = round(time.perf_counter() - t_total_start, 4)

        sources = [
            QuerySource(
                file_path=r.chunk.file_path,
                start_line=r.chunk.start_line,
                end_line=r.chunk.end_line,
                snippet=r.chunk.content,
                score=r.score,
                language=r.chunk.language,
            )
            for r in final_results
        ]

        return QueryResponse(
            query=query,
            answer="",
            sources=sources,
            total_results=len(sources),
            results=final_results,
            latencies=latencies,
        )

    def get_status(self) -> IndexStatusResponse:
        """Return status and metadata of current FAISS and BM25 index."""
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
