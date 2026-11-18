import time
from pathlib import Path
from fastapi import APIRouter, HTTPException, status

from backend.app.core.config import get_settings
from backend.app.models.schemas import (
    ChunkRequest,
    ChunkResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    IndexStatusResponse,
    QueryRequest,
    QueryResponse,
    ScanAndChunkRequest,
    ScanAndChunkResponse,
    ScanRequest,
    ScanSummary,
)
from backend.app.services.chunker import CodeChunker
from backend.app.services.llm import get_gemini_service
from backend.app.services.retriever import get_rag_service
from backend.app.services.scanner import RepoScanner, detect_language

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Check service health and configuration."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=settings.VERSION,
        project_name=settings.PROJECT_NAME,
    )


@router.post("/scan", response_model=ScanSummary, tags=["Repository"])
def scan_repository(request: ScanRequest) -> ScanSummary:
    """Scan a repository path for code and text files while applying exclusion rules."""
    try:
        scanner = RepoScanner(
            repo_path=request.repo_path,
            additional_exclusions=request.additional_exclusions,
            max_file_size_bytes=request.max_file_size_bytes,
        )
        return scanner.scan()
    except (FileNotFoundError, NotADirectoryError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while scanning repository: {str(e)}",
        )


@router.post("/chunk", response_model=ChunkResponse, tags=["Chunking"])
def chunk_code(request: ChunkRequest) -> ChunkResponse:
    """Chunk raw text or code snippets preserving line numbers and exact content."""
    try:
        chunker = CodeChunker(
            chunk_size_lines=request.chunk_size_lines,
            chunk_overlap_lines=request.chunk_overlap_lines,
        )
        file_path = request.file_path or "snippet.txt"
        language = request.language or detect_language(Path(file_path))

        chunks = chunker.chunk_text(
            content=request.content,
            file_path=file_path,
            language=language,
        )

        total_lines = len(request.content.splitlines()) if request.content else 0

        return ChunkResponse(
            file_path=file_path,
            language=language,
            total_lines=total_lines,
            total_chunks=len(chunks),
            chunks=chunks,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while chunking content: {str(e)}",
        )


@router.post("/scan-and-chunk", response_model=ScanAndChunkResponse, tags=["Pipeline"])
def scan_and_chunk(request: ScanAndChunkRequest) -> ScanAndChunkResponse:
    """Scan an entire repository and extract line-preserving code chunks for all discovered files."""
    try:
        scanner = RepoScanner(
            repo_path=request.repo_path,
            additional_exclusions=request.additional_exclusions,
            max_file_size_bytes=request.max_file_size_bytes,
        )
        scan_summary = scanner.scan()

        chunker = CodeChunker(
            chunk_size_lines=request.chunk_size_lines,
            chunk_overlap_lines=request.chunk_overlap_lines,
        )

        all_chunks = []
        for file_info in scan_summary.files:
            file_chunks = chunker.chunk_file(
                file_path=file_info.absolute_path,
                relative_to=scan_summary.scanned_path,
                language=file_info.language,
            )
            all_chunks.extend(file_chunks)

        return ScanAndChunkResponse(
            scan_summary=scan_summary,
            total_chunks=len(all_chunks),
            chunks=all_chunks,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during scan and chunk: {str(e)}",
        )


@router.post("/index", response_model=IndexResponse, tags=["RAG Retrieval"])
def index_repository(request: IndexRequest) -> IndexResponse:
    """Scan, chunk, embed, and store codebase chunks in the FAISS index."""
    try:
        rag_service = get_rag_service()
        return rag_service.index_repository(
            repo_path=request.repo_path,
            persist=request.persist,
            additional_exclusions=request.additional_exclusions,
            chunk_size_lines=request.chunk_size_lines,
            chunk_overlap_lines=request.chunk_overlap_lines,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during repository indexing: {str(e)}",
        )


@router.post("/query", response_model=QueryResponse, tags=["RAG Retrieval"])
def query_repository(request: QueryRequest) -> QueryResponse:
    """Perform semantic search against indexed code chunks and generate an answer using Gemini."""
    try:
        rag_service = get_rag_service()
        gemini_service = get_gemini_service()

        retrieval_response = rag_service.search(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            file_filter=request.file_filter,
            enable_reranker=request.enable_reranker,
            enable_hybrid=request.enable_hybrid,
        )

        chunks = [r.chunk for r in retrieval_response.results]
        t_llm_start = time.perf_counter()
        answer = gemini_service.generate_answer(
            question=request.query,
            chunks=chunks,
        )
        t_llm_end = time.perf_counter()
        llm_duration = round(t_llm_end - t_llm_start, 4)

        if not hasattr(retrieval_response, "latencies") or retrieval_response.latencies is None:
            retrieval_response.latencies = {}
        retrieval_response.latencies["llm_generation_sec"] = llm_duration
        retrieval_response.latencies["total_query_sec"] = round(
            retrieval_response.latencies.get("total_retrieval_sec", 0.0) + llm_duration, 4
        )

        retrieval_response.answer = answer
        return retrieval_response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during retrieval: {str(e)}",
        )


@router.get("/index/status", response_model=IndexStatusResponse, tags=["RAG Retrieval"])
def get_index_status() -> IndexStatusResponse:
    """Get status, chunk count, and metadata of the active FAISS index."""
    try:
        rag_service = get_rag_service()
        return rag_service.get_status()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch index status: {str(e)}",
        )


