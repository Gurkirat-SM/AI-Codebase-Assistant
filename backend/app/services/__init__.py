"""Services package for repository scanning, chunking, embeddings, and retrieval."""
from backend.app.services.bm25 import BM25Index, tokenize_code
from backend.app.services.chunker import CodeChunker
from backend.app.services.embedder import CodeEmbedder, get_embedder
from backend.app.services.hybrid import reciprocal_rank_fusion
from backend.app.services.llm import GeminiService, get_gemini_service
from backend.app.services.reranker import CodeReranker, get_reranker
from backend.app.services.retriever import RagService, get_rag_service
from backend.app.services.scanner import RepoScanner
from backend.app.services.vector_store import FaissVectorStore
from backend.app.services.weighting import classify_file_type, compute_file_weight, detect_query_intent

__all__ = [
    "RepoScanner",
    "CodeChunker",
    "CodeEmbedder",
    "get_embedder",
    "FaissVectorStore",
    "BM25Index",
    "tokenize_code",
    "reciprocal_rank_fusion",
    "CodeReranker",
    "get_reranker",
    "RagService",
    "get_rag_service",
    "GeminiService",
    "get_gemini_service",
    "classify_file_type",
    "detect_query_intent",
    "compute_file_weight",
]
