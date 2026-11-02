"""Services package for repository scanning, chunking, embeddings, and retrieval."""
from backend.app.services.chunker import CodeChunker
from backend.app.services.embedder import CodeEmbedder, get_embedder
from backend.app.services.llm import GeminiService, get_gemini_service
from backend.app.services.retriever import RagService, get_rag_service
from backend.app.services.scanner import RepoScanner
from backend.app.services.vector_store import FaissVectorStore

__all__ = [
    "RepoScanner",
    "CodeChunker",
    "CodeEmbedder",
    "get_embedder",
    "FaissVectorStore",
    "RagService",
    "get_rag_service",
    "GeminiService",
    "get_gemini_service",
]
