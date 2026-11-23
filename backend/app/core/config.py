from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable override support."""

    PROJECT_NAME: str = "AI Codebase Assistant"
    VERSION: str = "0.1.0"
    API_PREFIX: str = "/api"

    # Server configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Scanner configuration
    MAX_FILE_SIZE_BYTES: int = 1_048_576  # 1 MB default limit
    SCANNER_IGNORE_DIRS: List[str] = [
        ".git", ".venv", "venv", "ENV", "node_modules", "__pycache__",
        ".pytest_cache", ".idea", ".vscode", "dist", "build", "target",
        "bin", "obj", ".next", "coverage", ".mypy_cache", ".ruff_cache",
    ]
    SCANNER_IGNORE_FILE_PATTERNS: List[str] = [
        "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll", "*.exe", "*.dylib",
        "*.class", "*.jar", "*.war", "*.zip", "*.tar", "*.gz", "*.rar",
        "*.7z", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico", "*.svg",
        "*.webp", "*.pdf", "*.woff", "*.woff2", "*.ttf", "*.eot", "*.mp4",
        "*.mp3", "*.wav", "*.lock", "package-lock.json", "pnpm-lock.yaml",
        "yarn.lock",
    ]

    # Chunker configuration
    CHUNK_SIZE_LINES: int = 60
    CHUNK_OVERLAP_LINES: int = 15
    ENABLE_CODE_AWARE_CHUNKING: bool = True
    MAX_CODE_CHUNK_LINES: int = 120

    # RAG / Embedding configuration
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    INDEX_STORAGE_DIR: str = ".index"
    DEFAULT_TOP_K: int = 5
    EMBEDDING_BATCH_SIZE: int = 32

    # Adaptive Retrieval & Multi-File Architecture Depth
    ENABLE_ADAPTIVE_RETRIEVAL: bool = True
    MAX_ADAPTIVE_TOP_K: int = 15
    ARCHITECTURE_TOP_K: int = 12
    CROSS_MODULE_TOP_K: int = 8

    # Hybrid Search configuration
    ENABLE_HYBRID_SEARCH: bool = True
    BM25_WEIGHT: float = 0.5
    VECTOR_WEIGHT: float = 0.5
    RRF_K: int = 60

    # Intelligent Retrieval Weighting
    ENABLE_FILE_TYPE_WEIGHTING: bool = True

    # Reranker configuration
    ENABLE_RERANKER: bool = True
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_TOP_N_CANDIDATES: int = 20

    # Generation & Context Budget
    MAX_CONTEXT_CHARS: int = 24_000

    # Gemini LLM configuration
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-3.8-flash"

    # CORS configuration
    CORS_ORIGINS: List[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached instance of application settings."""
    return Settings()
