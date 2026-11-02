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
        ".git",
        ".venv",
        "venv",
        "ENV",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".idea",
        ".vscode",
        "dist",
        "build",
        "target",
        "bin",
        "obj",
        ".next",
        "coverage",
        ".mypy_cache",
        ".ruff_cache",
    ]
    SCANNER_IGNORE_FILE_PATTERNS: List[str] = [
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "*.so",
        "*.dll",
        "*.exe",
        "*.dylib",
        "*.class",
        "*.jar",
        "*.war",
        "*.zip",
        "*.tar",
        "*.gz",
        "*.rar",
        "*.7z",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.gif",
        "*.ico",
        "*.svg",
        "*.webp",
        "*.pdf",
        "*.woff",
        "*.woff2",
        "*.ttf",
        "*.eot",
        "*.mp4",
        "*.mp3",
        "*.wav",
        "*.lock",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
    ]

    # Chunker configuration
    CHUNK_SIZE_LINES: int = 60
    CHUNK_OVERLAP_LINES: int = 15

    # RAG / Embedding configuration
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    INDEX_STORAGE_DIR: str = ".index"
    DEFAULT_TOP_K: int = 5
    EMBEDDING_BATCH_SIZE: int = 32

    # Gemini LLM configuration
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash"

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

