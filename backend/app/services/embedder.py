from functools import lru_cache
from typing import List, Optional
import numpy as np

from backend.app.core.config import get_settings
from backend.app.models.schemas import CodeChunk


def format_chunk_for_embedding(chunk: CodeChunk) -> str:
    """Format code chunk with rich contextual headers for semantic embedding."""
    return f"File: {chunk.file_path}\nLanguage: {chunk.language}\n{chunk.content}"


class CodeEmbedder:
    """Embedder using sentence-transformers to produce normalized code and query embeddings."""

    def __init__(self, model_name: Optional[str] = None, batch_size: Optional[int] = None):
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self._model = None
        self._dimension: Optional[int] = None

    @property
    def model(self):
        """Lazily load SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Return the vector embedding dimension for the model."""
        if self._dimension is None:
            if hasattr(self.model, "get_embedding_dimension"):
                self._dimension = self.model.get_embedding_dimension()
            else:
                self._dimension = self.model.get_sentence_embedding_dimension()
        return self._dimension

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate normalized embeddings for a list of string texts."""
        if not texts:
            dim = self.dimension
            return np.empty((0, dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_chunks(self, chunks: List[CodeChunk]) -> np.ndarray:
        """Generate normalized embeddings for code chunks."""
        if not chunks:
            return np.empty((0, self.dimension), dtype=np.float32)

        formatted_texts = [format_chunk_for_embedding(c) for c in chunks]
        return self.embed_texts(formatted_texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate a single normalized embedding vector (1 x D) for a query string."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty")

        embeddings = self.model.encode(
            [query.strip()],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


@lru_cache
def get_embedder() -> CodeEmbedder:
    """Return cached singleton instance of CodeEmbedder."""
    return CodeEmbedder()
