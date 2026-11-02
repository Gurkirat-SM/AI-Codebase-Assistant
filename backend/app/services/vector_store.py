import json
from pathlib import Path
from typing import List, Optional
import faiss
import numpy as np

from backend.app.models.schemas import CodeChunk, QueryResult


class FaissVectorStore:
    """Vector store utilizing FAISS IndexFlatIP for cosine similarity search over code chunks."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index: faiss.Index = faiss.IndexFlatIP(dimension)
        self.chunks: List[CodeChunk] = []
        self.repo_path: Optional[str] = None
        self.model_name: Optional[str] = None

    @property
    def total_chunks(self) -> int:
        """Return total number of indexed chunks."""
        return len(self.chunks)

    def add_chunks(
        self,
        chunks: List[CodeChunk],
        embeddings: np.ndarray,
        repo_path: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> int:
        """Add code chunks and corresponding embeddings into the FAISS index."""
        if len(chunks) == 0:
            return 0

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding count ({embeddings.shape[0]}) does not match chunk count ({len(chunks)})"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) does not match index dimension ({self.dimension})"
            )

        # Ensure float32 format
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

        self.index.add(vectors)
        self.chunks.extend(chunks)

        if repo_path:
            self.repo_path = repo_path
        if model_name:
            self.model_name = model_name

        return len(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        file_filter: Optional[str] = None,
    ) -> List[QueryResult]:
        """Search for top-k relevant code chunks using cosine similarity."""
        if self.index.ntotal == 0 or len(self.chunks) == 0:
            return []

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)

        # When file_filter is active, search a wider window to ensure top_k candidates after filtering
        k_search = min(max(top_k * 4, 20), self.index.ntotal) if file_filter else min(top_k, self.index.ntotal)

        distances, indices = self.index.search(query_vector, k_search)

        results: List[QueryResult] = []
        filter_lower = file_filter.lower() if file_filter else None

        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            similarity_score = float(score)

            # Apply score threshold if specified
            if score_threshold is not None and similarity_score < score_threshold:
                continue

            chunk = self.chunks[idx]

            # Apply file filter if specified
            if filter_lower and filter_lower not in chunk.file_path.lower():
                continue

            results.append(
                QueryResult(
                    chunk=chunk,
                    score=similarity_score,
                )
            )

            if len(results) >= top_k:
                break

        return results

    def save(self, directory: str | Path) -> Path:
        """Persist FAISS index and chunk metadata to disk."""
        save_dir = Path(directory).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        index_file = save_dir / "index.faiss"
        metadata_file = save_dir / "metadata.json"

        # Write FAISS index
        faiss.write_index(self.index, str(index_file))

        # Write metadata JSON
        metadata = {
            "dimension": self.dimension,
            "total_chunks": len(self.chunks),
            "repo_path": self.repo_path,
            "model_name": self.model_name,
            "chunks": [chunk.model_dump() for chunk in self.chunks],
        }

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return save_dir

    def load(self, directory: str | Path) -> bool:
        """Load FAISS index and chunk metadata from disk."""
        save_dir = Path(directory).resolve()
        index_file = save_dir / "index.faiss"
        metadata_file = save_dir / "metadata.json"

        if not index_file.exists() or not metadata_file.exists():
            return False

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.dimension = metadata["dimension"]
        self.repo_path = metadata.get("repo_path")
        self.model_name = metadata.get("model_name")
        self.chunks = [CodeChunk(**c) for c in metadata.get("chunks", [])]

        # Read FAISS index
        self.index = faiss.read_index(str(index_file))
        return True

    def clear(self):
        """Reset index and stored metadata."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks.clear()
        self.repo_path = None

