import numpy as np
import pytest

from backend.app.models.schemas import CodeChunk
from backend.app.services.embedder import CodeEmbedder, format_chunk_for_embedding


def test_format_chunk_for_embedding():
    """Verify code chunk text formatting for semantic embedding."""
    chunk = CodeChunk(
        chunk_id="app/db.py:1-5",
        file_path="app/db.py",
        start_line=1,
        end_line=5,
        content="def connect(): pass",
        language="Python",
        line_count=5,
        chunk_index=0,
    )
    formatted = format_chunk_for_embedding(chunk)
    assert "File: app/db.py" in formatted
    assert "Language: Python" in formatted
    assert "def connect(): pass" in formatted


def test_embedder_empty_inputs():
    """Verify embedding empty lists and invalid queries."""
    embedder = CodeEmbedder()

    empty_texts = embedder.embed_texts([])
    assert empty_texts.shape == (0, 384)

    empty_chunks = embedder.embed_chunks([])
    assert empty_chunks.shape == (0, 384)

    with pytest.raises(ValueError, match="cannot be empty"):
        embedder.embed_query("")

    with pytest.raises(ValueError, match="cannot be empty"):
        embedder.embed_query("   ")


def test_embedder_normalization_and_dimension():
    """Verify that generated embeddings have dimension 384 and unit L2 norm."""
    embedder = CodeEmbedder()
    assert embedder.dimension == 384

    query_vec = embedder.embed_query("find database connection pool")
    assert query_vec.shape == (1, 384)
    assert query_vec.dtype == np.float32

    # L2 norm must be approximately 1.0
    norm = np.linalg.norm(query_vec[0])
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_embedder_semantic_similarity():
    """Verify that semantically related concepts have higher cosine similarity than unrelated concepts."""
    embedder = CodeEmbedder()

    auth_chunk = CodeChunk(
        chunk_id="auth.py:1-4",
        file_path="auth.py",
        start_line=1,
        end_line=4,
        content="def login(username, password):\n    validate_credentials(username, password)",
        language="Python",
        line_count=4,
        chunk_index=0,
    )

    math_chunk = CodeChunk(
        chunk_id="math.py:1-3",
        file_path="math.py",
        start_line=1,
        end_line=3,
        content="def calculate_circle_area(radius):\n    return 3.14159 * radius * radius",
        language="Python",
        line_count=3,
        chunk_index=0,
    )

    chunk_embeddings = embedder.embed_chunks([auth_chunk, math_chunk])
    auth_emb = chunk_embeddings[0]
    math_emb = chunk_embeddings[1]

    query_vec = embedder.embed_query("how does user authentication and login work?")[0]

    # Cosine similarity is dot product of normalized vectors
    sim_auth = np.dot(query_vec, auth_emb)
    sim_math = np.dot(query_vec, math_emb)

    assert sim_auth > sim_math

