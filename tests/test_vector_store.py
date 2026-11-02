from pathlib import Path
import numpy as np
import pytest

from backend.app.models.schemas import CodeChunk
from backend.app.services.vector_store import FaissVectorStore


def create_sample_chunk(file_path: str, start_line: int, end_line: int, content: str) -> CodeChunk:
    return CodeChunk(
        chunk_id=f"{file_path}:{start_line}-{end_line}",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        content=content,
        language="Python",
        line_count=end_line - start_line + 1,
        chunk_index=0,
    )


def test_vector_store_add_and_search():
    """Verify adding normalized embeddings and retrieving top-k chunks with cosine similarity."""
    dim = 4
    store = FaissVectorStore(dimension=dim)

    chunk1 = create_sample_chunk("auth.py", 1, 10, "def login(): pass")
    chunk2 = create_sample_chunk("db.py", 1, 10, "def connect_db(): pass")

    # Normalized vectors:
    # Vector 1 points along axis 0
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # Vector 2 points along axis 1
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    embeddings = np.vstack([v1, v2])

    count = store.add_chunks([chunk1, chunk2], embeddings, repo_path="/mock/repo", model_name="test-model")
    assert count == 2
    assert store.total_chunks == 2

    # Query closely aligned with v1
    query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    query = query / np.linalg.norm(query)

    results = store.search(query, top_k=2)
    assert len(results) == 2
    # chunk1 should be first because query is aligned with v1
    assert results[0].chunk.file_path == "auth.py"
    assert results[0].score > results[1].score


def test_vector_store_filters():
    """Verify score_threshold and file_filter work as expected."""
    dim = 4
    store = FaissVectorStore(dimension=dim)

    chunk1 = create_sample_chunk("src/auth.py", 1, 5, "auth code")
    chunk2 = create_sample_chunk("src/db.py", 1, 5, "db code")

    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    store.add_chunks([chunk1, chunk2], np.vstack([v1, v2]))

    # Query identical to v1
    query = v1.copy()

    # Score threshold filter
    high_threshold_results = store.search(query, top_k=5, score_threshold=0.5)
    assert len(high_threshold_results) == 1
    assert high_threshold_results[0].chunk.file_path == "src/auth.py"

    # File filter
    db_filter_results = store.search(query, top_k=5, file_filter="db.py")
    assert len(db_filter_results) == 1
    assert db_filter_results[0].chunk.file_path == "src/db.py"


def test_vector_store_persistence(tmp_path: Path):
    """Verify saving to disk and reloading a FAISS vector store."""
    dim = 4
    store = FaissVectorStore(dimension=dim)

    chunk = create_sample_chunk("main.py", 1, 20, "def main(): print('run')")
    v = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    store.add_chunks([chunk], v, repo_path="/app", model_name="mini-lm")

    # Save
    save_dir = tmp_path / "faiss_index"
    store.save(save_dir)
    assert (save_dir / "index.faiss").exists()
    assert (save_dir / "metadata.json").exists()

    # Load into new instance
    new_store = FaissVectorStore(dimension=dim)
    loaded = new_store.load(save_dir)
    assert loaded is True
    assert new_store.total_chunks == 1
    assert new_store.chunks[0].chunk_id == "main.py:1-20"
    assert new_store.repo_path == "/app"
    assert new_store.model_name == "mini-lm"


def test_vector_store_dimension_mismatch():
    """Verify ValueError is raised on mismatched embeddings count or dimension."""
    store = FaissVectorStore(dimension=4)
    chunk = create_sample_chunk("test.py", 1, 5, "test")

    # Count mismatch
    with pytest.raises(ValueError, match="does not match chunk count"):
        store.add_chunks([chunk], np.zeros((2, 4), dtype=np.float32))

    # Dimension mismatch
    with pytest.raises(ValueError, match="does not match index dimension"):
        store.add_chunks([chunk], np.zeros((1, 8), dtype=np.float32))

