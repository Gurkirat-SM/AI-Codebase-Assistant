from pathlib import Path
import pytest

from backend.app.services.embedder import CodeEmbedder
from backend.app.services.retriever import RagService
from backend.app.services.vector_store import FaissVectorStore


def test_rag_service_unindexed_error():
    """Verify ValueError is raised if query is executed before indexing any repository."""
    embedder = CodeEmbedder()
    store = FaissVectorStore(dimension=embedder.dimension)
    rag_service = RagService(embedder=embedder, vector_store=store)

    with pytest.raises(ValueError, match="No codebase has been indexed yet"):
        rag_service.search("hello")


def test_rag_service_index_and_retrieve(sample_repo: Path, tmp_path: Path):
    """Verify end-to-end repository indexing and semantic retrieval."""
    embedder = CodeEmbedder()
    storage_dir = tmp_path / "index_store"
    rag_service = RagService(embedder=embedder, storage_dir=storage_dir)

    # Initial status
    status_before = rag_service.get_status()
    assert status_before.is_indexed is False
    assert status_before.total_chunks == 0

    # Index sample repository
    index_res = rag_service.index_repository(
        repo_path=str(sample_repo),
        persist=True,
    )

    assert index_res.status == "success"
    assert index_res.total_chunks > 0
    assert index_res.embedding_dimension == 384
    assert (storage_dir / "index.faiss").exists()

    # Status after indexing
    status_after = rag_service.get_status()
    assert status_after.is_indexed is True
    assert status_after.total_chunks == index_res.total_chunks

    # Semantic search 1: search for javascript addition function
    query_js = "add two numbers function in javascript"
    search_res_js = rag_service.search(query=query_js, top_k=2)

    assert search_res_js.total_results >= 1
    top_chunk = search_res_js.results[0].chunk
    assert "add" in top_chunk.content
    assert top_chunk.file_path == "src/utils.js"
    assert search_res_js.results[0].score > 0.3

    # Semantic search 2: search for python main entry point
    query_py = "python entry point with sys"
    search_res_py = rag_service.search(query=query_py, top_k=2)

    assert search_res_py.total_results >= 1
    py_chunk = search_res_py.results[0].chunk
    assert "main" in py_chunk.content
    assert py_chunk.file_path == "src/main.py"

    # Search with file_filter
    filtered_res = rag_service.search(
        query="entry point",
        top_k=5,
        file_filter="utils.js",
    )
    for r in filtered_res.results:
        assert r.chunk.file_path == "src/utils.js"


def test_rag_service_empty_repo(tmp_path: Path):
    """Verify handling when indexing a repository with no valid code files."""
    empty_dir = tmp_path / "empty_repo"
    empty_dir.mkdir()

    embedder = CodeEmbedder()
    rag_service = RagService(embedder=embedder, storage_dir=tmp_path / "empty_index")
    res = rag_service.index_repository(str(empty_dir))

    assert res.status == "warning"
    assert res.total_chunks == 0

