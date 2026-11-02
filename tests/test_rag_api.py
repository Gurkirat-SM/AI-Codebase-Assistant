from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


def test_index_status_endpoint(test_client: TestClient):
    """Verify index status endpoint."""
    response = test_client.get("/api/index/status")
    assert response.status_code == 200
    data = response.json()
    assert "is_indexed" in data
    assert "total_chunks" in data
    assert "model_name" in data


@patch("backend.app.api.routes.get_gemini_service")
def test_index_and_query_endpoints(mock_get_gemini, test_client: TestClient, sample_repo: Path):
    """Verify POST /api/index and POST /api/query endpoints with mocked Gemini answer generation."""
    mock_gemini = MagicMock()
    mock_gemini.generate_answer.return_value = "In src/utils.js lines 1-3, function add(a, b) adds two numbers."
    mock_get_gemini.return_value = mock_gemini

    # Index repository
    index_res = test_client.post(
        "/api/index",
        json={
            "repo_path": str(sample_repo),
            "persist": False,
        },
    )
    assert index_res.status_code == 200
    index_data = index_res.json()
    assert index_data["status"] == "success"
    assert index_data["total_chunks"] > 0
    assert index_data["embedding_dimension"] == 384

    # Query repository
    query_res = test_client.post(
        "/api/query",
        json={
            "query": "function for adding numbers",
            "top_k": 3,
        },
    )
    assert query_res.status_code == 200
    query_data = query_res.json()
    assert query_data["query"] == "function for adding numbers"
    assert query_data["answer"] == "In src/utils.js lines 1-3, function add(a, b) adds two numbers."
    assert "sources" in query_data
    assert query_data["total_results"] >= 1
    assert len(query_data["sources"]) <= 3

    top_source = query_data["sources"][0]
    assert top_source["file_path"] == "src/utils.js"
    assert top_source["start_line"] >= 1
    assert top_source["end_line"] >= top_source["start_line"]
    assert "snippet" in top_source
    assert top_source["score"] > 0.0


def test_index_invalid_repo(test_client: TestClient):
    """Verify 400 Bad Request on invalid repository path."""
    response = test_client.post(
        "/api/index",
        json={"repo_path": "Z:\\invalid_nonexistent_repo"},
    )
    assert response.status_code == 400
    assert "does not exist" in response.json()["detail"]


def test_query_validation(test_client: TestClient):
    """Verify validation error when query is empty."""
    response = test_client.post(
        "/api/query",
        json={"query": ""},
    )
    assert response.status_code == 422

