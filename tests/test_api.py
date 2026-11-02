from pathlib import Path
from fastapi.testclient import TestClient


def test_root_endpoint(test_client: TestClient):
    """Verify root status response."""
    response = test_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "AI Codebase Assistant"
    assert "/docs" in data["docs"]


def test_health_check(test_client: TestClient):
    """Verify health endpoint response."""
    response = test_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_scan_endpoint(test_client: TestClient, sample_repo: Path):
    """Verify scan repository API endpoint."""
    response = test_client.post(
        "/api/scan",
        json={"repo_path": str(sample_repo)},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_files"] >= 3
    assert data["total_lines"] > 0
    assert "Python" in data["files_by_language"]

    rel_paths = [f["relative_path"] for f in data["files"]]
    assert "src/main.py" in rel_paths
    assert "image.png" not in rel_paths


def test_scan_invalid_path(test_client: TestClient):
    """Verify 400 error on non-existent repo path."""
    response = test_client.post(
        "/api/scan",
        json={"repo_path": "Z:\\invalid_path_does_not_exist"},
    )
    assert response.status_code == 400
    assert "does not exist" in response.json()["detail"]


def test_chunk_endpoint(test_client: TestClient):
    """Verify chunk code API endpoint."""
    code_text = "def a():\n    pass\n\ndef b():\n    pass\n"
    response = test_client.post(
        "/api/chunk",
        json={
            "file_path": "test.py",
            "content": code_text,
            "chunk_size_lines": 10,
            "chunk_overlap_lines": 2,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["file_path"] == "test.py"
    assert data["language"] == "Python"
    assert data["total_chunks"] == 1
    assert data["chunks"][0]["start_line"] == 1
    assert data["chunks"][0]["end_line"] == 5


def test_scan_and_chunk_endpoint(test_client: TestClient, sample_repo: Path):
    """Verify full scan and chunk pipeline endpoint."""
    response = test_client.post(
        "/api/scan-and-chunk",
        json={
            "repo_path": str(sample_repo),
            "chunk_size_lines": 20,
            "chunk_overlap_lines": 5,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_chunks"] >= 3
    assert len(data["chunks"]) == data["total_chunks"]

    # Verify each chunk possesses required metadata
    for chunk in data["chunks"]:
        assert "chunk_id" in chunk
        assert "file_path" in chunk
        assert chunk["start_line"] >= 1
        assert chunk["end_line"] >= chunk["start_line"]
        assert len(chunk["content"]) > 0

