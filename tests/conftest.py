import shutil
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture
def test_client() -> TestClient:
    """Fixture providing a FastAPI TestClient."""
    return TestClient(app)


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a mock repository with diverse files, ignored directories, and .gitignore."""
    repo_root = tmp_path / "mock_repo"
    repo_root.mkdir()

    # Valid code files
    src_dir = repo_root / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text(
        "import sys\n\ndef main():\n    print('Hello World')\n\nif __name__ == '__main__':\n    main()\n",
        encoding="utf-8",
    )
    (src_dir / "utils.js").write_text(
        "function add(a, b) {\n    return a + b;\n}\n\nmodule.exports = { add };\n",
        encoding="utf-8",
    )

    # Documentation and configs
    (repo_root / "README.md").write_text("# Mock Repo\n\nSample description.", encoding="utf-8")
    (repo_root / "Dockerfile").write_text("FROM python:3.13-slim\nWORKDIR /app\n", encoding="utf-8")

    # Default ignored directories and files
    git_dir = repo_root / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git config content", encoding="utf-8")

    node_modules = repo_root / "node_modules" / "some_pkg"
    node_modules.mkdir(parents=True)
    (node_modules / "index.js").write_text("console.log('ignored');", encoding="utf-8")

    pycache_dir = src_dir / "__pycache__"
    pycache_dir.mkdir()
    (pycache_dir / "main.cpython-313.pyc").write_bytes(b"\x00\x01\x02\x03\x04")

    # Binary file
    (repo_root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

    # .gitignore setup
    (repo_root / ".gitignore").write_text("*.log\nsecret_folder/\n", encoding="utf-8")
    (repo_root / "debug.log").write_text("2026-10-31: debug log info\n", encoding="utf-8")

    secret_folder = repo_root / "secret_folder"
    secret_folder.mkdir()
    (secret_folder / "secret.txt").write_text("confidential", encoding="utf-8")

    return repo_root

