from pathlib import Path
import pytest

from backend.app.services.scanner import (
    RepoScanner,
    detect_language,
    is_binary_file,
)


def test_detect_language():
    """Verify language detection across standard extensions and special filenames."""
    assert detect_language(Path("test.py")) == "Python"
    assert detect_language(Path("app.tsx")) == "TypeScript (React)"
    assert detect_language(Path("server.js")) == "JavaScript"
    assert detect_language(Path("style.css")) == "CSS"
    assert detect_language(Path("Dockerfile")) == "Dockerfile"
    assert detect_language(Path("dockerfile")) == "Dockerfile"
    assert detect_language(Path("unknown.xyz123")) == "Unknown"


def test_is_binary_file(tmp_path: Path):
    """Verify detection of binary versus text files."""
    text_file = tmp_path / "hello.txt"
    text_file.write_text("Hello world!\nLine 2", encoding="utf-8")
    assert is_binary_file(text_file) is False

    binary_file = tmp_path / "blob.bin"
    binary_file.write_bytes(b"\x00\x01\x02\x03\x04\xff")
    assert is_binary_file(binary_file) is True


def test_repo_scanner_exclusions(sample_repo: Path):
    """Verify that default exclusions (.git, node_modules, __pycache__, binary, .gitignore) work."""
    scanner = RepoScanner(sample_repo)
    summary = scanner.scan()

    relative_paths = [f.relative_path for f in summary.files]

    # Verify inclusion of valid source and doc files
    assert "src/main.py" in relative_paths
    assert "src/utils.js" in relative_paths
    assert "README.md" in relative_paths
    assert "Dockerfile" in relative_paths

    # Verify exclusion of git, node_modules, pycache
    assert not any(p == ".git" or p.startswith(".git/") for p in relative_paths)
    assert not any("node_modules" in p for p in relative_paths)
    assert not any("__pycache__" in p for p in relative_paths)

    # Verify exclusion of binary files
    assert "image.png" not in relative_paths
    assert not any(p.endswith(".pyc") for p in relative_paths)

    # Verify .gitignore rules are respected
    assert "debug.log" not in relative_paths
    assert not any("secret_folder" in p for p in relative_paths)

    # Verify language counts
    assert summary.files_by_language.get("Python", 0) >= 1
    assert summary.files_by_language.get("JavaScript", 0) >= 1
    assert summary.files_by_language.get("Dockerfile", 0) >= 1


def test_additional_exclusions(sample_repo: Path):
    """Verify that user-supplied additional exclusions are respected."""
    scanner = RepoScanner(sample_repo, additional_exclusions=["src/utils.js", "*.md"])
    summary = scanner.scan()

    relative_paths = [f.relative_path for f in summary.files]
    assert "src/main.py" in relative_paths
    assert "src/utils.js" not in relative_paths
    assert "README.md" not in relative_paths


def test_max_file_size_threshold(tmp_path: Path):
    """Verify that files exceeding the size threshold are excluded."""
    repo = tmp_path / "size_repo"
    repo.mkdir()

    small_file = repo / "small.txt"
    small_file.write_text("short" * 10, encoding="utf-8")

    large_file = repo / "large.txt"
    large_file.write_text("large" * 5000, encoding="utf-8")

    scanner = RepoScanner(repo, max_file_size_bytes=100)
    summary = scanner.scan()

    relative_paths = [f.relative_path for f in summary.files]
    assert "small.txt" in relative_paths
    assert "large.txt" not in relative_paths


def test_invalid_repo_path(tmp_path: Path):
    """Verify that invalid paths raise appropriate exceptions."""
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        RepoScanner(non_existent)

    regular_file = tmp_path / "file.txt"
    regular_file.write_text("content", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        RepoScanner(regular_file)
