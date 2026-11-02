from pathlib import Path
from backend.app.services.scanner import RepoScanner
from backend.app.services.chunker import CodeChunker


def test_scan_self_project():
    """Verify that RepoScanner can scan this actual codebase successfully."""
    project_root = Path(__file__).resolve().parent.parent
    scanner = RepoScanner(project_root)
    summary = scanner.scan()

    assert summary.total_files > 0
    assert summary.total_lines > 0
    assert "Python" in summary.files_by_language

    relative_paths = [f.relative_path for f in summary.files]
    # Check that .venv is ignored
    assert not any(p.startswith(".venv") for p in relative_paths)
    # Check that backend files are found
    assert any("backend/app/main.py" in p.replace("\\", "/") for p in relative_paths)


def test_chunk_self_project():
    """Verify that CodeChunker can chunk all scanned project files accurately."""
    project_root = Path(__file__).resolve().parent.parent
    scanner = RepoScanner(project_root)
    summary = scanner.scan()

    chunker = CodeChunker(chunk_size_lines=40, chunk_overlap_lines=10)
    total_chunks = 0
    for file_info in summary.files:
        chunks = chunker.chunk_file(
            file_path=file_info.absolute_path,
            relative_to=project_root,
            language=file_info.language,
        )
        total_chunks += len(chunks)
        for chunk in chunks:
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line
            assert len(chunk.content.splitlines()) == chunk.line_count

    assert total_chunks >= summary.total_files

