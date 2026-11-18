from pathlib import Path
import pytest

from backend.app.services.chunker import CodeChunker


def test_chunker_empty_content():
    """Verify empty text produces no chunks."""
    chunker = CodeChunker(chunk_size_lines=50, chunk_overlap_lines=10)
    assert chunker.chunk_text("") == []
    assert chunker.chunk_text(None) == []


def test_chunker_small_content():
    """Verify content with fewer lines than chunk_size produces exactly one chunk with exact lines."""
    chunker = CodeChunker(chunk_size_lines=10, chunk_overlap_lines=2)
    sample = "line 1\nline 2\nline 3\nline 4\n"
    chunks = chunker.chunk_text(sample, file_path="app.py", language="Python")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.file_path == "app.py"
    assert chunk.start_line == 1
    assert chunk.end_line == 4
    assert chunk.line_count == 4
    assert chunk.chunk_index == 0
    assert chunk.chunk_id == "app.py:1-4"
    assert chunk.content == sample
    assert chunk.language == "Python"


def test_chunker_line_preservation():
    """Verify that chunk content exactly preserves lines and line numbers."""
    # Generate 100 identifiable lines
    lines = [f"const line_{i} = {i};\n" for i in range(1, 101)]
    full_text = "".join(lines)

    chunk_size = 30
    overlap = 10
    chunker = CodeChunker(chunk_size_lines=chunk_size, chunk_overlap_lines=overlap)
    chunks = chunker.chunk_text(full_text, file_path="src/index.js", language="JavaScript")

    assert len(chunks) > 1

    # Check each chunk's line-by-line fidelity
    for chunk in chunks:
        expected_slice = lines[chunk.start_line - 1 : chunk.end_line]
        expected_content = "".join(expected_slice)
        assert chunk.content == expected_content
        assert chunk.line_count == len(expected_slice)
        assert chunk.end_line - chunk.start_line + 1 == chunk.line_count

    # Verify coverage: all 100 lines must be covered
    covered_lines = set()
    for chunk in chunks:
        for line_num in range(chunk.start_line, chunk.end_line + 1):
            covered_lines.add(line_num)

    assert covered_lines == set(range(1, 101))

    # Verify overlaps between adjacent chunks
    for i in range(len(chunks) - 1):
        c1 = chunks[i]
        c2 = chunks[i + 1]
        assert c2.start_line <= c1.end_line
        assert c2.chunk_index == c1.chunk_index + 1


def test_chunker_validation_errors():
    """Verify that invalid configurations raise ValueError."""
    with pytest.raises(ValueError):
        CodeChunker(chunk_size_lines=0)

    with pytest.raises(ValueError):
        CodeChunker(chunk_size_lines=-5)

    with pytest.raises(ValueError):
        CodeChunker(chunk_size_lines=20, chunk_overlap_lines=20)

    with pytest.raises(ValueError):
        CodeChunker(chunk_size_lines=20, chunk_overlap_lines=25)

    with pytest.raises(ValueError):
        CodeChunker(chunk_size_lines=20, chunk_overlap_lines=-1)


def test_chunk_file_method(tmp_path: Path):
    """Verify chunk_file correctly reads from disk and formats relative paths."""
    test_file = tmp_path / "sample.py"
    file_content = "\n".join([f"def func_{i}(): pass" for i in range(1, 51)]) + "\n"
    test_file.write_text(file_content, encoding="utf-8")

    chunker = CodeChunker(chunk_size_lines=20, chunk_overlap_lines=5)
    chunks = chunker.chunk_file(test_file, relative_to=tmp_path)

    assert len(chunks) >= 3
    assert chunks[0].file_path == "sample.py"
    assert chunks[0].language == "Python"
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 20


def test_code_aware_python_function_preservation():
    """Verify that complete Python functions are preserved without arbitrary mid-function cuts."""
    code = (
        "import os\n"
        "\n"
        "@decorator\n"
        "def compute_total(a: int, b: int) -> int:\n"
        "    # Line 1 of body\n"
        "    temp = a * 2\n"
        "    temp += b * 3\n"
        "    return temp\n"
        "\n"
        "def secondary_helper(x: int) -> int:\n"
        "    return x + 1\n"
    )
    chunker = CodeChunker(chunk_size_lines=15, chunk_overlap_lines=3, enable_code_aware=True)
    chunks = chunker.chunk_text(code, file_path="math_utils.py", language="Python")

    assert len(chunks) >= 1
    # Check that decorator and entire function body are kept together in a chunk
    found = False
    for c in chunks:
        if "compute_total" in c.content:
            found = True
            assert "@decorator" in c.content
            assert "return temp" in c.content
            # Check line correspondence
            expected = "".join(code.splitlines(keepends=True)[c.start_line - 1 : c.end_line])
            assert c.content == expected
    assert found


def test_code_aware_class_splitting_and_context():
    """Verify that large classes are split at method boundaries and keep method integrity."""
    class_code = (
        "class DataRepository:\n"
        "    '''Main data access repository.'''\n"
        "    DB_NAME = 'production'\n"
        "\n"
        "    def __init__(self, connection_string: str):\n"
        "        self.conn = connection_string\n"
        "        self.is_connected = False\n"
        "\n"
        "    def connect(self) -> bool:\n"
        "        self.is_connected = True\n"
        "        return True\n"
        "\n"
        "    def query_records(self, sql: str) -> list:\n"
        "        if not self.is_connected:\n"
        "            raise RuntimeError('Not connected')\n"
        "        return [1, 2, 3]\n"
    )
    # Small chunk_size_lines forces splitting of the class into methods
    chunker = CodeChunker(chunk_size_lines=8, chunk_overlap_lines=2, enable_code_aware=True)
    chunks = chunker.chunk_text(class_code, file_path="repo.py", language="Python")

    assert len(chunks) >= 2
    # Verify exact line slices for all chunks
    lines = class_code.splitlines(keepends=True)
    for c in chunks:
        expected = "".join(lines[c.start_line - 1 : c.end_line])
        assert c.content == expected
        assert c.line_count == len(lines[c.start_line - 1 : c.end_line])

    # Method boundaries must not be split
    query_chunk = next(c for c in chunks if "query_records" in c.content)
    assert "def query_records" in query_chunk.content
    assert "return [1, 2, 3]" in query_chunk.content


def test_code_aware_syntax_error_fallback():
    """Verify that Python files with syntax errors gracefully fall back to sliding window."""
    malformed_code = (
        "def broken(\n"
        "    this is invalid python syntax !!!\n"
        "    for x in:\n"
        "        pass\n"
        "return 42\n"
    )
    chunker = CodeChunker(chunk_size_lines=3, chunk_overlap_lines=1, enable_code_aware=True)
    chunks = chunker.chunk_text(malformed_code, file_path="broken.py", language="Python")

    assert len(chunks) >= 2
    lines = malformed_code.splitlines(keepends=True)
    for c in chunks:
        expected = "".join(lines[c.start_line - 1 : c.end_line])
        assert c.content == expected


