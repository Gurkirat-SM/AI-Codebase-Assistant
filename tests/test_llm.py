from unittest.mock import MagicMock, patch
import pytest

from backend.app.models.schemas import CodeChunk
from backend.app.services.llm import GeminiService, SYSTEM_INSTRUCTION


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


def test_build_prompt_structure():
    """Verify that build_prompt correctly formats files, lines, and user question."""
    service = GeminiService(api_key="fake-key")
    chunk1 = create_sample_chunk("auth.py", 10, 15, "def authenticate(): pass")
    chunk2 = create_sample_chunk("db.py", 1, 5, "def connect(): pass")

    prompt = service.build_prompt("How does login work?", [chunk1, chunk2])

    assert "How does login work?" in prompt
    assert "File: auth.py" in prompt
    assert "Lines: 10-15" in prompt
    assert "def authenticate(): pass" in prompt
    assert "File: db.py" in prompt
    assert "Lines: 1-5" in prompt
    assert "def connect(): pass" in prompt


def test_generate_answer_empty_chunks():
    """Verify that empty chunk list returns a graceful message without calling API."""
    mock_client = MagicMock()
    service = GeminiService(client=mock_client)

    result = service.generate_answer("What is this?", [])

    assert "No relevant code context was found" in result
    mock_client.models.generate_content.assert_not_called()


def test_generate_answer_missing_api_key():
    """Verify that missing API key returns an informative setup message."""
    service = GeminiService(api_key="", client=None)
    chunk = create_sample_chunk("main.py", 1, 5, "print('hi')")

    result = service.generate_answer("How does this work?", [chunk])

    assert "Gemini API key is not configured" in result


def test_generate_answer_successful_call():
    """Verify that GeminiService calls generate_content and returns response text."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "In auth.py lines 10-15, login is handled via authenticate()."
    mock_client.models.generate_content.return_value = mock_response

    service = GeminiService(api_key="fake-key", client=mock_client)
    chunk = create_sample_chunk("auth.py", 10, 15, "def authenticate(): pass")

    answer = service.generate_answer("Explain authentication", [chunk])

    assert answer == "In auth.py lines 10-15, login is handled via authenticate()."
    assert mock_client.models.generate_content.called

    call_kwargs = mock_client.models.generate_content.call_args[1]
    assert call_kwargs["model"] == service.model_name
    assert "Explain authentication" in call_kwargs["contents"]
    assert call_kwargs["config"].system_instruction == SYSTEM_INSTRUCTION


def test_generate_answer_api_error_handling():
    """Verify that API exceptions are caught and reported gracefully."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("429 ResourceExhausted: Quota exceeded")

    service = GeminiService(api_key="fake-key", client=mock_client)
    chunk = create_sample_chunk("app.py", 1, 5, "pass")

    answer = service.generate_answer("test query", [chunk])

    assert "An error occurred while generating answer with Gemini" in answer
    assert "Quota exceeded" in answer
