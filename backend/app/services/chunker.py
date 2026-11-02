from pathlib import Path
from typing import List, Optional

from backend.app.core.config import get_settings
from backend.app.models.schemas import CodeChunk
from backend.app.services.scanner import detect_language


class CodeChunker:
    """Line-preserving code chunker that maintains precise file path and line number mapping."""

    def __init__(
        self,
        chunk_size_lines: Optional[int] = None,
        chunk_overlap_lines: Optional[int] = None,
    ):
        settings = get_settings()
        self.chunk_size_lines = (
            chunk_size_lines if chunk_size_lines is not None else settings.CHUNK_SIZE_LINES
        )
        self.chunk_overlap_lines = (
            chunk_overlap_lines if chunk_overlap_lines is not None else settings.CHUNK_OVERLAP_LINES
        )

        if self.chunk_size_lines <= 0:
            raise ValueError("chunk_size_lines must be greater than 0")
        if self.chunk_overlap_lines < 0:
            raise ValueError("chunk_overlap_lines cannot be negative")
        if self.chunk_overlap_lines >= self.chunk_size_lines:
            raise ValueError("chunk_overlap_lines must be less than chunk_size_lines")

    def chunk_text(
        self,
        content: str,
        file_path: str = "snippet.txt",
        language: Optional[str] = None,
    ) -> List[CodeChunk]:
        """Split text into overlapping, line-preserving chunks with exact 1-based line numbers."""
        if not content:
            return []

        # Splitlines with keepends=True to preserve exact newlines and whitespace
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)
        if total_lines == 0:
            return []

        if language is None:
            language = detect_language(Path(file_path))

        chunks: List[CodeChunk] = []
        stride = max(1, self.chunk_size_lines - self.chunk_overlap_lines)

        start_idx = 0
        chunk_index = 0

        while start_idx < total_lines:
            end_idx = min(start_idx + self.chunk_size_lines, total_lines)

            # If this is not the first chunk and remaining lines are few,
            # adjust start_idx so the last chunk has adequate context up to total_lines
            chunk_lines = lines[start_idx:end_idx]
            chunk_content = "".join(chunk_lines)

            start_line_1based = start_idx + 1
            end_line_1based = end_idx

            chunk_id = f"{file_path}:{start_line_1based}-{end_line_1based}"

            chunks.append(
                CodeChunk(
                    chunk_id=chunk_id,
                    file_path=file_path,
                    start_line=start_line_1based,
                    end_line=end_line_1based,
                    content=chunk_content,
                    language=language,
                    line_count=len(chunk_lines),
                    chunk_index=chunk_index,
                )
            )

            chunk_index += 1

            if end_idx >= total_lines:
                break

            start_idx += stride

        return chunks

    def chunk_file(
        self,
        file_path: str | Path,
        relative_to: Optional[str | Path] = None,
        language: Optional[str] = None,
    ) -> List[CodeChunk]:
        """Read a file from disk and return line-preserving chunks."""
        path_obj = Path(file_path).resolve()
        if not path_obj.exists() or not path_obj.is_file():
            raise FileNotFoundError(f"File not found: {path_obj}")

        if relative_to is not None:
            try:
                display_path = path_obj.relative_to(Path(relative_to).resolve()).as_posix()
            except ValueError:
                display_path = path_obj.as_posix()
        else:
            display_path = path_obj.name

        if language is None:
            language = detect_language(path_obj)

        with open(path_obj, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return self.chunk_text(content=content, file_path=display_path, language=language)

