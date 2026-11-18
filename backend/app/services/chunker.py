import ast
from pathlib import Path
from typing import List, Optional

from backend.app.core.config import get_settings
from backend.app.models.schemas import CodeChunk
from backend.app.services.scanner import detect_language


class _CodeBlock:
    """Represents a syntactically coherent code block with exact line spans."""

    def __init__(
        self,
        start_line: int,
        end_line: int,
        block_type: str,
        name: str = "",
        class_name: Optional[str] = None,
        parent_context: Optional[str] = None,
    ):
        self.start_line = start_line
        self.end_line = end_line
        self.block_type = block_type
        self.name = name
        self.class_name = class_name
        self.parent_context = parent_context

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    def __repr__(self) -> str:
        return (
            f"<_CodeBlock {self.block_type}:{self.name} "
            f"lines {self.start_line}-{self.end_line} class={self.class_name}>"
        )


class CodeChunker:
    """Code-aware and line-preserving code chunker.

    Supports AST-based semantic chunking for Python (preserving complete classes,
    methods, functions, decorators, and imports) with a robust, line-preserving
    sliding-window fallback for other languages and unparseable code.
    """

    def __init__(
        self,
        chunk_size_lines: Optional[int] = None,
        chunk_overlap_lines: Optional[int] = None,
        enable_code_aware: Optional[bool] = None,
        max_chunk_lines: Optional[int] = None,
    ):
        settings = get_settings()
        self.chunk_size_lines = (
            chunk_size_lines if chunk_size_lines is not None else settings.CHUNK_SIZE_LINES
        )
        self.chunk_overlap_lines = (
            chunk_overlap_lines if chunk_overlap_lines is not None else settings.CHUNK_OVERLAP_LINES
        )
        self.enable_code_aware = (
            enable_code_aware if enable_code_aware is not None else settings.ENABLE_CODE_AWARE_CHUNKING
        )
        self.max_chunk_lines = (
            max_chunk_lines if max_chunk_lines is not None else settings.MAX_CODE_CHUNK_LINES
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
        """Split text into line-preserving code chunks.

        If Python is detected and code-aware chunking is enabled, uses AST parsing
        to detect class, function, and method boundaries. Otherwise falls back to
        the line-preserving sliding-window algorithm.
        """
        if not content:
            return []

        lines = content.splitlines(keepends=True)
        total_lines = len(lines)
        if total_lines == 0:
            return []

        if language is None:
            language = detect_language(Path(file_path))

        # Single-chunk fast-path if content is already small
        if total_lines <= self.chunk_size_lines:
            return [
                CodeChunk(
                    chunk_id=f"{file_path}:1-{total_lines}",
                    file_path=file_path,
                    start_line=1,
                    end_line=total_lines,
                    content="".join(lines),
                    language=language,
                    line_count=total_lines,
                    chunk_index=0,
                )
            ]

        # Use Python AST code-aware chunker when applicable
        is_python = language.lower() == "python" or file_path.lower().endswith((".py", ".pyw"))
        if is_python and self.enable_code_aware:
            try:
                chunks = self._chunk_python_ast(lines, file_path, language)
                if chunks:
                    return chunks
            except Exception:
                # Fall back gracefully to sliding-window on any AST or parsing error
                pass

        return self._chunk_sliding_window(lines, file_path, language)

    def _chunk_sliding_window(
        self,
        lines: List[str],
        file_path: str,
        language: str,
    ) -> List[CodeChunk]:
        """Classic line-preserving sliding-window chunker with exact line bounds."""
        total_lines = len(lines)
        chunks: List[CodeChunk] = []
        stride = max(1, self.chunk_size_lines - self.chunk_overlap_lines)

        start_idx = 0
        chunk_index = 0

        while start_idx < total_lines:
            end_idx = min(start_idx + self.chunk_size_lines, total_lines)
            chunk_lines = lines[start_idx:end_idx]
            chunk_content = "".join(chunk_lines)

            start_line_1based = start_idx + 1
            end_line_1based = end_idx

            chunks.append(
                CodeChunk(
                    chunk_id=f"{file_path}:{start_line_1based}-{end_line_1based}",
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

    def _chunk_python_ast(
        self,
        lines: List[str],
        file_path: str,
        language: str,
    ) -> List[CodeChunk]:
        """Extract code-aware chunks preserving classes, methods, and functions."""
        source_code = "".join(lines)
        tree = ast.parse(source_code)
        total_lines = len(lines)

        if not tree.body:
            return self._chunk_sliding_window(lines, file_path, language)

        raw_blocks: List[_CodeBlock] = []

        # Find the line where the first top-level definition begins
        first_def_line = total_lines + 1
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                node_start = self._get_node_start_line(node)
                first_def_line = min(first_def_line, node_start)
                break

        # Module preamble (imports, module docstrings, module-level constants)
        if first_def_line > 1:
            raw_blocks.append(
                _CodeBlock(
                    start_line=1,
                    end_line=first_def_line - 1,
                    block_type="preamble",
                    name="module_preamble",
                )
            )

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = self._get_node_start_line(node)
                end = node.end_lineno or start
                raw_blocks.append(
                    _CodeBlock(
                        start_line=start,
                        end_line=end,
                        block_type="function",
                        name=node.name,
                    )
                )

            elif isinstance(node, ast.ClassDef):
                class_start = self._get_node_start_line(node)
                class_end = node.end_lineno or class_start
                class_span = class_end - class_start + 1

                # If the entire class fits comfortably in chunk_size_lines, keep it as one unit
                if class_span <= self.chunk_size_lines:
                    raw_blocks.append(
                        _CodeBlock(
                            start_line=class_start,
                            end_line=class_end,
                            block_type="class",
                            name=node.name,
                            class_name=node.name,
                        )
                    )
                else:
                    # Class is large: decompose into class header and method blocks
                    methods = [
                        m for m in node.body
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]
                    if not methods:
                        # Class without methods: treat as a single block
                        raw_blocks.append(
                            _CodeBlock(
                                start_line=class_start,
                                end_line=class_end,
                                block_type="class",
                                name=node.name,
                                class_name=node.name,
                            )
                        )
                    else:
                        first_method_start = self._get_node_start_line(methods[0])
                        header_end = max(class_start, first_method_start - 1)
                        raw_blocks.append(
                            _CodeBlock(
                                start_line=class_start,
                                end_line=header_end,
                                block_type="class_header",
                                name=f"{node.name}_header",
                                class_name=node.name,
                                parent_context=f"class {node.name}",
                            )
                        )

                        for method in methods:
                            m_start = self._get_node_start_line(method)
                            m_end = method.end_lineno or m_start
                            raw_blocks.append(
                                _CodeBlock(
                                    start_line=m_start,
                                    end_line=m_end,
                                    block_type="method",
                                    name=f"{node.name}.{method.name}",
                                    class_name=node.name,
                                    parent_context=f"class {node.name}",
                                )
                            )

            else:
                # Other top-level statements (e.g. if __name__ == '__main__':)
                start = getattr(node, "lineno", 1)
                end = getattr(node, "end_lineno", start)
                raw_blocks.append(
                    _CodeBlock(
                        start_line=start,
                        end_line=end,
                        block_type="statement",
                        name="statement",
                    )
                )

        if not raw_blocks:
            return self._chunk_sliding_window(lines, file_path, language)

        # Merge adjacent or contiguous blocks into target chunk sizes,
        # splitting blocks that exceed max_chunk_lines.
        assembled_chunks: List[CodeChunk] = []
        chunk_index = 0

        current_start: Optional[int] = None
        current_end: Optional[int] = None
        current_class_name: Optional[str] = None
        current_context: Optional[str] = None

        def emit_chunk(start: int, end: int, cls_name: Optional[str], ctx: Optional[str]):
            nonlocal chunk_index
            s_line = max(1, start)
            e_line = min(total_lines, end)
            if e_line < s_line:
                return

            chunk_lines = lines[s_line - 1 : e_line]
            content_str = "".join(chunk_lines)

            # Strip leading/trailing blank lines only if entire chunk is empty
            if not content_str.strip():
                return

            chunk_id = f"{file_path}:{s_line}-{e_line}"
            assembled_chunks.append(
                CodeChunk(
                    chunk_id=chunk_id,
                    file_path=file_path,
                    start_line=s_line,
                    end_line=e_line,
                    content=content_str,
                    language=language,
                    line_count=len(chunk_lines),
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

        for block in raw_blocks:
            # If this individual block exceeds max_chunk_lines, split it internally
            if block.line_count > self.max_chunk_lines:
                # Flush pending accumulation first
                if current_start is not None and current_end is not None:
                    emit_chunk(current_start, current_end, current_class_name, current_context)
                    current_start, current_end, current_class_name, current_context = None, None, None, None

                # Sub-split oversized block with sliding window
                sub_lines = lines[block.start_line - 1 : block.end_line]
                stride = max(1, self.chunk_size_lines - self.chunk_overlap_lines)
                for sub_offset in range(0, len(sub_lines), stride):
                    sub_end_offset = min(sub_offset + self.chunk_size_lines, len(sub_lines))
                    sub_start_line = block.start_line + sub_offset
                    sub_end_line = block.start_line + sub_end_offset - 1
                    emit_chunk(sub_start_line, sub_end_line, block.class_name, block.parent_context)
                    if sub_end_offset >= len(sub_lines):
                        break
                continue

            # Check if adding this block to the accumulated chunk would exceed chunk_size_lines
            if current_start is None:
                current_start = block.start_line
                current_end = block.end_line
                current_class_name = block.class_name
                current_context = block.parent_context
            else:
                combined_span = block.end_line - current_start + 1
                # If fits in target size and keeps reasonable class affinity:
                if combined_span <= self.chunk_size_lines:
                    current_end = block.end_line
                    if not current_class_name and block.class_name:
                        current_class_name = block.class_name
                else:
                    # Flush accumulated chunk and start a new one
                    emit_chunk(current_start, current_end, current_class_name, current_context)
                    current_start = block.start_line
                    current_end = block.end_line
                    current_class_name = block.class_name
                    current_context = block.parent_context

        if current_start is not None and current_end is not None:
            emit_chunk(current_start, current_end, current_class_name, current_context)

        return assembled_chunks if assembled_chunks else self._chunk_sliding_window(lines, file_path, language)

    def _get_node_start_line(self, node: ast.AST) -> int:
        """Return the start line of an AST node including any decorators."""
        start = getattr(node, "lineno", 1)
        decorator_list = getattr(node, "decorator_list", [])
        if decorator_list:
            for dec in decorator_list:
                dec_line = getattr(dec, "lineno", start)
                if dec_line < start:
                    start = dec_line
        return start

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
