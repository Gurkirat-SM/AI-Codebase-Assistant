import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pathspec

from backend.app.core.config import get_settings
from backend.app.models.schemas import FileInfo, ScanSummary

# Extension to language name mapping
EXTENSION_LANGUAGE_MAP: Dict[str, str] = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript (React)",
    ".ts": "TypeScript",
    ".tsx": "TypeScript (React)",
    ".vue": "Vue",
    ".svelte": "Svelte",
    ".html": "HTML",
    ".htm": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".sass": "Sass",
    ".less": "Less",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".xml": "XML",
    ".md": "Markdown",
    ".markdown": "Markdown",
    ".rst": "reStructuredText",
    ".txt": "Plain Text",
    ".rs": "Rust",
    ".go": "Go",
    ".java": "Java",
    ".kt": "Kotlin",
    ".c": "C",
    ".h": "C/C++ Header",
    ".cpp": "C++",
    ".hpp": "C/C++ Header",
    ".cc": "C++",
    ".cxx": "C++",
    ".cs": "C#",
    ".php": "PHP",
    ".rb": "Ruby",
    ".sh": "Shell",
    ".bash": "Shell",
    ".zsh": "Shell",
    ".ps1": "PowerShell",
    ".sql": "SQL",
    ".graphql": "GraphQL",
    ".gql": "GraphQL",
    ".proto": "Protocol Buffers",
    ".lua": "Lua",
    ".swift": "Swift",
    ".r": "R",
    ".dart": "Dart",
}

SPECIAL_FILENAME_LANGUAGE_MAP: Dict[str, str] = {
    "dockerfile": "Dockerfile",
    "makefile": "Makefile",
    "gemfile": "Ruby",
    "cmakelists.txt": "CMake",
}


def detect_language(path: Path) -> str:
    """Detect language from filename or extension."""
    name_lower = path.name.lower()
    if name_lower in SPECIAL_FILENAME_LANGUAGE_MAP:
        return SPECIAL_FILENAME_LANGUAGE_MAP[name_lower]
    ext = path.suffix.lower()
    return EXTENSION_LANGUAGE_MAP.get(ext, "Unknown")


def is_binary_file(file_path: Path, sample_size: int = 8192) -> bool:
    """Heuristically determine if a file is binary by checking for null bytes and decoding."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(sample_size)
            if not chunk:
                return False  # Empty file is treated as text
            if b"\x00" in chunk:
                return True
            # Attempt decoding chunk as UTF-8
            try:
                chunk.decode("utf-8")
            except UnicodeDecodeError:
                # If UTF-8 fails, try latin-1 but check control characters
                try:
                    text = chunk.decode("latin-1")
                    # Check ratio of non-printable ASCII control characters
                    non_printable = sum(1 for c in text if ord(c) < 32 and c not in "\r\n\t\f\b")
                    if non_printable / len(text) > 0.3:
                        return True
                except Exception:
                    return True
        return False
    except (OSError, PermissionError):
        return True


class RepoScanner:
    """Scanner that walks a codebase repository with exclusion rules and gitignore support."""

    def __init__(
        self,
        repo_path: str | Path,
        additional_exclusions: Optional[List[str]] = None,
        max_file_size_bytes: Optional[int] = None,
    ):
        self.root_path = Path(repo_path).resolve()
        if not self.root_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {self.root_path}")
        if not self.root_path.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {self.root_path}")

        settings = get_settings()
        self.max_file_size = (
            max_file_size_bytes if max_file_size_bytes is not None else settings.MAX_FILE_SIZE_BYTES
        )
        self.ignore_dirs: Set[str] = set(settings.SCANNER_IGNORE_DIRS)
        self.ignore_patterns: List[str] = list(settings.SCANNER_IGNORE_FILE_PATTERNS)

        if additional_exclusions:
            for item in additional_exclusions:
                cleaned = item.strip()
                if cleaned:
                    if "/" not in cleaned and "\\" not in cleaned and not cleaned.startswith("*"):
                        self.ignore_dirs.add(cleaned)
                    self.ignore_patterns.append(cleaned)

        self._gitignore_spec: Optional[pathspec.PathSpec] = self._load_gitignore_spec()
        self._built_in_spec: pathspec.PathSpec = pathspec.PathSpec.from_lines(
            "gitignore", self.ignore_patterns
        )

    def _load_gitignore_spec(self) -> Optional[pathspec.PathSpec]:
        """Load .gitignore if present in repository root."""
        gitignore_path = self.root_path / ".gitignore"
        if gitignore_path.exists() and gitignore_path.is_file():
            try:
                with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
                    patterns = f.read().splitlines()
                return pathspec.PathSpec.from_lines("gitignore", patterns)
            except Exception:
                return None
        return None

    def _should_ignore_dir(self, dir_name: str, rel_path: Path) -> bool:
        """Check if directory should be skipped during walk."""
        if dir_name in self.ignore_dirs:
            return True
        if dir_name.startswith(".") and dir_name != ".":
            return True

        norm_rel = rel_path.as_posix()
        if self._gitignore_spec and self._gitignore_spec.match_file(norm_rel):
            return True
        if self._built_in_spec.match_file(norm_rel):
            return True
        return False

    def _should_ignore_file(self, file_name: str, rel_path: Path) -> bool:
        """Check if file matches ignore rules."""
        norm_rel = rel_path.as_posix()
        # Direct pattern match
        if self._gitignore_spec and self._gitignore_spec.match_file(norm_rel):
            return True
        if self._built_in_spec.match_file(norm_rel) or self._built_in_spec.match_file(file_name):
            return True
        return False

    def scan(self) -> ScanSummary:
        """Traverse directory and return metadata on all valid text and code files."""
        discovered_files: List[FileInfo] = []
        files_by_language: Dict[str, int] = defaultdict(int)
        total_lines = 0
        total_size = 0

        for current_dir, dir_names, file_names in os.walk(self.root_path):
            current_path = Path(current_dir)
            rel_dir = current_path.relative_to(self.root_path)

            # Prune directories in place so os.walk does not descend into ignored folders
            dir_names[:] = [
                d for d in dir_names
                if not self._should_ignore_dir(d, rel_dir / d)
            ]

            for file_name in file_names:
                file_path = current_path / file_name
                rel_file_path = file_path.relative_to(self.root_path)

                if self._should_ignore_file(file_name, rel_file_path):
                    continue

                try:
                    stat_info = file_path.stat()
                except OSError:
                    continue

                file_size = stat_info.st_size

                # Skip files exceeding maximum threshold
                if file_size > self.max_file_size:
                    continue

                # Skip binary files
                if is_binary_file(file_path):
                    continue

                # Count lines and detect language
                line_count = 0
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        line_count = sum(1 for _ in f)
                except Exception:
                    continue

                language = detect_language(file_path)
                files_by_language[language] += 1
                total_lines += line_count
                total_size += file_size

                discovered_files.append(
                    FileInfo(
                        relative_path=rel_file_path.as_posix(),
                        absolute_path=str(file_path),
                        size_bytes=file_size,
                        line_count=line_count,
                        extension=file_path.suffix.lower(),
                        language=language,
                    )
                )

        # Sort files alphabetically by relative path for deterministic results
        discovered_files.sort(key=lambda x: x.relative_path)

        return ScanSummary(
            scanned_path=str(self.root_path),
            total_files=len(discovered_files),
            total_lines=total_lines,
            total_size_bytes=total_size,
            files_by_language=dict(files_by_language),
            files=discovered_files,
        )
