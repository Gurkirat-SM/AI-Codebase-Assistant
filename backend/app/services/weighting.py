import re
from pathlib import Path
from typing import Dict, Optional, Set

from backend.app.core.config import get_settings

TEST_PATTERNS = {
    "test", "tests", "testing", "pytest", "unittest", "mock", "assert",
    "fixture", "coverage", "spec", "suite", "verify",
}

CONFIG_PATTERNS = {
    "config", "configuration", "settings", "env", "environment", "docker",
    "dockerfile", "port", "setup", "dependencies", "requirements", "install",
    "build", "package", "toml", "yaml", "yml", "json", "ini", "deploy",
    "deployment", "secrets", "cors", "database_url",
}

DOC_PATTERNS = {
    "architecture", "overview", "what is", "how to install", "readme",
    "usage", "getting started", "guide", "documentation", "explain",
    "system design", "purpose", "summary", "how does the project",
}

CODE_PATTERNS = {
    "function", "class", "method", "implement", "logic", "algorithm",
    "process", "handle", "service", "endpoint", "route", "scanner",
    "chunker", "embedder", "vector", "retriever", "query", "code",
}

DOC_EXTENSIONS = {".md", ".markdown", ".rst", ".txt"}
CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml", ".properties"}
CONFIG_FILENAMES = {
    "dockerfile", "makefile", "gemfile", "cmakelists.txt", "procfile",
    "pytest.ini", "requirements.txt", ".env.example", ".gitignore",
}


def classify_file_type(file_path: str) -> str:
    """Classify a relative or absolute file path into a category:
    'test', 'documentation', 'configuration', or 'code'.
    """
    normalized = file_path.replace("\\", "/").lower()
    path_obj = Path(normalized)
    name = path_obj.name
    parts = normalized.split("/")

    # 1. Test files
    if any(p in ("tests", "test", "__tests__") for p in parts[:-1]):
        return "test"
    if name.startswith("test_") or name.endswith(("_test.py", ".spec.ts", ".spec.js", ".test.ts", ".test.js")):
        return "test"

    # 2. Documentation files
    if path_obj.suffix in DOC_EXTENSIONS:
        return "documentation"
    if any(p in ("docs", "doc", "documentation") for p in parts):
        return "documentation"
    if name.startswith(("readme", "license", "contributing", "changelog", "architecture")):
        return "documentation"

    # 3. Configuration files
    if name in CONFIG_FILENAMES:
        return "configuration"
    if path_obj.suffix in CONFIG_EXTENSIONS:
        return "configuration"
    if "config" in name or ".env" in name or "docker" in name:
        return "configuration"

    # 4. Source code implementation (default)
    return "code"


def detect_query_intent(query: str) -> Dict[str, float]:
    """Analyze query text to determine weight affinities for different file types.
    Returns a dict mapping category name ('code', 'test', 'configuration', 'documentation')
    to its intent weight.
    """
    q_lower = query.lower()
    words: Set[str] = set(re.findall(r"[a-z0-9_]+", q_lower))

    test_score = sum(1.0 for w in TEST_PATTERNS if w in words or w in q_lower)
    config_score = sum(1.0 for w in CONFIG_PATTERNS if w in words or w in q_lower)
    doc_score = sum(1.0 for w in DOC_PATTERNS if w in words or w in q_lower)
    code_score = sum(1.0 for w in CODE_PATTERNS if w in words or w in q_lower)

    # Base neutral affinities
    weights = {
        "code": 1.0,
        "test": 0.90,
        "configuration": 0.95,
        "documentation": 0.95,
    }

    if test_score > 0 and test_score >= config_score and test_score >= doc_score:
        weights["test"] = 1.35
        weights["code"] = 1.05
        weights["configuration"] = 0.90
        weights["documentation"] = 0.85

    elif config_score > 0 and config_score >= test_score and config_score >= doc_score:
        weights["configuration"] = 1.40
        weights["documentation"] = 1.10
        weights["code"] = 0.95
        weights["test"] = 0.80

    elif doc_score > 0 and doc_score >= test_score and doc_score >= config_score:
        weights["documentation"] = 1.40
        weights["code"] = 1.05
        weights["configuration"] = 1.00
        weights["test"] = 0.80

    elif code_score > 0:
        weights["code"] = 1.25
        weights["configuration"] = 0.90
        weights["documentation"] = 0.85
        weights["test"] = 0.80

    return weights


def compute_file_weight(
    file_path: str,
    query: str,
    enable_weighting: Optional[bool] = None,
) -> float:
    """Compute query-dependent relevance multiplier for a file."""
    settings = get_settings()
    enabled = enable_weighting if enable_weighting is not None else settings.ENABLE_FILE_TYPE_WEIGHTING
    if not enabled:
        return 1.0

    category = classify_file_type(file_path)
    intent_weights = detect_query_intent(query)
    return intent_weights.get(category, 1.0)

