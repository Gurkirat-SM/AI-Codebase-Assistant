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
    "explain architecture", "how does the system",
}

CODE_PATTERNS = {
    "function", "class", "method", "implement", "logic", "algorithm",
    "process", "handle", "service", "endpoint", "route", "scanner",
    "chunker", "embedder", "vector", "retriever", "query", "code",
    # Explicit source-verification triggers always boost code
    "verify against source", "trace execution", "explain implementation",
    "how does it work", "source code", "source-code",
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
        # Testing questions: prioritize test suites and verification specs
        weights["test"] = 1.45
        weights["code"] = 1.05
        weights["configuration"] = 0.85
        weights["documentation"] = 0.75

    elif config_score > 0 and config_score >= test_score and config_score >= doc_score:
        # Configuration questions: prioritize configs and setup guides
        weights["configuration"] = 1.40
        weights["documentation"] = 1.15
        weights["code"] = 0.95
        weights["test"] = 0.75

    elif doc_score > 0 and doc_score >= test_score and doc_score >= config_score:
        # Architecture & design questions: source code is primary proof; docs are supplementary.
        # Explicit "verify against source code" triggers maximum code prioritization.
        explicit_code_verification = any(
            phrase in q_lower
            for phrase in (
                "source code", "source-code", "in code", "verify against code",
                "actual code", "verify the architecture", "verify against source",
                "trace execution", "explain implementation", "how does it work",
            )
        )
        if explicit_code_verification:
            weights["code"] = 1.70
            weights["documentation"] = 0.80
            weights["configuration"] = 1.05
            weights["test"] = 0.65
        else:
            weights["code"] = 1.45
            weights["documentation"] = 1.10
            weights["configuration"] = 1.10
            weights["test"] = 0.70

    elif code_score > 0:
        # Implementation questions: source code definitions are primary;
        # strongly penalize README and test files in ranking.
        weights["code"] = 1.60
        weights["configuration"] = 0.75
        weights["documentation"] = 0.60
        weights["test"] = 0.65

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


def determine_adaptive_top_k(query: str, requested_top_k: int) -> int:
    """Determine effective retrieval depth based on question complexity.
    For architecture, workflow, and cross-module questions, retrieve more chunks
    from multiple relevant files. Simple questions retain standard top_k.
    Bounded by configurable MAX_ADAPTIVE_TOP_K.
    """
    settings = get_settings()
    if not getattr(settings, "ENABLE_ADAPTIVE_RETRIEVAL", True):
        return requested_top_k

    q_lower = query.lower()
    is_arch = any(
        w in q_lower
        for w in (
            "architecture", "system design", "overview", "structure",
            "how the project works", "how it works", "end to end",
            "end-to-end", "components", "high level", "high-level",
        )
    )
    is_flow = any(
        w in q_lower
        for w in (
            "flow", "pipeline", "workflow", "lifecycle", "interaction",
            "interact", "communicate", "between", "cross-module",
            "data flow", "execution flow", "connect",
        )
    )

    max_k = getattr(settings, "MAX_ADAPTIVE_TOP_K", 15)
    arch_k = getattr(settings, "ARCHITECTURE_TOP_K", 12)
    cross_k = getattr(settings, "CROSS_MODULE_TOP_K", 8)

    if is_arch:
        adaptive_k = max(requested_top_k, arch_k)
    elif is_flow:
        adaptive_k = max(requested_top_k, cross_k)
    else:
        adaptive_k = requested_top_k

    return min(adaptive_k, max_k)

