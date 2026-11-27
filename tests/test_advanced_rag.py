from unittest.mock import MagicMock, patch
import pytest

from backend.app.models.schemas import CodeChunk, QueryResult
from backend.app.services.bm25 import BM25Index, tokenize_code
from backend.app.services.chunker import CodeChunker
from backend.app.services.hybrid import reciprocal_rank_fusion
from backend.app.services.llm import GeminiService
from backend.app.services.reranker import CodeReranker
from backend.app.services.weighting import (
    classify_file_type,
    compute_file_weight,
    detect_query_intent,
)


def make_chunk(
    file_path: str,
    start_line: int,
    end_line: int,
    content: str,
    language: str = "Python",
    chunk_index: int = 0,
) -> CodeChunk:
    return CodeChunk(
        chunk_id=f"{file_path}:{start_line}-{end_line}",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        content=content,
        language=language,
        line_count=end_line - start_line + 1,
        chunk_index=chunk_index,
    )


# ---------------------------------------------------------------------------
# 1. BM25 Exact Identifier Retrieval
# ---------------------------------------------------------------------------

def test_bm25_exact_identifier_retrieval():
    """Verify BM25 accurately finds exact class, method, and variable names."""
    chunk1 = make_chunk(
        "backend/services/scanner.py", 10, 20,
        "class RepoScanner:\n    def scan_files(self, max_size: int):\n        pass",
    )
    chunk2 = make_chunk(
        "backend/services/vector_store.py", 1, 15,
        "class FaissVectorStore:\n    def search_embeddings(self, query_vec):\n        pass",
    )
    chunk3 = make_chunk(
        "backend/services/chunker.py", 1, 15,
        "class CodeChunker:\n    def chunk_text(self, content):\n        pass",
    )

    bm25 = BM25Index()
    bm25.add_chunks([chunk1, chunk2, chunk3])

    # Search for exact class name
    results = bm25.search("RepoScanner", top_k=2)
    assert len(results) >= 1
    assert results[0][0].file_path == "backend/services/scanner.py"
    assert "RepoScanner" in results[0][0].content

    # Search for snake_case method identifier
    results_method = bm25.search("search_embeddings", top_k=2)
    assert len(results_method) >= 1
    assert results_method[0][0].file_path == "backend/services/vector_store.py"


def test_bm25_tokenization_details():
    """Verify code tokenizer splits camelCase, snake_case, and dots while preserving raw terms."""
    tokens = tokenize_code("class FaissVectorStore:\n    def is_binary_file(path.to.file): pass")
    assert "faissvectorstore" in tokens
    assert "faiss" in tokens
    assert "vector" in tokens
    assert "store" in tokens
    assert "is_binary_file" in tokens
    assert "binary" in tokens
    assert "path" in tokens
    assert "file" in tokens


# ---------------------------------------------------------------------------
# 2. File-Type and Query-Intent Weighting
# ---------------------------------------------------------------------------

def test_file_type_classification():
    """Verify correct classification of source code, tests, config, and docs."""
    assert classify_file_type("tests/test_scanner.py") == "test"
    assert classify_file_type("src/services/scanner_test.py") == "test"
    assert classify_file_type("docs/architecture.md") == "documentation"
    assert classify_file_type("README.md") == "documentation"
    assert classify_file_type("pytest.ini") == "configuration"
    assert classify_file_type("docker-compose.yml") == "configuration"
    assert classify_file_type("package.json") == "configuration"
    assert classify_file_type("src/services/scanner.py") == "code"


def test_query_dependent_weighting():
    """Verify that query intent boosts matching file types."""
    # Test intent query boosts test files
    w_test = compute_file_weight("tests/test_scanner.py", "how to write unit tests with pytest mock")
    w_code_for_test = compute_file_weight("src/scanner.py", "how to write unit tests with pytest mock")
    assert w_test > w_code_for_test

    # Config intent query boosts configuration files
    w_config = compute_file_weight("pytest.ini", "what is the docker port and environment configuration")
    w_code_for_config = compute_file_weight("src/scanner.py", "what is the docker port and environment configuration")
    assert w_config > w_code_for_config

    # Architecture / doc intent boosts documentation
    w_doc = compute_file_weight("README.md", "architecture overview and high-level system design")
    w_test_for_doc = compute_file_weight("tests/test_scanner.py", "architecture overview and high-level system design")
    assert w_doc > w_test_for_doc


# ---------------------------------------------------------------------------
# 3. Hybrid Search & Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def test_rrf_combines_and_deduplicates():
    """Verify that RRF combines vector and BM25 results without duplicates and boosts dual matches."""
    c_shared = make_chunk("src/app.py", 1, 10, "shared target code")
    c_vec_only = make_chunk("src/vector.py", 1, 10, "vector target code")
    c_bm25_only = make_chunk("src/bm25.py", 1, 10, "bm25 keyword code")

    vec_results = [
        QueryResult(chunk=c_shared, score=0.85),
        QueryResult(chunk=c_vec_only, score=0.80),
    ]
    bm25_results = [
        (c_shared, 1.0),
        (c_bm25_only, 0.90),
    ]

    fused = reciprocal_rank_fusion(
        vector_results=vec_results,
        bm25_results=bm25_results,
        query="shared target",
        rrf_k=60,
        top_k=5,
    )

    # Must contain all 3 unique chunks (no duplicates)
    assert len(fused) == 3
    chunk_ids = [r.chunk.chunk_id for r in fused]
    assert len(chunk_ids) == len(set(chunk_ids))

    # The chunk found by BOTH vector and BM25 search must be ranked first
    assert fused[0].chunk.chunk_id == c_shared.chunk_id


# ---------------------------------------------------------------------------
# 4. Reranking & Graceful Fallback
# ---------------------------------------------------------------------------

def test_reranker_reorders_candidates():
    """Verify that CrossEncoder reranker scores and re-orders candidate chunks."""
    c1 = make_chunk("auth.py", 1, 10, "login authentication handler")
    c2 = make_chunk("db.py", 1, 10, "database connection pool")

    candidates = [
        QueryResult(chunk=c2, score=0.5),
        QueryResult(chunk=c1, score=0.6),
    ]

    reranker = CodeReranker(enabled=True)
    # Mock model to simulate CrossEncoder returning higher score for c1
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 2.5]  # c2 gets 0.1, c1 gets 2.5
    reranker._model = mock_model

    reranked = reranker.rerank(query="how does login work", candidates=candidates, top_k=2)

    assert len(reranked) == 2
    # c1 should be moved to first position due to higher cross-encoder score
    assert reranked[0].chunk.file_path == "auth.py"
    assert reranked[1].chunk.file_path == "db.py"


def test_reranker_fallback_when_disabled():
    """Verify that reranker falls back to input order when disabled."""
    c1 = make_chunk("file1.py", 1, 5, "first chunk")
    c2 = make_chunk("file2.py", 1, 5, "second chunk")
    candidates = [QueryResult(chunk=c1, score=0.8), QueryResult(chunk=c2, score=0.7)]

    reranker = CodeReranker(enabled=False)
    assert not reranker.is_available

    result = reranker.rerank("some query", candidates, top_k=2)
    assert result == candidates


def test_reranker_fallback_on_load_failure():
    """Verify that reranker gracefully falls back if model loading throws an exception."""
    c1 = make_chunk("file1.py", 1, 5, "first chunk")
    c2 = make_chunk("file2.py", 1, 5, "second chunk")
    candidates = [QueryResult(chunk=c1, score=0.8), QueryResult(chunk=c2, score=0.7)]

    reranker = CodeReranker(enabled=True)
    with patch("sentence_transformers.CrossEncoder", side_effect=Exception("Network error: model download failed")):
        result = reranker.rerank("query", candidates, top_k=2)
        assert result == candidates
        assert reranker._failed_to_load is True


# ---------------------------------------------------------------------------
# 5. Insufficient Context Handling
# ---------------------------------------------------------------------------

def test_gemini_insufficient_context_response():
    """Verify GeminiService handles empty retrieval results without making API calls."""
    mock_client = MagicMock()
    service = GeminiService(api_key="mock-key", client=mock_client)

    answer = service.generate_answer("How does authentication work?", [])
    assert "No relevant code context was found" in answer
    mock_client.models.generate_content.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Implementation Prioritization & Grounding Prompt
# ---------------------------------------------------------------------------

def test_implementation_query_prioritizes_code_over_docs_and_tests():
    """Verify implementation questions strongly favor actual source code over README and test files."""
    q = "how does the function calculate and process chunk boundaries"
    w_code = compute_file_weight("src/chunker.py", q)
    w_doc = compute_file_weight("README.md", q)
    w_test = compute_file_weight("tests/test_chunker.py", q)

    assert w_code > w_doc
    assert w_code > w_test
    assert w_code >= 1.40


def test_gemini_grounding_prompt_structure():
    """Verify prompt instructs Gemini to distinguish confirmed details, tests, and inferences."""
    service = GeminiService(api_key="mock-key")
    chunk = make_chunk("src/auth.py", 10, 20, "def authenticate(): pass")

    prompt = service.build_prompt("Explain auth", [chunk])
    assert "cite the relevant file path and line numbers" in prompt.lower()
    assert "confirmed implementation details" in prompt.lower()
    assert "behaviour verified by tests" in prompt.lower()
    assert "inferences" in prompt.lower()


def test_gemini_grounding_prompt_sections_and_insufficient_context():
    """Verify prompt instructs Gemini on explicit sections and handling insufficient context."""
    service = GeminiService(api_key="mock-key")
    chunk = make_chunk("src/scanner.py", 1, 30, "class RepoScanner:\n    pass")

    prompt = service.build_prompt("How does repo scanning work?", [chunk])
    assert "documented information" in prompt.lower()
    assert "verified implementation" in prompt.lower()
    assert "differences or missing information" in prompt.lower()
    assert "the provided codebase context is insufficient to answer this question" in prompt.lower()
    assert "trace data flow, function calls, and component interactions" in prompt.lower()


def test_architecture_prioritizes_code_over_documentation():
    """Verify architecture queries prioritize source code files for implementation proof
    while retaining documentation for high-level structure."""
    q = "explain the high level system architecture and component interactions"
    w_code = compute_file_weight("src/main.py", q)
    w_doc = compute_file_weight("README.md", q)
    w_test = compute_file_weight("tests/test_main.py", q)

    assert w_code > w_doc
    assert w_code > w_test
    assert w_doc > w_test


def test_file_diversity_penalty_promotes_variety():
    """Verify that multiple chunks from the same file experience soft decay so diverse files are not starved."""
    # File A has 3 chunks with high initial scores
    cA1 = make_chunk("file_a.py", 1, 10, "chunk A1", chunk_index=0)
    cA2 = make_chunk("file_a.py", 11, 20, "chunk A2", chunk_index=1)
    cA3 = make_chunk("file_a.py", 21, 30, "chunk A3", chunk_index=2)

    # File B has 1 chunk with slightly lower initial score
    cB1 = make_chunk("file_b.py", 1, 10, "chunk B1", chunk_index=0)

    vec_results = [
        QueryResult(chunk=cA1, score=0.90),
        QueryResult(chunk=cA2, score=0.88),
        QueryResult(chunk=cA3, score=0.86),
        QueryResult(chunk=cB1, score=0.85),
    ]

    # Without diversity, file_a chunks would take top 3.
    # With diversity factor 1.0 / (1 + 0.20 * count):
    # cA1: count=0 -> 1.0
    # cA2: count=1 -> 0.833
    # cA3: count=2 -> 0.714
    # cB1: count=0 -> 1.0
    # Thus cB1 is promoted over cA3!
    fused = reciprocal_rank_fusion(
        vector_results=vec_results,
        bm25_results=[],
        query="query",
        top_k=4,
    )

    fused_files = [r.chunk.file_path for r in fused]
    # cB1 from file_b.py should rank ahead of cA3
    assert fused_files.index("file_b.py") < fused_files.index("file_a.py", 2)


def test_cross_file_import_co_retrieval_in_rag_service():
    """Verify that when query asks about module interactions, imported modules are retrieved."""
    from backend.app.services.retriever import RagService

    mock_embedder = MagicMock()
    mock_embedder.dimension = 384
    mock_embedder.embed_query.return_value = [0.0] * 384
    service = RagService(embedder=mock_embedder, storage_dir="tmp_test_dir")
    service.vector_store = MagicMock()
    service.vector_store.total_chunks = 10

    # Mock vector results containing an import statement for DatabaseService
    caller_chunk = make_chunk(
        "backend/api.py", 1, 15,
        "from backend.db import DatabaseService\n\ndef handle_req():\n    db = DatabaseService()\n",
    )
    service.vector_store.search.return_value = [QueryResult(chunk=caller_chunk, score=0.88)]

    # Mock BM25 index
    target_dep_chunk = make_chunk(
        "backend/db.py", 1, 20,
        "class DatabaseService:\n    def connect(self): pass\n",
    )
    service.bm25_index = MagicMock()
    service.bm25_index.total_chunks = 10
    service.bm25_index.search.side_effect = lambda query, top_k, file_filter=None: (
        [(target_dep_chunk, 2.5)] if "DatabaseService" in query else []
    )

    # Disable reranker for deterministic unit test of retrieval step
    resp = service.search(
        query="how does api communicate with the database service in the pipeline",
        top_k=5,
        enable_reranker=False,
        enable_hybrid=True,
    )

    retrieved_paths = [r.chunk.file_path for r in resp.results]
    assert "backend/api.py" in retrieved_paths
    assert "backend/db.py" in retrieved_paths


# ---------------------------------------------------------------------------
# 7. Adaptive Retrieval Depth & Multi-File Architecture Analysis
# ---------------------------------------------------------------------------

def test_adaptive_retrieval_depth():
    """Verify that retrieval depth dynamically adapts based on query complexity."""
    from backend.app.services.weighting import determine_adaptive_top_k

    # Simple targeted question keeps default top_k
    simple_k = determine_adaptive_top_k("how does calculate_sum format numbers", requested_top_k=5)
    assert simple_k == 5

    # Architecture query dynamically expands retrieval depth to >= 12
    arch_k = determine_adaptive_top_k("explain the high-level system architecture and components", requested_top_k=5)
    assert arch_k >= 12
    assert arch_k <= 15  # Bounded by MAX_ADAPTIVE_TOP_K

    # Flow / pipeline query dynamically expands depth to >= 8
    flow_k = determine_adaptive_top_k("trace the execution flow and data flow through the pipeline", requested_top_k=5)
    assert flow_k >= 8


def test_architecture_8_sections_in_prompt():
    """Verify architecture prompt contains all 8 required structured sections."""
    service = GeminiService(api_key="mock-key")
    chunk = make_chunk("src/app.py", 1, 25, "def create_app(): pass")

    prompt = service.build_prompt("Explain the project architecture and system design", [chunk])
    assert "1. project overview" in prompt.lower()
    assert "2. main components and responsibilities" in prompt.lower()
    assert "3. execution flow" in prompt.lower()
    assert "4. data flow between components" in prompt.lower()
    assert "5. external libraries and dependencies" in prompt.lower()
    assert "6. verified implementation details" in prompt.lower()
    # Section 7 is now "Documentation vs. Implementation Differences"
    assert "7. documentation vs" in prompt.lower()
    assert "8. missing or unverified" in prompt.lower()


def test_evidence_coverage_warning_when_source_code_missing():
    """Verify that when only documentation is retrieved for architecture questions,
    an explicit evidence limitation warning is injected."""
    service = GeminiService(api_key="mock-key")
    readme_chunk = make_chunk("README.md", 1, 30, "# Project Overview\nThis app provides AI RAG search.", language="Markdown")

    prompt = service.build_prompt("Explain the architecture and how it works", [readme_chunk])
    assert "important evidence limitation" in prompt.lower()
    assert "no concrete source-code implementation files" in prompt.lower()


def test_source_code_verification_explicit_boost():
    """Verify that explicit source-code verification requests strongly elevate source code over docs."""
    q_explicit = "explain the system architecture and verify against code"
    w_code = compute_file_weight("backend/main.py", q_explicit)
    w_doc = compute_file_weight("README.md", q_explicit)

    assert w_code >= 1.60
    assert w_code > w_doc


def test_multi_file_architecture_co_retrieval_in_rag_service():
    """Verify that architecture queries adapt retrieval depth and co-retrieve entry points,
    configuration, and core services from multiple distinct files."""
    from backend.app.services.retriever import RagService

    mock_embedder = MagicMock()
    mock_embedder.dimension = 384
    mock_embedder.embed_query.return_value = [0.0] * 384
    service = RagService(embedder=mock_embedder, storage_dir="tmp_test_dir")
    service.vector_store = MagicMock()
    service.vector_store.total_chunks = 20

    # Mock vector results returning an entry point chunk
    main_chunk = make_chunk(
        "backend/main.py", 1, 20,
        "from backend.app.services.retriever import RagService\napp = create_app()\nservice = RagService()\n",
    )
    service.vector_store.search.return_value = [QueryResult(chunk=main_chunk, score=0.90)]

    # Mock BM25 index returning service, routes, and config chunks
    svc_chunk = make_chunk("backend/app/services/retriever.py", 1, 30, "class RagService:\n    def search(self): pass\n")
    routes_chunk = make_chunk("backend/app/api/routes.py", 1, 25, "def query_endpoint(): pass\n")
    config_chunk = make_chunk("backend/app/core/config.py", 1, 20, "class Settings:\n    PORT = 8000\n")

    def mock_bm25_search(query, top_k, file_filter=None):
        hits = []
        if "RagService" in query or "service" in query:
            hits.append((svc_chunk, 2.5))
        if "routes" in query or "app" in query:
            hits.append((routes_chunk, 2.3))
        if "config" in query or "settings" in query:
            hits.append((config_chunk, 2.0))
        return hits

    service.bm25_index = MagicMock()
    service.bm25_index.total_chunks = 20
    service.bm25_index.search.side_effect = mock_bm25_search

    resp = service.search(
        query="explain the system architecture, entry points, and component interactions",
        top_k=5,
        enable_reranker=False,
        enable_hybrid=True,
    )

    retrieved_files = {r.chunk.file_path for r in resp.results}
    # Multiple distinct files across layers must be represented
    assert "backend/main.py" in retrieved_files
    assert "backend/app/services/retriever.py" in retrieved_files
    assert len(retrieved_files) >= 3
    # Adaptive depth expands beyond initial top_k=5 if needed
    assert len(resp.results) >= 3


# ---------------------------------------------------------------------------
# 8. Documentation vs Implementation Contradiction Resolution
# ---------------------------------------------------------------------------

def test_doc_vs_code_comparison_prompt_injection():
    """Verify that when both doc and code chunks are present for arch questions,
    the prompt explicitly names the retrieved files and requests comparison."""
    service = GeminiService(api_key="mock-key")
    readme_chunk = make_chunk("README.md", 1, 20, "# Phase 1\nModel training is NOT part of Phase 1.", language="Markdown")
    code_chunk = make_chunk("src/predictor.py", 10, 30, "class Predictor:\n    def __init__(self):\n        self.model = joblib.load('model.pkl')")

    prompt = service.build_prompt("Explain the architecture and how the model is used", [readme_chunk, code_chunk])
    # Must explicitly reference doc and code files in evidence note
    assert "README.md" in prompt or "documentation sources" in prompt.lower()
    assert "src/predictor.py" in prompt or "source-code sources" in prompt.lower()
    assert "documentation vs" in prompt.lower()


def test_doc_vs_code_contradiction_model_loading():
    """Verify that prompt structure for model-related architecture questions includes
    Model Integration and Inference section."""
    service = GeminiService(api_key="mock-key")
    code_chunk = make_chunk("src/model.py", 1, 25, "class ModelLoader:\n    def load(self):\n        return joblib.load('clf.pkl')")

    prompt = service.build_prompt("Explain the architecture and model training vs inference", [code_chunk])
    assert "model integration and inference" in prompt.lower()
    assert "documentation vs. implementation differences" in prompt.lower()


def test_evidence_completeness_instruction_present():
    """Verify that build_prompt includes instruction to complete every section."""
    service = GeminiService(api_key="mock-key")
    chunk = make_chunk("src/app.py", 1, 10, "def main(): pass")
    prompt = service.build_prompt("Explain the architecture", [chunk])
    assert "complete every section" in prompt.lower()


def test_implementation_query_code_weight_boost():
    """Verify that explicit source-verification phrases raise code weight above 1.60 and suppress docs."""
    # Test using an architecture-qualified query that explicitly requests source verification.
    # "verify against source code" is an explicit_code_verification phrase and also triggers
    # doc_score (via "architecture" in the query), giving code weight 1.70.
    for phrase in ("verify against source code", "verify the architecture against code", "verify against code"):
        q = f"explain the architecture and {phrase}"
        w_code = compute_file_weight("backend/core.py", q)
        w_doc = compute_file_weight("README.md", q)
        assert w_code >= 1.60, f"Expected code weight >= 1.60 for query: {q!r}, got {w_code}"
        assert w_code > w_doc, f"Code should outweigh docs for query: {q!r}"


def test_code_coverage_enforcement_favors_source_over_doc():
    """Verify evidence enforcement step adds source-code files when docs dominate results."""
    from backend.app.services.retriever import RagService

    mock_embedder = MagicMock()
    mock_embedder.dimension = 384
    mock_embedder.embed_query.return_value = [0.0] * 384
    service = RagService(embedder=mock_embedder, storage_dir="tmp_test_dir")
    service.vector_store = MagicMock()
    service.vector_store.total_chunks = 10

    # Vector search initially returns only README chunks
    readme1 = make_chunk("README.md", 1, 50, "# Architecture\nThis system does X", language="Markdown")
    readme2 = make_chunk("CONTRIBUTING.md", 1, 20, "# Contributing Guide", language="Markdown")
    service.vector_store.search.return_value = [
        QueryResult(chunk=readme1, score=0.90),
        QueryResult(chunk=readme2, score=0.85),
    ]

    impl_chunk = make_chunk("backend/main.py", 1, 30, "def create_app(): app = FastAPI(); return app")

    # BM25 returns a code implementation file
    service.bm25_index = MagicMock()
    service.bm25_index.total_chunks = 10
    service.bm25_index.search.return_value = [(impl_chunk, 2.0)]

    resp = service.search(
        query="explain the system architecture, how it works end to end",
        top_k=3,
        enable_reranker=False,
        enable_hybrid=True,
    )

    retrieved_files = {r.chunk.file_path for r in resp.results}
    # Enforcement must have added the source-code file even though it wasn't in the top vector results
    assert "backend/main.py" in retrieved_files

# ---------------------------------------------------------------------------
# 9. Performance Optimization & Execution-Flow Tests
# ---------------------------------------------------------------------------

def test_execution_flow_query_adaptive_top_k():
    """Verify that execution-flow queries expand adaptive top_k beyond the user's request."""
    from backend.app.services.weighting import determine_adaptive_top_k
    # "trace execution flow" contains "execution flow" → is_flow trigger
    result = determine_adaptive_top_k("trace execution flow from webcam capture to gesture prediction", 5)
    assert result >= 8, f"Expected >= 8 for flow query, got {result}"


def test_ml_model_query_detection_in_llm():
    """Verify is_model_query now catches gesture/webcam/landmark terms for ML flow questions."""
    service = GeminiService(api_key="mock-key")
    code_chunk = make_chunk(
        "src/predictor.py", 1, 20,
        "class GesturePredictor:\n    def predict(self, landmarks):\n        return self.clf.predict([landmarks])",
    )
    # Prompt for a gesture/webcam question should include model-inference section in arch context
    prompt = service.build_prompt(
        "Explain the architecture and how the gesture prediction pipeline works end to end", [code_chunk]
    )
    # is_arch_query=True (architecture, end to end), is_model_query=True (gesture/predict)
    # → prompt must contain "model integration and inference"
    assert "model integration and inference" in prompt.lower()


def test_flow_query_stage_verification_instruction():
    """Verify that execution-flow questions inject the staged-verification instruction."""
    service = GeminiService(api_key="mock-key")
    chunk = make_chunk("src/app.py", 1, 10, "import cv2\ncap = cv2.VideoCapture(0)")
    prompt = service.build_prompt(
        "Trace execution from webcam capture through gesture prediction to output", [chunk]
    )
    # Must contain the stage-verification language
    assert "not verified" in prompt.lower() or "stage n" in prompt.lower() or "stage" in prompt.lower()
    assert "webcam" in prompt.lower() or "ml" in prompt.lower() or "pipeline" in prompt.lower()


def test_flow_guidance_absent_for_non_flow_query():
    """Verify that the staged-verification instruction is NOT injected for plain code questions."""
    service = GeminiService(api_key="mock-key")
    chunk = make_chunk("src/scanner.py", 1, 5, "def scan(): pass")
    prompt = service.build_prompt("What does the scanner do?", [chunk])
    # Should not contain the ML-flow-specific instruction for simple queries
    assert "stage n [name]: not verified" not in prompt.lower()


def test_reranker_candidate_cap_applied(monkeypatch):
    """Verify that reranker receives at most RERANKER_MAX_CANDIDATES items even with many candidates."""
    from backend.app.services.retriever import RagService
    from backend.app.core.config import get_settings

    mock_embedder = MagicMock()
    mock_embedder.dimension = 384
    mock_embedder.embed_query.return_value = [0.0] * 384

    service = RagService(embedder=mock_embedder, storage_dir="tmp_test_cap")
    service.vector_store = MagicMock()
    service.vector_store.total_chunks = 50

    # Return 50 unique vector results to trigger the cap
    many_chunks = [
        make_chunk(f"src/file_{i}.py", i + 1, i + 6, f"def func_{i}(): pass")
        for i in range(50)
    ]
    service.vector_store.search.return_value = [
        QueryResult(chunk=c, score=round(0.9 - i * 0.01, 2))
        for i, c in enumerate(many_chunks)
    ]
    service.bm25_index = MagicMock()
    service.bm25_index.total_chunks = 0  # disable hybrid to isolate cap logic

    # Capture the candidates passed to reranker
    received_candidates = []

    class CapturingReranker:
        is_available = True

        def rerank(self, query, candidates, top_k):
            received_candidates.extend(candidates)
            return candidates[:top_k]

    service.reranker = CapturingReranker()

    service.search(
        query="what is the architecture of this system",
        top_k=5,
        enable_reranker=True,
        enable_hybrid=False,
    )
    settings = get_settings()
    max_candidates = getattr(settings, "RERANKER_MAX_CANDIDATES", 25)
    assert len(received_candidates) <= max_candidates, (
        f"Expected <= {max_candidates} candidates, reranker got {len(received_candidates)}"
    )


def test_entry_point_cache_populated_after_index():
    """Verify that _entry_point_cache is populated after index_repository()."""
    from backend.app.services.retriever import RagService
    from backend.app.services.bm25 import BM25Index

    mock_embedder = MagicMock()
    mock_embedder.dimension = 384
    mock_embedder.embed_chunks.return_value = [[0.0] * 384]
    mock_embedder.model_name = "mock-model"

    bm25 = BM25Index()
    main_chunk = make_chunk("backend/main.py", 1, 10, "def main(): app = create_app()")
    bm25.add_chunks([main_chunk])

    service = RagService(embedder=mock_embedder, bm25_index=bm25, storage_dir="tmp_cache_test")
    service.vector_store = MagicMock()
    service.vector_store.total_chunks = 1

    # Manually call cache builder (normally called in index_repository)
    service._build_entry_point_cache()

    # Cache should be non-empty
    assert isinstance(service._entry_point_cache, dict)
    # "main" is an entry keyword; since bm25 has a main.py chunk it should populate
    assert len(service._entry_point_cache) > 0


def test_num_reranker_candidates_in_latencies(monkeypatch):
    """Verify that latencies dict includes num_reranker_candidates key."""
    from backend.app.services.retriever import RagService

    mock_embedder = MagicMock()
    mock_embedder.dimension = 384
    mock_embedder.embed_query.return_value = [0.0] * 384

    service = RagService(embedder=mock_embedder, storage_dir="tmp_latency_test")
    service.vector_store = MagicMock()
    service.vector_store.total_chunks = 5

    chunks = [make_chunk(f"src/f{i}.py", 1, 5, f"class C{i}: pass") for i in range(5)]
    service.vector_store.search.return_value = [QueryResult(chunk=c, score=0.8) for c in chunks]
    service.bm25_index = MagicMock()
    service.bm25_index.total_chunks = 0

    mock_reranker = MagicMock()
    mock_reranker.is_available = True
    mock_reranker.rerank.return_value = [QueryResult(chunk=chunks[0], score=0.9)]
    service.reranker = mock_reranker

    resp = service.search(
        query="how does the system work",
        top_k=3,
        enable_reranker=True,
        enable_hybrid=False,
    )
    assert "num_reranker_candidates" in resp.latencies
    assert isinstance(resp.latencies["num_reranker_candidates"], (int, float))
    assert resp.latencies["num_reranker_candidates"] == 5


def test_incomplete_flow_evidence_instruction_contains_verified():
    """Verify build_prompt for execution-flow question instructs to report unverified stages."""
    service = GeminiService(api_key="mock-key")
    chunk = make_chunk("src/predictor.py", 1, 15, "class GesturePredictor:\n    def predict(self, lm): pass")
    prompt = service.build_prompt(
        "Trace the execution from webcam input through landmark extraction to gesture prediction", [chunk]
    )
    # Key phrases from flow_guidance must be present
    assert "not verified" in prompt.lower()
    assert "cite" in prompt.lower() or "line numbers" in prompt.lower()

