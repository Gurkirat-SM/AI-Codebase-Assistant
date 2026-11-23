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

