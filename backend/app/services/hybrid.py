from typing import Dict, List, Optional, Tuple

from backend.app.core.config import get_settings
from backend.app.models.schemas import CodeChunk, QueryResult
from backend.app.services.weighting import compute_file_weight


def reciprocal_rank_fusion(
    vector_results: List[QueryResult],
    bm25_results: List[Tuple[CodeChunk, float]],
    query: str,
    rrf_k: Optional[int] = None,
    vector_weight: Optional[float] = None,
    bm25_weight: Optional[float] = None,
    enable_weighting: Optional[bool] = None,
    top_k: int = 15,
) -> List[QueryResult]:
    """Combine dense vector similarity search and sparse BM25 results using
    Reciprocal Rank Fusion (RRF) with query-dependent file weighting and deduplication.

    Args:
        vector_results: QueryResult items from FAISS vector search (already sorted).
        bm25_results: (CodeChunk, score) tuples from BM25 search (already sorted).
        query: User question for intent-dependent weighting.
        rrf_k: Smoothing constant k for RRF (default 60).
        vector_weight: Weight multiplier for vector search ranks.
        bm25_weight: Weight multiplier for BM25 ranks.
        enable_weighting: Whether to apply query-dependent file-type weighting.
        top_k: Number of fused candidate results to return.

    Returns:
        List of deduplicated QueryResult items sorted by fused score descending.
    """
    settings = get_settings()
    k = rrf_k if rrf_k is not None else settings.RRF_K
    w_vec = vector_weight if vector_weight is not None else settings.VECTOR_WEIGHT
    w_bm25 = bm25_weight if bm25_weight is not None else settings.BM25_WEIGHT

    chunk_map: Dict[str, CodeChunk] = {}
    rrf_scores: Dict[str, float] = {}
    vector_score_map: Dict[str, float] = {}

    # 1. Accumulate vector search reciprocal ranks and preserve cosine similarity scores
    for rank, item in enumerate(vector_results, start=1):
        cid = item.chunk.chunk_id
        chunk_map[cid] = item.chunk
        vector_score_map[cid] = item.score
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (w_vec / (k + rank))

    # 2. Accumulate BM25 search reciprocal ranks
    for rank, (chunk, _) in enumerate(bm25_results, start=1):
        cid = chunk.chunk_id
        chunk_map[cid] = chunk
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (w_bm25 / (k + rank))

    if not rrf_scores:
        return []

    # 3. Apply file-type and query-intent weighting
    weighted_scores: Dict[str, float] = {}
    for cid, base_rrf in rrf_scores.items():
        chunk = chunk_map[cid]
        weight = compute_file_weight(
            file_path=chunk.file_path,
            query=query,
            enable_weighting=enable_weighting,
        )
        weighted_scores[cid] = base_rrf * weight

    # 4. Sort descending by final fused score with soft file-diversity decay
    # to avoid monopolization by multiple redundant chunks from the same file
    sorted_items = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
    file_seen_counts: Dict[str, int] = {}
    diversified_scores: List[Tuple[str, float]] = []

    for cid, score in sorted_items:
        fpath = chunk_map[cid].file_path
        count = file_seen_counts.get(fpath, 0)
        file_seen_counts[fpath] = count + 1
        diversity_factor = 1.0 / (1.0 + 0.20 * count)
        diversified_scores.append((cid, score * diversity_factor))

    diversified_scores.sort(key=lambda x: x[1], reverse=True)
    top_items = diversified_scores[:top_k]

    max_rrf = top_items[0][1] if top_items else 1.0
    fused_results: List[QueryResult] = []

    for cid, raw_score in top_items:
        # Preserve original vector similarity if available, or compute calibrated score from BM25 rank
        if cid in vector_score_map:
            calibrated_score = vector_score_map[cid]
        else:
            norm_rrf = raw_score / max_rrf if max_rrf > 0 else 0.5
            calibrated_score = 0.50 + 0.40 * norm_rrf

        clamped_score = max(0.0, min(1.0, round(calibrated_score, 4)))
        fused_results.append(
            QueryResult(
                chunk=chunk_map[cid],
                score=clamped_score,
            )
        )

    return fused_results
