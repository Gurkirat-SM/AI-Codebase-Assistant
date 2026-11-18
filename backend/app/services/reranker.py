import logging
import math
from functools import lru_cache
from typing import List, Optional, Tuple

from backend.app.core.config import get_settings
from backend.app.models.schemas import QueryResult

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """Map real-valued cross-encoder logit to [0.0, 1.0] probability."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class CodeReranker:
    """Lazy-loaded Cross-Encoder reranker for precision reranking of candidate chunks.

    Supports graceful fallback if the model cannot be loaded (e.g. offline mode,
    missing weights, memory constraints) or if reranking is disabled in settings.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self.enabled = enabled if enabled is not None else settings.ENABLE_RERANKER
        self._model = None
        self._failed_to_load = False

    @property
    def is_available(self) -> bool:
        """Check if reranker is enabled and model can be loaded."""
        if not self.enabled or self._failed_to_load:
            return False
        return self._get_model() is not None

    def _get_model(self):
        """Lazily load CrossEncoder model with error recovery."""
        if self._model is not None:
            return self._model
        if self._failed_to_load or not self.enabled:
            return None

        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading CrossEncoder reranker model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
            return self._model
        except Exception as exc:
            self._failed_to_load = True
            logger.warning(
                "Could not load CrossEncoder model '%s': %s. "
                "Retrieval will fall back gracefully to hybrid search without reranking.",
                self.model_name, exc,
            )
            return None

    def rerank(
        self,
        query: str,
        candidates: List[QueryResult],
        top_k: int = 5,
    ) -> List[QueryResult]:
        """Score candidate chunks with Cross-Encoder and return top-k reranked results.

        Falls back seamlessly to candidate order if reranker is disabled or model is unavailable.
        """
        if not candidates or top_k <= 0:
            return []

        if len(candidates) == 1:
            return candidates[:top_k]

        model = self._get_model()
        if model is None:
            return candidates[:top_k]

        try:
            pairs: List[Tuple[str, str]] = [
                (query, f"File: {c.chunk.file_path}\n{c.chunk.content}")
                for c in candidates
            ]

            raw_scores = model.predict(pairs)
            scored: List[Tuple[QueryResult, float]] = []

            for candidate, raw_score in zip(candidates, raw_scores):
                prob = _sigmoid(float(raw_score))
                # Calibrate score: preserve base similarity while allowing cross-encoder confidence to boost
                calibrated_score = max(candidate.score, round(prob, 4))
                scored.append(
                    (
                        QueryResult(
                            chunk=candidate.chunk,
                            score=calibrated_score,
                        ),
                        float(raw_score),  # strictly sort by Cross-Encoder relevance score
                    )
                )

            scored.sort(key=lambda x: x[1], reverse=True)
            return [item[0] for item in scored[:top_k]]

        except Exception as exc:
            logger.warning("Reranking failed with error: %s. Using hybrid rank.", exc)
            return candidates[:top_k]


@lru_cache
def get_reranker() -> CodeReranker:
    """Return cached singleton instance of CodeReranker."""
    return CodeReranker()
