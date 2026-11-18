import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend.app.models.schemas import CodeChunk


def tokenize_code(text: str) -> List[str]:
    """Code-aware tokenizer extracting exact terms, snake_case parts,
    camelCase/PascalCase parts, and path components.
    """
    if not text:
        return []

    tokens: List[str] = []
    # Match alphanumeric sequences, words, and identifiers
    raw_words = re.findall(r"[A-Za-z0-9_.-]+", text)

    for word in raw_words:
        w_lower = word.lower()
        tokens.append(w_lower)

        # Split snake_case, dot.notation, and kebab-case
        sub_tokens = re.split(r"[_.-]+", word)
        for s in sub_tokens:
            if s and s.lower() != w_lower:
                tokens.append(s.lower())

        # Split camelCase and PascalCase identifiers
        camel_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|\d+", word)
        for c in camel_parts:
            if c and c.lower() != w_lower:
                tokens.append(c.lower())

    return tokens


class BM25Index:
    """Lightweight, code-aware BM25 index using standard Robertson-Spärck Jones / Lucene IDF."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[CodeChunk] = []
        self.doc_lens: List[int] = []
        self.doc_term_freqs: List[Counter] = []
        self.doc_freqs: Counter = Counter()
        self.avg_dl: float = 0.0
        self.corpus_size: int = 0

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def add_chunks(self, chunks: List[CodeChunk]) -> int:
        """Tokenize and index code chunks."""
        if not chunks:
            return 0

        self.chunks.extend(chunks)
        self.corpus_size = len(self.chunks)

        for chunk in chunks:
            # Include file path in the indexed content for path-based matching
            text_to_index = f"{chunk.file_path} {chunk.content}"
            tokens = tokenize_code(text_to_index)
            dl = len(tokens)
            self.doc_lens.append(dl)

            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)

            for term in tf:
                self.doc_freqs[term] += 1

        self.avg_dl = sum(self.doc_lens) / max(1, self.corpus_size)
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        file_filter: Optional[str] = None,
    ) -> List[Tuple[CodeChunk, float]]:
        """Search chunks by BM25 relevance to query string.
        Returns list of (CodeChunk, normalized_score) sorted descending by score.
        """
        if self.corpus_size == 0 or not query.strip():
            return []

        query_tokens = tokenize_code(query)
        if not query_tokens:
            return []

        filter_lower = file_filter.lower() if file_filter else None
        scored_candidates: List[Tuple[int, float]] = []

        for idx, tf in enumerate(self.doc_term_freqs):
            chunk = self.chunks[idx]
            if filter_lower and filter_lower not in chunk.file_path.lower():
                continue

            dl = self.doc_lens[idx]
            score = 0.0

            for term in query_tokens:
                if term in tf:
                    df = self.doc_freqs[term]
                    # Robertson-Spärck Jones / Lucene IDF: strictly positive
                    idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))
                    freq = tf[term]
                    denom = freq + self.k1 * (1.0 - self.b + self.b * (dl / max(1.0, self.avg_dl)))
                    score += idf * (freq * (self.k1 + 1.0)) / denom

            if score > 0.0:
                scored_candidates.append((idx, score))

        if not scored_candidates:
            return []

        # Sort descending by BM25 raw score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored_candidates[:top_k]

        max_score = top_candidates[0][1] if top_candidates else 1.0
        results: List[Tuple[CodeChunk, float]] = []

        for idx, raw_score in top_candidates:
            # Normalize to [0.0, 1.0] relative to top match
            norm_score = raw_score / max_score if max_score > 0 else 0.0
            results.append((self.chunks[idx], norm_score))

        return results

    def save(self, directory: str | Path) -> Path:
        """Persist BM25 index and chunk metadata to disk."""
        save_dir = Path(directory).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        bm25_file = save_dir / "bm25.json"

        data = {
            "k1": self.k1,
            "b": self.b,
            "corpus_size": self.corpus_size,
            "avg_dl": self.avg_dl,
            "doc_lens": self.doc_lens,
            "doc_freqs": dict(self.doc_freqs),
            "doc_term_freqs": [dict(tf) for tf in self.doc_term_freqs],
            "chunks": [chunk.model_dump() for chunk in self.chunks],
        }

        with open(bm25_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        return save_dir

    def load(self, directory: str | Path) -> bool:
        """Load BM25 index and chunk metadata from disk."""
        save_dir = Path(directory).resolve()
        bm25_file = save_dir / "bm25.json"
        if not bm25_file.exists():
            return False

        try:
            with open(bm25_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.k1 = data.get("k1", 1.5)
            self.b = data.get("b", 0.75)
            self.corpus_size = data.get("corpus_size", 0)
            self.avg_dl = data.get("avg_dl", 0.0)
            self.doc_lens = data.get("doc_lens", [])
            self.doc_freqs = Counter(data.get("doc_freqs", {}))
            self.doc_term_freqs = [Counter(tf) for tf in data.get("doc_term_freqs", [])]
            self.chunks = [CodeChunk(**c) for c in data.get("chunks", [])]
            return True
        except Exception:
            return False

    def clear(self):
        """Reset BM25 index."""
        self.chunks.clear()
        self.doc_lens.clear()
        self.doc_term_freqs.clear()
        self.doc_freqs.clear()
        self.avg_dl = 0.0
        self.corpus_size = 0

