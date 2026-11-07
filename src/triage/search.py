"""Search over the issue snapshot: BM25 (keywords), vectors (meaning), or both (hybrid).

All three share one interface, `search(query, *, before, exclude, k) -> list[Hit]`,
and the same rule: only issues opened *before* the one being triaged are searchable.
Otherwise the agent could "find" a later issue that was itself closed as a duplicate
of this one, which would make duplicate detection look better than it is.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .data import Issue

TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOP = set("a an the and or of to in on for with is are was be it this that i we you not no "
           "when what how can do does from by at as if but have has my our".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP and len(t) > 1]


@dataclass
class Hit:
    number: int
    title: str
    state: str
    score: float


class _Filtered:
    """Shared filtering: drop the triaged issue and anything opened at or after it."""

    issues: dict[int, Issue]

    def _allowed(self, n: int, before: str | None, exclude: int | None) -> bool:
        return n != exclude and not (before and self.issues[n].created_at >= before)

    def _hits(self, ranked: list[tuple[int, float]], before, exclude, k, digits=2) -> list[Hit]:
        hits = []
        for n, s in ranked:
            if s <= 0 or not self._allowed(n, before, exclude):
                continue
            iss = self.issues[n]
            hits.append(Hit(n, iss.title, iss.state, round(float(s), digits)))
            if len(hits) == k:
                break
        return hits


# --- keywords -------------------------------------------------------------------

class _BM25(BM25Okapi):
    """BM25 with Lucene's IDF, which never goes negative.

    rank_bm25's default IDF is negative for words in more than half the documents,
    which makes genuine matches score below zero on small corpora.
    """

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))


class IssueIndex(_Filtered):
    """BM25 over titles (counted twice, so they weigh more) and bodies."""

    def __init__(self, issues: dict[int, Issue]):
        self.issues = issues
        self.numbers = sorted(issues)
        docs = [tokenize(issues[n].title + " " + issues[n].title + " " + issues[n].body) for n in self.numbers]
        self.bm25 = _BM25(docs)

    def ranked(self, query: str) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        return sorted(zip(self.numbers, scores), key=lambda x: -x[1])

    def search(self, query: str, *, before: str | None = None, exclude: int | None = None, k: int = 5) -> list[Hit]:
        return self._hits(self.ranked(query), before, exclude, k)


# --- meaning --------------------------------------------------------------------

class Embedder(Protocol):
    name: str

    def passages(self, texts: Sequence[str]) -> np.ndarray: ...
    def query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    """A small local embedding model (ONNX, no PyTorch, no API key). Downloads once."""

    def __init__(self, name: str = "BAAI/bge-small-en-v1.5", cache_dir: Path | None = None):
        import os
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless on Windows; just noisy
        from fastembed import TextEmbedding
        self.name = name
        self.model = TextEmbedding(name, cache_dir=str(cache_dir) if cache_dir else None)

    def passages(self, texts: Sequence[str]) -> np.ndarray:
        # Small batches: long issues at the default batch size need gigabytes of memory on a laptop CPU.
        return np.array(list(self.model.passage_embed(list(texts), batch_size=16)), dtype=np.float32)

    def query(self, text: str) -> np.ndarray:
        return np.array(next(iter(self.model.query_embed([text]))), dtype=np.float32)


def _unit(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12, None)


class VectorIndex(_Filtered):
    """Cosine similarity between the query and each issue's title + start of the body.

    Embeddings are cached to disk, keyed by model and snapshot contents, so they're
    computed once per snapshot.
    """

    MAX_CHARS = 1200  # ~300 tokens: the title and opening paragraph carry the problem, and it keeps embedding fast

    def __init__(self, issues: dict[int, Issue], embedder: Embedder, cache_dir: Path | None = None):
        self.issues = issues
        self.embedder = embedder
        self.numbers = sorted(issues)
        texts = [f"{issues[n].title}\n\n{issues[n].body[:self.MAX_CHARS]}" for n in self.numbers]
        key = hashlib.sha256((embedder.name + "\x00" + "\x00".join(texts)).encode("utf-8")).hexdigest()[:16]
        path = cache_dir / f"vectors-{key}.npy" if cache_dir else None
        if path and path.exists():
            self.matrix = np.load(path)
        else:
            self.matrix = _unit(embedder.passages(texts))
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, self.matrix)

    def ranked(self, query: str) -> list[tuple[int, float]]:
        sims = self.matrix @ _unit(self.embedder.query(query))
        return sorted(zip(self.numbers, sims.tolist()), key=lambda x: -x[1])

    def search(self, query: str, *, before: str | None = None, exclude: int | None = None, k: int = 5) -> list[Hit]:
        return self._hits(self.ranked(query), before, exclude, k)


# --- both -----------------------------------------------------------------------

class HybridIndex(_Filtered):
    """Reciprocal rank fusion of BM25 and vectors.

    Each method ranks the allowed issues; an issue's score is the sum of 1/(c + rank)
    over both lists. Ranks, not raw scores, so the two scales never need tuning
    against each other. c=60 is the usual default from the original RRF paper.
    """

    def __init__(self, keyword: IssueIndex, vector: VectorIndex, c: int = 60, depth: int = 50):
        self.issues = keyword.issues
        self.keyword, self.vector, self.c, self.depth = keyword, vector, c, depth

    def search(self, query: str, *, before: str | None = None, exclude: int | None = None, k: int = 5) -> list[Hit]:
        fused: dict[int, float] = {}
        for ranked in (self.keyword.ranked(query), self.vector.ranked(query)):
            allowed = [n for n, s in ranked if s > 0 and self._allowed(n, before, exclude)][:self.depth]
            for rank, n in enumerate(allowed, 1):
                fused[n] = fused.get(n, 0.0) + 1.0 / (self.c + rank)
        return self._hits(sorted(fused.items(), key=lambda x: -x[1]), before, exclude, k, digits=4)


# --- choosing one -----------------------------------------------------------------

SEARCH_METHODS = ("hybrid", "bm25", "vector")


def build_index(issues: dict[int, Issue], method: str | None = None, *, embedder: Embedder | None = None,
                cache_dir: Path | None = None):
    """The index behind the agent's search tool, chosen by SEARCH_METHOD (default: hybrid).

    Hybrid won the retrieval eval (`python -m evals.retrieval`): with the agent's own
    queries, BM25 found 6 of 8 known originals in the top 5, vectors and hybrid 8 of 8.
    Hybrid keeps BM25's exact matching for error messages and names.
    """
    from .config import settings

    method = method or settings.search_method
    if method not in SEARCH_METHODS:
        raise ValueError(f"Unknown SEARCH_METHOD {method!r}; use one of {SEARCH_METHODS}")
    keyword = IssueIndex(issues)
    if method == "bm25":
        return keyword
    vector = VectorIndex(issues, embedder or FastEmbedder(cache_dir=settings.models_dir),
                         cache_dir=cache_dir or settings.cache_dir)
    return vector if method == "vector" else HybridIndex(keyword, vector)
