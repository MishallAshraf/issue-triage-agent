"""Search details: tokenising, non-negative scores, filters, vectors and hybrid fusion."""

import numpy as np

from triage.search import HybridIndex, IssueIndex, VectorIndex, tokenize


class WordEmbedder:
    """A tiny fake embedding: one dimension per word in a fixed vocabulary.
    Lets the vector and hybrid code be tested without downloading a model."""

    name = "fake-words"
    VOCAB = ["traces", "page", "blank", "screen", "crash", "dark", "theme", "mode", "empty"]

    def __init__(self):
        self.calls = 0

    def _vec(self, text):
        words = set(tokenize(text))
        return np.array([1.0 if w in words else 0.0 for w in self.VOCAB], dtype=np.float32) + 1e-3

    def passages(self, texts):
        self.calls += 1
        return np.stack([self._vec(t) for t in texts])

    def query(self, text):
        return self._vec(text)


def test_vector_search_finds_by_meaning_and_respects_the_filters(snap):
    index = VectorIndex(snap.issues, WordEmbedder())
    assert index.search("dark theme", k=1)[0].number == 11
    hits = index.search("blank traces screen", before=snap.issues[12].created_at, exclude=12)
    assert 12 not in [h.number for h in hits] and hits[0].number == 10


def test_vectors_are_cached_and_reused(snap, tmp_path):
    first = WordEmbedder()
    VectorIndex(snap.issues, first, cache_dir=tmp_path)
    second = WordEmbedder()
    index = VectorIndex(snap.issues, second, cache_dir=tmp_path)
    assert first.calls == 1 and second.calls == 0, "second index loads from disk"
    assert index.matrix.shape == (3, len(WordEmbedder.VOCAB))


def test_hybrid_merges_both_rankings_and_keeps_the_leakage_guard(snap):
    hybrid = HybridIndex(IssueIndex(snap.issues), VectorIndex(snap.issues, WordEmbedder()))
    hits = hybrid.search("blank screen traces", before=snap.issues[12].created_at, exclude=12)
    assert [h.number for h in hits][:1] == [10] and 12 not in [h.number for h in hits]
    top = hybrid.search("dark theme", k=1)[0]
    assert top.number == 11 and top.score > 0


def test_tokenize_drops_stopwords_case_and_single_letters():
    assert tokenize("The Traces page is a BLANK screen, x") == ["traces", "page", "blank", "screen"]


def test_common_words_never_get_a_negative_score(snap):
    # "traces" is in 2 of the 3 issues. rank_bm25's default IDF would push it below zero.
    index = IssueIndex(snap.issues)
    assert all(v >= 0 for v in index.bm25.idf.values())
    assert [h.number for h in index.search("traces")] and all(h.score > 0 for h in index.search("traces"))


def test_unrelated_query_returns_nothing(snap):
    assert IssueIndex(snap.issues).search("kubernetes helm chart") == []


def test_k_limits_results_and_best_match_comes_first(snap):
    hits = IssueIndex(snap.issues).search("blank screen traces crash", k=1)
    assert len(hits) == 1 and hits[0].number in {10, 12}


def test_before_and_exclude_filters(snap):
    index = IssueIndex(snap.issues)
    assert 12 not in [h.number for h in index.search("traces", before="2026-01-15T00:00:00Z")]
    assert 10 not in [h.number for h in index.search("traces", exclude=10)]


def test_build_index_picks_the_method(snap, tmp_path):
    from triage.search import build_index
    assert isinstance(build_index(snap.issues, "bm25"), IssueIndex)
    assert isinstance(build_index(snap.issues, "vector", embedder=WordEmbedder(), cache_dir=tmp_path), VectorIndex)
    assert isinstance(build_index(snap.issues, "hybrid", embedder=WordEmbedder(), cache_dir=tmp_path), HybridIndex)


def test_build_index_refuses_an_unknown_method(snap):
    import pytest
    from triage.search import build_index
    with pytest.raises(ValueError, match="SEARCH_METHOD"):
        build_index(snap.issues, "magic")
