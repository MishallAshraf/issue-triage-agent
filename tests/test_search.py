"""BM25 search details: tokenising, non-negative scores, filters."""

from triage.search import IssueIndex, tokenize


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
