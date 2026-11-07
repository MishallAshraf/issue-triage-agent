"""Keyword search over the issue snapshot (BM25).

Only issues opened *before* the one being triaged are searchable. Otherwise the
agent could "find" a later issue that was itself closed as a duplicate of this one,
which would make duplicate detection look better than it is.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .data import Issue

TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOP = set("a an the and or of to in on for with is are was be it this that i we you not no "
           "when what how can do does from by at as if but have has my our".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP and len(t) > 1]


class _BM25(BM25Okapi):
    """BM25 with Lucene's IDF, which never goes negative.

    rank_bm25's default IDF is negative for words in more than half the documents,
    which makes genuine matches score below zero on small corpora.
    """

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))


@dataclass
class Hit:
    number: int
    title: str
    state: str
    score: float


class IssueIndex:
    def __init__(self, issues: dict[int, Issue]):
        self.issues = issues
        self.numbers = sorted(issues)
        docs = [tokenize(issues[n].title + " " + issues[n].title + " " + issues[n].body) for n in self.numbers]
        self.bm25 = _BM25(docs)

    def search(self, query: str, *, before: str | None = None, exclude: int | None = None, k: int = 5) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.numbers, scores), key=lambda x: -x[1])
        hits = []
        for n, s in ranked:
            iss = self.issues[n]
            if n == exclude or s <= 0:
                continue
            if before and iss.created_at >= before:
                continue
            hits.append(Hit(n, iss.title, iss.state, round(float(s), 2)))
            if len(hits) == k:
                break
        return hits
