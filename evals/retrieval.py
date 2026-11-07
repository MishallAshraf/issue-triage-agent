"""Measure search on its own, without the agent or any LLM: for each known duplicate,
is the original in the top k results?

    python -m evals.retrieval

Uses every pair in the snapshot where maintainers wrote "Duplicate of #N" and the
original was opened first (otherwise the leakage guard rightly hides it). Each pair
is searched three ways:

  title         the duplicate's title, as a short query
  title+body    the title plus the start of the body, as a long query
  agent         the queries the agent actually wrote in the recorded live run

and with three methods: BM25, vectors, and hybrid (reciprocal rank fusion of both).
Free and fast, so it's the right place to choose a search method before paying for
a live agent run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from triage.config import settings
from triage.data import load
from triage.search import FastEmbedder, HybridIndex, IssueIndex, VectorIndex

HERE = Path(__file__).parent
KS = (1, 5, 10)


def usable_pairs(snap) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    pairs = sorted((n, i.duplicate_of) for n, i in snap.issues.items() if i.duplicate_of in snap.issues)
    ok = [(d, o) for d, o in pairs if snap.issues[o].created_at < snap.issues[d].created_at]
    return ok, [p for p in pairs if p not in ok]


def agent_queries(issue: int) -> list[str]:
    path = HERE / "recordings" / f"{issue}.json"
    if not path.exists():
        return []
    run = json.loads(path.read_text(encoding="utf-8"))
    return [s["args"]["query"] for s in run["trajectory"] if s["tool"] == "search_similar_issues"]


def rank_of(index, query: str, dup, original: int, depth: int = 50) -> int | None:
    hits = index.search(query, before=dup.created_at, exclude=dup.number, k=depth)
    return next((i for i, h in enumerate(hits, 1) if h.number == original), None)


def best_rank(index, queries: list[str], dup, original: int) -> int | None:
    """Over several queries (the agent may search more than once), the best rank reached."""
    ranks = [r for q in queries if (r := rank_of(index, q, dup, original)) is not None]
    return min(ranks) if ranks else None


def summarize(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {f"recall@{k}": sum(r is not None and r <= k for r in ranks) / n for k in KS}
    out["mrr"] = sum(1 / r for r in ranks if r) / n
    return out


def main() -> int:
    snap = load()
    pairs, skipped = usable_pairs(snap)
    keyword = IssueIndex(snap.issues)
    vector = VectorIndex(snap.issues, FastEmbedder(cache_dir=settings.cache_dir.parent / "models"),
                         cache_dir=settings.cache_dir)
    methods = {"bm25": keyword, "vector": vector, "hybrid": HybridIndex(keyword, vector)}

    queries = {
        "title": lambda d: [d.title],
        "title+body": lambda d: [f"{d.title}\n{d.body[:1000]}"],
        "agent": lambda d: agent_queries(d.number),
    }

    print(f"{len(pairs)} duplicate pairs (skipped {len(skipped)} where the original came later: {skipped})\n")
    report = {"pairs": pairs, "skipped": skipped, "results": {}, "ranks": {}}
    print(f"{'query':12} {'method':8} " + " ".join(f"{'R@' + str(k):>6}" for k in KS) + f" {'MRR':>6}   ranks")
    for qname, make in queries.items():
        for mname, index in methods.items():
            ranks = []
            for d, o in pairs:
                qs = make(snap.issues[d])
                ranks.append(best_rank(index, qs, snap.issues[d], o) if qs else None)
            s = summarize(ranks)
            report["results"][f"{qname}/{mname}"] = s
            report["ranks"][f"{qname}/{mname}"] = dict(zip([d for d, _ in pairs], ranks))
            cells = " ".join(f"{s[f'recall@{k}']:6.2f}" for k in KS)
            print(f"{qname:12} {mname:8} {cells} {s['mrr']:6.2f}   {['-' if r is None else r for r in ranks]}")
        print()

    out = settings.results_dir / "retrieval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Pairs in order: {[d for d, _ in pairs]}  ('-' = not in the top 50)\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
