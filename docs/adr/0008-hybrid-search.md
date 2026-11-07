# 0008: Hybrid search (BM25 + vectors), chosen by measuring search on its own

**Status:** accepted

## Context
In the first live run (BM25 search, `openai/gpt-4o-mini`), the agent found 5 of 8 known duplicates. For two of the three misses (#568 and #3306), the original never appeared in the search results: the agent wrote short queries ("edit dataset"), and the original described the same problem in different words. BM25 only matches words. The third miss (#10494) was different: the original was ranked first, and the agent still didn't mark it.

A live agent run costs money and, on 8 duplicate cases, can't tell a real gain from luck (one case is 12.5 points). So search needed its own measurement.

## Decision
- **Measure search alone first.** `python -m evals.retrieval` checks, for every known pair where the original came first (8 of the 11 in the snapshot), whether the original is in the top k. It tries three kinds of query (the title, the title plus body, and the agent's own recorded queries) against BM25, vectors and hybrid. No LLM, no cost.
- **Results (recall@5):** with the agent's own queries, BM25 found 6 of 8, vectors 8 of 8, hybrid 8 of 8. On titles, hybrid had the best mean reciprocal rank (0.94 vs 0.92 for vectors and 0.88 for BM25).
- **Use hybrid** as the default (`SEARCH_METHOD=hybrid`): reciprocal rank fusion of BM25 and vectors. It matched vectors where it mattered and keeps BM25's exact matching for error messages, function names and model names, which embeddings can blur.
- **Embeddings:** `BAAI/bge-small-en-v1.5` through fastembed. It runs locally (ONNX, no PyTorch, no API key), is about 65 MB, and gives the same vectors every run. Vectors are cached per snapshot in `data/cache/`.
- The "earlier issues only" rule ([0002](0002-search-only-sees-earlier-issues.md)) applies to every method, before fusion.

## What happened in the live run
Search did its job: in the next live run the original was ranked first in 7 of 8 cases and third in the other. But end-to-end duplicate recall fell from 5 of 8 to 4 of 8, because the agent often didn't mark a candidate it was shown. Retrieval recall and end-to-end recall are different numbers; the follow-up is in [0009](0009-duplicate-check-by-a-stronger-model.md) and [the experiments log](../experiments.md).

## Consequences
- The retrieval misses behind two of the three missed duplicates are fixed at the search level. The reasoning miss (#10494) is not a search problem and stays.
- More similar-looking candidates can mean more false duplicates. The live rerun checks this against the baseline.
- The first run embeds all 641 issues once (about 5 minutes on a laptop CPU, with small batches to keep memory low). Later runs load the cache in seconds.
- CI's replay gate doesn't need the model, since it re-scores recordings. A manual live run in CI downloads it.
- 8 pairs is a small sample. The BM25 gap on the agent's queries (two originals not in the top 50) is a clear finding; the difference between vectors and hybrid is within one pair.

## Alternatives considered
- **Vectors only.** Same recall here, but weaker on exact identifiers, which GitHub issues are full of.
- **Keep BM25 and prompt the agent to write longer or multiple queries.** Cheaper, and still worth doing, but it relies on the model behaving; the search fix holds whatever the agent writes.
- **An embedding API (OpenAI, Voyage).** Better models, but adds a key and a per-call cost, and isn't needed at this size.
