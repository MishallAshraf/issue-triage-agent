# Design decisions

Short notes on the choices that shape this repo: what I chose, why, and what I gave up. One page each. Copy `template.md` for a new one.

| # | Decision |
|---|----------|
| [0001](0001-approval-gate-in-the-graph.md) | Every write waits for a human, enforced in the graph, not the prompt |
| [0002](0002-search-only-sees-earlier-issues.md) | Search only sees issues opened before the one being triaged |
| [0003](0003-mask-before-tracing.md) | Mask emails and tokens before anything is sent to Langfuse |
| [0004](0004-narrow-calibrated-judge.md) | The LLM judge answers one question, and is calibrated before it's trusted |
| [0005](0005-ci-replays-recordings.md) | CI re-scores recorded runs; live runs are manual |
| [0006](0006-run-summary-on-the-trace.md) | The code that ran the agent states what happened, on the trace |
| [0007](0007-tracing-fails-open.md) | Tracing fails open: it never breaks a run |
| [0008](0008-hybrid-search.md) | Hybrid search (BM25 + vectors), chosen by measuring search on its own |
| [0009](0009-duplicate-check-by-a-stronger-model.md) | A stronger model checks every proposed duplicate, in code, before a human sees it |
