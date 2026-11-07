# Observing the agent

With `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env`, every triage run becomes one Langfuse trace. Without them everything still runs, just untraced. The code is in `src/triage/tracing.py`.

## What one trace holds

```
triage-issue                      agent observation (input: repo + issue, output: run summary)
├── LangGraph run                 from the LangChain callback handler
│   ├── agent                     one per model turn
│   │   └── ChatOpenAI / ChatAnthropic   prompt, response, tool calls, tokens, cost, latency
│   ├── tools                     get_issue, search_similar_issues, list_labels, writes
│   ├── approval                  the pause for a human; the resume carries the answers
│   └── ...
└── scores                        attached to the trace (see below)
```

- **Run summary (output of the root observation):** the final message, the outbox, which writes ran, which were rejected, and which tool calls failed. It's built from the run itself, not rebuilt from the nested spans ([ADR 0006](adr/0006-run-summary-on-the-trace.md)).
- **Metadata:** repo, issue number, model, plus the run's counts: `steps`, `tool_calls`, `writes_approved`, `writes_rejected`, `tool_errors`, `stopped_by_cap`. These are on every run, CLI included, so they work as filters even without eval scores. **Tags:** `cli` for terminal runs; `eval` plus the approval policy (`approve_all`, `reject_all`, `reject_duplicates`) for eval runs.
- **Masking:** emails and token-like strings are replaced before anything is sent ([ADR 0003](adr/0003-mask-before-tracing.md)).
- The CLI prints the trace id at the end, and each eval recording stores its `trace_id`, so a bad case in a report links straight to its trace.

## Scores on each eval trace

| Score | Type | Meaning |
|-------|------|---------|
| `gate_respected` | 0 or 1 | Every write that ran was approved first. Anything but 1 is a bug. |
| `label_f1` | 0 to 1 | Proposed labels vs the maintainers' labels. Only on cases that have expected labels. |
| `duplicate` | category | `correct`, `missed`, `wrong_original` or `false_duplicate`. |
| `faithful` | 0 or 1 | The LLM judge's verdict on the final summary; its reasoning is the score comment. |

## Finding a bad run

1. In Langfuse, filter traces by tag `eval` and by `faithful = 0`, or `duplicate` not equal to `correct`. For any run, CLI included, filter metadata on `writes_rejected > 0` or `tool_errors > 0`.
2. Open the trace. The root output tells you what happened; then read the `agent` steps in order to see why: what it searched for, what came back, what it proposed.
3. For a missed duplicate, check the `search_similar_issues` results first. If the original isn't in the hits, it's a search problem (BM25 misses rewording), not a reasoning problem.
4. For an unfaithful summary, compare the last tool results with the final message; the judge's comment usually names the mismatch.
5. Sort by latency or cost to find runs that loop. The step cap stops them at 12 steps, and the `within_step_cap` check reports it.

## When Langfuse is down

Tracing fails open ([ADR 0007](adr/0007-tracing-fails-open.md)): if Langfuse can't be reached or a call fails, the run carries on and a `Tracing: ... failed` warning is logged. Errors from the agent itself still raise as normal.

## What isn't traced

- Approved writes only go to a local outbox; nothing reaches GitHub, so there's nothing external to trace.
- `evals/calibrate.py` runs the judge on the probe bank without tracing, since those are constructed cases, not agent runs.
