# 0006: The code that ran the agent states what happened, on the trace

**Status:** accepted

## Context
A Langfuse trace is built for debugging: nested LangGraph steps, LLM calls and tool calls, in whatever shape the callback handler produces. To answer "what did this run actually do?" from it, you have to rebuild the story: find the write calls, match each to its result, work out which were rejected. That rebuild breaks whenever the trace shape changes (a library update, a renamed node), and it can't recover facts the trace never recorded. Simple questions like "show me runs where a write was rejected" also have nothing to filter on.

## Decision
At the end of every traced run, CLI and eval alike, `agent.summarize()` builds one structured record from the run itself and puts it on the root `triage-issue` observation:

- **output:** the final summary, the outbox, the writes that ran, the writes that were rejected, and the tool calls that failed
- **metadata:** the counts (steps, tool calls, writes approved, writes rejected, tool errors, whether the step cap was hit)

The record is plain data. Anything that reads it (a person, a dashboard, a later scoring script) decides how to present it.

## Consequences
- Opening a trace shows the whole story at the top, before any drilling down.
- Runs can be filtered in Langfuse by `writes_rejected`, `tool_errors` or `steps`, and this works for CLI runs that have no eval scores.
- If the trace's nested shape changes, the summary doesn't.
- The summary repeats some of what's in the nested spans. That's a small cost for a stable record.
- Covered by `tests/test_tracing.py`.

## Alternatives considered
- **Rebuild the story from the nested spans when needed.** Fragile, and it can't recover facts that were never recorded.
- **Eval scores only.** CLI runs would have nothing to filter on, and scores don't say *which* write was rejected.
