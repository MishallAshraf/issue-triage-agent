# 0003: Mask emails and tokens before anything is sent to Langfuse

**Status:** accepted

## Context
Traces hold the full prompts and tool results, which include issue text. The issues are public, but people still paste emails and API tokens into bug reports. Sending those to a third-party tracing service copies them somewhere new, with its own retention.

## Decision
The Langfuse client is created with a `mask` function that replaces email addresses and common token shapes (`sk-`, `ghp_`, `github_pat_`, `xoxb-` and similar) with placeholders. It walks nested dicts and lists, so tool arguments and results are covered as well as messages. Masking happens on the client, before data leaves the machine. Tracing is off unless Langfuse keys are set.

## Consequences
- Traces stay useful for debugging (structure, timings, decisions) without carrying the sensitive strings.
- Regex masking is best effort: an unusual token format will get through. It reduces the risk; it doesn't remove it.
- Covered by `tests/test_tracing.py`.

## Alternatives considered
- **No masking, since the data is public.** Public isn't the same as fine to copy into another system.
- **Don't trace message content at all.** Much safer, but then a trace can't explain why the agent did something, which defeats the point.
