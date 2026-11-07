# 0001: Every write waits for a human, enforced in the graph

**Status:** accepted

## Context
The agent can label issues, close them as duplicates and post comments. A wrong duplicate close or an odd comment is visible to the person who opened the issue, so these actions need a person's OK. A prompt that says "ask before writing" is a request, not a guarantee: the model can ignore it.

## Decision
The LangGraph routing sends any turn that contains a write tool call to an `approval` node, which pauses the graph with `interrupt()`. The caller answers approve or reject per call. The tool node only runs a write if its call id was approved; a rejected call gets a tool message saying it was not run, and the agent has to carry on. Approved writes still only go to a local outbox, never to GitHub.

## Consequences
- The guarantee holds whatever the model does, and it's testable without an API key (`tests/test_agent.py`).
- `gate_respected` is a rule check on every eval run, with a 100% threshold.
- Each write turn costs a human a decision. Fine for triage; it would need batching for high volume.
- The agent must handle "no" well. The golden set rejects writes on purpose to test that.

## Alternatives considered
- **Prompt-only instruction.** Not enforceable.
- **Approve the whole run at the end.** The agent would carry on as if writes had happened, and its later steps would build on actions that may be rejected.
- **No writes at all (suggest only).** Safer, but then there's no approval flow to observe or evaluate.
