# 0007: Tracing fails open

**Status:** accepted

## Context
Tracing is there to help, not to be a reason the agent can't run. Langfuse can be unreachable, keys can be wrong, or a score call can fail on a network blip. If any of those raised an exception, a triage run (or a whole eval run halfway through 28 cases) would die because of the observer, not the thing being observed.

## Decision
Every tracing call in `src/triage/tracing.py` catches its own errors, logs a warning and carries on: creating the client, starting the trace, recording the output, posting a score, closing the trace and flushing. If the trace can't start, the run continues untraced. Errors raised by the agent itself are never caught. They pass through the trace (which records them) and propagate as normal.

## Consequences
- A Langfuse outage costs trace data, never a run.
- A failure shows up as a warning in the logs rather than a crash, so a run that went quietly untraced is easy to miss. The warning text names what failed.
- Covered by `tests/test_tracing.py`: broken spans, a trace that can't start, a failing flush, and an agent error that must still propagate.

## Alternatives considered
- **Let tracing errors raise.** Simpler, but the observer can take down the agent.
- **Disable tracing for the whole process after the first failure.** Hides a short blip for the rest of a long run; per-call handling recovers on its own.
