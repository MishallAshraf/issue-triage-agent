"""Langfuse tracing. Optional: with no Langfuse keys in .env, everything runs untraced.

One trace per triage run. LangGraph's steps, LLM calls and tool calls come in through
the LangChain callback handler; the run itself is an "agent" observation so scores
(from the rule checks and the judge) can be attached to the trace.

Fail-open: tracing must never break the thing it observes. If Langfuse can't be
reached or a call fails, the run carries on and a warning is logged. Errors from
the agent itself are never swallowed.
"""

from __future__ import annotations

import logging
import re
import sys
from contextlib import ExitStack, contextmanager
from typing import Any, Iterator

from .config import settings

log = logging.getLogger(__name__)

# Issue text is public, but it can still contain emails or pasted tokens. Mask them
# before anything leaves the machine.
_MASKS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<email>"),
    (re.compile(r"\b(?:sk|pk|ghp|gho|github_pat|xox[abp])[-_][A-Za-z0-9_-]{16,}\b"), "<token>"),
]


def mask(*, data: Any, **_: Any) -> Any:
    if isinstance(data, str):
        for rx, repl in _MASKS:
            data = rx.sub(repl, data)
        return data
    if isinstance(data, dict):
        return {k: mask(data=v) for k, v in data.items()}
    if isinstance(data, list):
        return [mask(data=v) for v in data]
    return data


def _warn(what: str, err: Exception) -> None:
    log.warning("Tracing: %s failed (%s: %s); continuing without it.", what, type(err).__name__, err)


_client = None


def client():
    global _client
    if _client is None and settings.tracing_enabled:
        try:
            from langfuse import Langfuse
            _client = Langfuse(mask=mask)
        except Exception as e:
            _warn("starting the Langfuse client", e)
    return _client


class Run:
    """Handle for one traced run. Every method is a no-op when tracing is off, and never raises."""

    def __init__(self, span=None, handler=None):
        self.span = span
        self.callbacks = [handler] if handler else []

    @property
    def trace_id(self) -> str | None:
        try:
            return self.span.trace_id if self.span else None
        except Exception:
            return None

    def finish(self, summary: dict | Any) -> None:
        """Record what happened. A dict from `agent.summarize` also puts its counts in the
        trace metadata, so runs can be filtered by rejected writes, errors or steps."""
        if not self.span:
            return
        try:
            meta = summary.get("counts") if isinstance(summary, dict) else None
            self.span.update(output=summary, **({"metadata": meta} if meta else {}))
        except Exception as e:
            _warn("recording the run output", e)

    def score(self, name: str, value: float | str, comment: str | None = None) -> None:
        if not self.span:
            return
        try:
            dtype = "CATEGORICAL" if isinstance(value, str) else "NUMERIC"
            self.span.score_trace(name=name, value=value, data_type=dtype, comment=comment)
        except Exception as e:
            _warn(f"posting score {name!r}", e)


def _close(stack: ExitStack, *exc: Any) -> None:
    try:
        stack.__exit__(*exc)
    except Exception as e:
        _warn("closing the trace", e)


@contextmanager
def traced_run(issue: int, *, tags: list[str] | None = None, metadata: dict | None = None) -> Iterator[Run]:
    lf = client()
    if lf is None:
        yield Run()
        return

    stack = ExitStack()
    try:
        from langfuse import propagate_attributes
        from langfuse.langchain import CallbackHandler

        meta = {"repo": settings.repo, "issue": str(issue), "model": settings.model,
                "search": settings.search_method,
                "duplicate_model": settings.duplicate_model or "none", **(metadata or {})}
        stack.enter_context(propagate_attributes(trace_name="triage-issue", tags=tags or [], metadata=meta))
        span = stack.enter_context(lf.start_as_current_observation(
            name="triage-issue", as_type="agent", input={"repo": settings.repo, "issue": issue}))
        run = Run(span, CallbackHandler())
    except Exception as e:
        _warn("starting the trace", e)
        _close(stack, None, None, None)
        yield Run()
        return

    try:
        yield run
    except BaseException:
        _close(stack, *sys.exc_info())  # the trace records the error; the error still propagates
        raise
    else:
        _close(stack, None, None, None)


def flush() -> None:
    try:
        if client():
            client().flush()
    except Exception as e:
        _warn("flushing traces", e)
