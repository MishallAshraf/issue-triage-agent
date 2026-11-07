"""Langfuse tracing. Optional: with no Langfuse keys in .env, everything runs untraced.

One trace per triage run. LangGraph's steps, LLM calls and tool calls come in through
the LangChain callback handler; the run itself is an "agent" observation so scores
(from the rule checks and the judge) can be attached to the trace.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Iterator

from .config import settings

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


_client = None


def client():
    global _client
    if _client is None and settings.tracing_enabled:
        from langfuse import Langfuse
        _client = Langfuse(mask=mask)
    return _client


class Run:
    """Handle for one traced run. Every method is a no-op when tracing is off."""

    def __init__(self, span=None, handler=None):
        self.span = span
        self.callbacks = [handler] if handler else []

    @property
    def trace_id(self) -> str | None:
        return self.span.trace_id if self.span else None

    def finish(self, output: Any) -> None:
        if self.span:
            self.span.update(output=output)

    def score(self, name: str, value: float | str, comment: str | None = None) -> None:
        if self.span:
            dtype = "CATEGORICAL" if isinstance(value, str) else "NUMERIC"
            self.span.score_trace(name=name, value=value, data_type=dtype, comment=comment)


@contextmanager
def traced_run(issue: int, *, tags: list[str] | None = None, metadata: dict | None = None) -> Iterator[Run]:
    lf = client()
    if lf is None:
        yield Run()
        return
    from langfuse import propagate_attributes
    from langfuse.langchain import CallbackHandler

    meta = {"repo": settings.repo, "issue": str(issue), "model": settings.model, **(metadata or {})}
    with propagate_attributes(trace_name="triage-issue", tags=tags or [], metadata=meta):
        with lf.start_as_current_observation(name="triage-issue", as_type="agent",
                                             input={"repo": settings.repo, "issue": issue}) as span:
            yield Run(span, CallbackHandler())


def flush() -> None:
    if client():
        client().flush()
