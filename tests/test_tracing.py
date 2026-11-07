import pytest
from langchain_core.messages import AIMessage

from conftest import ScriptedModel, call
from triage import tracing
from triage.agent import run_triage, summarize_run


def test_mask_redacts_emails_and_tokens_in_nested_data():
    data = {"body": "mail me at jane.doe@example.com", "msgs": ["key ghp_abcdefghijklmnop1234 here"]}
    out = tracing.mask(data=data)
    assert out == {"body": "mail me at <email>", "msgs": ["key <token> here"]}


def test_untraced_run_is_a_no_op(monkeypatch):
    monkeypatch.setattr(tracing, "client", lambda: None)
    with tracing.traced_run(1) as run:
        run.score("x", 1.0)
        run.finish("done")
    assert run.callbacks == [] and run.trace_id is None


# --- the run summary that goes on the root observation ---

def test_summary_separates_ran_rejected_and_failed_calls(ws):
    script = [
        AIMessage("", tool_calls=[call("apply_labels", {"number": 12, "labels": ["bug"]}, "w1"),
                                  call("mark_duplicate", {"number": 12, "original": 10, "reason": "same"}, "w2"),
                                  call("post_comment", {"number": 12, "body": "Thanks!"}, "w3")]),
        AIMessage("", tool_calls=[call("get_issue", {"number": 999}, "r1")]),
        AIMessage("Applied bug. The duplicate mark and comment were rejected."),
    ]
    run = run_triage(ws, ScriptedModel(script=script), lambda p: {"w1": "approve", "w2": "reject", "w3": "reject"})
    s = summarize_run(run)
    assert s["writes"]["ran"] == [{"tool": "apply_labels", "args": {"number": 12, "labels": ["bug"]}}]
    assert [w["tool"] for w in s["writes"]["rejected"]] == ["mark_duplicate", "post_comment"]
    assert [e["tool"] for e in s["tool_errors"]] == ["get_issue"]
    assert s["counts"] == {"steps": 3, "tool_calls": 4, "writes_approved": 1, "writes_rejected": 2,
                           "duplicates_blocked_by_check": 0, "tool_errors": 1, "stopped_by_cap": False}
    assert s["outbox"] == run["outbox"] and s["final"] == run["final"]


# --- fail-open: a broken tracing backend never breaks the run ---

class BrokenSpan:
    trace_id = "t1"

    def update(self, **kw):
        raise ConnectionError("langfuse down")

    def score_trace(self, **kw):
        raise ConnectionError("langfuse down")


class Recorder:
    trace_id = "t2"

    def __init__(self):
        self.updates = []

    def update(self, **kw):
        self.updates.append(kw)


def test_failing_score_and_finish_do_not_raise():
    run = tracing.Run(BrokenSpan())
    run.score("faithful", 1.0)
    run.finish({"final": "x", "counts": {"steps": 1}})
    assert run.trace_id == "t1"


def test_finish_puts_counts_in_metadata():
    span = Recorder()
    tracing.Run(span).finish({"final": "x", "counts": {"steps": 2, "writes_rejected": 1}})
    assert span.updates == [{"output": {"final": "x", "counts": {"steps": 2, "writes_rejected": 1}},
                             "metadata": {"steps": 2, "writes_rejected": 1}}]


def test_trace_that_cannot_start_falls_back_to_untraced(monkeypatch):
    class BrokenClient:
        def start_as_current_observation(self, **kw):
            raise ConnectionError("langfuse down")

    monkeypatch.setattr(tracing, "client", lambda: BrokenClient())
    with tracing.traced_run(1) as run:
        run.score("x", 1.0)
    assert run.span is None and run.callbacks == []


def test_agent_errors_are_never_swallowed(monkeypatch):
    monkeypatch.setattr(tracing, "client", lambda: None)
    with pytest.raises(ValueError, match="agent bug"):
        with tracing.traced_run(1):
            raise ValueError("agent bug")


def test_flush_failure_does_not_raise(monkeypatch):
    class BrokenFlush:
        def flush(self):
            raise ConnectionError("langfuse down")

    monkeypatch.setattr(tracing, "client", lambda: BrokenFlush())
    tracing.flush()
