from langchain_core.messages import AIMessage

from conftest import ScriptedModel, call
from triage.agent import run_triage, trajectory


def script_full_triage():
    return [
        AIMessage("", tool_calls=[call("get_issue", {"number": 12}, "r1"), call("list_labels", {}, "r2")]),
        AIMessage("", tool_calls=[call("search_similar_issues", {"query": "traces page blank crash"}, "r3")]),
        AIMessage("", tool_calls=[call("apply_labels", {"number": 12, "labels": ["bug"]}, "w1"),
                                  call("mark_duplicate", {"number": 12, "original": 10, "reason": "same crash"}, "w2")]),
        AIMessage("Labelled #12 as bug and marked it a duplicate of #10."),
    ]


def test_reads_run_without_approval_and_writes_wait_for_it(ws):
    seen = []
    run = run_triage(ws, ScriptedModel(script=script_full_triage()), lambda p: seen.append(p) or "approve")
    assert len(seen) == 1, "exactly one approval prompt, for the turn with write calls"
    assert {p["tool"] for p in seen[0]} == {"apply_labels", "mark_duplicate"}
    assert [a["tool"] for a in run["outbox"]] == ["apply_labels", "mark_duplicate"]
    assert run["final"].startswith("Labelled #12")


def test_rejected_writes_never_run(ws):
    run = run_triage(ws, ScriptedModel(script=script_full_triage()), lambda p: "reject")
    assert run["outbox"] == []
    results = [s["result"] for s in trajectory(run["messages"]) if s["tool"] in {"apply_labels", "mark_duplicate"}]
    assert all("NOT run" in r for r in results)


def test_per_call_decisions(ws):
    run = run_triage(ws, ScriptedModel(script=script_full_triage()), lambda p: {"w1": "approve", "w2": "reject"})
    assert [a["tool"] for a in run["outbox"]] == ["apply_labels"]


def test_search_cannot_see_later_issues(ws):
    hits = ws.index.search("traces page blank crash", before=ws.issue.created_at, exclude=12)
    assert [h.number for h in hits] == [10]
    ws.issue_number = 10
    assert ws.index.search("traces page blank crash", before=ws.issue.created_at, exclude=10) == []


def test_unknown_labels_are_refused(ws):
    script = [AIMessage("", tool_calls=[call("apply_labels", {"number": 12, "labels": ["made-up"]}, "w1")]),
              AIMessage("Done.")]
    run = run_triage(ws, ScriptedModel(script=script), lambda p: "approve")
    assert run["outbox"] == []
    assert "do not exist" in trajectory(run["messages"])[0]["result"]


def test_step_cap_stops_a_looping_agent(ws):
    loop = [AIMessage("", tool_calls=[call("list_labels", {}, "r")])]
    run = run_triage(ws, ScriptedModel(script=loop), lambda p: "approve", max_steps=3)
    assert run["stopped_by_cap"] and run["steps"] == 3


def test_triaged_issue_labels_are_hidden(ws):
    script = [AIMessage("", tool_calls=[call("get_issue", {"number": 12}, "r1")]), AIMessage("Done.")]
    run = run_triage(ws, ScriptedModel(script=script), lambda p: "approve")
    assert '"labels"' not in trajectory(run["messages"])[0]["result"]


# --- the duplicate check by a stronger model ---

from triage.agent import summarize_run
from triage.verify import DuplicateVerdict
from evals.checks import duplicate_outcome


class FakeVerifier:
    def __init__(self, same: bool):
        self.same, self.calls = same, []

    def __call__(self, new, earlier):
        self.calls.append((new.number, earlier.number))
        return DuplicateVerdict(reasoning="Different crash." if not self.same else "Same crash.", duplicate=self.same)


def script_mark_duplicate():
    return [AIMessage("", tool_calls=[call("mark_duplicate", {"number": 12, "original": 10, "reason": "same"}, "w1")]),
            AIMessage("Done.")]


def test_duplicate_blocked_by_the_check_never_reaches_the_human(ws):
    asked, verifier = [], FakeVerifier(same=False)
    run = run_triage(ws, ScriptedModel(script=script_mark_duplicate()), lambda p: asked.append(p) or "approve",
                     verifier=verifier)
    assert verifier.calls == [(12, 10)] and asked == [] and run["outbox"] == []
    result = trajectory(run["messages"])[0]["result"]
    assert result.startswith("Not run: the duplicate check") and "Different crash." in result
    assert summarize_run(run)["counts"]["duplicates_blocked_by_check"] == 1


def test_duplicate_passing_the_check_still_needs_the_human(ws):
    asked = []
    run = run_triage(ws, ScriptedModel(script=script_mark_duplicate()), lambda p: asked.append(p) or "approve",
                     verifier=FakeVerifier(same=True))
    assert len(asked) == 1 and [a["tool"] for a in run["outbox"]] == ["mark_duplicate"]


def test_other_writes_in_the_same_turn_still_go_to_the_human(ws):
    script = [AIMessage("", tool_calls=[call("apply_labels", {"number": 12, "labels": ["bug"]}, "w1"),
                                        call("mark_duplicate", {"number": 12, "original": 10, "reason": "x"}, "w2")]),
              AIMessage("Done.")]
    asked = []
    run = run_triage(ws, ScriptedModel(script=script), lambda p: asked.append(p) or "approve",
                     verifier=FakeVerifier(same=False))
    assert [p["tool"] for p in asked[0]] == ["apply_labels"] and [a["tool"] for a in run["outbox"]] == ["apply_labels"]


def test_blocked_duplicates_do_not_count_as_proposed_but_human_rejections_do(ws):
    blocked = run_triage(ws, ScriptedModel(script=script_mark_duplicate()), lambda p: "approve",
                         verifier=FakeVerifier(same=False))
    rejected = run_triage(ws, ScriptedModel(script=script_mark_duplicate()), lambda p: "reject")
    rec = lambda r: {"trajectory": trajectory(r["messages"])}
    assert duplicate_outcome(rec(blocked), None) == "correct"
    assert duplicate_outcome(rec(blocked), 10) == "missed"
    assert duplicate_outcome(rec(rejected), 10) == "correct"


def test_no_verifier_means_no_check(ws):
    asked = []
    run_triage(ws, ScriptedModel(script=script_mark_duplicate()), lambda p: asked.append(p) or "approve")
    assert len(asked) == 1
