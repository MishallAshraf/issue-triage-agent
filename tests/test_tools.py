"""The guards inside the tools themselves, called directly (no agent, no model)."""

import json

from triage.tools import WRITE_TOOLS, build_tools


def tools(ws):
    return {t.name: t for t in build_tools(ws)}


def test_write_tools_are_exactly_the_three_that_change_things(ws):
    assert WRITE_TOOLS == {"apply_labels", "mark_duplicate", "post_comment"}
    assert WRITE_TOOLS <= set(tools(ws))


def test_mark_duplicate_refuses_an_issue_that_does_not_exist(ws):
    out = tools(ws)["mark_duplicate"].invoke({"number": 12, "original": 999, "reason": "x"})
    assert out.startswith("Error") and ws.outbox == []


def test_mark_duplicate_refuses_the_issue_itself(ws):
    out = tools(ws)["mark_duplicate"].invoke({"number": 12, "original": 12, "reason": "x"})
    assert out.startswith("Error") and ws.outbox == []


def test_mark_duplicate_records_a_valid_call_in_the_outbox_only(ws):
    tools(ws)["mark_duplicate"].invoke({"number": 12, "original": 10, "reason": "same crash"})
    assert ws.outbox == [{"tool": "mark_duplicate", "number": 12, "original": 10, "reason": "same crash"}]


def test_apply_labels_is_all_or_nothing(ws):
    out = tools(ws)["apply_labels"].invoke({"number": 12, "labels": ["bug", "made-up"]})
    assert "made-up" in out and ws.outbox == [], "one bad label means nothing is applied"


def test_other_issues_show_their_labels_but_the_triaged_one_does_not(ws):
    t = tools(ws)["get_issue"]
    assert json.loads(t.invoke({"number": 10}))["labels"] == ["bug"]
    assert "labels" not in json.loads(t.invoke({"number": 12}))


def test_get_issue_reports_a_missing_issue(ws):
    assert tools(ws)["get_issue"].invoke({"number": 999}).startswith("Error")


def test_search_tool_never_returns_the_triaged_or_a_later_issue(ws):
    ws.issue_number = 11  # opened after #10, before #12
    hits = json.loads(tools(ws)["search_similar_issues"].invoke({"query": "traces page blank crash dark mode"}))
    assert {h["number"] for h in hits} <= {10}


def test_search_results_show_rank_and_snippet_not_raw_scores(ws):
    hits = json.loads(tools(ws)["search_similar_issues"].invoke({"query": "traces page blank crash"}))
    assert hits[0] == {"rank": 1, "number": 10, "title": "Traces page crashes on empty project", "state": "open",
                       "snippet": "Opening traces for a new project shows a blank screen."}
    assert all("score" not in h for h in hits)


def test_long_snippets_are_cut_at_a_word():
    from triage.tools import SNIPPET_CHARS, _snippet
    s = _snippet("word " * 100)
    assert len(s) <= SNIPPET_CHARS + 4 and s.endswith(" ...") and "  " not in s
