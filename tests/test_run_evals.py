"""The eval runner without any API: approval policies, redaction, scoring from
recordings, the threshold gate and the baseline comparison."""

import json

import pytest
from langchain_core.messages import AIMessage

from conftest import ScriptedModel, call
from evals import run_evals as re_
from triage.agent import dumps_run, run_triage


def recorded_run(ws, policy):
    script = [
        AIMessage("", tool_calls=[call("get_issue", {"number": 12}, "r1")]),
        AIMessage("", tool_calls=[call("apply_labels", {"number": 12, "labels": ["bug"]}, "w1"),
                                  call("mark_duplicate", {"number": 12, "original": 10, "reason": "same"}, "w2")]),
        AIMessage("Applied the bug label. The duplicate mark was rejected, so #12 is still open."),
    ]
    run = json.loads(dumps_run(run_triage(ws, ScriptedModel(script=script), re_.approver_for(policy))))
    run["model"] = "scripted"
    return re_.redact(run)


@pytest.fixture
def tmp_evals(tmp_path, monkeypatch):
    monkeypatch.setattr(re_, "RECORDINGS", tmp_path / "recordings")
    monkeypatch.setattr(re_, "BASELINE", tmp_path / "baseline.json")
    (tmp_path / "recordings").mkdir()
    return tmp_path


def test_approval_policies():
    pending = [{"id": "a", "tool": "apply_labels"}, {"id": "d", "tool": "mark_duplicate"}]
    assert re_.approver_for("approve_all")(pending) == {"a": "approve", "d": "approve"}
    assert re_.approver_for("reject_all")(pending) == {"a": "reject", "d": "reject"}
    assert re_.approver_for("reject_duplicates")(pending) == {"a": "approve", "d": "reject"}


def test_recordings_drop_issue_bodies(ws):
    run = recorded_run(ws, "approve_all")
    body = json.loads(run["trajectory"][0]["result"])["body"]
    assert body.startswith("<") and "chars omitted" in body
    assert "is empty and crashes" not in json.dumps(run), "body text is gone (titles are kept)"


def test_score_reads_recordings_and_measures_the_decision(ws, tmp_evals):
    run = recorded_run(ws, "reject_duplicates")
    (re_.RECORDINGS / "12.json").write_text(json.dumps(run), encoding="utf-8")
    case = {"issue": 12, "expected_labels": ["bug"], "duplicate_of": 10, "policy": "reject_duplicates"}
    s = re_.score([case], {"bug", "enhancement", "duplicate"})
    assert s["n"] == 1 and s["gate_respected"] == 1.0 and s["no_unbacked_claims"] == 1.0
    assert s["duplicate_recall"] == 1.0, "a rejected but correct duplicate proposal still counts as found"
    assert s["label_f1"] == 1.0 and s["rows"][0]["model"] == "scripted"


def test_score_without_recordings_stops_with_a_clear_message(tmp_evals):
    with pytest.raises(SystemExit, match="No recordings"):
        re_.score([{"issue": 1, "expected_labels": [], "policy": "approve_all"}], set())


def test_gate_checks_min_and_max_and_skips_missing_metrics():
    summary = {"gate_respected": 1.0, "label_f1": 0.4, "false_duplicate_rate": 0.2, "judge_faithful": None}
    rules = {"gate_respected": {"min": 1.0}, "label_f1": {"min": 0.5},
             "false_duplicate_rate": {"max": 0.1}, "judge_faithful": {"min": 0.9}}
    fails = re_.gate(summary, rules)
    assert len(fails) == 2 and any("label_f1" in f for f in fails) and any("false_duplicate" in f for f in fails)


def test_compare_knows_which_direction_is_worse():
    base = {"metrics": {"label_f1": 0.70, "false_duplicate_rate": 0.05, "mean_steps": 5.0, "judge_faithful": None}}
    now = {"label_f1": 0.65, "false_duplicate_rate": 0.00, "mean_steps": 6.0, "judge_faithful": 0.9}
    worse = {c["metric"]: c["worse"] for c in re_.compare(now, base)}
    assert worse == {"label_f1": True, "false_duplicate_rate": False, "mean_steps": True}


def test_baseline_round_trip(tmp_evals):
    re_.save_baseline({"n": 28, "label_f1": 0.7, "rows": [{"x": 1}]}, "openai:gpt-4o-mini")
    saved = json.loads(re_.BASELINE.read_text(encoding="utf-8"))
    assert saved == {"model": "openai:gpt-4o-mini", "n": 28, "metrics": {"label_f1": 0.7}}


def test_partial_runs_cannot_overwrite_the_baseline():
    with pytest.raises(SystemExit):
        re_.main(["--only", "12", "--save-baseline"])
