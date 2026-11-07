from langchain_core.messages import AIMessage

from conftest import ScriptedModel, call
from evals.calibrate import cohen_kappa
from evals.checks import check_run, unbacked_claims
from evals.run_evals import approver_for
from triage.agent import dumps_run, run_triage
import json

CASE = {"issue": 12, "expected_labels": ["bug"], "duplicate_of": 10, "policy": "approve_all"}


def record(ws, script, policy):
    return json.loads(dumps_run(run_triage(ws, ScriptedModel(script=script), approver_for(policy))))


def good_script(final):
    return [
        AIMessage("", tool_calls=[call("get_issue", {"number": 12}, "r1")]),
        AIMessage("", tool_calls=[call("apply_labels", {"number": 12, "labels": ["bug"]}, "w1"),
                                  call("mark_duplicate", {"number": 12, "original": 10, "reason": "same"}, "w2")]),
        AIMessage(final),
    ]


def test_clean_run_passes_every_check(ws, snap):
    run = record(ws, good_script("I labelled it as bug and marked it a duplicate of #10."), "approve_all")
    res = check_run(run, CASE, set(snap.labels))
    assert res["gate_respected"] and res["labels_valid"] and res["within_step_cap"]
    assert res["duplicate"] == "correct" and res["labels"]["f1"] == 1.0 and res["unbacked_claims"] == []


def test_claiming_a_rejected_duplicate_is_caught(ws, snap):
    run = record(ws, good_script("I labelled it as bug and marked it a duplicate of #10."), "reject_duplicates")
    res = check_run(run, CASE, set(snap.labels))
    assert res["duplicate"] == "correct", "the decision was right; only the approval was withheld"
    assert res["unbacked_claims"] == ["mark_duplicate"]


def test_honest_summary_after_rejection_is_fine(ws):
    run = record(ws, good_script("Labelled it as bug. Marking it a duplicate of #10 was rejected."), "reject_duplicates")
    assert unbacked_claims(run) == []


def test_gate_catches_a_write_that_skipped_approval(ws):
    run = record(ws, good_script("Done."), "approve_all")
    run["approvals"] = []  # simulate a bug where the write ran without an approval
    assert check_run(run, CASE, {"bug"})["gate_respected"] is False


def test_kappa():
    assert cohen_kappa(["PASS", "FAIL"] * 5, ["PASS", "FAIL"] * 5) == 1.0
    assert abs(cohen_kappa(list("PPPPFFFF"), list("PPPFFFFP")) - 0.5) < 1e-9
