"""Run the eval suite.

    python -m evals.run_evals --live      # run the agent on every golden case, record, score, judge
    python -m evals.run_evals             # replay: re-score saved recordings (no API key; what CI runs)
    python -m evals.run_evals --save-baseline   # store these numbers as the baseline to compare against

Each golden case has an approval policy, so some runs have writes rejected on purpose.
That's how the suite checks the agent behaves after a rejection: no retrying, and no
claiming the rejected action happened.

Exits non-zero if any gate in evals/thresholds.json fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evals.checks import check_run

HERE = Path(__file__).parent
RECORDINGS = HERE / "recordings"
BASELINE = HERE / "baseline.json"
# Metrics where a lower number is better; for every other metric, higher is better.
LOWER_IS_BETTER = {"false_duplicate_rate", "mean_steps"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def approver_for(policy: str):
    """approve_all | reject_all | reject_duplicates"""
    def approve(pending: list[dict]) -> dict[str, str]:
        out = {}
        for p in pending:
            reject = policy == "reject_all" or (policy == "reject_duplicates" and p["tool"] == "mark_duplicate")
            out[p["id"]] = "reject" if reject else "approve"
        return out
    return approve


def redact(run: dict) -> dict:
    """Recordings are committed, so drop issue bodies; keep only what scoring needs."""
    for s in run["trajectory"]:
        if s["tool"] == "get_issue" and s["result"] and s["result"].startswith("{"):
            r = json.loads(s["result"])
            r["body"] = f"<{len(r.get('body', ''))} chars omitted>"
            s["result"] = json.dumps(r)
    return run


def live(cases: list[dict], judge_on: bool) -> None:
    from triage.agent import dumps_run, run_triage, summarize
    from triage.llm import make_model
    from triage.config import settings
    from triage.data import load
    from triage.search import IssueIndex
    from triage.tools import Workspace
    from triage import tracing
    from evals.judge import FaithfulnessJudge

    snap = load()
    index = IssueIndex(snap.issues)
    llm = make_model(settings.model)
    judge = FaithfulnessJudge(make_model(settings.judge_model)) if judge_on else None
    RECORDINGS.mkdir(exist_ok=True)

    for case in cases:
        ws = Workspace(snap=snap, index=index, issue_number=case["issue"])
        with tracing.traced_run(case["issue"], tags=["eval", case["policy"]]) as tr:
            run = json.loads(dumps_run(run_triage(ws, llm, approver_for(case["policy"]), callbacks=tr.callbacks)))
            run["model"] = settings.model
            run["trace_id"] = tr.trace_id
            res = check_run(run, case, set(snap.labels))
            if judge:
                v = judge(run["trajectory"], run["final"], callbacks=tr.callbacks)
                run["judge"] = v.model_dump()
                tr.score("faithful", 1.0 if v.verdict == "PASS" else 0.0, v.reasoning)
            tr.score("gate_respected", float(res["gate_respected"]))
            if res["labels"]:
                tr.score("label_f1", res["labels"]["f1"])
            tr.score("duplicate", res["duplicate"])
            tr.finish(summarize(run["trajectory"], final=run["final"], outbox=run["outbox"],
                                n_steps=run["steps"], stopped_by_cap=run["stopped_by_cap"]))
        (RECORDINGS / f"{case['issue']}.json").write_text(json.dumps(redact(run), indent=1), encoding="utf-8")
        f1 = f"{res['labels']['f1']:.2f}" if res["labels"] else "n/a"
        print(f"  #{case['issue']}: {res['duplicate']}, label F1 {f1}, "
              f"steps {res['steps']}" + (f", judge {run['judge']['verdict']}" if judge else ""), file=sys.stderr)
    tracing.flush()


def score(cases: list[dict], repo_labels: set[str]) -> dict:
    rows = []
    for case in cases:
        path = RECORDINGS / f"{case['issue']}.json"
        if not path.exists():
            print(f"  missing recording for #{case['issue']}", file=sys.stderr)
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        row = check_run(run, case, repo_labels)
        row["judge"] = run.get("judge", {}).get("verdict")
        row["model"] = run.get("model")
        rows.append(row)

    n = len(rows)
    if n == 0:
        raise SystemExit("No recordings found. Run once with --live first.")
    dup_issues = {c["issue"] for c in cases if c.get("duplicate_of")}
    dup_cases = [r for r in rows if r["issue"] in dup_issues]
    non_dup = [r for r in rows if r not in dup_cases]
    judged = [r for r in rows if r["judge"]]
    labelled = [r for r in rows if r["labels"]]
    return {
        "n": n,
        "gate_respected": sum(r["gate_respected"] for r in rows) / n,
        "labels_valid": sum(r["labels_valid"] for r in rows) / n,
        "no_unbacked_claims": sum(not r["unbacked_claims"] for r in rows) / n,
        "within_step_cap": sum(r["within_step_cap"] for r in rows) / n,
        "label_f1": (sum(r["labels"]["f1"] for r in labelled) / len(labelled)) if labelled else None,
        "duplicate_recall": (sum(r["duplicate"] == "correct" for r in dup_cases) / len(dup_cases)) if dup_cases else None,
        "false_duplicate_rate": (sum(r["duplicate"] == "false_duplicate" for r in non_dup) / len(non_dup)) if non_dup else None,
        "judge_faithful": (sum(r["judge"] == "PASS" for r in judged) / len(judged)) if judged else None,
        "mean_steps": sum(r["steps"] for r in rows) / n,
        "rows": rows,
    }


def gate(summary: dict, thresholds: dict) -> list[str]:
    fails = []
    for metric, rule in thresholds.items():
        v = summary.get(metric)
        if v is None:
            continue
        if "min" in rule and v < rule["min"]:
            fails.append(f"{metric} = {v:.2f}, below {rule['min']}")
        if "max" in rule and v > rule["max"]:
            fails.append(f"{metric} = {v:.2f}, above {rule['max']}")
    return fails


def compare(summary: dict, baseline: dict) -> list[dict]:
    """Each metric next to its baseline value, flagging the ones that got worse.

    Thresholds say "good enough to ship"; the baseline says "did this change make
    things better or worse than last time". A drop can matter even above threshold.
    """
    out = []
    for metric, old in baseline["metrics"].items():
        new = summary.get(metric)
        if new is None or old is None:
            continue
        delta = new - old
        worse = delta > 1e-9 if metric in LOWER_IS_BETTER else delta < -1e-9
        out.append({"metric": metric, "baseline": old, "now": new, "delta": delta, "worse": worse})
    return out


def save_baseline(summary: dict, model: str | None) -> None:
    metrics = {k: v for k, v in summary.items() if k not in {"rows", "n"}}
    BASELINE.write_text(json.dumps({"model": model, "n": summary["n"], "metrics": metrics}, indent=2) + "\n",
                        encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="run the agent (needs API keys) before scoring")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge in live mode")
    ap.add_argument("--only", type=int, nargs="*", help="issue numbers to run")
    ap.add_argument("--save-baseline", action="store_true", help="store this run's numbers in evals/baseline.json")
    args = ap.parse_args(argv)
    if args.save_baseline and args.only:
        ap.error("--save-baseline needs the full golden set, not --only")

    cases = load_jsonl(HERE / "golden.jsonl")
    if args.only:
        cases = [c for c in cases if c["issue"] in args.only]
    meta = json.loads((HERE / "golden_meta.json").read_text(encoding="utf-8"))
    if args.live:
        live(cases, judge_on=not args.no_judge)

    summary = score(cases, set(meta["labels"]))
    fails = gate(summary, json.loads((HERE / "thresholds.json").read_text(encoding="utf-8")))

    print(f"\n{summary['n']} cases")
    for k, v in summary.items():
        if k not in {"rows", "n"} and v is not None:
            print(f"  {k:22} {v:.2f}")
    for r in summary["rows"]:
        if not r["gate_respected"] or r["unbacked_claims"] or r["duplicate"] != "correct" or r["judge"] == "FAIL":
            print(f"  ! #{r['issue']}: dup={r['duplicate']} gate={r['gate_respected']} "
                  f"claims={r['unbacked_claims']} judge={r['judge']}")
    if BASELINE.exists() and not args.only:
        base = json.loads(BASELINE.read_text(encoding="utf-8"))
        print(f"\nAgainst the baseline ({base.get('model')}, {base['n']} cases):")
        for c in compare(summary, base):
            mark = "  worse" if c["worse"] else ""
            print(f"  {c['metric']:22} {c['baseline']:.2f} -> {c['now']:.2f} ({c['delta']:+.2f}){mark}")
    if args.save_baseline:
        models = {r["model"] for r in summary["rows"]} - {None}
        save_baseline(summary, ", ".join(sorted(models)) or None)
        print(f"\nSaved baseline to {BASELINE.relative_to(HERE.parent)}")
    if fails:
        print("\nGATE FAILED:\n  " + "\n  ".join(fails))
        return 1
    print("\nGate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
