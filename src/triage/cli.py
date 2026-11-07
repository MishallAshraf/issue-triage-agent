"""Triage one issue from the terminal, approving each write yourself.

    python -m triage.cli 1234              # asks you before every write
    python -m triage.cli 1234 --auto reject   # dry run: no write is ever approved
"""

from __future__ import annotations

import argparse
import json
import sys

from .agent import make_model, run_triage, trajectory
from .config import settings
from .data import load
from .search import IssueIndex
from .tools import Workspace
from . import tracing


def ask(pending: list[dict]) -> dict[str, str]:
    answers = {}
    for p in pending:
        print(f"\n  The agent wants to run {p['tool']}:")
        print("  " + json.dumps(p["args"], indent=2).replace("\n", "\n  "))
        a = input("  Approve? [y/N] ").strip().lower()
        answers[p["id"]] = "approve" if a in {"y", "yes"} else "reject"
    return answers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("issue", type=int)
    ap.add_argument("--auto", choices=["approve", "reject"], help="answer every approval automatically")
    args = ap.parse_args(argv)

    snap = load()
    if args.issue not in snap.issues:
        print(f"#{args.issue} is not in the cache for {snap.repo}.", file=sys.stderr)
        return 1
    ws = Workspace(snap=snap, index=IssueIndex(snap.issues), issue_number=args.issue)
    approver = (lambda p: args.auto) if args.auto else ask

    print(f"Triaging {snap.repo}#{args.issue}: {ws.issue.title}  [{settings.model}]")
    with tracing.traced_run(args.issue, tags=["cli"]) as run:
        result = run_triage(ws, make_model(settings.model), approver, callbacks=run.callbacks)
        run.finish({"final": result["final"], "outbox": result["outbox"]})
    tracing.flush()

    print("\nSteps:")
    for s in trajectory(result["messages"]):
        print(f"  {s['tool']}({json.dumps(s['args'])[:80]})")
    print("\nActions that ran (local outbox only; GitHub is never changed):")
    for a in result["outbox"] or [{"none": True}]:
        print("  " + json.dumps(a))
    print("\nAgent summary:\n  " + result["final"])
    if run.trace_id:
        print(f"\nLangfuse trace id: {run.trace_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
