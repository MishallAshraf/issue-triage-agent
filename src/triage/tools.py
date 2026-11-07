"""The agent's tools.

Read tools run freely. Write tools are listed in WRITE_TOOLS: the graph pauses for
human approval before running them, and even then they only append to a local
outbox. Nothing is ever written to the real GitHub repository.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from langchain_core.tools import StructuredTool

from .data import Snapshot
from .search import IssueIndex

WRITE_TOOLS = {"apply_labels", "mark_duplicate", "post_comment"}


@dataclass
class Workspace:
    """Everything one triage run can see and change."""

    snap: Snapshot
    index: IssueIndex
    issue_number: int
    outbox: list[dict] = field(default_factory=list)  # write actions that actually ran

    @property
    def issue(self):
        return self.snap.issues[self.issue_number]


def build_tools(ws: Workspace) -> list[StructuredTool]:
    def get_issue(number: int) -> str:
        """Read an issue's title, body and state. Labels are hidden for the issue being triaged."""
        iss = ws.snap.issues.get(number)
        if iss is None:
            return f"Error: issue #{number} not found."
        out = {"number": iss.number, "title": iss.title, "state": iss.state,
               "created_at": iss.created_at, "body": iss.body[:3000]}
        if number != ws.issue_number:
            out["labels"] = iss.labels
        return json.dumps(out)

    def search_similar_issues(query: str, k: int = 5) -> str:
        """Keyword search over earlier issues in the repository. Use it to look for duplicates."""
        hits = ws.index.search(query, before=ws.issue.created_at, exclude=ws.issue_number, k=min(k, 10))
        return json.dumps([h.__dict__ for h in hits])

    def list_labels() -> str:
        """List the labels that exist in the repository, with descriptions."""
        return json.dumps(ws.snap.labels)

    def apply_labels(number: int, labels: list[str]) -> str:
        """Add existing labels to an issue. Requires human approval."""
        unknown = [lb for lb in labels if lb not in ws.snap.labels]
        if unknown:
            return f"Error: labels do not exist in this repository: {unknown}. Nothing was applied."
        ws.outbox.append({"tool": "apply_labels", "number": number, "labels": labels})
        return f"Applied labels {labels} to #{number}."

    def mark_duplicate(number: int, original: int, reason: str) -> str:
        """Close an issue as a duplicate of an earlier one. Requires human approval."""
        if original not in ws.snap.issues:
            return f"Error: issue #{original} not found. Nothing was changed."
        if original == number:
            return "Error: an issue cannot duplicate itself. Nothing was changed."
        ws.outbox.append({"tool": "mark_duplicate", "number": number, "original": original, "reason": reason})
        return f"Marked #{number} as a duplicate of #{original}."

    def post_comment(number: int, body: str) -> str:
        """Post a reply on the issue for the person who opened it. Requires human approval."""
        ws.outbox.append({"tool": "post_comment", "number": number, "body": body})
        return f"Posted a comment on #{number}."

    fns = [get_issue, search_similar_issues, list_labels, apply_labels, mark_duplicate, post_comment]
    return [StructuredTool.from_function(f) for f in fns]
