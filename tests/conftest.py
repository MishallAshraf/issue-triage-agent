"""Shared test fixtures: a tiny fake repo snapshot and a scripted chat model.

The scripted model lets the tests check the graph's control flow (approval gate,
rejections, step cap) without an API key or network.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from triage.data import Issue, Snapshot
from triage.search import IssueIndex
from triage.tools import Workspace


class ScriptedModel(BaseChatModel):
    """Returns the given AIMessages in order, ignoring its input."""

    script: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self.script[min(self.calls, len(self.script) - 1)].model_copy(deep=True)
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name: str, args: dict, id: str) -> dict:
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


@pytest.fixture
def snap() -> Snapshot:
    issues = {
        10: Issue(10, "Traces page crashes on empty project", "Opening traces for a new project shows a blank screen.",
                  ["bug"], "open", "2026-01-01T00:00:00Z"),
        11: Issue(11, "Add dark mode", "Please support a dark theme.", ["enhancement"], "open", "2026-01-02T00:00:00Z"),
        12: Issue(12, "Blank screen on traces page for a new project", "Traces page is empty and crashes.",
                  ["bug", "duplicate"], "closed", "2026-02-01T00:00:00Z", duplicate_of=10),
    }
    return Snapshot("acme/repo", "2026-03-01T00:00:00Z", {"bug": "Something is broken", "enhancement": "New feature",
                                                          "duplicate": "Already reported"}, issues)


@pytest.fixture
def ws(snap) -> Workspace:
    return Workspace(snap=snap, index=IssueIndex(snap.issues), issue_number=12)
