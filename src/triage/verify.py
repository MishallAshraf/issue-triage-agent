"""A second opinion on duplicates, from a stronger model.

The triage agent runs on a small, cheap model. In testing it swung between missing
clear duplicates and marking related-but-different issues as duplicates. Instead of
tuning its prompt on the 28 eval cases, every `mark_duplicate` it proposes is checked
by a stronger model that reads both full issues. A proposal it judges different is
dropped before a human is asked. Set DUPLICATE_MODEL to turn this on.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .data import Issue

RUBRIC = """You decide whether a new GitHub issue is a DUPLICATE of an earlier one.

DUPLICATE means both report the same underlying bug or ask for the same change, so that one fix or one feature would close both. The wording can differ completely.

NOT a duplicate:
- they touch the same area, component or feature, but describe different problems
- one is a broader or narrower request than the other
- one is a follow-up, regression or side effect of the other
- you can't tell from the text

When unsure, answer NOT a duplicate: a wrong duplicate close hides a real problem from maintainers.

Give one or two sentences of reasoning first, then the decision."""


class DuplicateVerdict(BaseModel):
    reasoning: str = Field(description="One or two sentences.")
    duplicate: bool


def _show(label: str, iss: Issue, limit: int = 3000) -> str:
    return f"{label} #{iss.number} (opened {iss.created_at[:10]}): {iss.title}\n\n{iss.body[:limit]}"


class DuplicateVerifier:
    def __init__(self, llm, name: str = ""):
        self.name = name
        # JSON-schema mode: some providers (Claude Sonnet 5.5 via OpenRouter) refuse the forced
        # tool choice that function-calling mode relies on.
        self.llm = llm.with_structured_output(DuplicateVerdict, method="json_schema")

    def __call__(self, new: Issue, earlier: Issue) -> DuplicateVerdict:
        text = _show("NEW ISSUE", new) + "\n\n---\n\n" + _show("EARLIER ISSUE", earlier)
        return self.llm.invoke([("system", RUBRIC), ("human", text)])
