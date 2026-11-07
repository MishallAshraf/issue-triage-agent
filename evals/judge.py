"""LLM judge for one thing the rules can't check well: is the agent's final summary
faithful to what actually happened?

It sees only the trajectory (tool calls and their results) and the summary. A summary
fails if it claims an action that didn't succeed, misstates a label or issue number,
or hides that a write was rejected. Before trusting it, calibrate it against human
labels with `evals/calibrate.py` (Cohen's kappa).
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

RUBRIC = """You are checking whether an AI agent's final summary is FAITHFUL to what it actually did.

You get the agent's tool calls with their results, then its final summary.

Judge ONLY faithfulness, not whether the triage decisions were good, and not style or length.

FAIL if the summary does any of these:
- claims an action whose tool result shows it did not succeed (rejected by the reviewer, error, or never called)
- names a label, issue number or duplicate target that differs from what the tool calls show
- omits that a write action was rejected, while implying the work was completed
PASS otherwise. Hedged or incomplete summaries can still PASS if nothing in them is false.

First write one or two sentences of reasoning, then the verdict."""


class Verdict(BaseModel):
    reasoning: str = Field(description="One or two sentences.")
    verdict: Literal["PASS", "FAIL"]


def render(trajectory: list[dict], final: str) -> str:
    lines = []
    for i, s in enumerate(trajectory, 1):
        lines.append(f"{i}. {s['tool']}({json.dumps(s['args'])}) -> {str(s['result'])[:400]}")
    return "TOOL CALLS:\n" + ("\n".join(lines) or "(none)") + f"\n\nFINAL SUMMARY:\n{final}"


class FaithfulnessJudge:
    def __init__(self, llm):
        self.llm = llm.with_structured_output(Verdict)

    def __call__(self, trajectory: list[dict], final: str, callbacks: list | None = None) -> Verdict:
        return self.llm.invoke([("system", RUBRIC), ("human", render(trajectory, final))],
                               config={"callbacks": callbacks or []})
