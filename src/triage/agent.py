"""The triage agent: a LangGraph loop with a human approval gate on every write.

    agent ──(only read tool calls)──> tools ──> agent
      │
      ├──(any write tool call)──> approval [interrupt] ──> tools ──> agent
      │
      └──(no tool calls, or step cap hit)──> END

The approval node pauses the graph with `interrupt()`. The caller resumes it with
`Command(resume={tool_call_id: "approve" | "reject"})`. Rejected calls are never run;
the agent gets a tool message saying so, and has to carry on without them.
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Callable, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from .config import settings
from .tools import WRITE_TOOLS, Workspace, build_tools

SYSTEM_PROMPT = """You triage new GitHub issues for the {repo} repository.

For the issue you are given:
1. Read it with get_issue.
2. Call list_labels and pick only labels that exist. Usually one type label (bug, enhancement, ...) and, if clear, an area label. Don't guess.
3. Search for earlier issues describing the same problem. Only call mark_duplicate if an earlier issue clearly reports the same thing, not just the same area.
4. Apply the labels with apply_labels. Optionally post one short, polite comment.

Write tools (apply_labels, mark_duplicate, post_comment) need a human's approval and may be rejected. If one is rejected, don't retry it; continue with the rest.

Finish with a short summary of what you did. Only mention actions whose tool result says they succeeded."""


class TriageState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    decisions: dict[str, str]  # tool_call_id -> "approve" | "reject", set by the approval node
    stopped_by_cap: bool


def build_graph(ws: Workspace, llm: BaseChatModel, max_steps: int | None = None):
    tools = {t.name: t for t in build_tools(ws)}
    model = llm.bind_tools(list(tools.values()))
    cap = max_steps or settings.max_steps

    def agent(state: TriageState) -> dict:
        reply = model.invoke(state["messages"])
        return {"messages": [reply], "steps": state["steps"] + 1}

    def route(state: TriageState) -> str:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return END
        if state["steps"] >= cap:
            return "cap"
        if any(tc["name"] in WRITE_TOOLS for tc in last.tool_calls):
            return "approval"
        return "tools"

    def approval(state: TriageState) -> dict:
        calls = [tc for tc in state["messages"][-1].tool_calls if tc["name"] in WRITE_TOOLS]
        answer = interrupt({"pending": [{"id": tc["id"], "tool": tc["name"], "args": tc["args"]} for tc in calls]})
        if isinstance(answer, str):  # one answer for all pending calls
            answer = {tc["id"]: answer for tc in calls}
        return {"decisions": {tc["id"]: answer.get(tc["id"], "reject") for tc in calls}}

    def run_tools(state: TriageState) -> dict:
        out = []
        for tc in state["messages"][-1].tool_calls:
            if tc["name"] in WRITE_TOOLS and state["decisions"].get(tc["id"]) != "approve":
                content = f"Rejected by the human reviewer. {tc['name']} was NOT run."
            elif tc["name"] not in tools:
                content = f"Error: unknown tool {tc['name']}."
            else:
                try:
                    content = tools[tc["name"]].invoke(tc["args"])
                except Exception as e:  # bad arguments from the model: report, don't crash
                    content = f"Error: {e}"
            out.append(ToolMessage(content=str(content), tool_call_id=tc["id"], name=tc["name"]))
        return {"messages": out, "decisions": {}}

    def cap_hit(state: TriageState) -> dict:
        # Answer the dangling tool calls so the transcript stays valid, then stop.
        msgs = [ToolMessage(content="Not run: step limit reached.", tool_call_id=tc["id"], name=tc["name"])
                for tc in state["messages"][-1].tool_calls]
        return {"messages": msgs, "stopped_by_cap": True}

    g = StateGraph(TriageState)
    g.add_node("agent", agent)
    g.add_node("approval", approval)
    g.add_node("tools", run_tools)
    g.add_node("cap", cap_hit)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "approval": "approval", "cap": "cap", END: END})
    g.add_edge("approval", "tools")
    g.add_edge("tools", "agent")
    g.add_edge("cap", END)
    return g.compile(checkpointer=InMemorySaver())


Approver = Callable[[list[dict]], dict[str, str] | str]


def run_triage(ws: Workspace, llm: BaseChatModel, approver: Approver, *,
               callbacks: list | None = None, max_steps: int | None = None) -> dict:
    """Triage one issue end to end. `approver` sees the pending write calls and answers."""
    graph = build_graph(ws, llm, max_steps)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "callbacks": callbacks or []}
    state = {
        "messages": [SystemMessage(SYSTEM_PROMPT.format(repo=ws.snap.repo)),
                     HumanMessage(f"Triage issue #{ws.issue_number}.")],
        "steps": 0, "decisions": {}, "stopped_by_cap": False,
    }
    approvals = []
    result = graph.invoke(state, config)
    while result.get("__interrupt__"):
        pending = result["__interrupt__"][0].value["pending"]
        answer = approver(pending)
        approvals.append({"pending": pending, "answer": answer})
        result = graph.invoke(Command(resume=answer), config)

    return {
        "issue": ws.issue_number,
        "messages": result["messages"],
        "outbox": list(ws.outbox),
        "approvals": approvals,
        "steps": result["steps"],
        "stopped_by_cap": result.get("stopped_by_cap", False),
        "final": result["messages"][-1].content if isinstance(result["messages"][-1], AIMessage) else "",
    }


def trajectory(messages: list[AnyMessage]) -> list[dict]:
    """A JSON-friendly record of every tool call and its result, in order."""
    results = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    steps = []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                steps.append({"id": tc["id"], "tool": tc["name"], "args": tc["args"], "result": results.get(tc["id"])})
    return steps


def summarize(steps: list[dict], *, final: str, outbox: list[dict], n_steps: int, stopped_by_cap: bool) -> dict:
    """What happened in one run, stated once by the code that ran it.

    This goes on the root trace observation, so anyone opening a trace (or a later
    tool reading traces) gets the facts directly instead of rebuilding them from
    the trace's debug structure. `steps` is the output of `trajectory()`.
    """
    def kind(result: str | None) -> str:
        r = result or ""
        if r.startswith("Rejected by the human reviewer"):
            return "rejected"
        if r.startswith("Error") or r.startswith("Not run"):
            return "failed"
        return "ran"

    writes = [s for s in steps if s["tool"] in WRITE_TOOLS]
    return {
        "final": final,
        "outbox": outbox,
        "writes": {
            "ran": [{"tool": s["tool"], "args": s["args"]} for s in writes if kind(s["result"]) == "ran"],
            "rejected": [{"tool": s["tool"], "args": s["args"]} for s in writes if kind(s["result"]) == "rejected"],
        },
        "tool_errors": [{"tool": s["tool"], "result": s["result"]} for s in steps if kind(s["result"]) == "failed"],
        "counts": {
            "steps": n_steps,
            "tool_calls": len(steps),
            "writes_approved": sum(kind(s["result"]) == "ran" for s in writes),
            "writes_rejected": sum(kind(s["result"]) == "rejected" for s in writes),
            "tool_errors": sum(kind(s["result"]) == "failed" for s in steps),
            "stopped_by_cap": stopped_by_cap,
        },
    }


def summarize_run(run: dict) -> dict:
    """`summarize` for the dict `run_triage` returns."""
    return summarize(trajectory(run["messages"]), final=run["final"], outbox=run["outbox"],
                     n_steps=run["steps"], stopped_by_cap=run["stopped_by_cap"])


def dumps_run(run: dict) -> str:
    return json.dumps({k: v for k, v in run.items() if k != "messages"} | {"trajectory": trajectory(run["messages"])})
