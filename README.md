# issue-triage-agent

An agent that triages new GitHub issues (labels, duplicate detection, a reply), built to show how I make an agent observable and how I check its quality. Every write action needs a human's approval, and every run is traced and scored.

It runs on a snapshot of public issues from [Arize-ai/phoenix](https://github.com/Arize-ai/phoenix), because maintainers there mark duplicates with "Duplicate of #N", which gives real ground truth. The agent never writes to GitHub: approved actions go to a local outbox.

## How it works

```
agent ──(read tool calls only)──> tools ──> agent
  │
  ├──(any write tool call)──> approval  [pauses: interrupt()] ──> tools ──> agent
  │
  └──(no tool calls, or step cap)──> end
```

- **LangGraph** state graph. Read tools (`get_issue`, `search_similar_issues`, `list_labels`) run freely. Write tools (`apply_labels`, `mark_duplicate`, `post_comment`) pause the graph with `interrupt()` until a person answers approve/reject for each call. A rejected call is never run; the agent is told so and has to carry on.
- **Guards in the tools, not just the prompt.** Labels that don't exist are refused. The search only sees issues opened *before* the one being triaged, so the agent can't "find" a later issue that was closed as a duplicate of this one. The triaged issue's own labels are hidden. A step cap stops a looping agent.
- **Search** is BM25 over titles and bodies.
- **Model** is set by `TRIAGE_MODEL`, e.g. `openai:gpt-4o-mini` or `anthropic:<model>`.

## Observing it

With Langfuse keys in `.env`, each run is one trace: the graph steps, every LLM call (tokens, cost, latency) and every tool call, with scores attached to the trace (`gate_respected`, `label_f1`, `duplicate`, `faithful`). Emails and token-like strings in issue text are masked before anything is sent. Without keys, everything runs untraced.

## Checking quality

The golden set (`evals/golden.jsonl`, 28 cases) is built from what maintainers actually did: 8 issues they closed as duplicates of a named earlier issue (expected: the same original) and 20 issues they labelled (expected: those labels, no duplicate). Some cases reject writes on purpose, to test what the agent does after a "no".

Two layers of checks:

1. **Rules** (`evals/checks.py`), deterministic and free:
   - every write that ran was approved first
   - only existing labels were tried
   - label precision/recall and duplicate outcome (correct / missed / wrong original / false duplicate), scored on what the agent *decided*, so rejected cases still measure the decision
   - the final summary doesn't claim actions that never ran
   - the step cap wasn't hit
2. **LLM judge** (`evals/judge.py`) for one narrow question: is the final summary faithful to what actually happened? The rule check catches the obvious cases; on the probe bank it flags 12 of the 23 unfaithful summaries and none of the faithful ones. It misses wrong issue numbers and quietly omitted rejections, which is what the judge is for.

**The judge is calibrated before it's trusted.** `evals/probes.jsonl` is a bank of 34 constructed trajectories and summaries covering each failure type plus borderline cases, labelled by me. `python -m evals.calibrate` reports Cohen's kappa between the judge and my labels, and lists each disagreement.

<!-- RESULTS: filled in after the first live run -->

## CI

`.github/workflows/evals.yml` runs the unit tests and then the eval gate on **recorded** runs on every PR, with no API calls, so it's free and deterministic. Thresholds live in `evals/thresholds.json`; the safety ones (approval gate, valid labels, no unbacked claims) must be 100%. A manual `workflow_dispatch` with `live: true` runs the agent for real using repository secrets.

Recordings drop issue bodies; the issue cache itself is gitignored, since issue text belongs to its authors.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e . --no-deps
cp .env.example .env        # add an LLM key; Langfuse and GITHUB_TOKEN are optional

python -m triage.data               # fetch the issue snapshot (resumes if rate-limited)
python -m triage.cli 1234           # triage one issue, approving each write yourself
python -m evals.build_golden        # rebuild the golden set from the snapshot
python -m evals.run_evals --live    # run, record, score, judge
python -m evals.run_evals           # re-score recordings (what CI does)
python -m evals.label_probes        # label the judge probes by hand
python -m evals.calibrate           # judge vs human: Cohen's kappa
pytest -q
```

## Limits

- The golden set is small (28 cases, only 8 duplicates), so the metrics are rough; one case is several percentage points. Phoenix has 30+ issues closed as duplicates, but only 11 name the original, and I didn't guess the rest.
- The snapshot holds the latest ~600 issues plus the duplicate pairs, so for older duplicates the search has fewer distractors than it would in the full repo.
- Duplicate ground truth only covers issues maintainers explicitly marked. An "unmarked" pair might still be a duplicate, so the false-duplicate rate is an upper bound.
- BM25 misses duplicates written in different words; embeddings would be the next thing to try.
