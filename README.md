# issue-triage-agent

An agent that triages new GitHub issues (labels, duplicate detection, a reply), built to show how I make an agent observable and how I check its quality. Every write action needs a human's approval, and every run is traced and scored.

It runs on a snapshot of public issues from [Arize-ai/phoenix](https://github.com/Arize-ai/phoenix), because maintainers there mark duplicates with "Duplicate of #N", which gives real ground truth. The agent never writes to GitHub: approved actions go to a local outbox.

## How it works

```
agent ──(read tool calls only)──> tools ──> agent
  │
  ├──(any write tool call)──> [duplicate check] ──> approval  [pauses: interrupt()] ──> tools ──> agent
  │
  └──(no tool calls, or step cap)──> end
```

- **LangGraph** state graph. Read tools (`get_issue`, `search_similar_issues`, `list_labels`) run freely. Write tools (`apply_labels`, `mark_duplicate`, `post_comment`) pause the graph with `interrupt()` until a person answers approve/reject for each call. A rejected call is never run; the agent is told so and has to carry on.
- **Guards in the tools, not just the prompt.** Labels that don't exist are refused. The search only sees issues opened *before* the one being triaged, so the agent can't "find" a later issue that was closed as a duplicate of this one. The triaged issue's own labels are hidden. A step cap stops a looping agent.
- **Search** is hybrid: BM25 (keywords) and a small local embedding model (meaning), merged by reciprocal rank fusion. It was chosen by measuring search on its own first (`python -m evals.retrieval`, no LLM): with the agent's own queries, BM25 found 6 of 8 known originals in the top 5, hybrid 8 of 8. `SEARCH_METHOD=bm25` switches back.
- **Duplicate check by a stronger model.** The agent runs on a small model that finds candidates well but judges them unevenly. Every `mark_duplicate` it proposes is first checked by a stronger model (`DUPLICATE_MODEL`, currently Claude Sonnet 5.5) that reads both full issues. Proposals it judges different are dropped before a human is asked; the rest still need approval. This raised duplicate recall from 5 to 6 of 8 without adding false duplicates, without tuning the prompt to the eval set.
- **Models** are set by `TRIAGE_MODEL`, `DUPLICATE_MODEL` and `JUDGE_MODEL`, as `openai:<model>`, `anthropic:<model>` or `openrouter:<model>`.

## Observing it

With Langfuse keys in `.env`, each run is one trace: the graph steps, every LLM call (tokens, cost, latency) and every tool call, with scores attached to the trace (`gate_respected`, `label_f1`, `duplicate`, `faithful`). The top of each trace holds a summary built by the code that ran the agent (which writes ran, which were rejected, which tools failed), with counts in the metadata for filtering. Emails and token-like strings in issue text are masked before anything is sent. Tracing fails open: if Langfuse is down, the run carries on untraced. Without keys, everything runs untraced. [docs/observability.md](docs/observability.md) covers what a trace holds and how to find a bad run.

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

**The judge is calibrated before it's trusted.** `evals/probes.jsonl` is a bank of 34 constructed trajectories and summaries covering each failure type plus borderline cases. I labelled 16 by hand; the other 18 were labelled by Claude on the same rules, and each probe records who labelled it. `python -m evals.calibrate --models ...` reports Cohen's kappa between each judge and the labels, on the human-labelled probes alone and on all 34, and lists every disagreement.

## Results

Current setup: `gpt-4o-mini` agent, hybrid search, Claude Sonnet 5.5 duplicate check, Claude Sonnet 5.5 judge. 28 golden cases, all through OpenRouter.

| Check | Result | Gate |
|-------|--------|------|
| Every write approved before it ran | 100% | 100% |
| Only existing labels tried | 100% | 100% |
| No claims of actions that didn't happen (rules) | 100% | 100% |
| Stayed under the step cap | 100% | 100% |
| Duplicate recall | 6 of 8 (0.75) | ≥ 0.6 |
| False duplicate rate | 1 of 20 (0.05) | ≤ 0.1 |
| Label F1 | 0.56 | ≥ 0.5 |
| Judge: summary faithful | 28 of 28 | ≥ 0.9 |

**Getting there took four runs** ([docs/experiments.md](docs/experiments.md)). BM25 search missed duplicates written in different words; measured on its own, hybrid search put every original in the top 5 (BM25: 6 of 8). But end-to-end recall then *fell*, because the agent didn't mark candidates it was shown. Pushing the prompt harder found 7 of 8 duplicates and wrongly marked 6 of 20 other issues. Instead of tuning the prompt further on the same 28 cases, a stronger model now checks each proposed duplicate: it blocked 7 of 8 wrong proposals and passed 6 of 7 right ones.

**Judge calibration** (`evals/calibration.json`):

| Judge | Human-labelled probes (16) | All probes (34) |
|-------|----------------------------|-----------------|
| `gpt-4o-mini` | kappa 1.00 | kappa 0.93 |
| Claude Sonnet 5.5 | kappa 1.00 | kappa 1.00 |

Both agree with every human label, so the probe bank doesn't separate them: it's easier than real runs. Real runs did. `gpt-4o-mini` failed three faithful summaries in live runs (for example, reading "the duplicate marking was rejected" as a false claim), and changed its verdict on one probe between two identical calibration runs. Sonnet was stable and, re-judging run 4, passed all 28 summaries, which matches my own reading. So Sonnet is the judge. The next improvement is harder probes built from those real judge mistakes.

What the numbers don't show: the set is small (one duplicate case is 12.5 points), the false-duplicate rate is an upper bound (the one "false" duplicate asks for the same eval suite as its match; maintainers just never linked them), and label F1 is held down partly because the agent adds area labels that maintainers often skip.

## CI

`.github/workflows/evals.yml` runs the unit tests and then the eval gate on **recorded** runs on every PR, with no API calls, so it's free and deterministic. Thresholds live in `evals/thresholds.json`; the safety ones (approval gate, valid labels, no unbacked claims) must be 100%. A manual `workflow_dispatch` with `live: true` runs the agent for real using repository secrets.

Thresholds say "good enough"; they don't say "worse than last time". So `evals/baseline.json` holds the numbers from a reference run (`--save-baseline`), and every run prints each metric next to it and flags the ones that dropped.

Recordings drop issue bodies; the issue cache itself is gitignored, since issue text belongs to its authors.

## Design decisions

The main choices, with the reasons and what they cost, are in [docs/adr/](docs/adr/README.md): the approval gate, search that only sees earlier issues, masking before tracing, the narrow calibrated judge, replaying recordings in CI, the run summary on each trace, tracing that fails open, hybrid search, and the duplicate check. [docs/experiments.md](docs/experiments.md) records every live run: what changed, what it showed, and what went wrong.

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
python -m evals.run_evals --save-baseline   # keep these numbers as the baseline
python -m evals.retrieval           # search alone: BM25 vs vectors vs hybrid, recall@k (no LLM)
python -m evals.label_probes        # label the judge probes by hand
python -m evals.calibrate --models openrouter:openai/gpt-4o-mini openrouter:anthropic/claude-sonnet-5.5   # judges vs labels: kappa
python -m evals.run_evals --rejudge # re-judge saved recordings with JUDGE_MODEL (no agent rerun)
pytest -q
```

## Limits

- The golden set is small (28 cases, only 8 duplicates), so the metrics are rough; one case is several percentage points. Phoenix has 30+ issues closed as duplicates, but only 11 name the original, and I didn't guess the rest.
- The snapshot holds the latest ~600 issues plus the duplicate pairs, so for older duplicates the search has fewer distractors than it would in the full repo.
- Duplicate ground truth only covers issues maintainers explicitly marked. An "unmarked" pair might still be a duplicate, so the false-duplicate rate is an upper bound.
- Retrieval is measured on 8 pairs, so the comparison between vectors and hybrid is within one pair. The clear finding is BM25 missing rewordings of short agent queries.
