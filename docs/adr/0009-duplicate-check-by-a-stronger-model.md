# 0009: A stronger model checks every proposed duplicate, in code, before a human sees it

**Status:** accepted

## Context
After hybrid search ([0008](0008-hybrid-search.md)), the right earlier issue was in the agent's results for all 8 known duplicates. The remaining problem was the agent's judgement (`gpt-4o-mini`):

- With a cautious prompt it missed duplicates whose original was ranked first (4 of 8 found).
- When the prompt was changed to open candidates and accept different wording, it found 7 of 8, but also marked 6 of 20 ordinary issues as duplicates. It also stopped adding labels after marking a duplicate.

More prompt tuning would have meant fitting the agent to the same 28 cases it's scored on. With a set this small, that improves the score more than the agent.

## Decision
Split the job: the cheap agent proposes, a stronger model filters, and the human approves.

- A `check_duplicates` node runs between the agent and the approval step whenever the agent proposes `mark_duplicate`. It sends both full issues (title and the first 3,000 characters of the body) to a stronger model with a narrow rubric: same underlying problem or request, so one fix would close both. Related area, broader or narrower scope, follow-ups, or "can't tell" all mean no. When unsure, say no.
- A proposal judged different is dropped with the model's reason, and the human is never asked about it. A proposal judged a duplicate still needs human approval.
- It's a separate node because `interrupt()` re-runs its node on resume, which would call the stronger model twice.
- It's turned on with `DUPLICATE_MODEL`. The current setting is `openrouter:anthropic/claude-sonnet-5.5`: a different provider from the agent, so the two are less likely to share blind spots.
- Scoring: a proposal blocked by the check doesn't count as proposed, since the whole system decided against it. A proposal the human rejects still counts, because the human is part of the test, not the system.

## Results (run 4 vs run 1, same 28 cases)
- Duplicate recall 5 of 8 → 6 of 8; false duplicates 1 of 20 → 1 of 20.
- Of 15 proposals, the check blocked 7 of the 8 wrong ones and passed 6 of the 7 right ones.
- The one wrong proposal it passed (#16294 → #16183) asks for the same eval suite; it's arguably a real duplicate that maintainers never linked.
- The one right proposal it blocked (#568 → #495, a first-time tutorial vs contextual help on pages) is a judgement call; it followed "when unsure, say no".
- Mean steps 5.0 → 7.0, plus about 15 stronger-model calls per run. Still cents per run.

## Consequences
- Precision comes from code plus a stronger model, not from prompt wording tuned to the eval set.
- The agent can be tuned for recall (finding candidates) without paying for it in false closes.
- Two models to configure and pay for instead of one.
- The check only sees what the agent proposes: a duplicate the agent never proposes (#3306 in run 4) is still missed.
- Covered by `tests/test_agent.py` with a fake verifier: blocked proposals never reach the human, passed ones still need approval, other writes in the same turn are unaffected, and scoring treats blocked and rejected proposals differently.

## Alternatives considered
- **Keep tuning the agent's prompt.** Cheapest, but it fits the agent to the 28 scored cases.
- **Run the whole agent on the stronger model.** Simpler, but every step pays the higher price, while only the duplicate decision needed it.
- **Leave it to the human reviewer.** The approval gate already allows this, but a reviewer shown many wrong suggestions starts approving without reading.
