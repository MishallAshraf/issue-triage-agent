# Experiments log

Every live run on the 28-case golden set, in order, with what changed and what it showed. Agent model for all runs: `openrouter:openai/gpt-4o-mini`. Judge: the same model (not yet calibrated; see the judge notes below).

| Run | What changed | Duplicate recall | False duplicates | Label F1 | Judge faithful | Mean steps | Gate |
|-----|--------------|------------------|------------------|----------|----------------|------------|------|
| 1 | BM25 search | 5 of 8 | 1 of 20 | 0.60 | 100% | 5.0 | pass |
| 2 | Hybrid search (BM25 + vectors) | 4 of 8 | 0 of 20 | 0.60 | 100% | 5.0 | fail |
| 3 | + search shows rank and snippet; prompt: open candidates, accept different wording | 7 of 8 | 6 of 20 | 0.44 | 96% | 6.7 | fail |
| 4 | + Claude Sonnet 5.5 checks each proposed duplicate | 6 of 8 | 1 of 20 | 0.56 | 93% | 7.0 | **pass** (current baseline) |

The safety checks (approval gate, valid labels, no claims of actions that didn't happen, step cap) were 100% in every run, including the cases where writes were rejected on purpose.

## What each run taught

**Run 1 (BM25).** Two of the three missed duplicates were search failures: the agent wrote one short query ("edit dataset"), and the original used different words, so it never appeared in the results. The third (#10494) was a reasoning miss: the original was ranked first and the agent didn't mark it.

**Search measured on its own** (`python -m evals.retrieval`, no LLM). For each of the 8 pairs: is the original in the top 5? With the agent's own queries from run 1: BM25 6 of 8, vectors 8 of 8, hybrid 8 of 8. Hybrid chosen ([ADR 0008](adr/0008-hybrid-search.md)).

**Run 2 (hybrid).** Search fixed: the original was ranked first in 7 of 8 cases and third in the other. But end-to-end recall dropped to 4 of 8. The agent saw the right match and didn't mark it. Two likely causes: hybrid's scores look tiny (0.03 vs BM25's 29), and the agent decided from titles alone. Lesson: retrieval recall and end-to-end recall are different numbers.

**Run 3 (agent-side fixes).** Recall jumped to 7 of 8, but false duplicates jumped to 6 of 20: "even in different words" made the agent too eager. It also stopped adding labels whenever it marked a duplicate, which is why label F1 fell. Two of the six were clearly wrong; four were arguable (same feature, never linked by maintainers).

**Run 4 (stronger-model check).** Instead of tuning the prompt further on the same 28 cases, a stronger model filters the agent's proposals ([ADR 0009](adr/0009-duplicate-check-by-a-stronger-model.md)). It blocked 7 of 8 wrong proposals and passed 6 of 7 right ones. Best balance so far, and the gate passes.

## Judge notes

In runs 3 and 4 the judge failed three summaries that were faithful:
- #12886 (runs 3 and 4): the summary said "the duplicate marking was rejected", which is what happened. The judge read the rejection as "never called".
- #17652 (run 4): the summary correctly said the duplicate check found the issues different.

So real faithfulness in run 4 was 100%, not 93%. This is why the judge is calibrated against hand labels (`python -m evals.calibrate`) before its numbers are trusted.

**Calibration.** 34 probes: 16 labelled by hand, 18 by Claude on the same rules (each probe records which). Both `gpt-4o-mini` and Claude Sonnet 5.5 agreed with all 16 human labels (kappa 1.00); on all 34, `gpt-4o-mini` scored 0.93 and Sonnet 1.00. Notes from doing it:
- One human label was a slip: probe 14 had the same pattern as probe 4 but was marked the opposite way. Both judges flagged it, and on review it was corrected to FAIL. Calibration checks the labels too.
- `gpt-4o-mini` gave a different verdict on probe 11 in two identical calibration runs. Sonnet was stable.
- The probe bank doesn't separate the two judges; the real runs did. Harder probes, built from the real judge mistakes above, are the next step.

**Switching the judge.** `JUDGE_MODEL` is now Claude Sonnet 5.5. `python -m evals.run_evals --rejudge` re-judged run 4's recordings without rerunning the agent: the two false failures passed, and judge-faithful went from 93% to 100%. A first attempt failed #10494, because recordings have issue text removed and the summary described what the original issue was about; the re-judge now restores the text from the local cache before judging.

## Open issues
- The agent sometimes skips `apply_labels` after proposing a duplicate (10 of 10 such runs in run 3, 4 of 15 in run 4). That's a behaviour bug, not tuning, and the obvious next fix.
- 8 duplicate cases is a small sample: one case is 12.5 points, and the same model at temperature 0 still made different calls on the same issue between runs (#1589, #16847). Repeating each case several times would firm up the numbers.
