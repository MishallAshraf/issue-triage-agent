# 0004: The LLM judge answers one question, and is calibrated before it's trusted

**Status:** accepted

## Context
Most of what matters can be checked with plain code: did writes wait for approval, were the labels real, did the agent pick the right original. One thing can't be checked well with rules: whether the final summary honestly describes what happened. A regex catches "marked as duplicate" after a rejected `mark_duplicate`. It misses a wrong issue number, or a summary that quietly leaves out a rejection.

## Decision
- Rules first (`evals/checks.py`): free, the same result every time, and run on every PR.
- One LLM judge (`evals/judge.py`) for faithfulness only, with a short rubric, PASS/FAIL output and one or two sentences of reasoning first. It doesn't grade label quality or style.
- Before its scores count, it's measured against my own labels on a bank of 34 constructed cases (`evals/probes.jsonl`) that cover each failure type plus borderline faithful ones. `evals/calibrate.py` reports Cohen's kappa and lists every disagreement.

## Consequences
- The judge's number comes with a measured agreement rate, instead of being taken on trust.
- The probes are constructed, because real runs rarely contain enough unfaithful summaries to measure a judge on. They show the judge works on the failure types I know about, not on ones I haven't thought of.
- Labelling the probes is manual work, and the tool hides the intended label while I answer so it doesn't anchor me.

## Alternatives considered
- **A helpfulness judge ("did the agent get the job done?").** It punishes correct behaviour. When a reviewer rejects a write, the right outcome is an agent that stops and says so, and a helpfulness judge would score that low. The golden set rejects writes on purpose, so this would mark the agent down exactly when it behaves well. Whether the agent made the *right* decision is a separate question, answered by the golden set's ground truth, not by the judge.
- **One judge scoring everything (1 to 10 overall quality).** Hard to calibrate, and hard to act on a low score.
- **Rules only.** They miss the subtle cases, which are the ones that mislead a reviewer.
- **Trust the judge without calibration.** No way to know how often it's wrong.
