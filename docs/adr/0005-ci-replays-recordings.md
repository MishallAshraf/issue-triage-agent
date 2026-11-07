# 0005: CI re-scores recorded runs; live runs are manual

**Status:** accepted

## Context
Running the agent and the judge on every golden case costs money and takes minutes. LLM outputs also vary a little between runs, so a gate on live runs can fail for no code reason. But the checks themselves, the thresholds and the scoring code change often, and those changes need testing on every PR.

## Decision
- `python -m evals.run_evals --live` runs the agent and judge, and writes one recording per case to `evals/recordings/` (issue bodies dropped).
- CI runs `python -m evals.run_evals` with no flags: it re-scores the committed recordings and fails if a threshold in `evals/thresholds.json` is missed. No API calls.
- A manual `workflow_dispatch` with `live: true` does a real run from repository secrets.
- `--save-baseline` stores a run's numbers in `evals/baseline.json`; later runs print what moved and flag what got worse.

## Consequences
- PR checks are free and give the same answer every time.
- Replay catches changes to scoring and thresholds, not changes in how the agent behaves. A prompt or model change needs a fresh live run and new recordings, committed with the change.
- Thresholds say "good enough"; the baseline says "better or worse than last time". Both are shown.

## Alternatives considered
- **Live evals on every PR.** Costly, and noisy enough to train people to ignore failures.
- **No eval gate in CI.** Then scoring bugs and threshold changes go unnoticed.
