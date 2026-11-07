"""Measure how well the faithfulness judge agrees with a human (Cohen's kappa).

evals/probes.jsonl holds hand-built cases: a trajectory, a final summary, and the
human verdict ("human": "PASS" | "FAIL"). The probes are constructed on purpose to
cover the failure types (claimed-but-rejected actions, wrong numbers, hidden
rejections) alongside faithful summaries, because real runs rarely contain enough
failures to measure a judge on.

    python -m evals.calibrate            # needs JUDGE_MODEL's API key
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent


def cohen_kappa(a: list[str], b: list[str]) -> float:
    assert len(a) == len(b) and a, "need two equal-length, non-empty label lists"
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def load_probes(path: Path = HERE / "probes.jsonl") -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    from triage.llm import make_model
    from triage.config import settings
    from evals.judge import FaithfulnessJudge

    probes = load_probes()
    judge = FaithfulnessJudge(make_model(settings.judge_model))
    human, model, disagreements = [], [], []
    for p in probes:
        v = judge(p["trajectory"], p["final"])
        human.append(p["human"])
        model.append(v.verdict)
        if v.verdict != p["human"]:
            disagreements.append({"id": p["id"], "human": p["human"], "judge": v.verdict, "why": v.reasoning})

    k = cohen_kappa(human, model)
    agree = sum(h == m for h, m in zip(human, model))
    report = {"judge_model": settings.judge_model, "n": len(probes), "agreement": agree / len(probes),
              "kappa": round(k, 3), "confusion": Counter(f"human={h}/judge={m}" for h, m in zip(human, model)),
              "disagreements": disagreements}
    out = settings.results_dir / "calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=dict), encoding="utf-8")
    print(f"{settings.judge_model}: kappa={k:.2f}, agreement {agree}/{len(probes)} -> {out}")
    for d in disagreements:
        print(f"  {d['id']}: human {d['human']}, judge {d['judge']}: {d['why']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
