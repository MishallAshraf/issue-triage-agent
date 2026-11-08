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


def agreement(rows: list[dict]) -> dict:
    """Kappa and raw agreement between the reference label and the judge, for a list of rows."""
    ref, got = [r["label"] for r in rows], [r["judge"] for r in rows]
    if not rows:
        return {"n": 0}
    return {"n": len(rows), "agreement": round(sum(a == b for a, b in zip(ref, got)) / len(rows), 3),
            "kappa": round(cohen_kappa(ref, got), 3),
            "confusion": dict(Counter(f"label={a}/judge={b}" for a, b in zip(ref, got)))}


def calibrate(model_spec: str, probes: list[dict]) -> dict:
    from triage.llm import make_model
    from evals.judge import FaithfulnessJudge

    judge = FaithfulnessJudge(make_model(model_spec))
    rows = []
    for p in probes:
        v = judge(p["trajectory"], p["final"])
        rows.append({"id": p["id"], "label": p["human"], "labeller": p.get("labeller", "human"),
                     "judge": v.verdict, "why": v.reasoning})
    return {
        "judge_model": model_spec,
        # Only the human-labelled probes measure agreement with a person; "all" adds probes
        # labelled by Claude on the same rules, which is weaker evidence.
        "human_labelled": agreement([r for r in rows if r["labeller"] == "human"]),
        "all": agreement(rows),
        "disagreements": [r for r in rows if r["judge"] != r["label"]],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    from triage.config import settings

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", help="judge models to compare (default: JUDGE_MODEL)")
    args = ap.parse_args(argv)
    probes = load_probes()
    if any(not p["human"] for p in probes):
        raise SystemExit("Some probes have no label yet. Run: python -m evals.label_probes")

    reports = [calibrate(m, probes) for m in (args.models or [settings.judge_model])]
    # Committed: verdicts and reasons on constructed probes only, no issue text.
    (HERE / "calibration.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    for r in reports:
        h, a = r["human_labelled"], r["all"]
        print(f"\n{r['judge_model']}")
        print(f"  human-labelled: kappa {h['kappa']:.2f}, agreement {h['agreement']:.0%} of {h['n']}")
        print(f"  all probes:     kappa {a['kappa']:.2f}, agreement {a['agreement']:.0%} of {a['n']}")
        for d in r["disagreements"]:
            print(f"  {d['id']} ({d['labeller']}): label {d['label']}, judge {d['judge']}: {d['why']}")
    print(f"\nSaved -> {HERE / 'calibration.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
