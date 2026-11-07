"""Label the probe bank by hand: shows each probe, you answer p (PASS) or f (FAIL).

It hides the generator's `intended` label until you answer, so it can't anchor you.
Progress is saved after every answer; rerun to continue.

    python -m evals.label_probes
"""

from __future__ import annotations

import json
from pathlib import Path

from evals.judge import RUBRIC, render

PATH = Path(__file__).parent / "probes.jsonl"


def main() -> int:
    probes = [json.loads(l) for l in PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    todo = [p for p in probes if not p["human"]]
    print(RUBRIC + f"\n\n{len(todo)} of {len(probes)} probes left. Answer p / f, or q to stop.\n")
    for p in todo:
        print("=" * 70 + f"\n{p['id']}\n" + render(p["trajectory"], p["final"]))
        while (a := input("\nPASS or FAIL? [p/f/q] ").strip().lower()) not in {"p", "f", "q"}:
            pass
        if a == "q":
            break
        p["human"] = "PASS" if a == "p" else "FAIL"
        if p["human"] != p["intended"]:
            print(f"  (noted: the generator expected {p['intended']} for '{p['kind']}')")
        PATH.write_text("".join(json.dumps(x) + "\n" for x in probes), encoding="utf-8")
    left = sum(1 for p in probes if not p["human"])
    print(f"\nSaved. {left} left." if left else "\nAll labelled. Next: python -m evals.calibrate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
