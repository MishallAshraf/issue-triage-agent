"""Build the golden set from what maintainers actually did.

- Duplicate cases: issues a maintainer closed with "Duplicate of #N", where #N is
  older and in the snapshot. Expected: mark_duplicate(original=N).
- Non-duplicate cases: issues a maintainer gave a type label (bug, enhancement, ...),
  not marked duplicate. Expected: those labels, and no mark_duplicate.

Only issue numbers and expected outcomes are committed, never issue text.
Approval policies are spread across cases so rejections get tested too.

    python -m evals.build_golden [--n-dup 12] [--n-other 18]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from triage.data import load

HERE = Path(__file__).parent

# Labels about process, not content: a triager isn't expected to predict these.
PROCESS = re.compile(r"(?i)duplicate|triage|stale|wontfix|won't fix|invalid|help wanted|good first|status|"
                     r"needs|priority|student|railway|^p\d\b|backlog|blocked|in progress|roadmap|release|^c/|size|effort")


def triage_labels(labels: list[str]) -> list[str]:
    return sorted(lb for lb in labels if not PROCESS.search(lb))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dup", type=int, default=12)
    ap.add_argument("--n-other", type=int, default=18)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    rnd = random.Random(args.seed)

    snap = load()
    iss = snap.issues
    dups = [i for i in iss.values()
            if i.duplicate_of in iss and iss[i.duplicate_of].created_at < i.created_at]
    others = [i for i in iss.values()
              if not i.duplicate_of and "duplicate" not in i.labels and triage_labels(i.labels)
              and len(i.body) > 80]
    dups = sorted(rnd.sample(dups, min(args.n_dup, len(dups))), key=lambda i: i.number)
    others = sorted(rnd.sample(others, min(args.n_other, len(others))), key=lambda i: i.number)

    cases = []
    for k, i in enumerate(dups):
        cases.append({"issue": i.number, "expected_labels": triage_labels(i.labels), "duplicate_of": i.duplicate_of,
                      "policy": "reject_duplicates" if k % 3 == 2 else "approve_all"})
    for k, i in enumerate(others):
        cases.append({"issue": i.number, "expected_labels": triage_labels(i.labels), "duplicate_of": None,
                      "policy": "reject_all" if k % 6 == 5 else "approve_all"})

    (HERE / "golden.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")
    (HERE / "golden_meta.json").write_text(json.dumps(
        {"repo": snap.repo, "snapshot_fetched_at": snap.fetched_at, "labels": sorted(snap.labels)}, indent=1),
        encoding="utf-8")
    print(f"{len(dups)} duplicate + {len(others)} other cases from {snap.repo} "
          f"({len(iss)} issues, {sum(1 for i in iss.values() if i.duplicate_of)} with a known original)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
