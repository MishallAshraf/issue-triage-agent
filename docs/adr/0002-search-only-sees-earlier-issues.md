# 0002: Search only sees issues opened before the one being triaged

**Status:** accepted

## Context
The golden set uses issues maintainers closed as "Duplicate of #N". The snapshot is taken later, so it also contains issues opened after the one being triaged, including ones that were themselves closed as duplicates of it. If search could see those, the agent could "find" the answer from the future, and duplicate recall would look better than it would be on a real new issue. The triaged issue's own labels leak the answer the same way.

## Decision
`IssueIndex.search` takes a `before` timestamp and an `exclude` number, and the tool always passes the triaged issue's `created_at` and number. `get_issue` hides labels for the triaged issue only.

## Consequences
- Eval numbers reflect what the agent could know when the issue was new.
- Older duplicates get a smaller pool of earlier issues to search, so they're slightly easier than in the full repo. This is listed in the README limits.
- Covered by `tests/test_search.py`, `tests/test_tools.py` and `tests/test_agent.py`.

## Alternatives considered
- **Search everything.** Simpler, but it leaks the answer.
- **Rebuild the index per case from earlier issues only.** Same effect, slower; filtering at query time is enough at this size.
