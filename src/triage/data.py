"""Fetch a public repository's issues and labels once, then work from a local cache.

The cache is gitignored: issue text belongs to its authors, so the repo ships the
fetch script and the golden-set issue numbers, not the text itself.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field

import httpx

from .config import settings

API = "https://api.github.com"
DUPLICATE_RE = re.compile(r"(?i)duplicate of\s+(?:[\w.-]+/[\w.-]+)?#(\d+)")


@dataclass
class Issue:
    number: int
    title: str
    body: str
    labels: list[str]
    state: str
    created_at: str
    comments: int = 0
    duplicate_of: int | None = None  # filled for issues maintainers marked as duplicates
    url: str = ""


@dataclass
class Snapshot:
    repo: str
    fetched_at: str
    labels: dict[str, str]  # name -> description
    issues: dict[int, Issue] = field(default_factory=dict)
    # Resume bookkeeping, so a rate-limited fetch can continue instead of restarting.
    dup_checked: list[int] = field(default_factory=list)
    complete: bool = False


def _client() -> httpx.Client:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "issue-triage-agent"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return httpx.Client(base_url=API, headers=headers, timeout=30, follow_redirects=True)


def _get(client: httpx.Client, path: str, **params) -> list | dict:
    r = client.get(path, params=params)
    if r.status_code == 403 and r.headers.get("x-ratelimit-remaining") == "0":
        reset = int(r.headers.get("x-ratelimit-reset", "0"))
        raise RuntimeError(f"GitHub rate limit hit; resets at {time.ctime(reset)}. Set GITHUB_TOKEN to raise it.")
    r.raise_for_status()
    return r.json()


def _issue(it: dict) -> Issue:
    return Issue(
        number=it["number"], title=it["title"], body=(it.get("body") or "")[:6000],
        labels=[lb["name"] for lb in it["labels"]], state=it["state"],
        created_at=it["created_at"], comments=it.get("comments", 0), url=it["html_url"],
    )


def fetch(repo: str, max_issues: int = 600, snap: Snapshot | None = None) -> Snapshot:
    """Fetch recent issues (not pull requests), labels, and duplicate links.

    Saves after each stage. Pass a partial snapshot to resume after a rate limit.
    """
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _client() as c:
        if snap is None or not snap.issues:
            labels = {}
            page = 1
            while True:
                batch = _get(c, f"/repos/{repo}/labels", per_page=100, page=page)
                labels.update({lb["name"]: lb.get("description") or "" for lb in batch})
                if len(batch) < 100:
                    break
                page += 1

            # The search API returns issues only (the issues endpoint mixes in pull requests).
            issues: dict[int, Issue] = {}
            page = 1
            while len(issues) < max_issues and page <= 10:
                batch = _get(c, "/search/issues", q=f"repo:{repo} is:issue", sort="created",
                             order="desc", per_page=100, page=page).get("items", [])
                if not batch:
                    break
                issues.update({it["number"]: _issue(it) for it in batch})
                page += 1
            snap = Snapshot(repo=repo, fetched_at=now, labels=labels, issues=issues)
            save(snap)

        # Issues maintainers closed as duplicates: find the original in the comments.
        # Two ways maintainers mark them: a "duplicate" label, or GitHub's "close as duplicate".
        dups = {}
        for q in (f"repo:{repo} is:issue label:duplicate", f"repo:{repo} is:issue reason:duplicate"):
            dups.update({it["number"]: it for it in _get(c, "/search/issues", q=q, per_page=100).get("items", [])})
        for it in dups.values():
            n = it["number"]
            if n in snap.dup_checked:
                continue
            snap.issues.setdefault(n, _issue(it))
            text = snap.issues[n].body
            for cm in _get(c, f"/repos/{repo}/issues/{n}/comments", per_page=100):
                text += "\n" + (cm.get("body") or "")
            m = DUPLICATE_RE.search(text)
            if m:
                snap.issues[n].duplicate_of = int(m.group(1))
            snap.dup_checked.append(n)
            save(snap)

        # Make sure every referenced original is in the snapshot, so search can find it.
        for iss in list(snap.issues.values()):
            o = iss.duplicate_of
            if o and o not in snap.issues:
                snap.issues[o] = _issue(_get(c, f"/repos/{repo}/issues/{o}"))

    snap.complete = True
    save(snap)
    return snap


def cache_path():
    return settings.cache_dir / (settings.repo.replace("/", "__") + ".json")


def save(snap: Snapshot) -> None:
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    data = asdict(snap)
    data["issues"] = {str(k): v for k, v in data["issues"].items()}
    cache_path().write_text(json.dumps(data, indent=1), encoding="utf-8")


def load() -> Snapshot:
    p = cache_path()
    if not p.exists():
        raise FileNotFoundError(f"No issue cache at {p}. Run: python -m triage.data")
    raw = json.loads(p.read_text(encoding="utf-8"))
    issues = {int(k): Issue(**v) for k, v in raw["issues"].items()}
    return Snapshot(repo=raw["repo"], fetched_at=raw["fetched_at"], labels=raw["labels"], issues=issues,
                    dup_checked=raw.get("dup_checked", []), complete=raw.get("complete", True))


if __name__ == "__main__":
    partial = load() if cache_path().exists() else None
    if partial and partial.complete:
        print(f"Cache already complete: {cache_path()} (delete it to refetch)", file=sys.stderr)
        snap = partial
    else:
        snap = fetch(settings.repo, snap=partial)
    n_dup = sum(1 for i in snap.issues.values() if i.duplicate_of)
    print(f"{settings.repo}: {len(snap.issues)} issues, {len(snap.labels)} labels, "
          f"{n_dup} with a known original -> {cache_path()}", file=sys.stderr)
