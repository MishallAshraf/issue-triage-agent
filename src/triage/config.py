"""Settings read from the environment (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    repo: str = os.getenv("TRIAGE_REPO", "Arize-ai/phoenix")
    model: str = os.getenv("TRIAGE_MODEL", "openai:gpt-4o-mini")
    judge_model: str = os.getenv("JUDGE_MODEL", os.getenv("TRIAGE_MODEL", "openai:gpt-4o-mini"))
    # A stronger model that checks every proposed duplicate before a human sees it. Off when empty.
    duplicate_model: str | None = os.getenv("DUPLICATE_MODEL") or None
    github_token: str | None = os.getenv("GITHUB_TOKEN") or None
    # Search behind the agent's search tool: "hybrid" (BM25 + vectors), "bm25" or "vector".
    search_method: str = os.getenv("SEARCH_METHOD", "hybrid")
    cache_dir: Path = ROOT / "data" / "cache"
    models_dir: Path = ROOT / "data" / "models"  # the local embedding model, downloaded once
    results_dir: Path = ROOT / "results"
    # Hard cap on agent steps, so a confused agent can't loop and burn money.
    max_steps: int = int(os.getenv("TRIAGE_MAX_STEPS", "12"))

    @property
    def tracing_enabled(self) -> bool:
        return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


settings = Settings()
