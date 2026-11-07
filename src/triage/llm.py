"""Which chat model to use. Kept apart from agent.py, so swapping providers never
touches the agent loop.

    make_model("openai:gpt-4o-mini")
    make_model("anthropic:claude-haiku-4-5")
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel


def make_model(spec: str) -> BaseChatModel:
    """'provider:model' -> a chat model at temperature 0, so runs are as repeatable as the API allows."""
    provider, _, name = spec.partition(":")
    if not name:
        raise ValueError(f"Model spec {spec!r} needs the form provider:model")
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=name, temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=name, temperature=0, max_tokens=2048)
    raise ValueError(f"Unknown model provider in {spec!r}; use openai:<model> or anthropic:<model>")
