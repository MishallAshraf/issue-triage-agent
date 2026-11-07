"""Which chat model to use. Kept apart from agent.py, so swapping providers never
touches the agent loop.

    make_model("openai:gpt-4o-mini")
    make_model("anthropic:claude-haiku-4-5")
    make_model("openrouter:openai/gpt-4o-mini")     # any model on openrouter.ai
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

OPENROUTER_URL = "https://openrouter.ai/api/v1"


def make_model(spec: str) -> BaseChatModel:
    """'provider:model' -> a chat model at temperature 0, so runs are as repeatable as the API allows."""
    provider, _, name = spec.partition(":")
    if not name:
        raise ValueError(f"Model spec {spec!r} needs the form provider:model")
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=name, temperature=0)
    if provider == "openrouter":
        # OpenRouter speaks the OpenAI API, so the same client works with a different URL and key.
        from langchain_openai import ChatOpenAI
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY is not set in .env")
        return ChatOpenAI(model=name, temperature=0, base_url=OPENROUTER_URL, api_key=key)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=name, temperature=0, max_tokens=2048)
    raise ValueError(f"Unknown model provider in {spec!r}; use openai:, anthropic: or openrouter:<model>")
