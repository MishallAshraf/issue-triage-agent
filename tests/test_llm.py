"""Model setup, without calling any API."""

import pytest

from triage.llm import OPENROUTER_URL, make_model


def test_openrouter_uses_the_openai_client_with_its_own_url_and_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    m = make_model("openrouter:openai/gpt-4o-mini")
    assert m.model_name == "openai/gpt-4o-mini" and m.temperature == 0
    assert str(m.openai_api_base).rstrip("/") == OPENROUTER_URL


def test_openrouter_without_a_key_says_so(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        make_model("openrouter:openai/gpt-4o-mini")


@pytest.mark.parametrize("spec", ["gpt-4o-mini", "mystery:model", "openai:"])
def test_bad_specs_are_refused(spec):
    with pytest.raises(ValueError):
        make_model(spec)
