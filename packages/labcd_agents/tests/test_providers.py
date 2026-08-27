import pytest

from labcd_agents.exceptions import UnknownProviderError
from labcd_agents.providers import LLMFactory, ProviderSpec


def test_resolve_provider_openai():
    assert LLMFactory.resolve_provider("gpt-4o-mini") == "openai"


def test_resolve_provider_groq():
    assert LLMFactory.resolve_provider("llama-3.3-70b-versatile") == "groq"


def test_resolve_provider_cerebras():
    assert LLMFactory.resolve_provider("gpt-oss-120b") == "cerebras"


def test_resolve_provider_nvidia_prefix():
    assert LLMFactory.resolve_provider("meta/llama-4-maverick-17b-128e-instruct") == "nvidia"


def test_resolve_provider_anthropic():
    assert LLMFactory.resolve_provider("claude-sonnet-5") == "anthropic"


def test_resolve_provider_unknown_returns_none():
    assert LLMFactory.resolve_provider("totally-unknown-model-xyz") is None


def test_create_unknown_model_raises():
    with pytest.raises(UnknownProviderError):
        LLMFactory.create("totally-unknown-model-xyz")


def test_create_calls_registered_builder():
    calls = []

    def fake_builder(model, temperature=0.0, seed=None, **kwargs):
        calls.append((model, temperature, seed, kwargs))
        return "fake-client"

    LLMFactory.register(ProviderSpec("fake", lambda m: m.startswith("fake-"), fake_builder))
    try:
        client = LLMFactory.create("fake-model-1", temperature=0.5, seed=7, extra="x")
        assert client == "fake-client"
        assert calls == [("fake-model-1", 0.5, 7, {"extra": "x"})]
    finally:
        LLMFactory._registry = [p for p in LLMFactory._registry if p.name != "fake"]


def test_create_with_explicit_provider_override():
    calls = []

    def fake_builder(model, temperature=0.0, seed=None, **kwargs):
        calls.append(model)
        return "fake-client"

    LLMFactory.register(ProviderSpec("fake2", lambda m: False, fake_builder))
    try:
        client = LLMFactory.create("anything", provider="fake2")
        assert client == "fake-client"
        assert calls == ["anything"]
    finally:
        LLMFactory._registry = [p for p in LLMFactory._registry if p.name != "fake2"]


def test_default_provider_fallback():
    LLMFactory.set_default_provider(None)
    with pytest.raises(UnknownProviderError):
        LLMFactory.create("unmatched-model")

    def fake_builder(model, temperature=0.0, seed=None, **kwargs):
        return "fallback-client"

    LLMFactory.register(ProviderSpec("fallback", lambda m: False, fake_builder))
    LLMFactory.set_default_provider("fallback")
    try:
        assert LLMFactory.create("unmatched-model") == "fallback-client"
    finally:
        LLMFactory.set_default_provider(None)
        LLMFactory._registry = [p for p in LLMFactory._registry if p.name != "fallback"]
