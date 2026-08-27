"""Provider-agnostic LLM client construction.

Every module's agent file contains its own copy of "given a model name,
figure out which LangChain chat model class to instantiate" - a hand-rolled
if/elif chain over hardcoded lists of OpenAI/Groq/Cerebras model names plus
NVIDIA NIM vendor prefixes (``meta/``, ``google/``, ...). :class:`LLMFactory`
replaces those per-module chains with a single, extensible registry:

- Each provider is a :class:`ProviderSpec` (a name, a "does this model
  string belong to me" matcher, and a builder function).
- Provider SDKs are imported lazily inside each builder, so importing
  ``labcd_agents`` never requires every provider package to be installed -
  only the ones you actually use.
- Future modules/providers register themselves with
  :func:`LLMFactory.register` instead of editing a shared if/elif chain.

Example:
    >>> from labcd_agents import LLMFactory
    >>> llm = LLMFactory.create("gpt-4o-mini", temperature=0.0, seed=42)
    >>> llm = LLMFactory.create("openai/gpt-oss-120b", provider="nvidia")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from .config import get_api_key
from .exceptions import UnknownProviderError

__all__ = ["ProviderSpec", "LLMFactory"]


@dataclass(frozen=True)
class ProviderSpec:
    """Describes one LLM provider: how to recognize its models, and how to build a client."""

    name: str
    matcher: Callable[[str], bool]
    builder: Callable[..., Any]
    description: str = ""


def _make_matcher(*, exact: Optional[List[str]] = None, prefixes: Optional[List[str]] = None,
                   contains: Optional[List[str]] = None) -> Callable[[str], bool]:
    exact_set = set(exact or [])
    prefix_list = list(prefixes or [])
    contains_list = list(contains or [])

    def matcher(model: str) -> bool:
        if model in exact_set:
            return True
        if any(model.startswith(p) for p in prefix_list):
            return True
        if any(c in model for c in contains_list):
            return True
        return False

    return matcher


# ---------------------------------------------------------------------------
# Default provider builders. Each lazily imports its LangChain integration so
# `import labcd_agents` doesn't require every provider SDK to be installed.
# ---------------------------------------------------------------------------


def _build_openai(model: str, temperature: float = 0.0, seed: Optional[int] = None,
                   json_mode: Optional[bool] = None, **kwargs: Any) -> Any:
    from langchain_openai import ChatOpenAI

    model_kwargs = kwargs.pop("model_kwargs", {}) or {}
    if json_mode is None:
        json_mode = "gpt-" in model
    if json_mode:
        model_kwargs.setdefault("response_format", {"type": "json_object"})

    api_key = kwargs.pop("api_key", None) or get_api_key("openai")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        seed=seed,
        api_key=api_key,
        model_kwargs=model_kwargs,
        **kwargs,
    )


def _build_groq(model: str, temperature: float = 0.0, seed: Optional[int] = None, **kwargs: Any) -> Any:
    from langchain_groq import ChatGroq

    model_kwargs = kwargs.pop("model_kwargs", {}) or {}
    if seed is not None:
        model_kwargs.setdefault("seed", seed)

    api_key = kwargs.pop("api_key", None) or get_api_key("groq")
    return ChatGroq(model=model, temperature=temperature, api_key=api_key, model_kwargs=model_kwargs, **kwargs)


def _build_cerebras(model: str, temperature: float = 0.0, seed: Optional[int] = None, **kwargs: Any) -> Any:
    from langchain_cerebras import ChatCerebras

    api_key = kwargs.pop("api_key", None) or get_api_key("cerebras")
    kwargs.setdefault("max_tokens", None)
    kwargs.setdefault("timeout", None)
    kwargs.setdefault("max_retries", 2)
    return ChatCerebras(model=model, temperature=temperature, seed=seed, api_key=api_key, **kwargs)


def _build_nvidia(model: str, temperature: float = 0.0, seed: Optional[int] = None, **kwargs: Any) -> Any:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    model_kwargs = kwargs.pop("model_kwargs", {}) or {}
    if seed is not None:
        model_kwargs.setdefault("seed", seed)

    api_key = kwargs.pop("api_key", None) or get_api_key("nvidia")
    return ChatNVIDIA(model=model, temperature=temperature, nvidia_api_key=api_key, model_kwargs=model_kwargs, **kwargs)


def _build_anthropic(model: str, temperature: float = 0.0, seed: Optional[int] = None, **kwargs: Any) -> Any:
    from langchain_anthropic import ChatAnthropic

    # Anthropic's API has no `seed` parameter; accepted for interface parity
    # with the other builders and intentionally ignored.
    api_key = kwargs.pop("api_key", None) or get_api_key("anthropic")
    return ChatAnthropic(model=model, temperature=temperature, api_key=api_key, **kwargs)


# Vendor-prefixed model IDs served through NVIDIA NIM/build.nvidia.com, e.g.
# "meta/llama-4-maverick-17b-128e-instruct" or "openai/gpt-oss-120b" when
# routed through NVIDIA rather than Groq/OpenAI directly.
_NVIDIA_PREFIXES = [
    "aisingapore/", "databricks/", "google/", "ibm/", "meta/", "microsoft/",
    "mistralai/", "seallms/", "snowflake/", "deepseek-ai/", "nvidia/",
]

_OPENAI_MODELS = [
    "gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106",
    "gpt-4", "gpt-4-turbo", "gpt-4-turbo-preview", "gpt-4-0125-preview",
    "gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini",
    "gpt-5.4", "gpt-5.4-mini", "gpt-5.5",
]

_GROQ_MODELS = [
    "groq/compound", "groq/compound-mini",
    "llama-3.1-8b-instant", "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile", "llama-3.2-90b-text-preview",
    "llama-3.2-11b-text-preview", "llama-3.2-3b-preview", "llama-3.2-1b-preview",
    "llama3-groq-70b-8192-tool-use-preview", "llama3-groq-8b-8192-tool-use-preview",
    "mixtral-8x7b-32768", "gemma2-9b-it", "gemma-7b-it",
    "deepseek-r1-distill-llama-70b", "deepseek-r1-distill-qwen-32b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-guard-4-12b",
    "moonshotai/kimi-k2-instruct", "moonshotai/kimi-k2-instruct-0905",
    "openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b",
    "qwen/qwen3-32b",
]

_CEREBRAS_MODELS = ["gpt-oss-120b", "llama-3.3-70b", "qwen-3-32b", "llama3.1-8b"]

_ANTHROPIC_PREFIXES = ["claude-"]

# Order matters: some model *names* are ambiguous across providers in the
# source modules themselves - e.g. "gpt-oss-120b" is served bare via Cerebras
# in SiloDesigner but as "openai/gpt-oss-120b" via Groq elsewhere, and both
# happen to start with "gpt-" (OpenAI's own prefix). Exact-list providers
# (groq, cerebras) are therefore matched before the broader prefix-based
# openai/nvidia/anthropic matchers, so a known open-model name always wins
# over an incidental "gpt-" prefix match.
_DEFAULT_PROVIDERS: List[ProviderSpec] = [
    ProviderSpec("cerebras", _make_matcher(exact=_CEREBRAS_MODELS), _build_cerebras,
                 "Open models served via Cerebras."),
    ProviderSpec("groq", _make_matcher(exact=_GROQ_MODELS), _build_groq,
                 "Open models served via Groq's low-latency inference API."),
    ProviderSpec("openai", _make_matcher(exact=_OPENAI_MODELS, prefixes=["gpt-", "o1-", "o3-"]), _build_openai,
                 "OpenAI models (gpt-*, o1-*, o3-*) via the OpenAI API."),
    ProviderSpec("anthropic", _make_matcher(prefixes=_ANTHROPIC_PREFIXES), _build_anthropic,
                 "Anthropic Claude models."),
    ProviderSpec("nvidia", _make_matcher(prefixes=_NVIDIA_PREFIXES), _build_nvidia,
                 "Vendor-namespaced models served via NVIDIA NIM (meta/, google/, ...)."),
]


class LLMFactory:
    """Registry-backed factory that turns a model name into a chat client.

    The registry is a class-level list so registrations made by one module
    (e.g. a future ``labcd_agents_local`` extension registering an Ollama
    provider) are visible to every other module in the process, mirroring
    how the legacy per-module if/elif chains behaved as a single source of
    truth within their own file.
    """

    _registry: List[ProviderSpec] = list(_DEFAULT_PROVIDERS)
    _default_provider: Optional[str] = None

    @classmethod
    def register(cls, spec: ProviderSpec, *, prepend: bool = False) -> None:
        """Register a new provider (or override an existing one by name).

        Args:
            spec: the provider to register.
            prepend: if True, this provider's matcher is tried before all
                existing ones (useful for narrowing/overriding a broader
                built-in match).
        """
        cls._registry = [p for p in cls._registry if p.name != spec.name]
        if prepend:
            cls._registry.insert(0, spec)
        else:
            cls._registry.append(spec)

    @classmethod
    def set_default_provider(cls, provider: Optional[str]) -> None:
        """Set a fallback provider used when no matcher recognizes a model.

        Mirrors the "unknown model -> default to Groq" fallback several
        modules applied, but opt-in and explicit rather than silent.
        """
        cls._default_provider = provider

    @classmethod
    def providers(cls) -> List[str]:
        return [p.name for p in cls._registry]

    @classmethod
    def resolve_provider(cls, model: str) -> Optional[str]:
        """Return the provider name that would handle ``model``, if any."""
        for spec in cls._registry:
            if spec.matcher(model):
                return spec.name
        return None

    @classmethod
    def create(
        cls,
        model: str,
        *,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Build a LangChain chat model client for ``model``.

        Args:
            model: model name/id (e.g. ``"gpt-4o-mini"``, ``"meta/llama-4-maverick..."``).
            temperature: sampling temperature.
            seed: optional deterministic seed, forwarded where the provider supports it.
            provider: force a specific provider by name instead of auto-detecting.
            **kwargs: forwarded to the provider's builder (e.g. ``api_key=``,
                ``max_tokens=``, ``model_kwargs=``, ``json_mode=`` for OpenAI).

        Raises:
            UnknownProviderError: if no registered provider recognizes
                ``model`` and no default provider was configured via
                :meth:`set_default_provider`.
        """
        provider_name = provider or cls.resolve_provider(model) or cls._default_provider
        if provider_name is None:
            raise UnknownProviderError(
                f"No provider recognizes model '{model}'. Pass provider=... explicitly, "
                "register a matching ProviderSpec via LLMFactory.register(), or call "
                "LLMFactory.set_default_provider(...) for a fallback."
            )

        spec = next((p for p in cls._registry if p.name == provider_name), None)
        if spec is None:
            raise UnknownProviderError(f"Unknown provider '{provider_name}'. Known: {cls.providers()}")

        return spec.builder(model=model, temperature=temperature, seed=seed, **kwargs)
