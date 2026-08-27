import os

from labcd_agents import LLMFactory, ensure_env_loaded

# ensure_env_loaded is a one-shot latch (first call wins). Prefer the monorepo
# root .env; fall back to the legacy sibling plantAgent-master/.env, then cwd.
# Climb: agents -> AgentAdaptive -> backend_core -> repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_ENV_CANDIDATES = (
    os.path.join(_REPO_ROOT, ".env"),
    os.path.normpath(os.path.join(_REPO_ROOT, "..", "plantAgent-master", ".env")),
)
for _env_path in _ENV_CANDIDATES:
    if os.path.isfile(_env_path):
        ensure_env_loaded(_env_path)
        break
else:
    ensure_env_loaded()  # python-dotenv default: walk from cwd

DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"

_ROLES = {
    "design": {
        "model_env": "OPENAI_MODEL",
        "key_env": "OPENAI_API_KEY",
        "max_tokens_env": "OPENAI_MAX_TOKENS",
        "max_tokens": 1500,
        "temperature": 0.2,
        "json_mode": True,
    },
    "tuner": {
        "model_env": "OPENAI_MODEL_TUNER",
        "key_env": "OPENAI_API_KEY_TUNER",
        "max_tokens_env": "OPENAI_MAX_TOKENS_TUNER",
        "max_tokens": 1500,
        "temperature": 0.2,
        "json_mode": True,
    },
    "clarifier": {
        "model_env": "OPENAI_MODEL_CLARIFIER",
        "key_env": "OPENAI_API_KEY_CLARIFIER",
        "max_tokens_env": "OPENAI_MAX_TOKENS_CLARIFIER",
        "max_tokens": 8000,
        "temperature": 0,
        "json_mode": True,
    },
    "reporter": {
        "model_env": "OPENAI_MODEL_REPORTER",
        "key_env": "OPENAI_API_KEY_REPORTER",
        "max_tokens_env": "OPENAI_MAX_TOKENS_REPORTER",
        "max_tokens": 500,
        "temperature": 0.7,
        "json_mode": True,
    },
}

_BASE_ROLE = "design"


def _config(role):
    try:
        return _ROLES[role]
    except KeyError:
        raise ValueError("unknown agent role %r: known roles are %s"
                         % (role, ", ".join(sorted(_ROLES)))) from None


def resolve_model(role):
    cfg = _config(role)
    base = os.environ.get(_ROLES[_BASE_ROLE]["model_env"]) or DEFAULT_OPENAI_MODEL
    if role == _BASE_ROLE:
        return base
    return os.environ.get(cfg["model_env"]) or base


def resolve_models():
    return {
        "agent": resolve_model("design"),
        "tuner": resolve_model("tuner"),
        "clarifier": resolve_model("clarifier"),
        "reporter": resolve_model("reporter"),
    }


def resolve_api_key(role):
    cfg = _config(role)
    if role == _BASE_ROLE:
        return os.environ.get(cfg["key_env"]) or None
    return (os.environ.get(cfg["key_env"])
            or os.environ.get(_ROLES[_BASE_ROLE]["key_env"]) or None)


def resolve_max_tokens(role):
    cfg = _config(role)
    raw = os.environ.get(cfg["max_tokens_env"])
    if raw is None:
        return cfg["max_tokens"]
    try:
        return int(raw)
    except (TypeError, ValueError):
        print("warning: %s=%r is not a number; using the default %d"
              % (cfg["max_tokens_env"], raw, cfg["max_tokens"]))
        return cfg["max_tokens"]


def resolve_provider(role):
    shared = os.environ.get("LLM_PROVIDER") or None
    if role == _BASE_ROLE:
        return shared
    return os.environ.get("LLM_PROVIDER_%s" % role.upper()) or shared


def build_llm(role, max_tokens=None, json_mode=None, **overrides):
    cfg = _config(role)
    model = resolve_model(role)
    provider = resolve_provider(role)

    kwargs = {
        "temperature": cfg["temperature"],
        "max_tokens": resolve_max_tokens(role) if max_tokens is None else max_tokens,
    }
    api_key = resolve_api_key(role)
    if api_key:
        kwargs["api_key"] = api_key

    wants_json = cfg["json_mode"] if json_mode is None else json_mode
    if (provider or LLMFactory.resolve_provider(model)) == "openai":
        # json_mode is an OpenAI-only kwarg. Other providers' builders don't accept it,
        # so it only gets sent once we know we're on openai.
        kwargs["json_mode"] = bool(wants_json)

    kwargs.update(overrides)
    return LLMFactory.create(model, provider=provider, **kwargs)
