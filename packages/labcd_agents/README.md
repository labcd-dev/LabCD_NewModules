# labcd_agents

A shared, reusable foundation for LLM-backed agents across LabCD backend
modules — providers, retries, token/cost tracking, prompt loading, and
JSON-response parsing, extracted from the duplicated logic in
`MuloDesigner`, `Recommender`, `Regularizer`, `SiloDesigner` and `Trimmer`.

> **This package does not touch any existing module.** It's a foundation
> for *future* modules and refactors — see [Migration path](#migration-path)
> for how an existing module could adopt it incrementally, opt-in.

## Why this exists

Every agent file in `backend_api/` independently reimplements the same
handful of concerns:

| Concern | Duplicated in |
|---|---|
| Picking an LLM client class from a model name | `GaAgent/base_agent.py`, `SiloDesigner/llm_agents.py`, `MuloDesigner/agents.py`, `Regularizer/agents.py` |
| Retry loop around `llm.invoke(...)` | `GaAgent/base_agent.py`, `SiloDesigner/llm_agents.py` |
| Per-model USD price table + vendor-prefix/regex fallback | `GaAgent/base_agent.py`, `SiloDesigner/llm_agents.py`, `Recommender/agents.py`, `Trimmer/agenticNodes/agents.py` |
| Extracting token usage from a response | all six files |
| Building `[system, user] + history` messages | `MuloDesigner/agents.py`, `Regularizer/agents.py`, `Recommender/agents.py` |
| Loading `*.yaml` prompt templates from a directory | `MuloDesigner/agents.py`, `Regularizer/agents.py`, `Recommender/agents.py` |
| Stripping `<think>` tags / markdown fences from a JSON response | `GaAgent/base_agent.py`, `SiloDesigner/llm_agents.py` |
| LangChain callback for token/cost tracking | `Trimmer/agenticNodes/agents.py` |

`labcd_agents` extracts each of these into one tested module, exposed
through a small public API.

## Installation

From the repository root:

```bash
pip install -e packages/labcd_agents
```

Provider SDKs are optional extras — install only what you need:

```bash
pip install -e "packages/labcd_agents[openai]"       # OpenAI only
pip install -e "packages/labcd_agents[openai,groq]"  # OpenAI + Groq
pip install -e "packages/labcd_agents[all]"          # every provider (matches requirements.txt)
```

Or add it to a module's own dependency list:

```
-e ../../packages/labcd_agents[openai,groq]
```

## Quick start

```python
from labcd_agents import BaseAgent

class SummarizerAgent(BaseAgent):
    def summarize(self, text: str) -> str:
        response, usage = self.invoke_llm(
            system_prompt="You are a concise technical summarizer.",
            user_prompt=text,
        )
        return response

agent = SummarizerAgent(model="gpt-4o-mini", temperature=0.0)
summary = agent.summarize("... long document ...")

print(agent.total_usage)   # TokenUsage(input_tokens=..., output_tokens=...)
print(agent.total_cost)    # 0.000123  (USD)
```

`BaseAgent` auto-selects a provider from the model name via `LLMFactory`
(OpenAI, Groq, Cerebras, NVIDIA NIM, or Anthropic). To be explicit, or to use
a model name that's ambiguous across providers:

```python
agent = SummarizerAgent(model="llama-3.3-70b", provider="cerebras")
```

## Public API

```python
from labcd_agents import (
    BaseAgent,             # retry + tracking wrapper around one LLM client
    LLMFactory,             # model name -> provider client
    ProviderSpec,           # register a new provider
    TokenTracker,           # accumulate TokenUsage across calls
    TokenUsage,              # (input_tokens, output_tokens) dataclass
    extract_usage,           # response -> TokenUsage (any known shape)
    CostCalculator,          # model + tokens -> USD cost
    ModelPrice,               # (input_per_million, output_per_million)
    PromptLibrary,             # load *.yaml prompt templates from a directory
    build_messages,             # build a [system?, user, *history] message list
    extract_response_text,       # response -> plain text (any known shape)
    extract_json_from_response,   # LLM text -> parsed JSON, robust to <think>/markdown
    round_floats,                  # round every float in a nested dict/list
    TokenCostCallbackHandler,       # LangChain callback wrapping TokenTracker+CostCalculator
    ensure_env_loaded, get_api_key, require_api_key,  # .env / API key helpers
    get_logger,                      # consistent logging.Logger factory
)
```

### `LLMFactory` — provider-agnostic client construction

```python
from labcd_agents import LLMFactory

llm = LLMFactory.create("gpt-4o-mini", temperature=0.0, seed=42)
llm = LLMFactory.create("meta/llama-4-maverick-17b-128e-instruct")  # -> NVIDIA NIM
llm = LLMFactory.create("gpt-oss-120b", provider="cerebras")         # force a provider
```

Register a new provider (e.g. a local Ollama model) without touching this
package:

```python
from labcd_agents import LLMFactory
from labcd_agents.providers import ProviderSpec

def _build_ollama(model, temperature=0.0, seed=None, **kwargs):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, temperature=temperature, **kwargs)

LLMFactory.register(ProviderSpec(
    name="ollama",
    matcher=lambda model: model.startswith("ollama/"),
    builder=_build_ollama,
))

llm = LLMFactory.create("ollama/llama3.1")
```

### `CostCalculator` — pricing with vendor-prefix + base-model fallback

```python
from labcd_agents import CostCalculator

calc = CostCalculator()
calc.compute_cost("gpt-4o-mini", input_tokens=1200, output_tokens=340)
calc.register("my-fine-tune", input_per_million=0.5, output_per_million=1.5)
```

### `TokenTracker` — accumulate usage across many calls

```python
from labcd_agents import TokenTracker, extract_usage

tracker = TokenTracker()
tracker.record(extract_usage(response), model="gpt-4o-mini", cost=0.002)
tracker.totals            # TokenUsage(...)
tracker.total_cost         # float
tracker.as_state_update()   # {"token_usage": {...}, "total_cost": ...} — drop-in for LangGraph state
```

### `PromptLibrary` — YAML prompt templates

```python
from labcd_agents import PromptLibrary

prompts = PromptLibrary("templates")   # loads every *.yaml in the directory
prompts.format("system_analyser", "analyse_system", equation="dx/dt = -x")
```

### `TokenCostCallbackHandler` — for LCEL chains / LangGraph nodes

For code that builds LCEL chains (`prompt | llm | parser`) rather than
calling `BaseAgent.invoke_llm` directly:

```python
from labcd_agents import TokenTracker, TokenCostCallbackHandler

tracker = TokenTracker()
handler = TokenCostCallbackHandler(tracker)
chain.invoke(inputs, config={"callbacks": [handler]})
tracker.totals
```

## Running the tests

```bash
cd packages/labcd_agents
pip install -e ".[dev]"
pytest
```

## Migration path

Existing modules (`MuloDesigner`, `Recommender`, `Regularizer`, `SiloDesigner`,
`Trimmer`) are **untouched** by this package and continue to work exactly as
before. For a *new* module (or a future opt-in rewrite of an existing one):

1. Add `labcd_agents` (with the provider extras you need) to the module's
   dependencies.
2. Subclass `BaseAgent` instead of writing a new `LLMBaseAgent`/`Agents`
   class from scratch.
3. Replace hand-rolled price tables / token parsing / retry loops with
   `self.invoke_llm(...)`, `self.total_cost`, `self.total_usage`.
4. Replace `_load_all_prompts(directory)` with `PromptLibrary(directory)`.
5. Replace `extract_json_from_response` copies with the shared one.

None of this requires editing the modules listed as "the problem" in this
package's origin — they can be migrated independently, whenever convenient,
each verified in isolation.

## Package layout

```
packages/labcd_agents/
├── pyproject.toml
├── README.md              (this file)
├── src/labcd_agents/
│   ├── __init__.py         # public API surface
│   ├── agent.py             # BaseAgent
│   ├── providers.py          # LLMFactory, ProviderSpec, default provider builders
│   ├── tokens.py               # TokenUsage, extract_usage, TokenTracker
│   ├── pricing.py                # ModelPrice, DEFAULT_PRICE_TABLE, CostCalculator
│   ├── messages.py                 # build_messages, extract_response_text
│   ├── prompts.py                    # PromptLibrary
│   ├── json_utils.py                   # extract_json_from_response, round_floats
│   ├── callbacks.py                      # TokenCostCallbackHandler
│   ├── config.py                           # env/.env + API key helpers
│   ├── logging_utils.py                      # get_logger
│   └── exceptions.py                           # LabCDAgentsError and subclasses
└── tests/                                        # unit tests (pytest)
```
