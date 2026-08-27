"""labcd_agents: shared LLM-agent foundation for LabCD backend modules.

This package extracts the patterns duplicated across
``MuloDesigner/GaAgent``, ``MuloDesigner``, ``Recommender``, ``Regularizer``,
``SiloDesigner`` and ``Trimmer``'s agent files into one reusable, tested
library:

- **Provider construction** — :class:`LLMFactory` picks the right LangChain
  chat model class (OpenAI, Groq, Cerebras, NVIDIA NIM, Anthropic, ...) from
  a model name, the way every module's ``_initialize_llm_client`` /
  ``_choose_model`` did independently.
- **Retries + invocation** — :class:`BaseAgent` provides ``invoke_llm`` /
  ``call``, matching ``LLMBaseAgent.invoke_llm`` and the various
  ``Agents._call_llm`` helpers, with built-in retry logic and logging.
- **Token tracking** — :class:`TokenTracker` / :func:`extract_usage`
  normalize the several usage-reporting shapes LangChain/OpenAI use into one
  :class:`TokenUsage` type.
- **Cost calculation** — :class:`CostCalculator` merges every module's
  per-model price table and vendor-prefix/regex fallback logic.
- **Prompt templates** — :class:`PromptLibrary` replaces the duplicated
  ``_load_all_prompts(directory)`` method.
- **JSON parsing** — :func:`extract_json_from_response` replaces the
  duplicated think-tag-stripping / markdown-fence / brace-matching parser.
- **LangChain callback integration** — :class:`TokenCostCallbackHandler`
  replaces ``TokenCostCallback`` for chain/graph-based (non-``invoke_llm``)
  call sites.

Quick start:
    >>> from labcd_agents import BaseAgent
    >>> class MyAgent(BaseAgent):
    ...     def summarize(self, text: str) -> str:
    ...         response, usage = self.invoke_llm(
    ...             system_prompt="You are a concise summarizer.",
    ...             user_prompt=text,
    ...         )
    ...         return response
    >>> agent = MyAgent(model="gpt-4o-mini")
    >>> agent.summarize("...")           # doctest: +SKIP
    >>> agent.total_cost                  # doctest: +SKIP
    0.000123

See the package README for a full guide, including how to register a new
provider and how a future backend module should depend on this package.
"""

from .agent import BaseAgent
from .callbacks import TokenCostCallbackHandler
from .config import ensure_env_loaded, get_api_key, require_api_key
from .exceptions import (
    LabCDAgentsError,
    LLMInvocationError,
    PromptNotFoundError,
    UnknownProviderError,
)
from .json_utils import extract_json_from_response, round_floats, strip_think_tags
from .logging_utils import get_logger
from .messages import build_messages, extract_response_text
from .pricing import CostCalculator, DEFAULT_PRICE_TABLE, ModelPrice
from .prompts import PromptLibrary
from .providers import LLMFactory, ProviderSpec
from .tokens import TokenTracker, TokenUsage, extract_usage

__version__ = "0.1.0"

__all__ = [
    # Agent
    "BaseAgent",
    # Providers
    "LLMFactory",
    "ProviderSpec",
    # Tokens
    "TokenTracker",
    "TokenUsage",
    "extract_usage",
    # Cost
    "CostCalculator",
    "ModelPrice",
    "DEFAULT_PRICE_TABLE",
    # Prompts
    "PromptLibrary",
    # Messages / responses
    "build_messages",
    "extract_response_text",
    "extract_json_from_response",
    "round_floats",
    "strip_think_tags",
    # Callbacks
    "TokenCostCallbackHandler",
    # Config / env
    "ensure_env_loaded",
    "get_api_key",
    "require_api_key",
    # Logging
    "get_logger",
    # Exceptions
    "LabCDAgentsError",
    "UnknownProviderError",
    "LLMInvocationError",
    "PromptNotFoundError",
]
