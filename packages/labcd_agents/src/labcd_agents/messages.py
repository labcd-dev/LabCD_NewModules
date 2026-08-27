"""Conversation / message construction helpers.

Consolidates the ``[SystemMessage(...) or HumanMessage(...)] + context_messages``
pattern duplicated in ``MuloDesigner.agents.Agents._call_llm``,
``Regularizer.agents.Agents._call_llm`` and ``Recommender.agents.agents.Agents._call_llm``,
as well as the plain-dict ``[{"role": "system", ...}, {"role": "user", ...}]``
pattern used by the ``LLMBaseAgent.invoke_llm`` implementations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

__all__ = ["build_messages", "extract_response_text"]

# A "message" here is either a LangChain message object or a plain
# {"role": ..., "content": ...} dict, matching the two conventions used
# across the existing agent modules.
MessageLike = Any


def build_messages(
    user_prompt: str,
    *,
    system_prompt: Optional[str] = None,
    context_messages: Optional[Sequence[MessageLike]] = None,
    as_dicts: bool = False,
) -> List[MessageLike]:
    """Build a message list for a single chat turn.

    Args:
        user_prompt: the human/user message content.
        system_prompt: optional system instruction, prepended first.
        context_messages: optional prior conversation turns to append after
            the new system/user messages (matching the ``_call_llm(...,
            context_messages=...)`` convention used across modules).
        as_dicts: if True, return plain ``{"role": ..., "content": ...}``
            dicts (what ``invoke_llm``-style methods pass straight to
            ``llm.invoke(...)``). If False (default), return LangChain
            ``SystemMessage`` / ``HumanMessage`` objects.

    Returns:
        A list of messages ready to pass to ``llm.invoke(...)``.
    """
    messages: List[MessageLike] = []

    if as_dicts:
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
    else:
        from langchain_core.messages import HumanMessage, SystemMessage

        if system_prompt is not None:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))

    if context_messages:
        messages.extend(context_messages)

    return messages


def extract_response_text(response: Any) -> str:
    """Pull plain text out of an LLM response, regardless of its shape.

    Handles LangChain ``AIMessage``-like objects (``.content``), raw OpenAI
    Responses API results (``.output_text`` or walking ``.output``), and
    falls back to ``str(response)``.

    Mirrors ``Recommender.agents.agents.Agents._extract_responses_text`` and
    the ``isinstance(response, AIMessage)`` branch in every ``invoke_llm``.
    """
    # LangChain AIMessage / any object exposing `.content` as a string.
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content

    # OpenAI Responses API convenience property.
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    # OpenAI Responses API: walk `.output` items for message/text content.
    output_items = getattr(response, "output", None)
    if output_items:
        chunks: List[str] = []
        for item in output_items:
            if getattr(item, "type", None) != "message":
                continue
            for part in getattr(item, "content", None) or []:
                text = getattr(part, "text", None)
                if text is None and isinstance(part, dict):
                    text = part.get("text")
                if text:
                    chunks.append(str(text))
        if chunks:
            return "".join(chunks)

    if isinstance(response, dict) and "content" in response:
        return str(response["content"])

    return str(response)
