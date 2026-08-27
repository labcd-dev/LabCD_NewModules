"""Response-parsing utilities shared by every module's agent layer.

Every module hand-rolls a near-identical ``extract_json_from_response``:
strip ``<think>...</think>`` reasoning traces, pull JSON out of a markdown
fence if present, otherwise fall back to a brace-matched substring, then
``json.loads`` it. This module consolidates that logic (plus the
``round_floats`` helper duplicated alongside it) into one tested
implementation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

__all__ = ["strip_think_tags", "extract_json_from_response", "round_floats"]

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_MARKDOWN_JSON_RE = re.compile(r"```json\s*\n?(.*?)\n?```", re.DOTALL)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks some models emit."""
    return _THINK_TAG_RE.sub("", text)


def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """Extract a JSON object from a raw LLM response string.

    Handles, in order of preference:

    1. Plain JSON: ``{"key": "value"}``
    2. Markdown-fenced JSON: ```` ```json\\n{"key": "value"}\\n``` ````
    3. A response containing ``<think>...</think>`` reasoning followed by
       either of the above.
    4. As a last resort, the largest ``{...}`` substring found anywhere in
       the text.

    Raises:
        json.JSONDecodeError: if no valid JSON could be parsed out of the
            response (including when ``response_text`` is empty).
    """
    if not response_text:
        raise json.JSONDecodeError("Empty response", "", 0)

    cleaned = strip_think_tags(response_text)

    match = _MARKDOWN_JSON_RE.search(cleaned)
    if match:
        json_text = match.group(1).strip()
    else:
        brace_match = _BRACE_RE.search(cleaned)
        json_text = brace_match.group(0).strip() if brace_match else cleaned.strip()

    return json.loads(json_text)


def round_floats(obj: Any, decimals: int = 4) -> Any:
    """Recursively round every float in a nested dict/list structure."""
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(item, decimals) for item in obj]
    return obj
