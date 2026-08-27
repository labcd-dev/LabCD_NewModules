from labcd_agents import extract_json_from_response

from . import llm_factory


def _extract_json_payload(raw_text):
    try:
        return extract_json_from_response(raw_text or ""), None
    except Exception as e:
        return None, e


def _empty_usage():
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "cached_input_tokens": 0}


def _extract_usage(message_or_response):
    # cached_input_tokens is a SUBSET of input_tokens (cache_read), not extra on top.
    # Matters here since we resend the same big system prompt every round, and those tokens bill cheaper.
    usage = getattr(message_or_response, "usage_metadata", None) or {}
    details = usage.get("input_token_details") or {}
    return {
        "input_tokens": usage.get("input_tokens", 0) or 0,
        "output_tokens": usage.get("output_tokens", 0) or 0,
        "total_tokens": usage.get("total_tokens", 0) or 0,
        "cached_input_tokens": details.get("cache_read", 0) or 0,
    }


def resolved_models():
    return llm_factory.resolve_models()


def _sum_usage(*usages):
    total = _empty_usage()
    for u in usages:
        for k in total:
            total[k] += u.get(k, 0) or 0
    return total


def _sum_usage_from_messages(messages):
    return _sum_usage(*[_extract_usage(m) for m in messages])


def _per_turn_usage(messages):
    turns = []
    for m in messages:
        usage = getattr(m, "usage_metadata", None)
        if not usage:
            continue
        tool_calls = getattr(m, "tool_calls", None) or []
        kind = ("tool_call: " + ", ".join(tc.get("name", "?") for tc in tool_calls)
                if tool_calls else "final summary")
        turns.append({
            "turn": len(turns) + 1,
            "kind": kind,
            "input_tokens": usage.get("input_tokens", 0) or 0,
            "output_tokens": usage.get("output_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
            "cached_input_tokens": (usage.get("input_token_details") or {}).get("cache_read", 0) or 0,
        })
    return turns


class _SyntheticMessage:
    def __init__(self, content):
        self.content = content
