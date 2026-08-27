"""PlantModelAgent: LabCD's plant-model chatbot.

Driven by ``promptTemplate.yaml``. The model always returns one of
three JSON shapes:

- ``{"status": "continue", "reply": "..."}``
- ``{"status": "draft", "reply": "...", "system_name": "...", "python_code": "...", "metadata": {...}}``
- ``{"status": "complete", "system_name": "...", "python_code": "...", "metadata": {...}}``

Product rule: once the plant is known, the agent sketches draft dynamics
code and iterates with the user until they accept it (or a max-draft limit
is hit). No hardcoded user-facing strings are ever returned by this agent —
every message shown to the user comes from the model (or is the raw model
text if JSON parsing fails after a repair attempt).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from labcd_agents import BaseAgent, PromptLibrary, extract_json_from_response

PROMPTS_DIR = Path(__file__).parent
PROMPT_FILE_NAME = "promptTemplate"

REQUIRED_CODE_KEYS = ("system_name", "python_code", "metadata")
REQUIRED_METADATA_KEYS = (
    "states", "state_meanings", "inputs", "outputs",
    "state_equations", "parameters", "system_type", "assumptions",
)

# After this many draft turns in one conversation, accept the latest draft
# as complete even without an explicit "finish" from the user.
DEFAULT_MAX_DRAFTS = 5

# Soft floor: do not accept complete before this many user turns unless the
# user message itself is an explicit accept/finish after a draft was shown.
DEFAULT_MIN_USER_TURNS_BEFORE_COMPLETION = 2

_REPAIR_NOTE = (
    "\n\nIMPORTANT — internal note: your previous reply was not valid JSON in one of "
    'the required shapes (continue / draft / complete). Reply again with ONLY a single '
    'JSON object. Prefer status "draft" if the system is already known from the history; '
    'otherwise status "continue".'
)

_FORCE_DRAFT_NOTE = (
    "\n\nIMPORTANT — internal note: do not emit status \"complete\" yet. The conversation "
    "is not finished. If you already know the plant, emit status \"draft\" with a real "
    "python_code body and a short reply asking the user what to change or whether to "
    'finish. Otherwise emit status "continue" with one concrete question. Output ONLY JSON.'
)

_DIGIT_RE = re.compile(r"\d")
_FINISH_RE = re.compile(
    r"\b(finish|done|ship\s*it|looks?\s+good|totally\s+good|accept|finalize|finalise|"
    r"good\s+to\s+go|that'?s\s+(fine|good|ok|okay)|perfect|approved)\b",
    re.IGNORECASE,
)

# Common Greek / unit glyphs the model likes to emit inside python_code.
# Mapped to plain ASCII so dynamics.py stays executable and editor-safe.
_ASCII_REPLACEMENTS = (
    ("θ", "theta"),
    ("Θ", "Theta"),
    ("ω", "omega"),
    ("Ω", "Omega"),
    ("τ", "tau"),
    ("α", "alpha"),
    ("β", "beta"),
    ("γ", "gamma"),
    ("δ", "delta"),
    ("Δ", "Delta"),
    ("φ", "phi"),
    ("Φ", "Phi"),
    ("ψ", "psi"),
    ("Ψ", "Psi"),
    ("ρ", "rho"),
    ("σ", "sigma"),
    ("Σ", "Sigma"),
    ("μ", "mu"),
    ("λ", "lambda"),
    ("π", "pi"),
    ("·", "*"),
    ("×", "*"),
    ("²", "^2"),
    ("³", "^3"),
    ("¹", "^1"),
    ("⁰", "^0"),
    ("⁻", "-"),
    ("≈", "~="),
    ("≤", "<="),
    ("≥", ">="),
    ("≠", "!="),
    ("→", "->"),
    ("←", "<-"),
    ("°", " deg"),
    ("\u00a0", " "),  # non-breaking space
)


def _to_ascii(text: str) -> str:
    """Replace common non-ASCII science glyphs, then strip any remaining non-ASCII."""
    if not text:
        return text
    out = text
    for src, dst in _ASCII_REPLACEMENTS:
        out = out.replace(src, dst)
    # Drop anything still outside printable ASCII (keep tab/newline).
    return "".join(ch if (32 <= ord(ch) <= 126) or ch in "\n\r\t" else "?" for ch in out)


class PlantModelAgent(BaseAgent):
    """Plant-model chatbot with continue / draft / complete structured turns.

    Args:
        max_drafts: after this many draft responses in one conversation,
            the latest draft is auto-accepted as complete.
        min_user_turns_before_completion: soft floor before accepting
            complete (bypassed when the user explicitly accepts a draft).
    """

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.2,
        provider: Optional[str] = None,
        prompts_dir: Path = PROMPTS_DIR,
        max_drafts: int = DEFAULT_MAX_DRAFTS,
        min_user_turns_before_completion: int = DEFAULT_MIN_USER_TURNS_BEFORE_COMPLETION,
        **base_agent_kwargs: Any,
    ) -> None:
        super().__init__(model=model, temperature=temperature, provider=provider, **base_agent_kwargs)
        self.prompt_library = PromptLibrary(str(prompts_dir))
        self._system_prompt: str = self.prompt_library.get_key(PROMPT_FILE_NAME, "system_prompt")
        self._user_prompt_template: str = self.prompt_library.get_key(PROMPT_FILE_NAME, "user_prompt_template")
        self.max_drafts = max(1, max_drafts)
        self.min_user_turns_before_completion = max(1, min_user_turns_before_completion)
        self._draft_count = 0
        self._latest_draft: Optional[Dict[str, Any]] = None

    def reset_conversation_state(self) -> None:
        """Clear draft counter / latest draft (call on UI reset)."""
        self._draft_count = 0
        self._latest_draft = None

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------
    @staticmethod
    def format_history(messages: Iterable[Dict[str, str]]) -> str:
        lines = []
        for message in messages:
            role = "User" if message.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {message.get('content', '')}")
        return "\n".join(lines)

    def render_user_prompt(self, history_text: str, user_message: str) -> str:
        return (
            self._user_prompt_template
            .replace("{{conversation_history}}", history_text or "(no previous turns)")
            .replace("{{user_message}}", user_message)
        )

    # ------------------------------------------------------------------
    # Turn execution
    # ------------------------------------------------------------------
    def step(
        self,
        history_messages: Iterable[Dict[str, str]],
        user_message: str,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Run one turn. Returns ``(display_text, final_result_or_None)``.

        ``display_text`` is always model-authored (reply field, or a rendered
        draft summary built only from model fields). Never a fixed template
        string invented by this class.
        """
        history_messages = list(history_messages)
        history_text = self.format_history(history_messages)
        user_prompt = self.render_user_prompt(history_text, user_message)
        user_turns = self._count_user_turns(history_messages) + 1
        user_accepts = bool(_FINISH_RE.search(user_message))

        response_text, _ = self.invoke_llm(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
        )
        parsed = self._parse_structured_response(response_text)

        # Repair invalid JSON once.
        if parsed is None:
            response_text, _ = self.invoke_llm(
                system_prompt=self._system_prompt + _REPAIR_NOTE,
                user_prompt=user_prompt,
            )
            parsed = self._parse_structured_response(response_text)
            if parsed is None:
                # Last resort: show the model's raw text, never a canned line.
                return response_text.strip() or "(empty model response)", None

        # User explicitly accepts after we already have a draft → complete.
        if user_accepts and self._latest_draft is not None and parsed.get("status") != "complete":
            # Prefer model's complete if it just emitted one; else promote latest draft.
            if parsed.get("status") == "complete" and self._has_code(parsed):
                return self._accept_complete(parsed)
            return self._accept_complete(self._latest_draft)

        status = parsed.get("status")

        if status == "continue":
            reply = _to_ascii((parsed.get("reply") or "").strip())
            if reply:
                return reply, None
            # Model sent continue without reply — one forced repair, then raw.
            response_text, _ = self.invoke_llm(
                system_prompt=self._system_prompt + _FORCE_DRAFT_NOTE,
                user_prompt=user_prompt,
            )
            parsed = self._parse_structured_response(response_text)
            if parsed is None:
                return response_text.strip() or "(empty model response)", None
            status = parsed.get("status")
            if status == "continue":
                reply = (parsed.get("reply") or "").strip()
                return reply or response_text.strip(), None
            # fall through if repair produced draft/complete

        if status == "draft" and self._has_code(parsed):
            parsed = self._sanitize_code_fields(parsed)
            self._draft_count += 1
            self._latest_draft = {
                "system_name": parsed["system_name"],
                "python_code": parsed["python_code"],
            }
            if "metadata" in parsed and isinstance(parsed["metadata"], dict):
                self._latest_draft["metadata"] = parsed["metadata"]
            display = self._format_draft_display(parsed)
            if self._draft_count >= self.max_drafts:
                return display, dict(self._latest_draft)
            return display, None

        if status == "complete" and self._has_code(parsed):
            # Accept if user is finishing, or min turns satisfied, or we already drafted.
            if (
                user_accepts
                or self._latest_draft is not None
                or user_turns >= self.min_user_turns_before_completion
            ):
                return self._accept_complete(parsed)
            # Premature complete: force a draft instead of inventing text.
            response_text, _ = self.invoke_llm(
                system_prompt=self._system_prompt + _FORCE_DRAFT_NOTE,
                user_prompt=(
                    f"{user_prompt}\n\n(Your previous reply claimed status complete too "
                    f"early. Rejected payload:\n{response_text}\n)"
                ),
            )
            parsed = self._parse_structured_response(response_text)
            if parsed is None:
                return response_text.strip() or "(empty model response)", None
            if parsed.get("status") == "draft" and self._has_code(parsed):
                parsed = self._sanitize_code_fields(parsed)
                self._draft_count += 1
                self._latest_draft = {
                    "system_name": parsed["system_name"],
                    "python_code": parsed["python_code"],
                }
                if "metadata" in parsed and isinstance(parsed["metadata"], dict):
                    self._latest_draft["metadata"] = parsed["metadata"]
                return self._format_draft_display(parsed), None
            if parsed.get("status") == "continue":
                reply = (parsed.get("reply") or "").strip()
                return reply or response_text.strip(), None
            if parsed.get("status") == "complete" and self._has_code(parsed):
                # Model insists — accept rather than loop forever.
                return self._accept_complete(parsed)
            return response_text.strip(), None

        # Unknown shape: show model text only.
        reply = (parsed.get("reply") or "").strip()
        if reply:
            return reply, None
        return response_text.strip() or "(empty model response)", None

    def _accept_complete(self, payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        payload = self._sanitize_code_fields(payload)
        final = {
            "system_name": payload["system_name"],
            "python_code": payload["python_code"],
        }
        if "metadata" in payload and isinstance(payload["metadata"], dict):
            final["metadata"] = payload["metadata"]
        self._latest_draft = final
        # Display text still comes from structured fields only (system_name).
        return f"Model ready — **{final['system_name']}**.", final

    @staticmethod
    def _sanitize_code_fields(data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure system_name, python_code, and metadata strings are pure ASCII."""
        out = dict(data)
        if isinstance(out.get("system_name"), str):
            out["system_name"] = _to_ascii(out["system_name"])
        if isinstance(out.get("python_code"), str):
            out["python_code"] = _to_ascii(out["python_code"])
        if isinstance(out.get("reply"), str):
            out["reply"] = _to_ascii(out["reply"])
        meta = out.get("metadata")
        if isinstance(meta, dict):
            out["metadata"] = PlantModelAgent._sanitize_metadata(meta)
        return out

    @staticmethod
    def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively ASCII-sanitize string fields inside metadata."""
        cleaned: Dict[str, Any] = {}
        for key, value in meta.items():
            if isinstance(value, str):
                cleaned[key] = _to_ascii(value)
            elif isinstance(value, list):
                cleaned[key] = [
                    _to_ascii(v) if isinstance(v, str) else v for v in value
                ]
            elif isinstance(value, dict):
                cleaned[key] = {
                    (_to_ascii(k) if isinstance(k, str) else k): (
                        _to_ascii(v) if isinstance(v, str) else v
                    )
                    for k, v in value.items()
                }
            else:
                cleaned[key] = value
        return cleaned

    @staticmethod
    def _format_draft_display(parsed: Dict[str, Any]) -> str:
        """Build the chat message for a draft turn from model fields only."""
        reply = (parsed.get("reply") or "").strip()
        name = parsed.get("system_name", "draft")
        code = parsed.get("python_code", "")
        parts = []
        if reply:
            parts.append(reply)
        parts.append(f"**Draft: {name}**")
        parts.append(f"```python\n{code}\n```")
        meta = parsed.get("metadata")
        if isinstance(meta, dict):
            states = meta.get("states") or []
            outputs = meta.get("outputs") or []
            system_type = meta.get("system_type") or "?"
            parts.append(
                f"_Metadata: {len(states)} states, outputs={outputs}, type={system_type}_"
            )
        parts.append("_Say what to change, or **finish** to accept this draft._")
        return "\n\n".join(parts)

    @staticmethod
    def _has_code(data: Dict[str, Any]) -> bool:
        """True if system_name + python_code present; metadata preferred but legacy OK."""
        if not all(
            isinstance(data.get(k), str) and data[k].strip()
            for k in ("system_name", "python_code")
        ):
            return False
        meta = data.get("metadata")
        if meta is None:
            # Legacy fallback: accept without metadata (warn downstream).
            return True
        if not isinstance(meta, dict):
            return False
        for key in REQUIRED_METADATA_KEYS:
            if key not in meta:
                return False
        return True

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_structured_response(text: str) -> Optional[Dict[str, Any]]:
        try:
            data = extract_json_from_response(text)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(data, dict):
            return None
        status = data.get("status")
        if status in ("continue", "draft", "complete"):
            return data
        # Legacy: bare final object without status.
        if all(k in data for k in REQUIRED_CODE_KEYS) and "reply" not in data:
            out = dict(data)
            out["status"] = "complete"
            return out
        return data

    @staticmethod
    def _count_user_turns(history_messages: List[Dict[str, str]]) -> int:
        return sum(1 for m in history_messages if m.get("role") == "user")