"""
================================================================================
frontend_streamlit/assistant_ui.py
================================================================================
The Diagnostics assistant's presence on every screen -- shared by both AgentMPC
frontends so the two behave identically.

The assistant has three faces:

  1. a TOAST, the moment a new problem is found, so it is noticed even when the
     place it applies to is scrolled out of view;
  2. an INLINE CARD at the exact place the problem applies -- next to the dt
     field, under the reference preview -- with what is wrong and what to do;
  3. an "Ask the assistant" DIALOG on every card, which brings in the LLM to
     explain that specific finding in the context of this plant and this run.

Findings come from deterministic checks (agents/mpc_suitability.py,
agents/dynamics_validator.py) so they are instant, free and fire on every
relevant widget change. The LLM is only spent when the user asks.

Every finding raised anywhere is also collected into one session INBOX, which
the Diagnostics tab lists -- so nothing raised on an earlier screen is lost.

Only native Streamlit components are used, so each app's own theme styles
them; the two frontends differ in look, not in capability.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

_ICON = {"error": ":material/error:", "warning": ":material/warning:", "info": ":material/info:"}
_BOX = {"error": "error", "warning": "warning", "info": "info"}
_VERDICT = {
    "suitable": ("success", "Suitable for MPC"),
    "caveats": ("warning", "Suitable, with caveats"),
    "not_suitable": ("error", "Not suitable as configured"),
}


def _state():
    st.session_state.setdefault("assistant_inbox", [])
    st.session_state.setdefault("assistant_toasted", [])
    st.session_state.setdefault("assistant_chats", {})
    return st.session_state


def _signature(finding: Dict[str, Any]) -> str:
    return f"{finding.get('where')}|{finding.get('code')}|{finding.get('message')}"


def record(findings: List[Dict[str, Any]]) -> None:
    """Add findings to the session inbox, once each."""
    ss = _state()
    known = {f["_sig"] for f in ss.assistant_inbox}
    for f in findings:
        sig = _signature(f)
        if sig not in known:
            ss.assistant_inbox.append({**f, "_sig": sig, "_at": time.strftime("%H:%M:%S")})
            known.add(sig)


def surface(findings: List[Dict[str, Any]], *, key: str, ask: Optional[Callable] = None,
            show_info: bool = False, toast: bool = True, inline: bool = True) -> None:
    """Raise findings AT the place they apply.

    ``key`` makes widget keys unique per call site. ``ask`` is the frontend's
    dialog opener (see ``assistant_dialog`` below); pass None to omit the
    button, e.g. where the LLM is not configured.

    A given finding toasts only once per session: widgets re-run the script
    constantly, and a toast on every rerun would turn the assistant into noise.
    """
    ss = _state()
    record(findings)
    for i, f in enumerate(findings):
        sig = _signature(f)
        sev = f.get("severity", "info")
        if toast and sev in ("error", "warning") and sig not in ss.assistant_toasted:
            st.toast(f"**{f['title']}**", icon=_ICON[sev])
            ss.assistant_toasted.append(sig)
        if not inline or (sev == "info" and not show_info):
            continue
        body = f"**{f['title']}** — {f['message']}"
        if f.get("suggestion"):
            body += f"\n\n→ {f['suggestion']}"
        getattr(st, _BOX[sev])(body, icon=_ICON[sev])
        if ask is not None and sev in ("error", "warning"):
            if st.button("Ask the assistant about this", key=f"assist_{key}_{i}_{f.get('code')}",
                         icon=":material/forum:"):
                st.session_state.assistant_focus = f
                ask()


def verdict_card(assessment: Dict[str, Any]) -> None:
    """The headline answer to "can MPC control this plant?"."""
    kind, label = _VERDICT.get(assessment.get("verdict"), ("info", "Not assessed"))
    counts = {s: sum(1 for f in assessment.get("findings", []) if f["severity"] == s)
              for s in ("error", "warning")}
    detail = []
    if counts["error"]:
        detail.append(f"{counts['error']} blocking problem(s)")
    if counts["warning"]:
        detail.append(f"{counts['warning']} caveat(s)")
    getattr(st, kind)(f"**{label}**" + (f" — {', '.join(detail)}" if detail else ""),
                      icon=_ICON.get(kind, ":material/check_circle:") if kind != "success"
                      else ":material/check_circle:")


def render_inbox(ask: Optional[Callable] = None, key: str = "inbox") -> None:
    """Everything the assistant raised this session, most severe first."""
    ss = _state()
    inbox = ss.assistant_inbox
    if not inbox:
        st.caption("Nothing raised so far this session.")
        return
    order = {"error": 0, "warning": 1, "info": 2}
    for i, f in enumerate(sorted(inbox, key=lambda f: (order.get(f["severity"], 9), f["_at"]))):
        with st.expander(f"{f['_at']} · {f['where']} · {f['title']}",
                         icon=_ICON.get(f["severity"])):
            st.write(f["message"])
            if f.get("suggestion"):
                st.markdown(f"→ {f['suggestion']}")
            if ask is not None and f["severity"] != "info":
                if st.button("Ask the assistant", key=f"{key}_{i}", icon=":material/forum:"):
                    st.session_state.assistant_focus = f
                    ask()
    if st.button("Clear the inbox", key=f"{key}_clear"):
        ss.assistant_inbox = []
        ss.assistant_toasted = []
        st.rerun()


_RULE_BADGE = {"error": ":red[error]", "warning": ":orange[warning]", "info": ":blue[info]",
               "basis": ":gray[definition]"}


def render_rules(groups: List[Dict[str, Any]], key: str = "rules") -> None:
    """Every rule the assistant applies, as formulas -- collapsed until clicked.

    One tab per group; inside each, every rule with the question it answers,
    its mathematics, the decision it leads to, and the strongest severity it
    can raise. The numbers come from the same constants the checks use, so
    this page cannot describe a rule the code no longer follows.
    """
    with st.expander("How the assistant decides \u2014 every rule, as formulas", expanded=False,
                     icon=":material/function:"):
        tabs = st.tabs([g["group"] for g in groups])
        for tab, group in zip(tabs, groups):
            with tab:
                st.caption(group["intro"])
                for rule in group["rules"]:
                    st.markdown(f"**{rule['title']}** \u00b7 {_RULE_BADGE.get(rule['severity'], '')}")
                    st.caption(rule["question"])
                    for expr in rule["latex"]:
                        st.latex(expr)
                    st.markdown(f"\u2192 {rule['decision']}")
                    st.divider()


def dialog_body(ask_llm: Callable[[str, Dict[str, Any], list], str], llm_ready: bool) -> None:
    """Body of the assistant dialog. Each frontend wraps this in its own
    @st.dialog so the decorator binds in that app's script run.

    ``ask_llm(question, finding, history) -> reply`` is supplied by the
    frontend, because the run context the LLM needs is different state in
    each app.
    """
    ss = _state()
    f = ss.get("assistant_focus")
    if not f:
        st.write("Pick a finding to ask about.")
        return
    getattr(st, _BOX.get(f["severity"], "info"))(
        f"**{f['title']}** — {f['message']}", icon=_ICON.get(f["severity"]))
    if f.get("suggestion"):
        st.markdown(f"**Suggested fix:** {f['suggestion']}")

    history = ss.assistant_chats.setdefault(_signature(f), [])
    for turn in history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

    if not llm_ready:
        st.info("The LLM is not configured, so only the deterministic finding above is available.")
        return

    with st.form(f"assistant_form_{abs(hash(_signature(f)))}", clear_on_submit=True):
        question = st.text_area(
            "Ask about this", height=90,
            placeholder="e.g. \"why can't it hold the pole still?\" or \"what dt should I use?\"")
        sent = st.form_submit_button("Ask", icon=":material/send:")
    if sent and question.strip():
        with st.spinner("Thinking..."):
            try:
                reply = ask_llm(question.strip(), f, list(history))
            except Exception as exc:  # noqa: BLE001
                reply = f"Couldn't get a reply: {exc}"
        history.append({"role": "user", "content": question.strip()})
        history.append({"role": "assistant", "content": reply})
        st.rerun(scope="fragment")


def finding_context(f: Dict[str, Any]) -> str:
    """The finding as prompt text, ahead of the frontend's own run context."""
    lines = [f"FINDING [{f.get('severity')}] ({f.get('where')}): {f.get('title')}",
             f"What was found: {f.get('message')}"]
    if f.get("suggestion"):
        lines.append(f"Suggested fix already shown to the user: {f['suggestion']}")
    if f.get("data"):
        lines.append(f"Numbers behind it: {f['data']}")
    return "\n".join(lines)


def as_issue(f: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """A finding in the category shape the Diagnostics Agent's chat expects."""
    return {f.get("code", "finding"): {"count": 1, "examples": [f.get("message", "")],
                                         "iterations": []}}
