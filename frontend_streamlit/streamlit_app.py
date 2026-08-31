import os
import sys
import io
import re
import json
import time
import threading
import contextlib

import matplotlib
# has to happen before pyplot gets imported or it tries to open a GUI backend and crashes headless
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st

# Repo root (parent of backend_core/ and frontend_streamlit/) must be on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

# Load repo-root .env before importing agents (ensure_env_loaded is one-shot).
try:
    from labcd_agents import ensure_env_loaded
    _env = os.path.join(_REPO_ROOT, ".env")
    ensure_env_loaded(_env if os.path.isfile(_env) else None)
except ImportError:
    pass

from backend_core.AgentAdaptive import agents
from backend_core.AgentAdaptive.agents import clarifier
from backend_core.AgentAdaptive.tools import system_spec
from backend_core.AgentAdaptive.tools import system_complexity
from backend_core.AgentAdaptive.tools import model_pricing
from backend_core.AgentAdaptive.tools import tuning_objectives
from backend_core.AgentAdaptive.tools import pdf_report
from backend_core.AgentAdaptive.tools import scoring as _scoring_mod

# matplotlib figures are one global registry, not per-thread - two users clicking
# at once would stomp on each other's plots. this lock makes runs go one at a time.
_PLOT_LOCK = threading.Lock()

# streamlit auto-closes all figures on every rerun, which was eating our worker
# thread's plots mid-run. only the thread holding the lock gets to close stuff.
_plot_lock_owner = [None]
_real_plt_close = plt.close


def _guarded_plt_close(*args, **kwargs):
    owner = _plot_lock_owner[0]
    if owner is not None and threading.get_ident() != owner:
        return None
    return _real_plt_close(*args, **kwargs)


plt.close = _guarded_plt_close


def normalize_latex_delimiters(text: str) -> str:

    if not text:
        return text
    text = re.sub(r"\\\[(.*?)\\\]", lambda m: "$$" + m.group(1) + "$$", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.*?)\\\)", lambda m: "$" + m.group(1) + "$", text, flags=re.DOTALL)
    return text


_ALIGN_ENV_RE = re.compile(
    r"\$?\$?\s*\\begin\{(aligned|align\*?|gather\*?)\}(.*?)\\end\{\1\}\s*\$?\$?",
    re.DOTALL,
)


def sanitize_latex_environments(text: str) -> str:
    if not text:
        return text

    def _fix(m):
        body = m.group(2)
        lines = re.split(r"\\\\(?:\[\d+pt\])?", body)
        out = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            line = line.replace("&", "")
            for cmd in (r"\quad", r"\qquad", r"\,", r"\!", r"\bigl", r"\bigr", r"\Bigl", r"\Bigr"):
                line = line.replace(cmd, "")
            out.append("$$" + line.strip() + "$$")
        return "\n\n".join(out)

    return _ALIGN_ENV_RE.sub(_fix, text)


_STABILITY_HEADING_RE = re.compile(
    r"^(?P<hashes>#{1,6})[ \t]*(?P<title>Stability\b[^\n]*)$", re.MULTILINE)


def split_stability_section(text):
    # stop at the next same-or-shallower heading - a deeper subheading is still
    # part of the proof, don't cut there.
    if not text:
        return None
    m = _STABILITY_HEADING_RE.search(text)
    if m is None:
        return None
    level = len(m.group("hashes"))
    tail = text[m.end():]
    end = len(text)
    for nxt in re.finditer(r"^(#{1,6})[ \t]*\S", tail, re.MULTILINE):
        if len(nxt.group(1)) <= level:
            end = m.end() + nxt.start()
            break
    return (text[:m.start()], m.group("title").strip(),
            text[m.end():end], text[end:])


def render_summary_with_stability_expander(text):
    parts = split_stability_section(text)
    if parts is None:
        st.markdown(text)
        return
    before, title, body, after = parts
    if before.strip():
        st.markdown(before)
    with st.expander(title, expanded=False):
        if body.strip():
            st.markdown(body)
        else:
            st.caption("No proof text was recorded for this run.")
    if after.strip():
        st.markdown(after)


def _final_metrics_from_pipeline(result, tuning_best):
    # prefer final_metrics but it's not always there - fall back to the best
    # tuning round's own metrics, slightly less accurate but it's something.
    if isinstance(result, dict):
        metrics = result.get("final_metrics")
        if isinstance(metrics, dict) and metrics:
            return metrics
    if isinstance(tuning_best, dict):
        metrics = tuning_best.get("metrics")
        if isinstance(metrics, dict) and metrics:
            return metrics
    return None


def _pipeline_worker(description, opts, box):
    # runs on a bg thread so the UI stays responsive - box hands results back
    # to the main thread since a bg thread can't touch st state directly.

    log_buffer = io.StringIO()
    summary = None
    abstract = None
    error = None
    tuning_log = []
    tuning_best = None
    usage = None
    final_metrics = None

    with _PLOT_LOCK:
        # claim ownership so _guarded_plt_close lets OUR close("all") calls through
        # while blocking streamlit's automatic ones for as long as this run lasts
        _plot_lock_owner[0] = threading.get_ident()
        try:
            plt.close("all")
            fignums_before = set(plt.get_fignums())

            try:
                with contextlib.redirect_stdout(log_buffer):
                    result, usage, tuning_log, tuning_best = agents.run_full_pipeline(
                        description, on_event=box["events"].append,
                        should_stop=box["stop_event"].is_set, **opts)
                    summary = result["messages"][-1].content
                    abstract = result.get("abstract")
                    final_metrics = _final_metrics_from_pipeline(result, tuning_best)
            except Exception as e:
                error = "%s: %s" % (type(e).__name__, e)

            figures = []
            for num in plt.get_fignums():
                if num not in fignums_before:
                    fig = plt.figure(num)
                    title = ""
                    if fig._suptitle is not None:
                        title = fig._suptitle.get_text()
                    else:
                        for ax in fig.axes:
                            ax_title = ax.get_title()
                            if ax_title:
                                title = ax_title
                                break
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
                    buf.seek(0)
                    figures.append({"png": buf.getvalue(), "title": title})
            plt.close("all")
        finally:
            _plot_lock_owner[0] = None

    box["result"] = dict(summary=summary, abstract=abstract, log=log_buffer.getvalue(), error=error,
                          figures=figures, usage=usage,
                          tuning_log=tuning_log, tuning_best=tuning_best,
                          cancelled=box["stop_event"].is_set(),
                          final_metrics=final_metrics,
                          clarification_record=opts.get("clarification_record"),
                          system_spec=opts.get("system_spec"),
                          tuning_objectives=opts.get("tuning_objectives"),
                          events=list(box["events"]))
    box["running"] = False


def _start_pipeline(description, opts, seed_events=None):
    # seed_events preloads the trace with what the clarifier chat already logged,
    # so the log doesn't start blank.
    box = {"events": list(seed_events or []), "result": None, "running": True,
           "stop_event": threading.Event()}
    st.session_state.result = None
    st.session_state.pdf_bytes = None
    st.session_state.pipeline_box = box
    threading.Thread(target=_pipeline_worker, args=(description, opts, box), daemon=True).start()
    return box


def _clarify_turn_worker(messages, round_num, box, force_finish=False):
    status, reply, dynamics, usage, error, updated = (
        "error", "", None, clarifier._empty_usage(), "", messages)
    try:
        status, reply, dynamics, usage, error, updated = clarifier.run_clarifier_turn(
            messages, on_event=box["events"].append, round_num=round_num,
            force_finish=force_finish)
    except Exception as e:
        error = "%s: %s" % (type(e).__name__, e)
    box["result"] = {"status": status, "reply": reply, "dynamics": dynamics,
                     "usage": usage, "error": error, "messages": updated}
    box["running"] = False


def _reduce_trace_events(events):
    # folds the flat raw event list into one row per (stage, round), so the UI
    # draws one expander per row instead of one per raw event.
    items, order = {}, []
    for ev in events:
        key = (ev.get("stage"), ev.get("round", 0))
        if key not in items:
            items[key] = {"stage": key[0], "round": key[1], "status": "pending",
                          "start_ts": None, "done_ts": None, "notes": []}
            order.append(key)
        item = items[key]
        kind = ev["kind"]
        extra = {k: v for k, v in ev.items() if k not in ("kind", "ts", "stage", "round")}
        if kind == "stage_start":
            item["status"] = "running"
            item["start_ts"] = ev.get("ts")
            item.update(extra)
        elif kind == "stage_done":
            item["status"] = "done"
            item["done_ts"] = ev.get("ts")
            item.update(extra)
        elif kind == "cancelled":
            item["status"] = "cancelled"
            item["done_ts"] = ev.get("ts")
            item.update(extra)
        elif kind == "note":
            item["notes"].append(ev.get("text", ""))
    return [items[k] for k in order]


def _trace_row_label(item):
    stage, rnd = item["stage"], item["round"]
    elapsed = None
    if item["start_ts"] is not None:
        end = item["done_ts"] if item["done_ts"] is not None else time.time()
        elapsed = max(0, int(end - item["start_ts"]))
    time_suffix = " (%ds)" % elapsed if elapsed else ""

    if stage == "clarify":
        round_suffix = " (round %d)" % rnd if rnd else ""
        if item["status"] == "running":
            return "Clarifier Agent is extracting your system…" + round_suffix + time_suffix
        n = item.get("n_questions")
        name = item.get("system_name") or ""
        if not n:
            return ("✓ Clarifier Agent: system complete%s"
                    % (", %s" % name if name else "")) + round_suffix + time_suffix
        return ("✓ Clarifier Agent: %d question(s) for you" % n) + round_suffix + time_suffix

    if stage == "design":
        if item["status"] == "running":
            return "Extracting the system structure…" + time_suffix
        if item.get("ok"):
            return "✓ Extraction complete" + time_suffix
        return "✗ Extraction failed" + time_suffix

    if stage == "build":
        head = "Running the full simulation" + (" (baseline for tuning)" if item.get("for_tuning") else "")
        if item["status"] == "running":
            return head + "…" + time_suffix
        if item["status"] == "cancelled":
            return head + " (stopped)" + time_suffix
        return "✓ " + head + " complete" + time_suffix

    if stage == "tuning":
        if item.get("skipped"):
            return "✓ Tuning not needed: target already met" + time_suffix
        maxr = item.get("max_rounds")
        head = "Tuning round %d%s" % (rnd, (" of %d" % maxr) if maxr else "")
        if item["status"] == "running":
            return head + "…" + time_suffix
        if item["status"] == "cancelled":
            return head + ", stopped" + time_suffix
        if item.get("tool_called") is False:
            return "✗ " + head + ": proposal failed" + time_suffix
        if item.get("met_target"):
            return "✓ " + head + ", target met" + time_suffix
        return "○ " + head + " (target not yet met)" + time_suffix

    return stage or "…"


def _render_notes(item):
    for n in item.get("notes") or []:
        st.markdown("- " + n)


def _render_clarify_body(item):
    _render_notes(item)
    if item["status"] == "running":
        if not item.get("notes"):
            st.write("Thinking about what to ask next…")
        return
    st.write("This turn is recorded in the chat above and, once the "
             "conversation ends, in 'Clarifications applied' below.")


def _render_design_body(item):
    _render_notes(item)
    if item["status"] == "running":
        if not item.get("notes"):
            st.write("Design Agent is deciding the control method and structuring the "
                     "system (states, inputs/outputs, uncertainty/disturbance)…")
    elif item.get("ok"):
        st.write("System structure extracted. The control law and simulation come next; "
                 "see the 'Running the full simulation' step below for the actual "
                 "result.")
    else:
        st.write("Extraction failed. See the error message in the Results section below "
                 "for details.")


def _as_pct(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN check: it's the only float that never equals itself
        return None
    return f


def _fmt_pct(value):
    f = _as_pct(value)
    return "n/a" if f is None else "%.1f%%" % f


def _score_fields(source):
    if not isinstance(source, dict):
        return None
    if "success" not in source and "tracking_pct_headline" not in source:
        nested = source.get("metrics")
        if isinstance(nested, dict) and ("success" in nested
                                          or "tracking_pct_headline" in nested):
            source = nested
        else:
            return None
    pct = source.get("tracking_pct")
    if not isinstance(pct, dict):
        pct = {}
    return {
        "has_success": "success" in source,
        "success": bool(source.get("success")),
        "reason": source.get("success_reason") or "",
        "checks": source.get("success_checks") or {},
        "target_frac": source.get("success_target_frac"),
        "target_from_tuner": source.get("mse_target_from_tuner"),
        "headline": _as_pct(source.get("tracking_pct_headline")),
        "mean": _as_pct(source.get("tracking_pct_mean")),
        "full": list(pct.get("full") or []),
        "steady": list(pct.get("steady") or []),
        "transient": list(pct.get("transient") or []),
        "trivial": list(source.get("task_trivial") or []),
    }


# read off the scoring module instead of hardcoding so this can't go stale.
# the literal tuple is just a fallback for an old scoring module version.
_HARD_CHECKS = tuple(getattr(_scoring_mod, "_SUCCESS_HARD_CHECKS",
                             ("finite", "bounded", "mse_target")))
_ADVISORY_CHECKS = tuple(getattr(_scoring_mod, "_SUCCESS_ADVISORY_CHECKS", ()))

_CHECK_TEXT = {
    "finite": (
        "Stayed a real number",
        "no NaN or infinite values appeared anywhere in the simulation.",
        "the simulation produced NaN or infinite values: the loop blew up."),
    "bounded": (
        "Stayed within a sensible range",
        "neither the output nor the control effort grew beyond a plausible size "
        "for this task.",
        "the output and/or the control effort grew far beyond any plausible size "
        "for this task."),
    "mse_target": (
        "MSE within target",
        "the steady-state MSE is at or below %s for every output.",
        "the steady-state MSE is still above %s for at least one output. "
        "The controller does not track the reference closely enough."),
}


def _target_frac_str(target_frac):
    return ("%.1f%%" % (100.0 * target_frac)
            if isinstance(target_frac, (int, float)) else "target")


def _target_phrase(target_frac, from_tuner):
    pct = "%s of the reference amplitude" % _target_frac_str(target_frac)
    return pct if from_tuner is False else "your target (%s)" % pct


def _check_rows(checks, target_frac=None, target_from_tuner=None):
    if not isinstance(checks, dict):
        return [], [], []
    groups = {"hard": [], "advisory": [], "other": []}
    seen = set()
    blew_up = checks.get("finite") is False  # checking "is False" here is deliberate, not just falsy.
    # a missing key means "no data", not "it failed"; those are different things
    target_phrase = _target_phrase(target_frac, target_from_tuner)

    def _row(name):
        # if the run went non-finite, everything except "finite" itself is really
        # "we never got to measure this", not a genuine pass or fail
        ok = None if (blew_up and name != "finite") else bool(checks.get(name))
        text = _CHECK_TEXT.get(name)
        if text is None:
            return (name, name, "", ok)
        criterion, on_pass, on_fail = text
        if ok is None:
            return (name, criterion,
                    "not tested: once the run produced non-finite values there "
                    "was nothing left to measure.", None)
        description = on_pass if ok else on_fail
        if name == "mse_target":
            description = description % target_phrase
        return (name, criterion, description, ok)

    for name in _HARD_CHECKS:
        if name in checks:
            groups["hard"].append(_row(name))
            seen.add(name)
    for name in _ADVISORY_CHECKS:
        if name in checks:
            groups["advisory"].append(_row(name))
            seen.add(name)
    for name in checks:
        if name not in seen:
            groups["other"].append(_row(name))
    return groups["hard"], groups["advisory"], groups["other"]


def _plain_reason(reason):
    text = (reason or "").strip()
    for name in set(_CHECK_TEXT) | set(_HARD_CHECKS) | set(_ADVISORY_CHECKS):
        prefix = name + ": "
        if text.startswith(prefix):
            body = text[len(prefix):].strip()
            return (body[:1].upper() + body[1:]) if body else text
    return text


def _check_line(row):
    name, criterion, description, ok = row
    flag = (":gray[**not tested**]" if ok is None
            else ":green[**PASS**]" if ok else ":red[**FAIL**]")
    tail = (": " + description) if description else ""
    span = "" if criterion == name else "  `%s`" % name
    return "%s  **%s**%s%s" % (flag, criterion, tail, span)


def render_run_scores(source, spec=None, show_table=True):
    fields = _score_fields(source)
    if fields is None:
        return False

    complexity = system_complexity.complexity_grade_from_spec(spec)

    headline, mean = fields["headline"], fields["mean"]
    if not fields["has_success"] and headline is None:
        return False

    checks = fields["checks"] if isinstance(fields["checks"], dict) else {}
    hard, advisory, other = _check_rows(checks, fields["target_frac"],
                                        fields["target_from_tuner"])
    blew_up = checks.get("finite") is False
    steady = fields["steady"]
    n_out = max(len(steady), len(fields["full"]), len(fields["transient"]))

    st.subheader("Run Scores")
    st.caption(
        "Whether this run passed depends on three checks: it stayed finite, "
        "it stayed bounded, and its steady-state MSE reached your target.")

    if fields["has_success"]:
        n_failed = sum(1 for row in hard if row[3] is False)
        n_hard = len(hard)
        cols = st.columns(3 if complexity is not None else 2)
        with cols[0]:
            st.metric("MSE score",
                      "not measured" if blew_up else
                      (_fmt_pct(headline) if headline is not None else "n/a"))
        with cols[1]:
            st.metric("Checks passed",
                      "%d / %d" % (n_hard - n_failed, n_hard) if n_hard else "n/a")
        if complexity is not None:
            with cols[2]:
                st.metric("System complexity", "%d / %d" % (complexity, system_complexity.MAX_GRADE),
                          help="just how complex the plant looks, not how good this run was")

        if hard or advisory or other:
            with st.container(border=True):
                for row in hard:
                    st.markdown(_check_line(row))
                if advisory:
                    for row in advisory:
                        st.markdown(_check_line(row))
                if other:
                    st.caption("Additional checks recorded by this run:")
                    for row in other:
                        st.markdown(_check_line(row))

    if show_table and steady and not blew_up:
        expanded = n_out > 1 or (fields["has_success"] and not fields["success"])
        with st.expander("Per-output detail and the three time windows",
                          expanded=expanded):
            st.markdown(
                "The same percentage, measured over three slices of the run:\n\n"
                "- **full run**: every sample, from the first instant to the "
                "last.\n"
                "- **steady state (last 20%)**, after the loop has settled. "
                "This is the window the headline figure above comes from.\n"
                "- **transient (first 20%)** (while the output is still on its "
                "way to the reference).")
            st.caption(
                "Expect the full-run and transient figures to be LOWER than the "
                "steady-state one, and that holds on a good design just as much as on a bad "
                "one. At the start of the run the output physically cannot "
                "already be sitting on the reference, and all three windows are "
                "scored against the same yardstick: the error a controller that "
                "never acted would have made. The very first instant therefore "
                "scores 0% by construction, and the early windows can only climb "
                "away from it. A low transient figure beside a high steady-state "
                "one is the normal shape of a design that works; it is the "
                "steady-state column that says whether it ended up where it was "
                "told to.")

            def _col(values):
                return [_fmt_pct(values[i]) if i < len(values) else "n/a"
                        for i in range(n_out)]

            columns = {
                "output": ["y%d" % (i + 1) for i in range(n_out)],
                "full run": _col(fields["full"]),
                "steady state (last 20%)": _col(steady),
                "transient (first 20%)": _col(fields["transient"]),
            }
            trivial = fields["trivial"]
            if any(trivial):
                columns["task"] = ["stay put" if (i < len(trivial) and trivial[i])
                                   else "tracking" for i in range(n_out)]
            st.table(columns)
            if any(trivial):
                st.caption(
                    "An output marked \"stay put\" was commanded to remain "
                    "exactly where it already was. There is no movement to score "
                    "against, so its percentage comes from a small absolute "
                    "tolerance instead: read a high figure there as \"it did not "
                    "drift\", not as a tracking achievement.")
    return True


def _render_objective_values(source, label="Your tuning priorities this round"):
    rows = source.get("objective_values") if isinstance(source, dict) else None
    if not rows:
        return False
    names, values, befores = [], [], []
    for row in rows:
        if not row or len(row) < 2:
            continue
        names.append(row[0])
        values.append(row[1])
        befores.append(row[2] if len(row) > 2 else None)
    if not names:
        return False
    st.markdown("**%s:**" % label)
    table = {"objective": names, "value": values}
    if any(b for b in befores):
        table["before tuning (round 0)"] = [b or "--" for b in befores]
    st.table(table)
    return True


def _render_round_scores(source):
    fields = _score_fields(source)
    if fields is None:
        return False
    bits = []
    if fields["has_success"]:
        if fields["success"]:
            bits.append("Loop held together: :green[**PASS**]")
        else:
            checks = fields["checks"] if isinstance(fields["checks"], dict) else {}
            detail = ""
            for name in _HARD_CHECKS:
                if checks.get(name) is False and name in _CHECK_TEXT:
                    text = _CHECK_TEXT[name][2]
                    if name == "mse_target":
                        # hardcoded True - this only renders for tuning rounds, which
                        # always use the tuner's own target slider.
                        text = text % _target_phrase(fields["target_frac"], True)
                    detail = ": " + text
                    break
            if not detail and fields["reason"]:
                detail = ": " + _plain_reason(fields["reason"])
            bits.append("Loop held together: :red[**FAIL**]" + detail)
    if fields["headline"] is not None:
        bits.append("reference tracking **%s** (worst output, once settled)"
                    % _fmt_pct(fields["headline"]))
    if not bits:
        return False
    st.markdown(" | ".join(bits))
    return True


def _render_tuning_body(item):
    _render_notes(item)
    if item.get("skipped"):
        if not item.get("notes"):
            st.write(item.get("reasoning") or
                     "Tuning was not needed since the initial design already met the target.")
        return
    if item["status"] == "running":
        current = item.get("current_tuning")
        if current:
            st.caption("Current parameters: " +
                        ", ".join("%s=%s" % (k, v) for k, v in current.items()))
        return
    if item["status"] == "cancelled":
        if not item.get("notes"):
            st.write(item.get("reasoning") or "Stopped before this round could finish.")
        return

    _render_round_scores(item)
    _render_objective_values(item)

    reasoning = item.get("reasoning")
    if reasoning:
        st.markdown("**Tuner Agent's reasoning:** " + reasoning)

    changed = item.get("changed") or {}
    if changed:
        st.markdown("**Parameters changed:**")
        st.table({
            "parameter": list(changed.keys()),
            "old value": [str(old) for old, _new in changed.values()],
            "new value": [str(new) for _old, new in changed.values()],
        })
    else:
        st.caption("No tuning parameters were changed this round.")

    if item.get("tool_called") is False:
        st.error("The tuner's proposal could not be parsed this round.")
    elif item.get("met_target"):
        st.success("Target met.")
    else:
        st.caption("Target not yet met.")


def _render_build_body(item):
    _render_notes(item)
    if item["status"] == "running":
        st.write("Running the full simulation" +
                 (" (this becomes the tuning loop's round-0 baseline)." if item.get("for_tuning") else "."))
        return
    if item["status"] == "cancelled":
        st.write("Stopped before the final build could finish.")
        return
    _render_round_scores(item)
    if item.get("report"):
        st.markdown("**Simulation metrics:**")
        st.code(item["report"], language=None)


def render_trace_rows(events):
    items = _reduce_trace_events(events)
    if not items:
        st.write("Starting Design Agent…")
        return
    for item in items:
        label = _trace_row_label(item)
        key = "trace_%s_%s" % (item["stage"], item["round"])
        with st.expander(label, expanded=False, key=key):
            if item["stage"] == "clarify":
                _render_clarify_body(item)
            elif item["stage"] == "design":
                _render_design_body(item)
            elif item["stage"] == "tuning":
                _render_tuning_body(item)
            elif item["stage"] == "build":
                _render_build_body(item)


@st.fragment(run_every=0.5)
def render_live_pipeline(box):
    if not box["running"]:
        # worker finished between polls - full rerun so the script re-enters at
        # the top and picks up box["result"] below.
        st.rerun()
        return

    render_trace_rows(box["events"])

    stop_requested = box["stop_event"].is_set()
    if stop_requested:
        st.button("Stopping…", disabled=True, key="stop_pipeline_btn")
        st.caption("Stop requested. The process will stop at most at the end of the "
                    "current stage; the design, log, and any plots produced so far will "
                    "be kept.")
    else:
        if st.button("Stop process", type="primary", key="stop_pipeline_btn",
                       use_container_width=False):
            box["stop_event"].set()
            st.rerun(scope="fragment")


def _launch_clarified_pipeline(state):
    # this is the one spot where the clarify chat hands off into the real pipeline.
    # triggered once the model says "complete"; builds opts from the finished spec + chat
    spec = system_spec.normalize_defaults(state.get("spec"))
    opts = dict(state["opts"])
    opts["system_spec"] = spec if spec["dynamics"]["states"] else None
    opts["clarification_record"] = state.get("record") or []
    opts["sim_overrides"] = (clarifier.sim_overrides_from_spec(spec)
                             if opts["system_spec"] else None)
    opts["clarifier_usage"] = state.get("usage")
    st.session_state.clarify = None
    st.session_state.system_spec = opts["system_spec"]
    _start_pipeline(state.get("description", ""), opts, seed_events=state.get("events"))


def _clarification_record_from_log(chat_log):
    # assumes strict assistant/user alternation - pairs each question with the
    # reply right after it for the report's Q&A table.
    record = []
    i, idx, n = 0, 0, len(chat_log)
    while i < n:
        turn = chat_log[i]
        if turn["role"] != "assistant":
            i += 1
            continue
        question = turn["text"]
        answer = chat_log[i + 1]["text"] if i + 1 < n and chat_log[i + 1]["role"] == "user" else ""
        idx += 1
        record.append({
            "id": "uncertainty-%d" % idx, "category": "uncertainty_split",
            "question": question, "answer_label": answer, "answer_value": answer,
            "answered": bool(answer), "default_label": "", "source": "user", "evidence": "",
        })
        i += 2
    return record


def _start_clarify_turn(state, force_finish=False):
    box = {"events": [], "result": None, "running": True}
    round_num = state.get("round", 0) + 1
    if round_num > clarifier.MAX_CLARIFY_TURNS:
        # safety valve so the chat can't go forever. the model still writes
        # the final answer itself; this just tells it to stop asking
        force_finish = True
    new_state = dict(state, mode="running", box=box, round=round_num)
    st.session_state.clarify = new_state
    threading.Thread(target=_clarify_turn_worker,
                     args=(state["messages"], round_num, box, force_finish),
                     daemon=True).start()


@st.fragment(run_every=0.5)
def render_live_clarifier(box):
    if not box["running"]:
        st.rerun()
        return

    render_trace_rows(box["events"])
    st.caption("One model call per turn: it asks whatever it still needs to, and "
                "decides for itself when it has heard enough. Nothing here is a "
                "scripted question.")


def render_clarify_chat(state):
    st.subheader("Uncertainty & disturbance")
    st.caption(
        "A real conversation, not a form. Answer in your own words: the agent "
        "decides what to ask and when it has enough to finish, and nothing here is "
        "a fixed question list.")

    for turn in state.get("chat_log", []):
        with st.chat_message("assistant" if turn["role"] == "assistant" else "user"):
            st.markdown(turn["text"])

    typed = st.chat_input("Your answer...")
    if typed:
        state = dict(state)
        state["chat_log"] = list(state.get("chat_log", [])) + [{"role": "user", "text": typed}]
        state["messages"] = list(state["messages"]) + [{"role": "user", "content": typed}]
        _start_clarify_turn(state)
        st.rerun()

    if st.button("Skip (let it finish with what it has so far)", key="clarify_skip_btn"):
        _start_clarify_turn(state, force_finish=True)
        st.rerun()


st.set_page_config(page_title="Agentic Nonlinear Control Designer", layout="wide")
st.title("Agentic Nonlinear Control Designer")
st.caption("LLM-Driven SMC and Backstepping Controller Design (backend_core.AgentAdaptive.controller & backend_core.AgentAdaptive.agents)")

def _model_selectbox(label, env_var, default_model=None, allow_inherit=False,
                     disabled=False, help=None):
    options = list(model_pricing.AVAILABLE_MODELS)
    current = os.environ.get(env_var, "") or ""
    if current and current not in options:
        # env var already set to something not in our price table (e.g. someone
        # exported a model manually). add it rather than silently overriding it
        options.append(current)
    if allow_inherit:
        options = [model_pricing.INHERIT_LABEL] + options
        index = options.index(current) if current in options else 0
    else:
        if default_model and default_model not in options:
            options.append(default_model)
        target = current or default_model
        index = options.index(target) if target in options else 0
    choice = st.selectbox(
        label, options, index=index, disabled=disabled, help=help,
        format_func=lambda m: (m if m == model_pricing.INHERIT_LABEL
                               else model_pricing.model_label(m)))
    return "" if choice == model_pricing.INHERIT_LABEL else choice


with st.sidebar:
    st.header("Design Agent")
    model_name = _model_selectbox(
        "Model", "OPENAI_MODEL",
        default_model=agents.DEFAULT_OPENAI_MODEL)

    st.divider()
    st.header("Clarifier")
    st.caption(
        "A real conversation about uncertainty and "
        "disturbance, also rewrites reference-signal"
        "text into a valid expression"
        "before handing off.")
    clarifier_model_name = _model_selectbox(
        "Model (Clarifier)", "OPENAI_MODEL_CLARIFIER", allow_inherit=True)

    st.divider()
    st.header("Tuner Agent")
    enable_tuning = st.checkbox(
        "Enable parameter tuning loop", value=False,
        help="turns on a loop that keeps tweaking the numbers to try to hit the target below")
    tuner_model_name = _model_selectbox(
        "Model (Tuner Agent)", "OPENAI_MODEL_TUNER", allow_inherit=True,
        disabled=not enable_tuning,
        help="picks the model for the tuner, it gets called once per round so cheaper = less cost")
    target_rms_frac = st.slider(
        "Target steady-state error (fraction of reference amplitude)",
        min_value=0.005, max_value=0.10, value=0.02, step=0.005, format="%.3f",
        disabled=not enable_tuning,
        help="tuning stops once the error gets this small, or when it runs out of rounds")
    max_tuning_rounds = st.number_input(
        "Max tuning rounds", min_value=1, max_value=10, value=4, step=1,
        disabled=not enable_tuning,
        help="how many tries the tuner gets before giving up")

    # two-stage picker (multiselect first, then one slider per picked item) instead
    # of showing all 10 objectives' sliders at once. sidebar's crowded enough already
    _objective_keys_by_label = {spec["label"]: key
                                for key, spec in tuning_objectives.OBJECTIVES.items()}
    picked_labels = st.multiselect(
        "Tuning priorities (optional)",
        options=list(_objective_keys_by_label.keys()),
        default=[],
        disabled=not enable_tuning,
        help="pick what you care about, or leave empty and let the tuner decide on its own")

    tuning_priorities = {}
    for _label in picked_labels:
        _key = _objective_keys_by_label[_label]
        tuning_priorities[_key] = st.select_slider(
            "%s: importance" % _label,
            options=list(range(tuning_objectives.WEIGHT_MIN,
                               tuning_objectives.WEIGHT_MAX + 1)),
            value=tuning_objectives.DEFAULT_WEIGHT,
            key="tuning_weight_%s" % _key,
            disabled=not enable_tuning,
            help=tuning_objectives.OBJECTIVES[_key]["help"] + " bigger number = more important")

    tuning_priorities = tuning_objectives.normalize_selection(tuning_priorities)
    st.caption(tuning_objectives.selection_summary(tuning_priorities))

    st.caption(
        "This UI calls agents.run_full_pipeline(...).")

if "plant_spec" not in st.session_state:
    st.session_state.plant_spec = None
if "sim_knobs_spec" not in st.session_state:
    st.session_state.sim_knobs_spec = None

# lock the wizard steps' widgets while a chat or run is in progress.
# editing the plant/sim inputs mid-run would desync from what's actually running
_wizard_locked = bool(
    st.session_state.get("clarify") is not None
    or (st.session_state.get("pipeline_box") and st.session_state.pipeline_box["running"]))

_DEFAULT_PLANT_JSON = """{
  "system_name": "Cart-Pole",
  "states": ["x1", "x2"],
  "state_meanings": [
    "pole angle from upright (rad)",
    "pole angular rate (rad/s)"
  ],
  "inputs": ["u"],
  "outputs": ["x1"],
  "state_equations": [
    "x2",
    "(-(u + 0.06*x2**2*sin(x1))*cos(x1) + 11.2815*sin(x1))/(0.06*sin(x1)**2 + 0.4)"
  ],
  "parameters": {},
  "system_type": "SISO",
  "assumptions": []
}"""

st.subheader("1. Plant (from the plant agent)")
if st.session_state.plant_spec is None:
    # --- Artifact store integration (unified LabCD flow) ---
    try:
        from backend_core.artifact_store import ArtifactStore
        _art_store = ArtifactStore(base_dir=os.path.join(_REPO_ROOT, "artifacts"))
        _art_list = _art_store.list_artifacts()
    except Exception:
        _art_store = None
        _art_list = []

    _load_mode = st.radio(
        "Plant source",
        options=["Paste JSON", "Load from artifact"],
        horizontal=True,
        key="plant_source_mode",
        disabled=_wizard_locked,
    )

    if _load_mode == "Load from artifact" and _art_list:
        _labels = ["%s (%s)" % (a["artifact_id"], a.get("system_name", "")) for a in _art_list]
        _ids = [a["artifact_id"] for a in _art_list]
        _sel = st.selectbox("Artifact", options=list(range(len(_ids))),
                            format_func=lambda i: _labels[i], key="adaptive_artifact_sel",
                            disabled=_wizard_locked)
        if st.button("Load artifact into Adaptive", disabled=_wizard_locked):
            try:
                full_spec = _art_store.get_adaptive_spec(_ids[_sel])
                # Plant only — references are owned by Adaptive, so step 2
                # (sim knobs form) must still collect reference expressions.
                st.session_state.plant_spec = system_spec.normalize_plant_spec(full_spec)
                st.session_state.sim_knobs_spec = None
                dyn = (full_spec.get("dynamics") or {})
                st.session_state["artifact_sim_defaults"] = {
                    "sim_time": dyn.get("sim_time"),
                    "solver_step": dyn.get("solver_step"),
                    "x0": list(dyn.get("x0") or []),
                }
                st.session_state["loaded_artifact_id"] = _ids[_sel]
                st.rerun()
            except Exception as e:
                st.error("Failed to load artifact: %s" % e)
    elif _load_mode == "Load from artifact":
        st.info("No artifacts found under artifacts/. Complete Plant + Pre-Launch first.")

    if _load_mode == "Paste JSON":
        plant_json_text = st.text_area(
            "Paste the plant JSON blob (states, inputs, outputs, dynamics, "
            "parameters):", value=_DEFAULT_PLANT_JSON, height=220,
            key="plant_json_raw", disabled=_wizard_locked)
        if st.button("Confirm plant", disabled=_wizard_locked):
            try:
                raw = json.loads(plant_json_text)
            except ValueError as e:
                st.error("Not valid JSON: %s" % e)
            else:
                plant_spec = system_spec.normalize_plant_spec(raw)
                plant_gaps = [g for g in system_spec.missing_items(plant_spec)
                             if g["category"] in ("dynamics", "states_inputs", "output")]
                if plant_gaps:
                    st.error("Plant JSON is incomplete:\n\n"
                             + "\n".join("- %s" % g["detail"] for g in plant_gaps))
                else:
                    st.session_state.plant_spec = plant_spec
                    st.rerun()
else:
    plant = st.session_state.plant_spec["dynamics"]
    _aid = st.session_state.get("loaded_artifact_id")
    _extra = (" (artifact: %s)" % _aid) if _aid else ""
    st.success("Plant confirmed%s. States: %s, inputs: %s, outputs: %s"
              % (_extra, ", ".join(plant["states"]), ", ".join(plant["inputs"]),
                 ", ".join(plant["outputs"])))

# Treat empty/missing references as incomplete so artifact loads (which no
# longer carry pre-launch trajectory) always open the sim-setup form.
_sim_knobs_ready = False
if st.session_state.sim_knobs_spec is not None:
    _sk_dyn = st.session_state.sim_knobs_spec.get("dynamics") or {}
    _sk_outs = _sk_dyn.get("outputs") or []
    _sk_refs = _sk_dyn.get("references") or []
    _sk_ref_outs = {
        r.get("output") for r in _sk_refs
        if isinstance(r, dict) and str(r.get("expr") or "").strip()
    }
    _sim_knobs_ready = bool(_sk_outs) and all(o in _sk_ref_outs for o in _sk_outs)

if st.session_state.plant_spec is not None and not _sim_knobs_ready:
    st.subheader("2. Simulation setup")
    plant = st.session_state.plant_spec["dynamics"]
    _art_defs = st.session_state.get("artifact_sim_defaults") or {}
    _def_sim = _art_defs.get("sim_time")
    _def_dt = _art_defs.get("solver_step")
    _def_x0 = _art_defs.get("x0") or []
    try:
        _sim_idx = list(system_spec.SIM_TIME_PRESETS).index(float(_def_sim))
    except (TypeError, ValueError):
        _sim_idx = 1
    try:
        _dt_idx = list(system_spec.SOLVER_STEP_PRESETS).index(float(_def_dt))
    except (TypeError, ValueError):
        _dt_idx = 1
    if _def_x0 and len(_def_x0) == len(plant["states"]):
        _x0_default = ", ".join(str(v) for v in _def_x0)
    else:
        _x0_default = ", ".join("0" for _ in plant["states"])
    st.caption(
        "Reference trajectory is set here (Adaptive-owned). "
        "Pre-Launch does not supply it."
    )
    with st.form("sim_knobs_form"):
        sim_time = st.radio("Simulation time (s)", system_spec.SIM_TIME_PRESETS,
                            index=_sim_idx, disabled=_wizard_locked)
        solver_step = st.radio("Solver step (s)", system_spec.SOLVER_STEP_PRESETS,
                               index=_dt_idx, disabled=_wizard_locked)
        x0_text = st.text_input(
            "Initial condition x0 (comma-separated, one per state: %s)"
            % ", ".join(plant["states"]),
            value=_x0_default, disabled=_wizard_locked)
        ref_texts = {}
        for out in plant["outputs"]:
            ref_texts[out] = st.text_input(
                "Reference for %s (plain text is fine, e.g. '0', 'step of amplitude 0.2', "
                "'0.5*sin(0.5*t)'; the Clarifier will normalize it)" % out,
                key="ref_%s" % out, disabled=_wizard_locked)
        submitted = st.form_submit_button("Confirm parameters", disabled=_wizard_locked)
    if submitted:
        try:
            x0 = [float(v.strip()) for v in x0_text.split(",") if v.strip()]
        except ValueError:
            x0 = []
        knobs_spec = system_spec.merge_sim_knobs(
            st.session_state.plant_spec, sim_time=sim_time, solver_step=solver_step,
            x0=x0, reference_exprs=ref_texts)
        knobs_gaps = [g for g in system_spec.missing_items(knobs_spec)
                     if g["category"] in ("initial_condition", "reference")]
        if knobs_gaps:
            st.error("Simulation setup is incomplete:\n\n"
                     + "\n".join("- %s" % g["detail"] for g in knobs_gaps))
        else:
            st.session_state.sim_knobs_spec = knobs_spec
            st.session_state.pop("artifact_sim_defaults", None)
            st.rerun()
elif st.session_state.sim_knobs_spec is not None:
    dyn = st.session_state.sim_knobs_spec["dynamics"]
    st.success("Parameters confirmed. sim_time=%.4g s, solver_step=%.4g s, x0=%s"
              % (dyn["sim_time"], dyn["solver_step"], dyn["x0"]))
    _refs = dyn.get("references") or []
    if _refs:
        _ref_bits = []
        for _r in _refs:
            if isinstance(_r, dict):
                _ref_bits.append("%s: %s" % (_r.get("output", "?"), _r.get("expr", "?")))
            else:
                _ref_bits.append(str(_r))
        st.caption("Reference trajectory: " + "; ".join(_ref_bits))
    else:
        st.caption("Reference trajectory: (none)")

if st.session_state.plant_spec is not None and not _wizard_locked:
    if st.button("Start over"):
        st.session_state.plant_spec = None
        st.session_state.sim_knobs_spec = None
        st.session_state.result = None
        st.session_state.pop("artifact_sim_defaults", None)
        st.session_state.pop("loaded_artifact_id", None)
        st.rerun()


if "result" not in st.session_state:
    st.session_state.result = None
if "pipeline_box" not in st.session_state:
    st.session_state.pipeline_box = None
if "clarify" not in st.session_state:
    st.session_state.clarify = None
if "system_spec" not in st.session_state:
    st.session_state.system_spec = None

pipeline_running = bool(st.session_state.pipeline_box and st.session_state.pipeline_box["running"])
clarify_active = st.session_state.clarify is not None

run_clicked = st.button("Design controller", type="primary", use_container_width=True,
                          disabled=(pipeline_running or clarify_active
                                   or st.session_state.sim_knobs_spec is None))

# bg pipeline thread just finished (render_live_pipeline's st.rerun() got us here).
# grab its result into session state now, on the main thread, before drawing anything
if (st.session_state.pipeline_box is not None
        and not st.session_state.pipeline_box["running"]
        and st.session_state.pipeline_box["result"] is not None):
    st.session_state.result = st.session_state.pipeline_box["result"]
    st.session_state.pipeline_box = None
    st.session_state.pdf_bytes = None
    pipeline_running = False

# clarify state machine: None (idle) -> "running" (model call in flight) ->
# "asking" (waiting on chat_input) -> loops until "complete" hands off to the pipeline.
clarify_state = st.session_state.clarify
if (clarify_state is not None and clarify_state["mode"] == "running"
        and not clarify_state["box"]["running"]
        and clarify_state["box"]["result"] is not None):
    result = clarify_state["box"]["result"]
    events = list(clarify_state.get("events") or []) + list(clarify_state["box"]["events"])
    usage = clarifier._sum_usage(clarify_state.get("usage") or clarifier._empty_usage(),
                                 result.get("usage") or clarifier._empty_usage())
    chat_log = list(clarify_state.get("chat_log") or [])
    clarify_state = dict(clarify_state, events=events, usage=usage)

    if result["status"] == "error":
        st.error("The Clarifier could not respond: %s" % result["error"])
        try:
            with open(clarifier.DEBUG_LOG, "r", encoding="utf-8") as _fh:
                _log = _fh.read()
            with st.expander("Debug: the exact exchange with the model", expanded=True):
                st.caption(clarifier.DEBUG_LOG)
                st.code(_log[-6000:], language="text")
        except Exception as _e:
            st.caption("(no debug log yet: %s)" % _e)
        clarify_state["mode"] = "asking"
        clarify_state["chat_log"] = chat_log
        st.session_state.clarify = clarify_state
    elif result["status"] == "complete":
        dyn = dict(clarify_state["spec"]["dynamics"])
        dyn["uncertainty"] = result["dynamics"]["uncertainty"]
        dyn["disturbance"] = result["dynamics"]["disturbance"]
        # References are owned by Adaptive (Clarifier / sim knobs). Accept
        # Clarifier refs when provided; otherwise keep whatever Adaptive
        # already has (may be empty until sim setup fills them).
        clarifier_refs = (result.get("dynamics") or {}).get("references")
        if clarifier_refs is not None:
            dyn["references"] = clarifier_refs
        spec = {"status": clarify_state["spec"]["status"],
                "system_name": clarify_state["spec"]["system_name"], "dynamics": dyn}
        record = _clarification_record_from_log(chat_log)
        launch_state = {"spec": spec, "opts": clarify_state["opts"], "record": record,
                        "events": events, "usage": usage, "description": ""}
        _launch_clarified_pipeline(launch_state)
        pipeline_running = True
    else:
        chat_log = chat_log + [{"role": "assistant", "text": result["reply"]}]
        clarify_state["mode"] = "asking"
        clarify_state["chat_log"] = chat_log
        clarify_state["messages"] = result["messages"]
        st.session_state.clarify = clarify_state

if run_clicked:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. Add it to the repo-root .env "
                 "(copy from .env.example; see backend_core/AgentAdaptive/agents/llm_factory.py), "
                 "or export it before launching Streamlit.")
    elif st.session_state.sim_knobs_spec is None:
        st.error("Confirm the plant and simulation setup above first.")
    else:
        os.environ["OPENAI_MODEL"] = model_name
        # clear any leftover per-agent key from an older session config so it
        # can't silently get reused now that everything shares one key
        os.environ.pop("OPENAI_API_KEY_TUNER", None)
        os.environ.pop("OPENAI_API_KEY_CLARIFIER", None)
        if tuner_model_name:
            os.environ["OPENAI_MODEL_TUNER"] = tuner_model_name
        else:
            os.environ.pop("OPENAI_MODEL_TUNER", None)
        if clarifier_model_name:
            os.environ["OPENAI_MODEL_CLARIFIER"] = clarifier_model_name
        else:
            os.environ.pop("OPENAI_MODEL_CLARIFIER", None)

        opts = dict(enable_tuning=enable_tuning, target_rms_frac=target_rms_frac,
                    max_tuning_rounds=max_tuning_rounds,
                    tuning_objectives=tuning_priorities)
        st.session_state.result = None
        st.session_state.pdf_bytes = None
        st.session_state.system_spec = None
        _clarify_state = {"messages": clarifier.start_conversation(st.session_state.sim_knobs_spec),
                          "chat_log": [], "spec": st.session_state.sim_knobs_spec,
                          "opts": opts, "round": 0, "usage": clarifier._empty_usage(),
                          "events": []}
        _start_clarify_turn(_clarify_state)
        st.rerun()


res = st.session_state.result
clarify_state = st.session_state.clarify
if clarify_state is not None and clarify_state["mode"] == "running":
    st.subheader("Thinking…")
    render_live_clarifier(clarify_state["box"])
elif clarify_state is not None and clarify_state["mode"] == "asking":
    render_clarify_chat(clarify_state)
elif pipeline_running:
    st.subheader("Processing")
    render_live_pipeline(st.session_state.pipeline_box)
elif res:
    if res["error"]:
        st.error(res["error"])
        if res["log"]:
            with st.expander("Console log up to the error"):
                st.code(res["log"])
    else:
        if res.get("cancelled"):
            st.warning("The process was stopped by the user \u2014 the result below reflects whatever had been produced up to that point "
                        "(some planned review/tuning rounds may not have run).")

        if render_run_scores(res.get("final_metrics"), spec=res.get("system_spec")):
            st.divider()

        finished_spec = res.get("system_spec") or st.session_state.get("system_spec")
        if finished_spec:
            spec_name = finished_spec.get("system_name") or "Extracted system"
            with st.expander("Extracted system: %s" % spec_name):
                st.caption(
                    "This JSON is what the Clarifier Agent handed to the Design Agent. "
                    "Everything in it was either taken from your description or confirmed "
                    "by you. The design method (SMC vs backstepping) is deliberately left "
                    "out; the Design Agent picks that from the structure.")
                st.code(system_spec.spec_to_json(finished_spec), language="json")

        clarification_record = res.get("clarification_record") or []
        if clarification_record:
            def _clar_source(row):
                source = row.get("source") or ("you" if row.get("answered") else "default")
                return {"user": "you", "description": "your description"}.get(source, source)

            has_read = any((r.get("source") == "description")
                            for r in clarification_record)
            with st.expander("Clarifications applied"):
                columns = {
                    "question": [r.get("question", "") for r in clarification_record],
                    "answer": [r.get("answer_label", "") for r in clarification_record],
                    "source": [_clar_source(r) for r in clarification_record],
                }
                if has_read:
                    columns["read from"] = [(r.get("evidence") or "")
                                            for r in clarification_record]
                st.table(columns)
                st.caption("Rows marked 'default' were not answered. The value shown was "
                            "assumed on your behalf and is recorded as an assumption in the "
                            "report.")
                if has_read:
                    st.caption("Rows marked 'your description' were never put to you: your "
                                "description already settled them, so the value was read from "
                                "the quoted sentence instead of asked about. Check the reading; "
                                "if a row is wrong, the fix is to reword that sentence.")

        if res.get("abstract"):
            st.subheader("Abstract")
            st.markdown(res["abstract"])

        st.subheader("Design Agent \u2014 Summary")
        render_summary_with_stability_expander(
            normalize_latex_delimiters(sanitize_latex_environments(res["summary"])))
        if res.get("tuning_log"):
            st.caption(
                "The RMS/Performance numbers above are from BEFORE tuning: "
                "the Design Agent writes this summary before the Tuner Agent runs. See the "
                "'Tuner Agent' section below for the actual final RMS "
                "after tuning.")

        if res.get("tuning_log"):
            st.subheader("Tuner Agent \u2014 Tuning Result")
            st.caption(
                "The Tuner Agent never sees or changes the system description, states, dynamics, "
                "or has_delta/has_disturbance. It only reads the simulation metrics and "
                "proposes new TUNING parameters, using its own model/API call, separate "
                "from the Design Agent above.")
            tlog = res["tuning_log"]
            best = res.get("tuning_best")

            _used_objectives = None
            if tlog and isinstance(tlog[0], dict) and tlog[0].get("round") == 0:
                _used_objectives = tlog[0].get("objectives")
            if not _used_objectives:
                _used_objectives = res.get("tuning_objectives")
            _used_objectives = tuning_objectives.normalize_selection(_used_objectives)
            if _used_objectives:
                st.caption("Tuning priorities in force for this run: "
                           + tuning_objectives.selection_summary(_used_objectives))

            if len(tlog) == 1 and tlog[0]["round"] == 0:
                r0 = tlog[0]["reasoning"]
                if r0 not in ("(initial design, before tuning)",):
                    if "already meets the target" in r0:
                        st.info(r0 + " so the tuning loop did not need to run.")
                    else:
                        st.warning("Tuner Agent did not run any tuning rounds: " + r0)

            if best is not None:
                best_entry = next((e for e in tlog if e["round"] == best["round"]), None)
                if best["round"] == 0:
                    st.info("Tuning made no improvement. The Design Agent's original design "
                            "(round 0) is still the best result.")
                else:
                    st.success(
                        "Target met" if best_entry and best_entry.get("met_target")
                        else "Best result reached after %d round(s) (target not fully met, "
                             "best attempt kept)." % best["round"]
                    )
                    if best_entry and best_entry.get("reasoning"):
                        st.markdown("**Tuner Agent's reasoning for this final result:** "
                                    + best_entry["reasoning"])

                _final_rows = (best_entry or {}).get("objective_values") or []
                _baseline_by_label = {
                    row[0]: row[1]
                    for row in ((tlog[0] or {}).get("objective_values") or [])
                    if row and len(row) > 1
                }
                if _final_rows:
                    _render_objective_values(
                        {"objective_values": [
                            (row[0], row[1], _baseline_by_label.get(row[0]))
                            for row in _final_rows if row and len(row) > 1]},
                        label="Your tuning priorities, final result")

                st.markdown("**Final tuning parameter values:**")
                st.table({
                    "parameter": list(best["tuning"].keys()),
                    "value": [str(v) for v in best["tuning"].values()],
                })

                if best_entry and best_entry.get("report"):
                    st.code(best_entry["report"], language=None)

            if len(tlog) > 1:
                with st.expander("Show all %d tuning rounds (full history)" % len(tlog)):
                    for entry in tlog:
                        label = "Round %d" % entry["round"]
                        if entry["round"] == 0:
                            label += " (Design Agent's original design, before any tuning)"
                        else:
                            label += " (Tuner Agent's proposal)"
                        target_note = " (target met)" if entry.get("met_target") else ""
                        st.markdown("---")
                        st.markdown("**" + label + target_note + "**")
                        _render_round_scores(entry)
                        _render_objective_values(
                            entry, label="Your tuning priorities, this round")
                        reasoning = entry.get("reasoning")
                        if reasoning and reasoning != "(initial design, before tuning)":
                            prefix = "Tuner Agent's reasoning: " if entry["round"] > 0 else ""
                            st.markdown(prefix + reasoning)
                        if entry["round"] > 0:
                            changed = entry.get("changed") or {}
                            if changed:
                                st.table({
                                    "parameter": list(changed.keys()),
                                    "old value": [str(old) for old, _new in changed.values()],
                                    "new value": [str(new) for _old, new in changed.values()],
                                })
                            else:
                                st.caption("No tuning parameters were changed this round.")
                        if entry.get("report"):
                            st.code(entry["report"], language=None)

        if res.get("usage"):
            st.subheader("Token Usage & Cost")
            u = res["usage"]

            cost_rows, cost_total = model_pricing.run_cost_rows(u)
            if cost_rows:
                st.metric("Estimated cost of this run",
                          model_pricing.format_cost(cost_total))
                st.table({
                    "agent": [r["label"] for r in cost_rows],
                    "model": [r["model"] for r in cost_rows],
                    "input": [format(r["input_tokens"], ",") for r in cost_rows],
                    "of which cached": [format(r["cached_input_tokens"], ",") for r in cost_rows],
                    "output": [format(r["output_tokens"], ",") for r in cost_rows],
                    "cost": [model_pricing.format_cost(r["cost"]) for r in cost_rows],
                })
                if cost_total is None:
                    st.caption(
                        "One or more agents ran on a model that is not in the "
                        "price table, so no run total can be given. The rows "
                        "above show which.")
                st.caption(
                    "Cached input is the part OpenAI served from its prompt "
                    "cache and bills at a reduced rate; it is already included "
                    "in the input column, not additional to it. Treat this as "
                    "an estimate: it prices the tokens this run reported, and "
                    "does not include anything your account is charged "
                    "separately.")

            has_tuner = bool(u.get("tuner", {}).get("total_tokens"))
            has_clarifier = bool(u.get("clarifier", {}).get("total_tokens"))
            n_cols = 2 + int(has_clarifier) + int(has_tuner)
            cols = st.columns(n_cols)
            next_col = 0
            if has_clarifier:
                with cols[next_col]:
                    st.metric("Clarifier Agent", "%s tokens" % format(u["clarifier"]["total_tokens"], ","),
                               help="input: %s  |  output: %s" % (format(u["clarifier"]["input_tokens"], ","),
                                                                    format(u["clarifier"]["output_tokens"], ",")))
                next_col += 1
            with cols[next_col]:
                st.metric("Design Agent", "%s tokens" % format(u["agent"]["total_tokens"], ","),
                           help="input: %s  |  output: %s" % (format(u["agent"]["input_tokens"], ","),
                                                                format(u["agent"]["output_tokens"], ",")))
            next_col += 1
            if has_tuner:
                with cols[next_col]:
                    st.metric("Tuner Agent", "%s tokens" % format(u["tuner"]["total_tokens"], ","),
                               help="input: %s  |  output: %s" % (format(u["tuner"]["input_tokens"], ","),
                                                                    format(u["tuner"]["output_tokens"], ",")))
                next_col += 1
            with cols[next_col]:
                st.metric("Total", "%s tokens" % format(u["total"]["total_tokens"], ","))

            if u.get("timeline"):
                with st.expander("Per-step breakdown (what actually happened, in order)"):
                    _ACTOR_NAME = {"clarify": "Clarifier Agent", "design": "Design Agent",
                                   "tuner": "Tuner Agent"}
                    timeline = u["timeline"]
                    for i, t in enumerate(timeline):
                        name = _ACTOR_NAME.get(t["actor"], t["actor"])
                        st.markdown(
                            "**%s**, %s: **%s tokens** (input: %s, output: %s), *%s*"
                            % (name, t["label"], format(t["tokens"], ","),
                               format(t["input_tokens"], ","), format(t["output_tokens"], ","),
                               t["detail"])
                        )
                        if i < len(timeline) - 1:
                            st.markdown(
                                "<div style='opacity:.4; padding:0 0 0 1.4rem; margin-top:-.4rem;'>↓</div>",
                                unsafe_allow_html=True)
                    st.caption(
                        "Each call re-sends the relevant conversation so far as input (the API is "
                        "stateless), so cost grows with every extra tuning round.")

        if res["figures"]:
            st.subheader("Plots")
            cols_per_row = 2
            for i in range(0, len(res["figures"]), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, fig in enumerate(res["figures"][i:i + cols_per_row]):
                    with cols[j]:
                        st.image(fig["png"], caption=fig.get("title") or None,
                                  use_container_width=True)
                        st.download_button(
                            "Download PNG", data=fig["png"],
                            file_name="plot_%d.png" % (i + j + 1),
                            mime="image/png",
                            key="dl_%d" % (i + j),
                        )

        if res.get("events"):
            st.subheader("Process Log")
            st.caption("What happened, step by step. Click any row to expand it; "
                        "this is the same trace shown while the run was in progress.")
            render_trace_rows(res["events"])

        with st.expander("Full log (agent reasoning, controller law, RMS numbers)"):
            st.code(res["log"])

        st.divider()
        st.subheader("Download Report")
        if st.button("Generate PDF report"):
            with st.spinner("Compiling report with LaTeX (formulas, tables, and plots)..."):
                try:
                    st.session_state.pdf_bytes = pdf_report.build_pdf_report(
                        normalize_latex_delimiters(sanitize_latex_environments(res["summary"])),
                        res["figures"],
                        usage=res.get("usage"),
                        log_text=res["log"],
                        tuning_log=res.get("tuning_log"),
                        tuning_best=res.get("tuning_best"),
                        clarification_record=res.get("clarification_record"),
                        final_metrics=res.get("final_metrics"),
                        abstract_markdown=res.get("abstract"),
                    )
                except RuntimeError as e:
                    st.session_state.pdf_bytes = None
                    st.error(str(e))
        if st.session_state.get("pdf_bytes"):
            st.download_button(
                "Download PDF", data=st.session_state.pdf_bytes,
                file_name="control_design_report.pdf", mime="application/pdf",
                use_container_width=True,
            )
else:
    st.info("Confirm the plant and simulation setup above, then click **Design controller**.")
