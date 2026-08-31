"""Thin HTTP service adapter for AgentMPC tuning jobs.

Orchestrates the multi-agent graph (``build_mpc_tuning_graph`` /
``build_ui_tuning_graph`` + ``initial_state``) without reimplementing agents
or the MPC numeric stack. Job state lives in ``job_store``; core calls go to
``backend_core.AgentMPC``.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

from backend_api.AgentMPC.job_store import InMemoryJobStore, JobRecord, default_job_store
from backend_api.AgentMPC.schemas import (
    MPCDynamicsInput,
    MPCJobCreateRequest,
    MPCJobCreateResponse,
    MPCJobOptions,
    MPCJobProgressEvent,
    MPCJobResultsResponse,
    MPCJobStatusResponse,
    MPCJobSummary,
)

_PLUGINS_DIR = (
    Path(__file__).resolve().parents[2]
    / "backend_core"
    / "AgentMPC"
    / "dynamics"
    / "plugins"
)

_LLM_CONFIGURED = False
_LLM_CONFIG_LOCK = threading.Lock()
_LLM_CONFIG_ERROR: str | None = None


def ensure_llm_configured(model: str | None = None) -> None:
    """Register ``configure_llm`` once for AgentMPC agents (process-wide).

    OpenAI-first backbone (API / production default):

    1. ``labcd_agents.LLMFactory`` with an **OpenAI** model when
       ``OPENAI_API_KEY`` is available (default ``gpt-4o-mini``).
    2. Direct ``langchain_openai.ChatOpenAI`` fallback.
    3. Groq only if explicitly requested via model name / ``LABCD_MPC_PROVIDER=groq``
       or when OpenAI is unavailable and ``GROQ_API_KEY`` is set.

    Env knobs:

    - ``LABCD_MPC_MODEL`` / ``DEFAULT_LLM_MODEL`` / ``OPENAI_MODEL`` — model id
    - ``LABCD_MPC_PROVIDER`` — force ``openai`` or ``groq``
    - ``OPENAI_API_KEY``, ``GROQ_API_KEY``

    Raises ``RuntimeError`` with an actionable message if no provider is available.
    """
    global _LLM_CONFIGURED, _LLM_CONFIG_ERROR

    from backend_core.AgentMPC.agents.llm_base import configure_llm, get_llm

    # Already usable?
    try:
        get_llm()
        _LLM_CONFIGURED = True
        return
    except RuntimeError:
        pass

    with _LLM_CONFIG_LOCK:
        try:
            get_llm()
            _LLM_CONFIGURED = True
            return
        except RuntimeError:
            pass

        force = (os.getenv("LABCD_MPC_PROVIDER") or "").strip().lower()
        default_openai_model = (
            os.getenv("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        model_name = (
            model
            or os.getenv("LABCD_MPC_MODEL")
            or os.getenv("DEFAULT_LLM_MODEL")
            or default_openai_model
        )

        # Treat Groq-style defaults from older Streamlit configs as OpenAI
        # unless the user forced groq.
        _groq_like = (
            "gpt-oss" in model_name
            or model_name.startswith("llama")
            or "groq" in model_name.lower()
        )
        if force != "groq" and _groq_like and os.getenv("OPENAI_API_KEY"):
            model_name = default_openai_model

        def _try_labcd(model_id: str) -> bool:
            global _LLM_CONFIG_ERROR
            try:
                from labcd_agents import LLMFactory, ensure_env_loaded, get_api_key

                repo_root = Path(__file__).resolve().parents[2]
                env_path = repo_root / ".env"
                ensure_env_loaded(str(env_path) if env_path.is_file() else None)
                provider = LLMFactory.resolve_provider(model_id)
                if not provider or not get_api_key(provider):
                    return False
                if force == "openai" and provider != "openai":
                    return False
                if force == "groq" and provider != "groq":
                    return False
                llm_instance = LLMFactory.create(
                    model_id, temperature=0.3, seed=42, max_retries=2
                )
                configure_llm(lambda: llm_instance)
                return True
            except ImportError:
                return False
            except Exception as exc:  # noqa: BLE001
                _LLM_CONFIG_ERROR = f"labcd_agents init failed: {exc}"
                return False

        def _try_openai_direct(model_id: str) -> bool:
            global _LLM_CONFIG_ERROR
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                return False
            try:
                from langchain_openai import ChatOpenAI

                # Strip provider prefix if present (e.g. openai/gpt-4o-mini)
                mid = model_id.split("/", 1)[-1] if model_id.startswith("openai/") else model_id
                if mid.startswith("gpt-oss") or mid.startswith("llama"):
                    mid = default_openai_model

                def _oai_factory(_model: str = mid, _key: str = openai_key):
                    return ChatOpenAI(
                        model=_model,
                        api_key=_key,
                        temperature=0.3,
                        max_retries=2,
                    )

                configure_llm(_oai_factory)
                return True
            except Exception as exc:  # noqa: BLE001
                _LLM_CONFIG_ERROR = f"ChatOpenAI init failed: {exc}"
                return False

        def _try_groq_direct(model_id: str) -> bool:
            global _LLM_CONFIG_ERROR
            groq_key = os.getenv("GROQ_API_KEY")
            if not groq_key:
                return False
            try:
                from langchain_groq import ChatGroq

                groq_model = os.getenv("GROQ_MODEL") or model_id
                # Retired Groq id — use a current default
                if "llama-3.3-70b-versatile" in groq_model:
                    groq_model = os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"

                def _groq_factory(
                    _model: str = groq_model, _key: str = groq_key
                ):
                    return ChatGroq(
                        model=_model,
                        api_key=_key,
                        temperature=0.3,
                        max_retries=2,
                    )

                configure_llm(_groq_factory)
                return True
            except Exception as exc:  # noqa: BLE001
                _LLM_CONFIG_ERROR = f"ChatGroq init failed: {exc}"
                return False

        # --- OpenAI first (unless forced to groq) ---
        if force != "groq":
            if _try_labcd(model_name if not _groq_like else default_openai_model):
                _LLM_CONFIGURED = True
                _LLM_CONFIG_ERROR = None
                return
            if _try_openai_direct(model_name):
                _LLM_CONFIGURED = True
                _LLM_CONFIG_ERROR = None
                return

        # --- Groq second ---
        if force != "openai":
            groq_model = model_name if force == "groq" or _groq_like else (
                os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"
            )
            if _try_labcd(groq_model):
                _LLM_CONFIGURED = True
                _LLM_CONFIG_ERROR = None
                return
            if _try_groq_direct(groq_model):
                _LLM_CONFIGURED = True
                _LLM_CONFIG_ERROR = None
                return

        # Last resort: opposite provider
        if force == "groq" and _try_openai_direct(default_openai_model):
            _LLM_CONFIGURED = True
            _LLM_CONFIG_ERROR = None
            return
        if force == "openai" and _try_groq_direct(
            os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"
        ):
            _LLM_CONFIGURED = True
            _LLM_CONFIG_ERROR = None
            return

        detail = _LLM_CONFIG_ERROR or "no provider matched"
        raise RuntimeError(
            "No LLM configured for AgentMPC. Set OPENAI_API_KEY (preferred) "
            "and/or GROQ_API_KEY in the environment or repo-root .env. "
            "Optional: LABCD_MPC_MODEL / OPENAI_MODEL / LABCD_MPC_PROVIDER=openai|groq. "
            f"Detail: {detail}"
        )



def llm_status() -> dict[str, Any]:
    """Lightweight readiness probe for /health (never raises)."""
    configured = False
    try:
        from backend_core.AgentMPC.agents import llm_base

        configured = llm_base._llm_factory is not None  # noqa: SLF001
    except Exception:  # noqa: BLE001
        configured = _LLM_CONFIGURED
    return {
        "llm_configured": bool(configured),
        "llm_error": _LLM_CONFIG_ERROR,
        "default_model": os.getenv("LABCD_MPC_MODEL")
        or os.getenv("DEFAULT_LLM_MODEL")
        or os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
    }


def _store(store: InMemoryJobStore | None = None) -> InMemoryJobStore:
    return store or default_job_store


def _options_dict(options: MPCJobOptions | dict[str, Any] | None) -> dict[str, Any]:
    if options is None:
        return MPCJobOptions().model_dump()
    if isinstance(options, MPCJobOptions):
        return options.model_dump()
    return MPCJobOptions(**options).model_dump()


def _options_model(raw: dict[str, Any] | None) -> MPCJobOptions:
    return MPCJobOptions(**(raw or {}))


def _resolve_plugin_path(dynamics: MPCDynamicsInput | dict[str, Any] | None) -> str:
    """Resolve dynamics input to a filesystem path for DynamicLoader."""
    if dynamics is None:
        # Default bundled example (same as run_agents.py).
        path = _PLUGINS_DIR / "example_pendulum.py"
        if path.is_file():
            return str(path)
        raise ValueError(
            "No dynamics provided and default plugin example_pendulum.py not found"
        )

    if isinstance(dynamics, dict):
        dynamics = MPCDynamicsInput(**dynamics)

    if dynamics.plugin_path:
        p = Path(dynamics.plugin_path).expanduser()
        if not p.is_absolute():
            # Try relative to repo root, then plugins dir.
            repo = Path(__file__).resolve().parents[2]
            candidates = [repo / p, _PLUGINS_DIR / p.name, p]
            for c in candidates:
                if c.is_file():
                    return str(c.resolve())
            return str(p.resolve())
        return str(p.resolve())

    if dynamics.plugin_id:
        stem = dynamics.plugin_id.removesuffix(".py")
        path = _PLUGINS_DIR / f"{stem}.py"
        if not path.is_file():
            raise ValueError(f"Unknown plugin_id: {dynamics.plugin_id!r} (looked for {path})")
        return str(path.resolve())

    if dynamics.source:
        fd, tmp = tempfile.mkstemp(suffix=".py", prefix="mpc_plugin_")
        os.close(fd)
        Path(tmp).write_text(dynamics.source, encoding="utf-8")
        return tmp

    raise ValueError(
        "dynamics must include plugin_path, plugin_id, or source "
        "(or omit dynamics to use the default example_pendulum plugin)"
    )



def _json_safe(value: Any) -> Any:
    """Convert numpy / nested graph artefacts into JSON-friendly Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # pydantic v2 / v1 models (e.g. param blobs in history)
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return _json_safe(value.model_dump())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return _json_safe(value.dict())
        except Exception:  # noqa: BLE001
            pass
    # numpy scalars / arrays
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _json_safe(value.item())
        if isinstance(value, np.ndarray):
            return _json_safe(value.tolist())
    except ImportError:
        pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _json_safe(value.tolist())
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def _metric_series(raw: Any) -> list[Any]:
    """Preserve length; map non-finite numbers to None so plots can skip points."""
    out: list[Any] = []
    for x in list(raw or []):
        if x is None:
            out.append(None)
            continue
        try:
            v = float(x)
        except (TypeError, ValueError):
            out.append(None)
            continue
        if v != v or v in (float("inf"), float("-inf")):
            out.append(None)
        else:
            out.append(v)
    return out


def _normalize_history(raw: Any) -> list[Any]:
    """Core graph ``history`` is List[str]; tolerate dicts or mixed."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        return [_json_safe(raw)]
    out: list[Any] = []
    for item in raw:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            out.append(_json_safe(item))
        else:
            out.append(_json_safe(item))
    return out


def _to_status_response(record: JobRecord) -> MPCJobStatusResponse:
    progress = []
    for ev in record.progress:
        extra = {
            k: v
            for k, v in ev.items()
            if k not in ("kind", "stage", "text", "round", "ts")
        }
        progress.append(
            MPCJobProgressEvent(
                kind=str(ev.get("kind") or ""),
                stage=str(ev.get("stage") or ""),
                text=str(ev.get("text") or ""),
                round=ev.get("round"),
                ts=ev.get("ts"),
                extra=extra,
            )
        )
    return MPCJobStatusResponse(
        job_id=record.job_id,
        status=record.status,  # type: ignore[arg-type]
        stage=record.stage,  # type: ignore[arg-type]
        message=record.message,
        error=record.error,
        iteration=record.iteration,
        max_iterations=record.max_iterations,
        progress=progress,
        created_at=record.created_at,
        updated_at=record.updated_at,
        user_id=record.user_id,
        project_id=record.project_id,
        options=_options_model(record.options),
        system_name=record.system_name,
    )


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x != x or x in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return x



def _series_list(raw: Any) -> list[Any]:
    """JSON-safe metric series; drop non-finite floats for plotting clients."""
    out: list[Any] = []
    for x in raw or []:
        f = _finite_float(x)
        if f is not None:
            out.append(f)
        elif isinstance(x, (str, int, bool)):
            out.append(x)
    return out


def _to_results(record: JobRecord) -> MPCJobResultsResponse:
    """Build results payload; never raise on messy graph artefacts."""
    mse_hist: list[Any] = []
    for x in record.mse_history or []:
        f = _finite_float(x)
        if f is not None:
            mse_hist.append(f)

    best_params = _json_safe(record.best_params) if record.best_params is not None else None
    # Schema accepts Any; still prefer a dict when possible.
    if best_params is not None and not isinstance(best_params, (dict, list, str, int, float, bool)):
        best_params = str(best_params)

    try:
        return MPCJobResultsResponse(
            job_id=record.job_id,
            status=record.status,  # type: ignore[arg-type]
            stage=record.stage,  # type: ignore[arg-type]
            best_params=best_params,
            best_mse=_finite_float(record.best_mse),
            iteration=int(record.iteration or 0),
            termination_reason=(
                str(record.termination_reason) if record.termination_reason else None
            ),
            mse_history=mse_hist or _metric_series((record.metrics or {}).get("mse_history") if isinstance(record.metrics, dict) else []),
            overshoot_history=_metric_series(
                getattr(record, "overshoot_history", None)
                or ((record.metrics or {}).get("overshoot_history") if isinstance(record.metrics, dict) else [])
            ),
            settling_history=_metric_series(
                getattr(record, "settling_history", None)
                or ((record.metrics or {}).get("settling_history") if isinstance(record.metrics, dict) else [])
            ),
            effort_history=_metric_series(
                getattr(record, "effort_history", None)
                or ((record.metrics or {}).get("effort_history") if isinstance(record.metrics, dict) else [])
            ),
            params_history=_json_safe(
                list(record.params_history or [])
                or (list((record.metrics or {}).get("params_history") or []) if isinstance(record.metrics, dict) else [])
            ),
            history=_normalize_history(record.history),
            report=str(record.report) if record.report else None,
            export_script=str(record.export_script) if record.export_script else None,
            metrics=_json_safe(record.metrics) if record.metrics is not None else None,
            error=str(record.error) if record.error else None,
        )
    except Exception as exc:  # noqa: BLE001
        # Last-resort minimal payload so the HTTP layer never 500s on results.
        return MPCJobResultsResponse(
            job_id=record.job_id,
            status=record.status,  # type: ignore[arg-type]
            stage=record.stage,  # type: ignore[arg-type]
            best_params=None,
            best_mse=None,
            iteration=int(record.iteration or 0),
            termination_reason=None,
            mse_history=[],
            params_history=[],
            history=[],
            report=None,
            export_script=None,
            metrics=None,
            error=f"results_serialization_error: {type(exc).__name__}: {exc}",
        )


def _make_on_event(job_id: str, store: InMemoryJobStore) -> Callable[[dict[str, Any]], None]:
    def on_event(fields: dict[str, Any]) -> None:
        store.append_progress(job_id, dict(fields))
        kind = fields.get("kind")
        stage = str(fields.get("stage") or "")
        iteration = fields.get("round") or fields.get("iteration")
        updates: dict[str, Any] = {}
        if kind == "stage_start" and stage:
            stage_map = {
                "scenarist": "scenarist",
                "actor": "actor",
                "evaluator": "evaluator",
                "terminator": "terminator",
                "critic": "critic",
                "juror": "juror",
            }
            mapped = stage_map.get(stage.lower())
            if mapped:
                updates["stage"] = mapped
                updates["message"] = f"{mapped} in progress"
        if iteration is not None:
            try:
                updates["iteration"] = int(iteration)
            except (TypeError, ValueError):
                pass
        if updates:
            store.update(job_id, **updates)

    return on_event


def _run_tuning_thread(job_id: str, store: InMemoryJobStore) -> None:
    """Background worker: load dynamics, build graph, invoke, store results."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    record = store.get(job_id)
    if record is None:
        return
    if record.cancel_requested:
        store.update(job_id, status="cancelled", stage="error", message="Cancelled before start")
        return

    options = record.options or {}
    on_event = _make_on_event(job_id, store)

    store.update(
        job_id,
        status="running",
        stage="actor" if options.get("use_ui_graph", True) else "scenarist",
        message="Starting MPC tuning graph",
        error=None,
    )
    on_event(
        {
            "kind": "stage_start",
            "stage": "actor" if options.get("use_ui_graph", True) else "scenarist",
            "text": "Graph started",
            "ts": time.time(),
        }
    )

    try:
        ensure_llm_configured(options.get("model"))

        from backend_core.AgentMPC.dynamics.loader import DynamicLoader
        from backend_core.AgentMPC.graph.workflow import (
            build_mpc_tuning_graph,
            build_ui_tuning_graph,
            initial_state,
        )
        from backend_core.AgentMPC.mpc.config import Config

        dyn_ref = record.dynamics_ref or {}
        plugin_path = _resolve_plugin_path(
            MPCDynamicsInput(**dyn_ref) if dyn_ref else None
        )
        plugin = DynamicLoader.load_from_path(plugin_path)
        dynamics = plugin.create_dynamics()

        cfg = Config()
        cfg.mpc.prediction_horizon = int(options.get("prediction_horizon") or 12)
        cfg.mpc.control_horizon = int(options.get("control_horizon") or 4)
        cfg.data.dt_mpc = float(options.get("dt_mpc") or 0.02)
        cfg.data.simulation_time = float(options.get("simulation_time") or 3.0)

        system_name = (
            options.get("system_name")
            or record.system_name
            or getattr(plugin, "source_name", None)
            or "mpc_system"
        )
        store.update(job_id, system_name=str(system_name))

        seed = options.get("seed_params")
        use_ui = bool(options.get("use_ui_graph", True))
        if use_ui:
            entry = "evaluator" if seed else "actor"
            graph = build_ui_tuning_graph(dynamics, cfg, entry_node=entry)
        else:
            graph = build_mpc_tuning_graph(dynamics, cfg)

        state = initial_state(
            dynamics,
            system_name=str(system_name),
            max_iterations=int(options.get("max_iterations") or 15),
            ui_scenario_level=int(options.get("ui_scenario_level") or 1),
            seed_params=seed,
            user_guidance=str(options.get("user_guidance") or ""),
            min_explore_iterations=int(options.get("min_explore_iterations") or 4),
            exploration_intensity=int(options.get("exploration_intensity") or 50),
            dt_mpc=float(options.get("dt_mpc") or 0.02),
        )

        if store.is_cancel_requested(job_id):
            store.update(job_id, status="cancelled", stage="error", message="Cancelled")
            return

        # Full invoke; cancel is cooperative via is_cancel_requested checks around the run.
        final_state = graph.invoke(state)

        if store.is_cancel_requested(job_id):
            store.update(job_id, status="cancelled", stage="error", message="Cancelled")
            return

        best_mse = final_state.get("best_mse")
        if best_mse is not None and best_mse == float("inf"):
            best_mse = None

        mse_h = _metric_series(final_state.get("mse_history"))
        overshoot_h = _metric_series(final_state.get("overshoot_history"))
        settling_h = _metric_series(final_state.get("settling_history"))
        effort_h = _metric_series(final_state.get("effort_history"))
        params_h = _json_safe(list(final_state.get("params_history") or []))

        # Actor often omits dt (null) unless the Juror/Actor retunes sample time.
        # Streamlit plots the *effective* cfg.data.dt_mpc each iteration — recover
        # that from state / options and back-fill params_history + dt_history.
        default_dt = None
        try:
            default_dt = float(
                final_state.get("dt_mpc")
                or (options.get("dt_mpc") if isinstance(options, dict) else None)
                or 0.02
            )
        except (TypeError, ValueError):
            default_dt = 0.02
        dt_h: list[Any] = []
        filled_params: list[Any] = []
        running_dt = default_dt
        for p in params_h:
            if isinstance(p, dict):
                p = dict(p)
                raw_dt = p.get("dt") if p.get("dt") is not None else p.get("dt_mpc")
                if raw_dt is not None:
                    try:
                        running_dt = float(raw_dt)
                    except (TypeError, ValueError):
                        pass
                else:
                    p["dt"] = running_dt
                dt_h.append(running_dt)
                filled_params.append(p)
            else:
                dt_h.append(running_dt)
                filled_params.append(p)
        params_h = filled_params
        if not dt_h and default_dt is not None and mse_h:
            dt_h = [default_dt] * len(mse_h)

        metrics = {
            "best_mse": best_mse,
            "iteration": final_state.get("iteration"),
            "termination_reason": final_state.get("termination_reason"),
            # Canonical place for notebook / React charts (always present)
            "mse_history": mse_h,
            "overshoot_history": overshoot_h,
            "settling_history": settling_h,
            "effort_history": effort_h,
            "dt_history": dt_h,
            "params_history": params_h,
            "best_overshoot": _json_safe(final_state.get("best_overshoot")),
            "best_settling": _json_safe(final_state.get("best_settling")),
            "best_effort": _json_safe(final_state.get("best_effort")),
            "dt_mpc": default_dt,
        }
        history = _normalize_history(final_state.get("history"))

        # Only pass JobRecord fields that exist (older stores may lack series attrs)
        fields = {
            "status": "completed",
            "stage": "done",
            "message": "Tuning completed",
            "best_params": _json_safe(
                final_state.get("best_params") or final_state.get("current_params")
            ),
            "best_mse": best_mse if best_mse is None else float(best_mse),
            "iteration": int(final_state.get("iteration") or 0),
            "termination_reason": str(final_state.get("termination_reason") or "") or None,
            "mse_history": mse_h,
            "params_history": params_h,
            "history": history,
            "metrics": _json_safe(metrics),
        }
        for key, val in (
            ("overshoot_history", overshoot_h),
            ("settling_history", settling_h),
            ("effort_history", effort_h),
        ):
            # probe field presence on a throwaway check via store record
            rec0 = store.get(job_id)
            if rec0 is not None and hasattr(rec0, key):
                fields[key] = val
        store.update(job_id, **fields)
        on_event(
            {
                "kind": "completed",
                "stage": "done",
                "text": "Tuning completed",
                "round": final_state.get("iteration"),
                "ts": time.time(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        store.update(
            job_id,
            status="failed",
            stage="error",
            message="Tuning failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        on_event(
            {
                "kind": "error",
                "stage": "error",
                "text": f"{type(exc).__name__}: {exc}",
                "ts": time.time(),
            }
        )


def _start_tuning_async(job_id: str, store: InMemoryJobStore) -> None:
    thread = threading.Thread(
        target=_run_tuning_thread,
        args=(job_id, store),
        daemon=True,
        name=f"mpc-job-{job_id}",
    )
    thread.start()


def submit_job(
    request: MPCJobCreateRequest,
    *,
    store: InMemoryJobStore | None = None,
) -> MPCJobCreateResponse:
    """Create a job and start the tuning graph in a background thread."""
    job_store = _store(store)
    options = _options_dict(request.options)
    dynamics_ref = None
    if request.dynamics is not None:
        dynamics_ref = request.dynamics.model_dump(exclude_none=True)

    # Validate dynamics resolution early so the client gets a 4xx-style error
    # path via the service (router maps ValueError if needed).
    try:
        _resolve_plugin_path(request.dynamics)
    except ValueError:
        # Still create the job as failed for auditability, or re-raise.
        # Prefer fail-fast on submit.
        raise

    system_name = options.get("system_name") or "mpc_system"
    record = job_store.create(
        dynamics_ref=dynamics_ref,
        options=options,
        user_id=request.user_id,
        project_id=request.project_id,
        system_name=str(system_name),
    )
    job_id = record.job_id
    job_store.update(
        job_id,
        status="running",
        stage="actor" if options.get("use_ui_graph", True) else "scenarist",
        message="Job queued; starting tuning",
    )
    _start_tuning_async(job_id, job_store)
    latest = job_store.get(job_id)
    assert latest is not None
    return MPCJobCreateResponse(
        job_id=job_id,
        status=latest.status,  # type: ignore[arg-type]
        stage=latest.stage,  # type: ignore[arg-type]
        message=latest.message or "Job started",
    )


def get_job(job_id: str, *, store: InMemoryJobStore | None = None) -> MPCJobStatusResponse:
    record = _store(store).get(job_id)
    if record is None:
        raise KeyError(job_id)
    return _to_status_response(record)


def list_jobs(
    user_id: int | None = None,
    *,
    store: InMemoryJobStore | None = None,
) -> list[MPCJobSummary]:
    records = _store(store).list_jobs(user_id=user_id)
    return [
        MPCJobSummary(
            job_id=r.job_id,
            status=r.status,  # type: ignore[arg-type]
            stage=r.stage,  # type: ignore[arg-type]
            system_name=r.system_name,
            created_at=r.created_at,
            updated_at=r.updated_at,
            user_id=r.user_id,
        )
        for r in records
    ]


def cancel_job(job_id: str, *, store: InMemoryJobStore | None = None) -> MPCJobStatusResponse:
    job_store = _store(store)
    record = job_store.get(job_id)
    if record is None:
        raise KeyError(job_id)
    if record.status in ("completed", "failed", "cancelled"):
        return _to_status_response(record)
    job_store.request_cancel(job_id)
    job_store.update(
        job_id,
        status="cancelled",
        stage="error",
        message="Cancel requested",
    )
    updated = job_store.get(job_id)
    assert updated is not None
    return _to_status_response(updated)


def get_results(job_id: str, *, store: InMemoryJobStore | None = None) -> MPCJobResultsResponse:
    record = _store(store).get(job_id)
    if record is None:
        raise KeyError(job_id)
    try:
        return _to_results(record)
    except Exception as exc:  # noqa: BLE001
        return MPCJobResultsResponse(
            job_id=job_id,
            status=getattr(record, "status", "failed") or "failed",  # type: ignore[arg-type]
            stage=getattr(record, "stage", "error") or "error",  # type: ignore[arg-type]
            error=f"results_serialization_error: {type(exc).__name__}: {exc}",
        )
