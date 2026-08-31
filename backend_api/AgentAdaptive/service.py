"""Thin HTTP service adapter for AgentAdaptive design jobs.

Orchestrates Clarifier turns and ``run_full_pipeline`` without reimplementing
control math or prompts. Job state lives in ``job_store``; core calls go to
``backend_core.AgentAdaptive``.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from backend_core.AgentAdaptive.agents import clarifier
from backend_core.AgentAdaptive.agents.tuner_agent import run_full_pipeline
from backend_core.AgentAdaptive.tools import system_spec as system_spec_mod

from backend_api.AgentAdaptive.job_store import InMemoryJobStore, JobRecord, default_job_store
from backend_api.AgentAdaptive.schemas import (
    AdaptiveClarifyRequest,
    AdaptiveClarifyResponse,
    AdaptiveJobCreateRequest,
    AdaptiveJobCreateResponse,
    AdaptiveJobOptions,
    AdaptiveJobProgressEvent,
    AdaptiveJobResultsResponse,
    AdaptiveJobStatusResponse,
    AdaptiveJobSummary,
)


def _store(store: InMemoryJobStore | None = None) -> InMemoryJobStore:
    return store or default_job_store


def _options_dict(options: AdaptiveJobOptions | dict[str, Any] | None) -> dict[str, Any]:
    if options is None:
        return AdaptiveJobOptions().model_dump()
    if isinstance(options, AdaptiveJobOptions):
        return options.model_dump()
    return AdaptiveJobOptions(**options).model_dump()


def _options_model(raw: dict[str, Any] | None) -> AdaptiveJobOptions:
    return AdaptiveJobOptions(**(raw or {}))


def _system_name(spec: dict[str, Any] | None) -> str | None:
    if not spec:
        return None
    name = spec.get("system_name")
    return str(name) if name else None


def _to_status_response(record: JobRecord) -> AdaptiveJobStatusResponse:
    progress = []
    for ev in record.progress:
        extra = {
            k: v
            for k, v in ev.items()
            if k not in ("kind", "stage", "text", "round", "ts")
        }
        progress.append(
            AdaptiveJobProgressEvent(
                kind=str(ev.get("kind") or ""),
                stage=str(ev.get("stage") or ""),
                text=str(ev.get("text") or ""),
                round=ev.get("round"),
                ts=ev.get("ts"),
                extra=extra,
            )
        )
    return AdaptiveJobStatusResponse(
        job_id=record.job_id,
        status=record.status,  # type: ignore[arg-type]
        stage=record.stage,  # type: ignore[arg-type]
        message=record.message,
        error=record.error,
        round=record.clarify_round,
        clarify_pending=record.status == "clarifying",
        last_clarifier_reply=record.last_clarifier_reply,
        progress=progress,
        created_at=record.created_at,
        updated_at=record.updated_at,
        user_id=record.user_id,
        project_id=record.project_id,
        options=_options_model(record.options),
    )


def _to_results(record: JobRecord) -> AdaptiveJobResultsResponse:
    return AdaptiveJobResultsResponse(
        job_id=record.job_id,
        status=record.status,  # type: ignore[arg-type]
        stage=record.stage,  # type: ignore[arg-type]
        abstract=record.abstract,
        report=record.report,
        method=record.method,
        final_metrics=record.final_metrics,
        tuning_log=list(record.tuning_log or []),
        tuning_best=record.tuning_best,
        system_spec=record.system_spec,
        clarification_record=list(record.clarification_record or []),
        usage=record.usage,
        series=record.series,
        error=record.error,
    )


def _clarification_record_from_log(chat_log: list[dict[str, str]]) -> list[dict[str, Any]]:
    record: list[dict[str, Any]] = []
    i, idx, n = 0, 0, len(chat_log)
    while i < n:
        turn = chat_log[i]
        if turn.get("role") != "assistant":
            i += 1
            continue
        question = turn.get("text") or ""
        answer = ""
        if i + 1 < n and chat_log[i + 1].get("role") == "user":
            answer = chat_log[i + 1].get("text") or ""
        idx += 1
        record.append(
            {
                "id": f"uncertainty-{idx}",
                "category": "uncertainty_split",
                "question": question,
                "answer_label": answer,
                "answer_value": answer,
                "answered": bool(answer),
                "default_label": "",
                "source": "user",
                "evidence": "",
            }
        )
        i += 2
    return record


def _make_on_event(job_id: str, store: InMemoryJobStore) -> Callable[[dict[str, Any]], None]:
    def on_event(fields: dict[str, Any]) -> None:
        store.append_progress(job_id, dict(fields))
        kind = fields.get("kind")
        stage = str(fields.get("stage") or "")
        # Map pipeline stages onto job status for poll clients.
        if kind == "stage_start":
            if stage == "design":
                store.update(job_id, status="designing", stage="design", message="Design in progress")
            elif stage == "build":
                store.update(job_id, status="building", stage="build", message="Building control law / simulating")
            elif stage in ("tune", "tuning"):
                store.update(job_id, status="tuning", stage="tune", message="Tuning in progress")
            elif stage == "clarify":
                store.update(job_id, status="clarifying", stage="clarify")
        elif kind == "cancelled":
            store.update(
                job_id,
                status="cancelled",
                stage="error",
                message=str(fields.get("reasoning") or "Cancelled"),
            )

    return on_event


def _run_pipeline_thread(job_id: str, store: InMemoryJobStore) -> None:
    # Prefer non-interactive matplotlib for HTTP workers.
    import os as _os
    _os.environ.setdefault("LABCD_ADAPTIVE_SHOW_PLOTS", "0")
    _os.environ.setdefault("MPLBACKEND", "Agg")
    record = store.get(job_id)
    if record is None:
        return
    if record.cancel_requested:
        store.update(job_id, status="cancelled", stage="error", message="Cancelled before design")
        return

    options = record.options or {}
    spec = record.system_spec
    if spec:
        spec = system_spec_mod.normalize_defaults(spec)

    on_event = _make_on_event(job_id, store)

    def should_stop() -> bool:
        return store.is_cancel_requested(job_id)

    store.update(
        job_id,
        status="designing",
        stage="design",
        message="Starting design pipeline",
        error=None,
    )

    try:
        sim_overrides = None
        if spec and spec.get("dynamics", {}).get("states"):
            sim_overrides = clarifier.sim_overrides_from_spec(spec)

        result, usage, tuning_log, tuning_best = run_full_pipeline(
            options.get("description") or "",
            enable_tuning=bool(options.get("enable_tuning")),
            target_rms_frac=float(options.get("target_rms_frac") or 0.02),
            max_tuning_rounds=int(options.get("max_tuning_rounds") or 4),
            on_event=on_event,
            should_stop=should_stop,
            clarification_record=record.clarification_record or None,
            sim_overrides=sim_overrides,
            clarifier_usage=record.clarifier_usage or None,
            system_spec=spec if spec and (spec.get("dynamics") or {}).get("states") else None,
        )

        if store.is_cancel_requested(job_id):
            store.update(job_id, status="cancelled", stage="error", message="Cancelled")
            return

        report = None
        abstract = None
        method = None
        final_metrics = None
        if isinstance(result, dict):
            messages = result.get("messages") or []
            if messages:
                last = messages[-1]
                report = getattr(last, "content", None) or (
                    last.get("content") if isinstance(last, dict) else str(last)
                )
            abstract = result.get("abstract")
            final_metrics = result.get("final_metrics")
            # Best-effort method from tuning / progress
            for ev in reversed(store.get(job_id).progress if store.get(job_id) else []):
                args = ev.get("args") if isinstance(ev, dict) else None
                if isinstance(args, dict) and args.get("method"):
                    method = str(args["method"])
                    break

        series = result.get("series") if isinstance(result, dict) else None
        store.update(
            job_id,
            status="completed",
            stage="done",
            message="Design completed",
            report=report,
            abstract=abstract,
            method=method,
            final_metrics=final_metrics,
            tuning_log=list(tuning_log or []),
            tuning_best=tuning_best,
            usage=usage,
            system_spec=spec,
            series=series if isinstance(series, dict) else None,
        )
    except Exception as exc:  # noqa: BLE001
        store.update(
            job_id,
            status="failed",
            stage="error",
            message="Pipeline failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def _start_pipeline_async(job_id: str, store: InMemoryJobStore) -> None:
    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(job_id, store),
        daemon=True,
        name=f"adaptive-job-{job_id}",
    )
    thread.start()


def submit_job(
    request: AdaptiveJobCreateRequest,
    *,
    store: InMemoryJobStore | None = None,
) -> AdaptiveJobCreateResponse:
    """Create a job. Starts clarifier or skips straight to the design pipeline."""
    job_store = _store(store)
    options = _options_dict(request.options)
    record = job_store.create(
        system_spec=request.system_spec,
        options=options,
        user_id=request.user_id,
        project_id=request.project_id,
    )
    job_id = record.job_id

    if options.get("skip_clarify"):
        job_store.update(
            job_id,
            status="designing",
            stage="design",
            message="Clarifier skipped; starting design",
        )
        _start_pipeline_async(job_id, job_store)
        return AdaptiveJobCreateResponse(
            job_id=job_id,
            status="designing",
            stage="design",
            message="Job started (clarify skipped)",
        )

    # Interactive clarification: seed messages and run first model turn.
    spec = request.system_spec or {}
    if spec:
        spec = system_spec_mod.normalize_defaults(spec)
        job_store.update(job_id, system_spec=spec)

    messages = clarifier.start_conversation(spec if spec else {"dynamics": {}})
    job_store.update(
        job_id,
        status="clarifying",
        stage="clarify",
        message="Clarifier started",
        clarify_messages=messages,
        clarify_round=0,
        clarifier_usage=clarifier._empty_usage(),
    )
    # First clarifier turn (ask initial questions) without a user answer yet.
    _run_first_clarify_turn(job_id, job_store)
    latest = job_store.get(job_id)
    assert latest is not None
    return AdaptiveJobCreateResponse(
        job_id=job_id,
        status=latest.status,  # type: ignore[arg-type]
        stage=latest.stage,  # type: ignore[arg-type]
        message=latest.message or "Job created",
    )


def _run_first_clarify_turn(job_id: str, store: InMemoryJobStore) -> None:
    record = store.get(job_id)
    if record is None:
        return
    round_num = record.clarify_round + 1
    on_event = _make_on_event(job_id, store)
    status, reply, dynamics, usage, error, updated = clarifier.run_clarifier_turn(
        record.clarify_messages,
        on_event=on_event,
        round_num=round_num,
        force_finish=False,
    )
    _apply_clarify_result(
        job_id,
        store,
        status=status,
        reply=reply,
        dynamics=dynamics,
        usage=usage,
        error=error,
        updated_messages=updated,
        round_num=round_num,
        user_text=None,
    )


def _apply_clarify_result(
    job_id: str,
    store: InMemoryJobStore,
    *,
    status: str,
    reply: str,
    dynamics: dict[str, Any] | None,
    usage: dict[str, Any],
    error: str,
    updated_messages: list[dict[str, str]],
    round_num: int,
    user_text: str | None,
) -> None:
    record = store.get(job_id)
    if record is None:
        return

    chat_log = list(record.clarify_chat_log)
    if user_text is not None:
        chat_log.append({"role": "user", "text": user_text})
    if reply:
        chat_log.append({"role": "assistant", "text": reply})

    merged_usage = clarifier._sum_usage(record.clarifier_usage or clarifier._empty_usage(), usage)

    if status == "error":
        store.update(
            job_id,
            status="failed",
            stage="error",
            message="Clarifier error",
            error=error or reply,
            clarify_messages=updated_messages,
            clarify_chat_log=chat_log,
            clarify_round=round_num,
            last_clarifier_reply=reply,
            clarifier_usage=merged_usage,
        )
        return

    if status == "complete":
        spec = dict(record.system_spec or {})
        dyn = dict(spec.get("dynamics") or {})
        if dynamics:
            dyn["uncertainty"] = dynamics.get("uncertainty") or []
            dyn["disturbance"] = dynamics.get("disturbance") or []
            if dynamics.get("references") is not None:
                dyn["references"] = dynamics["references"]
        spec["dynamics"] = dyn
        spec = system_spec_mod.normalize_defaults(spec)
        clarification_record = _clarification_record_from_log(chat_log)
        store.update(
            job_id,
            system_spec=spec,
            clarify_messages=updated_messages,
            clarify_chat_log=chat_log,
            clarify_round=round_num,
            last_clarifier_reply=reply,
            clarification_record=clarification_record,
            clarifier_usage=merged_usage,
            status="designing",
            stage="design",
            message="Clarifier complete; starting design",
        )
        _start_pipeline_async(job_id, store)
        return

    # continue — wait for next answer
    store.update(
        job_id,
        status="clarifying",
        stage="clarify",
        message="Awaiting clarification answer",
        clarify_messages=updated_messages,
        clarify_chat_log=chat_log,
        clarify_round=round_num,
        last_clarifier_reply=reply,
        clarifier_usage=merged_usage,
    )


def clarify_job(
    job_id: str,
    request: AdaptiveClarifyRequest,
    *,
    store: InMemoryJobStore | None = None,
) -> AdaptiveClarifyResponse:
    job_store = _store(store)
    record = job_store.get(job_id)
    if record is None:
        raise KeyError(job_id)
    if record.status not in ("clarifying", "queued"):
        return AdaptiveClarifyResponse(
            job_id=job_id,
            status=record.status,  # type: ignore[arg-type]
            stage=record.stage,  # type: ignore[arg-type]
            clarifier_status="error",
            reply=f"Job is not awaiting clarification (status={record.status})",
            round=record.clarify_round,
        )

    messages = list(record.clarify_messages)
    user_text = (request.answer or "").strip()
    if user_text and not request.force_finish:
        messages.append({"role": "user", "content": user_text})

    round_num = record.clarify_round + 1
    force_finish = bool(request.force_finish) or round_num > clarifier.MAX_CLARIFY_TURNS
    on_event = _make_on_event(job_id, job_store)

    status, reply, dynamics, usage, error, updated = clarifier.run_clarifier_turn(
        messages,
        on_event=on_event,
        round_num=round_num,
        force_finish=force_finish,
    )
    _apply_clarify_result(
        job_id,
        job_store,
        status=status,
        reply=reply,
        dynamics=dynamics,
        usage=usage,
        error=error,
        updated_messages=updated,
        round_num=round_num,
        user_text=user_text or None,
    )
    latest = job_store.get(job_id)
    assert latest is not None
    clarifier_status: str
    if status == "complete":
        clarifier_status = "complete"
    elif status == "error":
        clarifier_status = "error"
    else:
        clarifier_status = "continue"
    return AdaptiveClarifyResponse(
        job_id=job_id,
        status=latest.status,  # type: ignore[arg-type]
        stage=latest.stage,  # type: ignore[arg-type]
        clarifier_status=clarifier_status,  # type: ignore[arg-type]
        reply=reply or "",
        round=latest.clarify_round,
    )


def cancel_job(
    job_id: str,
    *,
    store: InMemoryJobStore | None = None,
) -> AdaptiveJobStatusResponse:
    job_store = _store(store)
    record = job_store.get(job_id)
    if record is None:
        raise KeyError(job_id)
    if record.status in ("completed", "failed", "cancelled"):
        return _to_status_response(record)
    job_store.request_cancel(job_id)
    # If still clarifying / queued, mark cancelled immediately (no running pipeline).
    if record.status in ("clarifying", "queued"):
        updated = job_store.update(
            job_id,
            status="cancelled",
            stage="error",
            message="Cancelled by client",
        )
        assert updated is not None
        return _to_status_response(updated)
    # Pipeline thread will observe cancel_requested via should_stop.
    updated = job_store.update(job_id, message="Cancel requested")
    assert updated is not None
    return _to_status_response(updated)


def get_job(
    job_id: str,
    *,
    store: InMemoryJobStore | None = None,
) -> AdaptiveJobStatusResponse:
    record = _store(store).get(job_id)
    if record is None:
        raise KeyError(job_id)
    return _to_status_response(record)


def get_results(
    job_id: str,
    *,
    store: InMemoryJobStore | None = None,
) -> AdaptiveJobResultsResponse:
    record = _store(store).get(job_id)
    if record is None:
        raise KeyError(job_id)
    return _to_results(record)


def list_jobs(
    user_id: int | None = None,
    *,
    store: InMemoryJobStore | None = None,
) -> list[AdaptiveJobSummary]:
    records = _store(store).list_jobs(user_id=user_id)
    return [
        AdaptiveJobSummary(
            job_id=r.job_id,
            status=r.status,  # type: ignore[arg-type]
            stage=r.stage,  # type: ignore[arg-type]
            system_name=_system_name(r.system_spec),
            created_at=r.created_at,
            updated_at=r.updated_at,
            user_id=r.user_id,
        )
        for r in records
    ]
