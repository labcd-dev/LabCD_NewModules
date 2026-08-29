"""Plant-model chat HTTP service adapter.

Adapted from LabCD_Application ``backend_api/http/services/plant_model_service.py``.
Drives ``backend_core.AgentPlant.PlantModelAgent`` and returns Application-compatible
response shapes.
"""

from __future__ import annotations

from typing import Literal

from backend_core.AgentPlant import (
    PlantModelAgent,
    PlantModelSessionState as AgentSessionState,
    apply_session_state,
    export_session_state,
)
from backend_api.AgentPlant.schemas import (
    PlantModelChatRequest,
    PlantModelChatResponse,
    PlantModelResult,
    PlantModelSessionStateOut,
    TokenUsageOut,
)


def _to_agent_session_state(
    state: PlantModelSessionStateOut | None,
) -> AgentSessionState | None:
    if state is None:
        return None
    latest = None
    if state.latest_draft is not None:
        latest = {
            "system_name": state.latest_draft.system_name,
            "python_code": state.latest_draft.python_code,
        }
    return AgentSessionState(draft_count=state.draft_count, latest_draft=latest)


def _from_agent_session_state(state: AgentSessionState) -> PlantModelSessionStateOut:
    latest = None
    if state.latest_draft is not None:
        latest = PlantModelResult(
            system_name=state.latest_draft["system_name"],
            python_code=state.latest_draft["python_code"],
        )
    return PlantModelSessionStateOut(draft_count=state.draft_count, latest_draft=latest)


def _infer_status(
    *,
    prev_draft_count: int,
    draft_count: int,
    final_result: dict | None,
) -> Literal["continue", "draft", "complete"]:
    if final_result is not None:
        return "complete"
    if draft_count > prev_draft_count:
        return "draft"
    return "continue"


def run_plant_model_chat(request: PlantModelChatRequest) -> PlantModelChatResponse:
    """Run one plant-model turn and return a structured response."""
    agent = PlantModelAgent(
        model=request.model,
        max_drafts=request.max_drafts,
        min_user_turns_before_completion=request.min_user_turns_before_completion,
    )
    apply_session_state(agent, _to_agent_session_state(request.session_state))
    prev_draft_count = agent._draft_count

    history = [{"role": m.role, "content": m.content} for m in request.messages]
    reply, final_payload = agent.step(history, request.user_message.strip())

    session_state = _from_agent_session_state(export_session_state(agent))
    status = _infer_status(
        prev_draft_count=prev_draft_count,
        draft_count=agent._draft_count,
        final_result=final_payload,
    )

    final_result = None
    if final_payload is not None:
        final_result = PlantModelResult(
            system_name=final_payload["system_name"],
            python_code=final_payload["python_code"],
        )

    usage_totals = agent.total_usage
    usage = TokenUsageOut(
        input_tokens=usage_totals.input_tokens,
        output_tokens=usage_totals.output_tokens,
        estimated_cost=agent.total_cost,
    )

    return PlantModelChatResponse(
        reply=reply,
        status=status,
        final_result=final_result,
        session_state=session_state,
        usage=usage,
        conversation_id=request.conversation_id,
    )
