# Assignment: FastAPI Layer for AgentMPC

## Purpose

Expose AgentMPC as a first-class service within the LabCD platform through a FastAPI layer that integrates cleanly with the existing `backend_api` architecture of the LabCD Application.

The goal is to make MPC auto-tuning available to the React frontend, external clients, and future orchestration flows under the same conventions already used by other LabCD pipelines (job lifecycle, authentication, project context, and result persistence).

---

## Context

AgentMPC currently runs as a self-contained module with a Streamlit interface. The platform is moving toward a unified FastAPI backend and a React + Tailwind frontend. This assignment brings AgentMPC into that shared service layer so it can be invoked, monitored, and reviewed through the same patterns as the rest of the product.

The work must remain consistent with the structure and practices of the full repository:

https://github.com/labcd-dev/LabCD_Application/tree/master/backend_api

---

## Expected Outcomes

1. **Service entry point**  
   AgentMPC is reachable via the platform API under a clear, versioned route family. Callers can submit a tuning job, track its progress, and retrieve results without depending on the Streamlit application.

2. **Job-oriented lifecycle**  
   Long-running tuning runs are treated as asynchronous jobs. Status, intermediate progress, and final artefacts (metrics, reports, figures) are available through standard job endpoints and are associated with the authenticated user and project context.

3. **Consistent integration**  
   The new surface reuses existing platform capabilities for authentication, project membership, configuration, and result storage. It does not introduce parallel mechanisms that would diverge from other modules already living under `backend_api`.

4. **Clear separation of concerns**  
   HTTP handling, request/response contracts, and orchestration remain thin. Core MPC logic continues to live in `backend_core`; the API layer is responsible for accepting work, driving the module, and returning structured outcomes.

5. **Ready for the product frontend**  
   The resulting API is sufficient for the React application to offer AgentMPC as a peer capability alongside the existing design pipelines, with a comparable user experience for submission, monitoring, and review of results.

---

## Scope Boundaries

- This assignment covers the API and orchestration surface for AgentMPC.
- It does not require redesign of the underlying tuning algorithms or agent logic.
- It does not include the React UI itself; the API must simply be ready for that UI to consume.
- Alignment with the existing `backend_api` layout and conventions is mandatory so that future merge into the full LabCD Application repository is straightforward.

---

## Success Criteria

- AgentMPC can be driven end-to-end through the platform API.
- Jobs are trackable, results are retrievable, and access is governed by the same rules as other LabCD services.
- The implementation follows the organisational patterns already established under `backend_api` (module package, HTTP layer, shared infrastructure).
- A subsequent frontend or orchestration service can adopt AgentMPC without special-case integration work.

---

*This assignment defines the strategic intent and the required outcomes. Implementation choices are left to the assigned team, provided the outcomes above are achieved and consistency with the target repository is maintained.*
