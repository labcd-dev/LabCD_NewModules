from . import llm_factory
from .prompt_loader import load_prompt
from .agent_io import _extract_json_payload, _extract_usage, _empty_usage

ABSTRACT_SYSTEM_PROMPT = load_prompt("report_abstract_prompt.yaml")


def write_abstract(system_name, method_label, agents_used, outcome_text, why=""):
    user_content = (
        "System name: %s\n"
        "Control method chosen: %s\n"
        "Design Agent's reasoning for that choice: %s\n"
        "Agents/stages that ran for this design: %s\n"
        "Run outcome: %s\n"
        % (system_name or "(unnamed system)", method_label, why or "(not given)",
           ", ".join(agents_used) or "(none recorded)", outcome_text)
    )
    try:
        llm = llm_factory.build_llm("reporter")
        resp = llm.invoke([
            {"role": "system", "content": ABSTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ])
        usage = _extract_usage(resp)
        payload, _err = _extract_json_payload(resp.content)
        if not isinstance(payload, dict) or not payload.get("abstract"):
            return None, usage
        return str(payload["abstract"]).strip(), usage
    except Exception:
        # a missing abstract should cost the report one section, not blow up the whole build
        return None, _empty_usage()
