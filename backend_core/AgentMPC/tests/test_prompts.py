"""
Run with: pytest backend_core/AgentMPC/tests/test_prompts.py -v

Guards the YAML prompt layout (backend_core/AgentMPC/prompts/*.yaml), which
replaced the string constants that used to live in agents/*.py.

None of these tests need an API key: they check that every prompt file is
present, that every template still renders with exactly the placeholders its
agent supplies, and that the two names the Streamlit UI imports directly are
still importable. Together those cover the ways this refactor can break
silently -- a renamed key, a placeholder typo'd in YAML, or an unescaped brace
-- none of which would surface until an actual tuning run made an LLM call.
"""

from string import Formatter

import pytest

from backend_core.AgentMPC.agents import (
    actor,
    animation_agent,
    config_advisor_agent,
    critic,
    diagnostics_agent,
    dynamics_validator,
    juror,
    report_agent,
    scenarist,
    terminator,
    trajectory_validator,
)
from backend_core.AgentMPC.agents.prompt_library import (
    PROMPTS_DIR,
    get_prompt,
    get_prompt_library,
    reload_prompts,
)

# (yaml file stem, key) for every prompt the package loads.
ALL_PROMPTS = [
    ("actor", "prompt"),
    ("critic", "prompt"),
    ("terminator", "prompt"),
    ("juror", "prompt"),
    ("scenarist", "prompt"),
    ("report_agent", "prompt"),
    ("animation_agent", "prompt"),
    ("diagnostics_agent", "prompt"),
    ("config_advisor_agent", "chat_system_prompt"),
    ("config_advisor_agent", "suggest_system_prompt"),
    ("dynamics_validator", "standard"),
    ("dynamics_validator", "fix_prompt"),
    ("trajectory_validator", "standard"),
    ("trajectory_validator", "fix_prompt"),
]

# Every PromptTemplate built from YAML. Rendering each one is the real check:
# PromptTemplate.format() raises on a placeholder the template needs but the
# caller doesn't pass, and on a stray brace the YAML introduced.
TEMPLATES = [
    ("actor", actor.actor_prompt),
    ("critic", critic.critic_prompt),
    ("terminator", terminator.terminator_prompt),
    ("juror", juror.juror_prompt),
    ("scenarist", scenarist.scenarist_prompt),
    ("report_agent", report_agent.report_prompt),
    ("animation_agent", animation_agent.animation_prompt),
    ("diagnostics_agent", diagnostics_agent._diagnostics_prompt),
]


def test_prompts_directory_exists():
    assert PROMPTS_DIR.is_dir(), f"missing prompt directory: {PROMPTS_DIR}"


@pytest.mark.parametrize("name,key", ALL_PROMPTS)
def test_every_prompt_is_present_and_non_empty(name, key):
    text = get_prompt(name, key)
    assert isinstance(text, str)
    assert text.strip(), f"{name}.yaml:{key} is empty"
    # The Python constants were all .strip()ed; `|-` block scalars must match,
    # or every prompt picks up a stray trailing newline.
    assert text == text.strip(), f"{name}.yaml:{key} has leading/trailing whitespace"


def _dummy_values(template):
    """One stand-in value per declared variable, respecting format specs.

    The Actor prompt renders its exploration bounds as ``{intensity_low:.2f}``,
    so a few fields need a number rather than a string. Everything else gets a
    marker string so the render can be asserted against.
    """
    numeric = {
        field
        for _, field, spec, _ in Formatter().parse(template.template)
        if field and spec and spec[-1] in "eEfFgGn%"
    }
    return {v: (1.0 if v in numeric else f"<{v}>") for v in template.input_variables}


@pytest.mark.parametrize("name,template", TEMPLATES)
def test_template_renders_with_its_declared_variables(name, template):
    values = _dummy_values(template)
    rendered = template.format(**values)
    assert rendered
    for var, value in values.items():
        if isinstance(value, str):
            assert value in rendered, f"{name}: '{var}' declared but never used in YAML"


@pytest.mark.parametrize(
    "prompt_text",
    [
        config_advisor_agent._CHAT_SYSTEM_PROMPT,
        config_advisor_agent._SUGGEST_SYSTEM_PROMPT,
    ],
)
def test_plain_format_prompts_have_balanced_braces(prompt_text):
    assert prompt_text.count("{") == prompt_text.count("}")


def test_fix_prompts_embed_the_standard_and_the_caller_inputs():
    """The validators' repair prompts interpolate three things; a renamed YAML
    placeholder would drop one silently rather than raise."""
    for module, standard in [
        (dynamics_validator, dynamics_validator.DYNAMICS_STANDARD),
        (trajectory_validator, trajectory_validator.TRAJECTORY_STANDARD),
    ]:
        rendered = module._fix_prompt("SOURCE_SENTINEL", "ERROR_SENTINEL")
        assert standard in rendered
        assert "SOURCE_SENTINEL" in rendered
        assert "ERROR_SENTINEL" in rendered


def test_ui_facing_standards_are_still_module_level_names():
    """frontend_streamlit/agent_mpc_app.py imports these two names directly and
    renders them with st.markdown -- they are user documentation as well as
    prompt material, so moving the text to YAML must not move the name."""
    assert dynamics_validator.DYNAMICS_STANDARD.strip()
    assert trajectory_validator.TRAJECTORY_STANDARD.strip()


def test_reload_prompts_rebuilds_the_library():
    first = get_prompt_library()
    assert get_prompt_library() is first
    reload_prompts()
    assert get_prompt_library() is not first
