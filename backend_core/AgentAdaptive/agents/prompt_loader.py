import os
import yaml

# prompts/ lives inside AgentAdaptive/ next to agents/, so two dirname() calls
# climb from here (backend_core/AgentAdaptive/agents/) up to AgentAdaptive/.
_PROMPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")


def load_prompt(yaml_filename):
    path = os.path.join(_PROMPT_DIR, yaml_filename)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["prompt"]
