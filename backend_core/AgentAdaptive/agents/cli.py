import os

from .tuner_agent import run_full_pipeline, _format_tuning_diff

DESCRIPTION_FILE = "system_description.txt"


def load_description(path: str) -> str:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "# Describe your system here in plain language: the physics,\n"
                "# what you can measure, what you can actuate, the initial\n"
                "# condition, and what you want it to track. Lines starting\n"
                "# with # are ignored. Save this file, then re-run the script.\n"
            )
        raise SystemExit(
            "'%s' was not found, so I created an empty one in the project "
            "root. Open it, write your system description, save, and run "
            "python -m backend_core.AgentAdaptive.agents.cli again." % path
        )
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    text = "".join(lines).strip()
    if not text:
        raise SystemExit("'%s' is empty. Write your system description in it and re-run." % path)
    return text


if __name__ == "__main__":
    # only exists so the module still runs end-to-end. No system_spec here means
    # it just prints its own "no spec" failure; real designs go through streamlit_app.py instead.
    # Climb agents -> AgentAdaptive -> backend_core -> repo root.
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    description_path = os.path.join(project_root, DESCRIPTION_FILE)
    user_text = load_description(description_path)

    print("=== system description read from %s ===" % DESCRIPTION_FILE)
    print(user_text)
    print("=========================================\n")

    enable_tuning = os.environ.get("ENABLE_TUNING", "1") != "0"
    result, usage, tuning_log, tuning_best = run_full_pipeline(
        user_text, enable_tuning=enable_tuning)

    print("\n=== agent's final summary ===")
    print(result["messages"][-1].content)

    if tuning_log:
        print("\n=== tuning rounds ===")
        for entry in tuning_log:
            print("\n-- round %s --" % entry["round"])
            print("reasoning: %s" % entry["reasoning"])
            print("changed params: %s" % _format_tuning_diff(entry.get("changed", {})))
            print("met target: %s" % entry["met_target"])

    if tuning_best is not None:
        print("\n=== final chosen tuning (round %s) ===" % tuning_best["round"])
        for k, v in tuning_best["tuning"].items():
            print("  %s = %s" % (k, v))
