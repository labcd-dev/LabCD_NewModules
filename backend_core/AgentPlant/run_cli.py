"""LabCD Plant-Model Agent — CLI.

Run from repository root:

    PYTHONPATH=. python backend_core/AgentPlant/run_cli.py
    PYTHONPATH=. python backend_core/AgentPlant/run_cli.py --model gpt-4o
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from labcd_agents import ensure_env_loaded
    _env = _REPO_ROOT / ".env"
    ensure_env_loaded(str(_env) if _env.is_file() else None)
except ImportError:
    pass

from backend_core.AgentPlant.agent import PlantModelAgent

DEFAULT_MODEL = os.getenv("LABCD_DEMO_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = 0.0


def print_banner() -> None:
    print("=" * 60)
    print("🧪  LabCD Plant-Model Agent  —  CLI Mode")
    print("=" * 60)
    print("Type your system description, or:")
    print("  finish / done / looks good  →  accept the current draft")
    print("  reset                       →  start a new conversation")
    print("  quit / exit                 →  leave")
    print("-" * 60)


def save_result(result: dict, filename: str = "dynamics.py") -> None:
    code = result.get("python_code", "")
    name = result.get("system_name", "unnamed_system")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# {name}\n")
        f.write(code)
        f.write("\n")
    print(f"\n💾  Saved to ./{filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LabCD Plant-Model Agent CLI")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--provider", default=None, help="Provider override (optional)")
    parser.add_argument("--max-drafts", type=int, default=5, help="Auto-accept after N drafts")
    parser.add_argument("--min-turns", type=int, default=2, help="Min user turns before auto-complete")
    args = parser.parse_args()

    print_banner()

    try:
        agent = PlantModelAgent(
            model=args.model,
            temperature=args.temperature,
            provider=args.provider,
            max_drafts=args.max_drafts,
            min_user_turns_before_completion=args.min_turns,
        )
    except Exception as exc:
        print(f"\n❌  Could not initialize the agent: {exc}")
        sys.exit(1)

    history: list[dict[str, str]] = []
    final_result: dict | None = None

    while True:
        if final_result is not None:
            # Conversation is locked — only reset or quit makes sense
            user_input = input("\n[complete] reset or quit? > ").strip().lower()
            if user_input in {"reset"}:
                history.clear()
                final_result = None
                agent.reset_conversation_state()
                print("\n🔄  Conversation reset. Describe your new system.")
                continue
            if user_input in {"quit", "exit", "q"}:
                break
            print("  (already complete — type 'reset' for a new model or 'quit')")
            continue

        try:
            user_input = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            break

        if user_input.lower() in {"reset"}:
            history.clear()
            final_result = None
            agent.reset_conversation_state()
            print("\n🔄  Conversation reset.")
            continue

        # Run the turn
        history_before = list(history)
        history.append({"role": "user", "content": user_input})

        try:
            reply, result = agent.step(history_before, user_input)
        except Exception as exc:
            reply = f"⚠️  LLM call failed: {exc}"
            result = None

        history.append({"role": "assistant", "content": reply})

        print(f"\nAgent > {reply}")

        if result is not None:
            final_result = result
            print("\n" + "=" * 60)
            print(f"✅  Model complete — {final_result.get('system_name', 'Unnamed')}")
            print("=" * 60)
            print(final_result.get("python_code", ""))
            print("=" * 60)
            save_result(final_result)
            print("   Type 'reset' to design a new system, or 'quit' to exit.")


if __name__ == "__main__":
    main()