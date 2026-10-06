"""Tries the plant-compatibility step on a real model (needs OPENAI_API_KEY in .env, a few cents a run).
--plant picks the test plant, --repeat N counts how often the fact check had to step in.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from labcd_agents import ensure_env_loaded  # noqa: E402
    _env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    ensure_env_loaded(_env if os.path.isfile(_env) else None)
except ImportError:
    pass

from backend_core.AgentAdaptive.agents import compat_agent  # noqa: E402

SPEC = {
    "status": "complete",
    "system_name": "Two-axis stage with a trim motor",
    "dynamics": {
        "states": ["x1", "x2", "x3", "x4"],
        "state_meanings": ["axis A position", "axis B position", "axis A velocity", "axis B velocity"],
        "state_equations": ["x3 + u1", "x4 + u2 + u3", "-x3 + u1", "-x4 + u3"],
        "inputs": ["u1", "u2", "u3"],
        "outputs": ["x1", "x2"],
        "x0": [0.0, 0.0, 0.0, 0.0],
        "references": [{"output": "x1", "expr": "0.1*sin(t)"}, {"output": "x2", "expr": "0.1"}],
        "parameters": {}, "uncertainty": [], "disturbance": [],
    },
}
UNICYCLE = {
    "status": "complete",
    "system_name": "Unicycle",
    "dynamics": {
        "states": ["x", "y", "th"], "state_meanings": ["x position", "y position", "heading"],
        "state_equations": ["v*cos(th)", "v*sin(th)", "om"],
        "inputs": ["v", "om"], "outputs": ["x", "y"], "x0": [0.0, 0.5, 0.0],
        "references": [{"output": "x", "expr": "0.5*sin(t)"}, {"output": "y", "expr": "0.5*cos(t)"}],
        "parameters": {}, "uncertainty": [], "disturbance": [],
    },
}
CONVERSATION = ("u1 and u2 are the main drive motors. u3 is a small trim motor that overheats "
                "if it works hard.")


def run_once(spec, conversation, quiet=False):
    say = (lambda *a, **k: None) if quiet else print
    events = []
    proposal = compat_agent.propose_conversions(
        spec, clarification_record=conversation, on_event=lambda e: events.append(e["text"]))
    say("status:", proposal["status"], "| source:", proposal["source"], "| model calls:", proposal["llm_calls"])
    say("\nprogress:")
    for line in events:
        say(" -", line)
    say("\n" + proposal["message"])
    for o in proposal["options"]:
        say("\n[%s] %s  params=%s" % (o["id"], o["title"], json.dumps(o["params"])))
        say("   rationale:", o["rationale"])
        say("   risks:    ", o["risks"])
    say("\ncritic:", proposal["critic"], "| llm_error:", proposal["llm_error"], "| notes:", proposal["notes"])
    if proposal["status"] != "needs_choice":
        return proposal, None
    explanation = compat_agent.explain(proposal, spec)
    say("\n--- explanation (%s) ---" % explanation["source"])
    say(explanation["headline"])
    for bullet in explanation["why_not_fit"]:
        say(" *", bullet)
    for oid, text in explanation["options"].items():
        say("  [%s] %s" % (oid, text))
    say(explanation["recommendation"])
    for r in explanation["replaced"]:
        say("replaced by the fact (%s): %s" % (r["field"], r["problem"]))
    if explanation["error"]:
        say("explainer error:", explanation["error"])
    total = proposal["usage"]["total_tokens"] + explanation["usage"]["total_tokens"]
    say("\ntokens used: %d" % total)
    return proposal, explanation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", choices=("three", "unicycle"), default="three",
                        help="three: 3 inputs 2 outputs; unicycle: singular decoupling matrix")
    parser.add_argument("--repeat", type=int, default=1, help="run N times and print only the totals")
    args = parser.parse_args()
    spec, conversation = (SPEC, CONVERSATION) if args.plant == "three" else (UNICYCLE, "")
    if args.repeat == 1:
        run_once(spec, conversation)
        return
    stats = {"runs": 0, "plan_texts_replaced": 0, "sentences_replaced": 0, "explainer_failed": 0, "tokens": 0}
    for i in range(args.repeat):
        proposal, explanation = run_once(spec, conversation, quiet=True)
        stats["runs"] += 1
        stats["plan_texts_replaced"] += sum("replaced by fixed text" in n for n in proposal["notes"])
        stats["tokens"] += proposal["usage"]["total_tokens"]
        if explanation:
            stats["sentences_replaced"] += len(explanation["replaced"])
            stats["explainer_failed"] += explanation["source"] != "llm"
            stats["tokens"] += explanation["usage"]["total_tokens"]
        print("run %d: replaced plan texts so far %d, replaced sentences so far %d"
              % (i + 1, stats["plan_texts_replaced"], stats["sentences_replaced"]))
    print("\n" + json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
