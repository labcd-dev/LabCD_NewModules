MODEL_PRICING = {
    "gpt-5-nano":   {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    "gpt-4o-mini":  {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-4o":       {"input": 2.50, "cached_input": 1.25,  "output": 10.00},
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "gpt-5.4":      {"input": 2.50, "cached_input": 0.25,  "output": 15.00},
    "gpt-5.5":      {"input": 5.00, "cached_input": 0.50,  "output": 30.00},
}

AVAILABLE_MODELS = tuple(MODEL_PRICING)

# not a real model id, never send this to the API.
# streamlit_app.py swaps it back to "" before it reaches the pipeline
INHERIT_LABEL = "(same as Design Agent)"


def model_label(model):
    p = MODEL_PRICING.get(model)
    if p is None:
        return model
    return "%s: $%s in / $%s out per 1M" % (
        model, _trim(p["input"]), _trim(p["output"]))


def _trim(value):
    return ("%.3f" % value).rstrip("0").rstrip(".")


def cost_for(model, input_tokens=0, output_tokens=0, cached_input_tokens=0):
    p = MODEL_PRICING.get(model)
    if p is None:
        return None
    # cached tokens are counted inside input_tokens too, so pull them out
    # first, otherwise they get billed twice
    cached = max(0, int(cached_input_tokens or 0))
    fresh = max(0, int(input_tokens or 0) - cached)
    return (fresh * p["input"]
            + cached * p["cached_input"]
            + max(0, int(output_tokens or 0)) * p["output"]) / 1_000_000.0


def cost_for_usage(model, usage):
    usage = usage or {}
    return cost_for(model,
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                    usage.get("cached_input_tokens", 0))


def format_cost(usd):
    # runs cost single-digit cents, so plain 2-decimal rounding would just
    # print "$0.00" every time: small amounts need the extra precision
    if usd is None:
        return "n/a"
    if usd == 0:
        return "$0"
    if usd < 0.0001:
        return "<$0.0001"
    if usd < 1:
        return "$%.4f" % usd
    return "$%.2f" % usd


def format_cost_plain(usd):
    # LaTeX treats "$" as the math-mode delimiter, so tables in the PDF use
    # this instead and put the currency unit in the column header
    return format_cost(usd).replace("$", "")


def run_cost_rows(usage):
    usage = usage or {}
    models = usage.get("models") or {}
    rows, total, complete = [], 0.0, True
    for actor, label in (("clarifier", "Clarifier Agent"),
                         ("agent", "Design Agent"),
                         ("tuner", "Tuner Agent"),
                         ("reporter", "Report Writer")):
        u = usage.get(actor) or {}
        if not u.get("total_tokens"):
            continue
        model = models.get(actor) or ""
        cost = cost_for_usage(model, u)
        if cost is None:
            # an unpriced model turns the whole total into "n/a" rather than
            # quietly understating the bill
            complete = False
        else:
            total += cost
        rows.append({
            "actor": actor, "label": label, "model": model or "unknown",
            "input_tokens": u.get("input_tokens", 0),
            "cached_input_tokens": u.get("cached_input_tokens", 0),
            "output_tokens": u.get("output_tokens", 0),
            "total_tokens": u.get("total_tokens", 0),
            "cost": cost,
        })
    return rows, (total if complete else None)


def format_run_cost_report(usage):
    rows, total = run_cost_rows(usage)
    if not rows:
        return ""
    out = ["", "=== token cost for this run ===",
           "%-16s %-14s %10s %10s %10s %12s"
           % ("agent", "model", "input", "cached", "output", "cost")]
    for r in rows:
        out.append("%-16s %-14s %10s %10s %10s %12s" % (
            r["label"], r["model"], format(r["input_tokens"], ","),
            format(r["cached_input_tokens"], ","),
            format(r["output_tokens"], ","), format_cost(r["cost"])))
    out.append("%-16s %-14s %10s %10s %10s %12s"
               % ("TOTAL", "", "", "", "", format_cost(total)))
    out.append("=" * 31)
    return "\n".join(out)
