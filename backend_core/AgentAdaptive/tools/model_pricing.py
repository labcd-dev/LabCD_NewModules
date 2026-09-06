from labcd_agents.pricing import CostCalculator, ModelPrice

AVAILABLE_MODELS = ("gpt-5-nano", "gpt-4o-mini", "gpt-4o", "gpt-5.4-mini", "gpt-5.4", "gpt-5.5")

INHERIT_LABEL = "(same as Design Agent)"

_CALCULATOR = CostCalculator(overrides={
    "gpt-4o": ModelPrice(2.50, 10.00, cached_input_per_million=1.25),
})


def _trim(value):
    return ("%.3f" % value).rstrip("0").rstrip(".")


def model_label(model):
    price = _CALCULATOR.resolve_price(model)
    if price is None:
        return model
    return "%s: $%s in / $%s out per 1M" % (
        model, _trim(price.input_per_million), _trim(price.output_per_million))


def cost_for_usage(model, usage):
    usage = usage or {}
    if _CALCULATOR.resolve_price(model) is None:
        return None
    return _CALCULATOR.compute_cost(model,
                                    usage.get("input_tokens", 0),
                                    usage.get("output_tokens", 0),
                                    usage.get("cached_input_tokens", 0))


def format_cost(usd):
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
