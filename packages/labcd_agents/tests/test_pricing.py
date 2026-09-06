from labcd_agents.pricing import CostCalculator, ModelPrice


def test_resolve_exact_match():
    calc = CostCalculator()
    price = calc.resolve_price("gpt-4o-mini")
    assert price == ModelPrice(0.15, 0.60, cached_input_per_million=0.075)


def test_resolve_strips_vendor_prefix():
    calc = CostCalculator()
    price = calc.resolve_price("openai/gpt-oss-120b")
    assert price is not None
    assert price.input_per_million == 0.25


def test_resolve_llama_regex_fallback():
    calc = CostCalculator()
    # Not an exact key, but should fall back to the "llama-4" base entry.
    price = calc.resolve_price("llama-4-scout-17b-16e-instruct")
    assert price == ModelPrice(0.60, 0.80)


def test_resolve_maverick_regex_fallback():
    calc = CostCalculator()
    price = calc.resolve_price("meta/llama-4-maverick-17b-128e-instruct")
    assert price == ModelPrice(0.50, 0.64)


def test_resolve_unknown_model_returns_none():
    calc = CostCalculator()
    assert calc.resolve_price("some-totally-unknown-model") is None


def test_compute_cost_known_model():
    calc = CostCalculator()
    cost = calc.compute_cost("gpt-4o", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 3.0 + 10.0


def test_compute_cost_unknown_model_is_zero():
    calc = CostCalculator()
    assert calc.compute_cost("mystery-model", 1000, 1000) == 0.0


def test_register_overrides_price():
    calc = CostCalculator()
    calc.register("my-custom-model", input_per_million=1.0, output_per_million=2.0)
    cost = calc.compute_cost("my-custom-model", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 3.0


def test_overrides_constructor_arg():
    calc = CostCalculator(overrides={"gpt-4o-mini": ModelPrice(9.0, 9.0)})
    price = calc.resolve_price("gpt-4o-mini")
    assert price == ModelPrice(9.0, 9.0)
