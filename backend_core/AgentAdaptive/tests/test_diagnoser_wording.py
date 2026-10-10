"""
Run from the repo root with:
    python -m pytest backend_core/AgentAdaptive/tests/test_diagnoser_wording.py -v
"""

from backend_core.AgentAdaptive.agents import diagnoser_agent as da
from backend_core.AgentAdaptive.tools import failure_mode as fm


def _ev(mode, **extra):
    ev = {"failure_mode": {"mode": mode, "chain": [], "ruled_out": []},
          "metrics": {"tracking_pct_headline": 75.25, "steady_rms_frac": [0.2475],
                      "success_target_frac": 0.02, "success_reason": "mse_target: x"},
          "plant": {"states": ["x1", "x2"], "inputs": ["u"]}}
    ev.update(extra)
    return ev


def _limits(frac=0.67, free=None):
    return {"limits": {"u": [-10.0, 10.0]},
            "saturation": {"any_input_fraction": frac, "first_saturated_at_s": 0.8},
            "without_limits": free}


def test_the_headline_for_a_tracking_miss_has_only_numbers_from_the_metrics():
    text = fm.headline(_ev("tracking_error_only"))
    assert text == ("The worst output follows its reference only 75% in steady state "
                    "(error 24.8% of the task scale, target 2.0%).")


def test_the_headline_for_each_limit_mode():
    ev = _ev("limit_too_small", input_limits=_limits())
    assert fm.headline(ev) == ("The input limit is too small: the command was clipped 67% of the run, "
                               "and the same design passes without the limit.")
    free = {"passes": False, "tracking_pct": 75.2, "failed_check": "mse_target", "peak_by_input": {"u": 12.0}}
    worse = _ev("limit_adds_to_failure", input_limits=_limits(0.6, free))
    worse["metrics"]["tracking_pct_headline"] = 0.0
    assert "tracking 0% with it, 75% without" in fm.headline(worse)


def test_the_headline_for_a_singular_reference_names_the_largest_request():
    ev = _ev("fails_even_without_limit", pre_run_warnings=["w"],
             control_spikes={"largest": [{"u": -627100.0}]})
    assert "singular" in fm.headline(ev) and "627100" in fm.headline(ev)
    plain = _ev("fails_even_without_limit", input_limits=_limits(
        free={"passes": False, "failed_check": "bounded", "peak_by_input": {}}))
    assert fm.headline(plain) == "The design fails even without the input limit (bounded)."


def test_no_headline_without_a_mode_so_the_models_stays():
    assert fm.headline({}) is None
    report = da._prune_for_evidence({"headline": "model words", "explanation": [], "suggestions": []}, {})
    assert report["headline"] == "model words"
    ev = _ev("tracking_error_only")
    assert da._prune_for_evidence({"headline": "model words", "explanation": [], "suggestions": []},
                                  ev)["headline"].startswith("The worst output")


def test_a_sentence_with_no_number_and_no_name_is_dropped():
    ev = _ev("tracking_error_only")
    report = {"explanation": ["The controller does not track the reference closely enough.",
                              "The control signal jumped far above its usual level.",
                              "x2 reaches 3.1 rad/s near the end."], "suggestions": []}
    out = da._prune_for_evidence(report, ev)["explanation"]
    assert out[-1] == "x2 reaches 3.1 rad/s near the end."
    assert not any("closely enough" in e or "usual level" in e for e in out)


def test_a_sentence_made_of_words_the_facts_already_use_is_dropped():
    chain = ["The input is limited to u in [-20, 20]; the plant clipped the command 62% of the run."]
    ev = _ev("limit_too_small", failure_mode={"mode": "limit_too_small", "chain": chain, "ruled_out": []})
    report = {"explanation": ["The input limit for 'u' was set to [-20, 20].",
                              "The command reached 66.76 at its peak.",
                              "The same design without the limits fails too, but the limits make it worse."],
              "suggestions": []}
    out = da._prune_for_evidence(report, ev)["explanation"]
    assert out == chain + ["The command reached 66.76 at its peak."]


def test_the_tuning_pointer_only_when_it_is_logical():
    off = _ev("tracking_error_only", tuning_ran=False)
    text = " ".join(da._prune_for_evidence({"explanation": [], "suggestions": []}, off)["explanation"])
    assert "Tuning was off for this run" in text
    ran = _ev("tracking_error_only", tuning_ran=True)
    assert "Tuning" not in " ".join(da._prune_for_evidence({"explanation": [], "suggestions": []}, ran)["explanation"])
    unknown = _ev("tracking_error_only")
    assert "Tuning" not in " ".join(da._prune_for_evidence({"explanation": [], "suggestions": []}, unknown)["explanation"])
    other = _ev("limit_too_small", tuning_ran=False, input_limits=_limits())
    assert "Tuning" not in " ".join(da._prune_for_evidence({"explanation": [], "suggestions": []}, other)["explanation"])
    blown = _ev("tracking_error_only", tuning_ran=False)
    blown["metrics"]["success_reason"] = "bounded: x"
    assert "Tuning" not in " ".join(da._prune_for_evidence({"explanation": [], "suggestions": []}, blown)["explanation"])


def test_a_sentence_that_repeats_the_headline_is_dropped():
    ev = _ev("tracking_error_only")
    report = {"explanation": ["The worst output follows its reference only 75 percent in steady state."],
              "suggestions": []}
    out = da._prune_for_evidence(report, ev)["explanation"]
    assert out == []


def test_the_tuning_pointer_also_for_a_limit_that_adds_to_a_failure_on_tracking():
    free = {"passes": False, "tracking_pct": 75.2, "failed_check": "mse_target", "peak_by_input": {"u": 12.0}}
    ev = _ev("limit_adds_to_failure", tuning_ran=False, input_limits=_limits(0.6, free))
    text = " ".join(da._prune_for_evidence({"explanation": [], "suggestions": []}, ev)["explanation"])
    assert "Tuning was off for this run" in text
    free["failed_check"] = "bounded"
    again = " ".join(da._prune_for_evidence({"explanation": [], "suggestions": []}, ev)["explanation"])
    assert "Tuning" not in again
