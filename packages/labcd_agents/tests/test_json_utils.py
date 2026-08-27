import json

import pytest

from labcd_agents.json_utils import extract_json_from_response, round_floats, strip_think_tags


def test_extract_plain_json():
    assert extract_json_from_response('{"a": 1}') == {"a": 1}


def test_extract_markdown_fenced_json():
    text = 'Here you go:\n```json\n{"a": 1, "b": 2}\n```\nThanks'
    assert extract_json_from_response(text) == {"a": 1, "b": 2}


def test_extract_json_with_think_tags():
    text = "<think>reasoning about it</think>\n```json\n{\"ok\": true}\n```"
    assert extract_json_from_response(text) == {"ok": True}


def test_extract_json_brace_fallback():
    text = 'Sure, the result is {"x": 42} as requested.'
    assert extract_json_from_response(text) == {"x": 42}


def test_extract_json_empty_raises():
    with pytest.raises(json.JSONDecodeError):
        extract_json_from_response("")


def test_extract_json_invalid_raises():
    with pytest.raises(json.JSONDecodeError):
        extract_json_from_response("not json at all, no braces")


def test_strip_think_tags():
    assert strip_think_tags("<think>hmm</think>rest") == "rest"


def test_round_floats_nested():
    data = {"a": 1.23456, "b": [1.1111, {"c": 2.98765}]}
    rounded = round_floats(data, decimals=2)
    assert rounded == {"a": 1.23, "b": [1.11, {"c": 2.99}]}
