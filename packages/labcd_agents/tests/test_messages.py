from types import SimpleNamespace

from labcd_agents.messages import build_messages, extract_response_text


def test_build_messages_as_dicts():
    messages = build_messages("hi", system_prompt="be nice", as_dicts=True)
    assert messages == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
    ]


def test_build_messages_langchain_objects():
    messages = build_messages("hi", system_prompt="be nice")
    assert [type(m).__name__ for m in messages] == ["SystemMessage", "HumanMessage"]
    assert messages[0].content == "be nice"
    assert messages[1].content == "hi"


def test_build_messages_no_system_prompt():
    messages = build_messages("hi", as_dicts=True)
    assert messages == [{"role": "user", "content": "hi"}]


def test_build_messages_with_context():
    context = [{"role": "assistant", "content": "prior turn"}]
    messages = build_messages("hi", as_dicts=True, context_messages=context)
    assert messages[-1] == {"role": "assistant", "content": "prior turn"}


def test_extract_response_text_from_content():
    response = SimpleNamespace(content="plain text")
    assert extract_response_text(response) == "plain text"


def test_extract_response_text_from_output_text():
    response = SimpleNamespace(output_text="from responses api")
    assert extract_response_text(response) == "from responses api"


def test_extract_response_text_from_output_items():
    part = SimpleNamespace(text="chunk one")
    item = SimpleNamespace(type="message", content=[part])
    response = SimpleNamespace(output_text=None, output=[item])
    assert extract_response_text(response) == "chunk one"


def test_extract_response_text_fallback_str():
    assert extract_response_text(12345) == "12345"
