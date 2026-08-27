import pytest

from labcd_agents.exceptions import PromptNotFoundError
from labcd_agents.prompts import PromptLibrary


@pytest.fixture
def prompt_dir(tmp_path):
    (tmp_path / "greeting.yaml").write_text(
        "hello_template: 'Hello, {name}!'\nschema:\n  type: object\n"
    )
    (tmp_path / "farewell.yml").write_text("bye_template: 'Bye, {name}.'\n")
    return tmp_path


def test_loads_all_yaml_files(prompt_dir):
    lib = PromptLibrary(str(prompt_dir))
    assert set(lib.names) == {"greeting", "farewell"}


def test_get_and_get_key(prompt_dir):
    lib = PromptLibrary(str(prompt_dir))
    assert lib.get("greeting")["hello_template"] == "Hello, {name}!"
    assert lib.get_key("greeting", "schema") == {"type": "object"}


def test_format(prompt_dir):
    lib = PromptLibrary(str(prompt_dir))
    assert lib.format("greeting", "hello_template", name="Ada") == "Hello, Ada!"


def test_missing_prompt_raises(prompt_dir):
    lib = PromptLibrary(str(prompt_dir))
    with pytest.raises(PromptNotFoundError):
        lib.get("does_not_exist")


def test_missing_key_raises(prompt_dir):
    lib = PromptLibrary(str(prompt_dir))
    with pytest.raises(PromptNotFoundError):
        lib.get_key("greeting", "no_such_key")


def test_contains(prompt_dir):
    lib = PromptLibrary(str(prompt_dir))
    assert "greeting" in lib
    assert "nope" not in lib
