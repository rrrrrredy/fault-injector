from pathlib import Path
from runpy import run_path


def test_human_plural():
    pluralize = run_path(str(Path.cwd() / "inflection.py"))["pluralize"]
    print(f"person control: {pluralize('person')}")
    actual = pluralize("human")
    print(f"plural of human: {actual}")
    assert actual == "humans"
