"""Project-authored public black-box tests. Only the selected functions enter a slice."""


def test_termcolor_visible():
    from termcolor import colored

    assert colored("ready", "red", force_color=True) == "\x1b[31mready\x1b[0m"
    assert colored("ready", "red", no_color=True) == "ready"


def test_ordered_visible():
    from ordered_set import OrderedSet

    queue = OrderedSet(["b", "a", "b"])
    assert queue.add("a") == 1
    assert list(queue.union(["c", "a"])) == ["b", "a", "c"]


def test_time_visible():
    from pytimeparse import parse

    assert parse("1 minute, 24 seconds") == 84
    assert parse("1.5 minutes") == 90


def test_blinker_visible():
    from blinker import Signal

    signal = Signal()
    seen = []
    def receive(sender, **data):
        seen.append((sender, data["value"]))
    signal.connect(receive, sender="inbox")
    signal.send("other", value=1)
    signal.send("inbox", value=2)
    assert seen == [("inbox", 2)]


def test_addict_visible():
    from addict import Dict

    record = Dict({"service": {"host": "local", "port": 80}})
    record.update({"service": {"port": 443}})
    assert record.service.host == "local"
    assert record["service"].port == 443


def test_docopt_visible():
    from docopt import docopt

    usage = "Usage: ship [--fast] <parcel>..."
    options = docopt(usage, ["--fast", "box", "letter"])
    assert options["--fast"] is True
    assert options["<parcel>"] == ["box", "letter"]
