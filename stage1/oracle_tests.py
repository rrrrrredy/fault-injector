"""Project-authored held-out assertions. This file is never given to the actor."""


def test_termcolor_heldout():
    from termcolor import colored

    assert colored(7, (1, 2, 3), force_color=True) == "\x1b[38;2;1;2;3m7\x1b[0m"
    assert colored("quiet", "blue", no_color=True, force_color=True) == "quiet"


def test_ordered_heldout():
    from ordered_set import OrderedSet

    queue = OrderedSet(["c", "b", "a", "d"])
    queue.discard("b")
    assert queue.index(["d", "a"]) == [2, 1]
    assert list(queue.intersection(["d", "c"])) == ["c", "d"]


def test_time_heldout():
    from pytimeparse import parse

    assert parse("1:30") == 90
    assert parse("1:30", granularity="minutes") == 5400
    assert parse("- 1 minute") == -60
    assert parse("nonsense") is None


def test_blinker_heldout():
    from blinker import Signal

    signal = Signal()
    seen = []
    def receive(sender):
        seen.append(sender)
    with signal.connected_to(receive):
        with signal.muted():
            signal.send("muted")
        signal.send("connected")
    signal.send("disconnected")
    assert seen == ["connected"]


def test_addict_heldout():
    from addict import Dict

    record = Dict({"queue": [{"id": 1}], "limits": {"soft": 3}})
    record.update({"queue": [{"id": 2}], "limits": {"hard": 7}})
    assert record.queue[0].id == 2
    assert record.limits.to_dict() == {"soft": 3, "hard": 7}
    record.freeze()
    try:
        record.limits.extra = 8
    except KeyError:
        pass
    else:
        raise AssertionError("frozen nested keys must reject new assignments")


def test_docopt_heldout():
    from docopt import DocoptExit, docopt

    usage = "Usage: ship (--air | --sea) [--] <parcel>"
    result = docopt(usage, ["--sea", "--", "-fragile"])
    assert (result["--air"], result["--sea"], result["<parcel>"]) == (False, True, "-fragile")
    try:
        docopt(usage, ["--sea", "--air", "box"])
    except DocoptExit:
        pass
    else:
        raise AssertionError("mutually exclusive transport modes must be rejected")
