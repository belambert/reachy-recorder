import pytest

from reachy_recorder.annotate import FixedTask, cycle_label, gaze_running, gaze_task

PANEL = {"enabled": True, "detector_ok": True, "locked": False, "label": ""}


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, "look around"),
        ({"locked": True, "label": "cat"}, "look at the cat"),
        ({"enabled": False}, None),
        ({"detector_ok": False, "locked": True, "label": "cat"}, None),
    ],
)
def test_gaze_task(overrides, expected):
    assert gaze_task(PANEL | overrides) == expected


def test_gaze_task_unreachable():
    assert gaze_task(None) is None


def test_fixed_task():
    assert FixedTask("wave").task() == "wave"


def test_gaze_running():
    assert gaze_running(PANEL)
    assert not gaze_running(PANEL | {"enabled": False})
    assert not gaze_running(None)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"phase": "moving", "cycle": 3}, (None, 3)),
        ({"phase": "holding", "cycle": 3}, (None, 3)),
        ({"phase": "scanning", "cycle": 3}, ("go", 3)),
        ({"phase": "tracking", "cycle": 4}, ("go", 4)),
        ({"phase": "tracking", "cycle": 4, "enabled": False}, (None, 0)),
    ],
)
def test_cycle_label(overrides, expected):
    assert cycle_label(PANEL | overrides, "go") == expected


def test_cycle_label_unreachable():
    assert cycle_label(None, "go") == (None, 0)
