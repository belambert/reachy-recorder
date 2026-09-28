import pytest

from reachy_recorder.annotate import FixedTask, gaze_task

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
