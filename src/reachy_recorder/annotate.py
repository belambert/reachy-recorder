"""Sources of the language instruction for each tick.

An annotator answers "what is the robot doing right now?" as an instruction,
or None when nothing should be recorded, plus a segment number. A change of
either ends an episode.
"""

from __future__ import annotations

import threading
from typing import Protocol

import requests


class Annotator(Protocol):
    """Anything that names the current task."""

    def task(self) -> str | None:
        """The instruction for this tick, or None to record nothing."""
        ...

    def segment(self) -> int:
        """Which stretch this tick belongs to; a change starts a new episode."""
        ...

    def close(self) -> None:
        """Release any background resources."""
        ...


class FixedTask:
    """The same instruction throughout, e.g. for a teleop session."""

    def __init__(self, task: str) -> None:
        """Label every tick with `task`."""
        self._task = task

    def task(self) -> str | None:
        """The fixed instruction."""
        return self._task

    def segment(self) -> int:
        """Always the one stretch."""
        return 0

    def close(self) -> None:
        """Nothing to release."""


def gaze_running(state: dict | None) -> bool:
    """Whether reachy-gaze is reachable, enabled, and its detector is up."""
    return state is not None and state["enabled"] and state["detector_ok"]


def gaze_task(state: dict | None) -> str | None:
    """Instruction for one reading of reachy-gaze's control panel."""
    if not gaze_running(state):
        return None
    assert state is not None
    return f"look at the {state['label']}" if state["locked"] else "look around"


def cycle_label(state: dict | None, task: str) -> tuple[str | None, int]:
    """`task` while reachy-gaze searches or tracks, and the cycle it's in.

    Each cycle opens with a scripted move to a random pose and a hold there;
    those phases are left out, since the camera can't explain them.
    """
    if not gaze_running(state):
        return None, 0
    assert state is not None
    looking = state["phase"] in ("scanning", "tracking")
    return (task if looking else None), state["cycle"]


class GazePanel:
    """Label ticks from what reachy-gaze reports it is doing.

    Polls on its own thread so a slow or unreachable panel never stalls the
    recording loop; a failed poll reads as "nothing to record".
    """

    def __init__(self, url: str, period: float = 0.25, timeout: float = 1.0) -> None:
        """Poll `url` (the panel's /state endpoint) every `period` seconds."""
        self.url = url
        self.period = period
        self.timeout = timeout
        self._state: dict | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def task(self) -> str | None:
        """The instruction from the latest poll."""
        return gaze_task(self._state)

    def segment(self) -> int:
        """Always the one stretch: task changes alone split episodes."""
        return 0

    def close(self) -> None:
        """Stop polling."""
        self._stop.set()
        self._thread.join(timeout=self.timeout + self.period)

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                r = requests.get(self.url, timeout=self.timeout)
                r.raise_for_status()
                state = r.json()
            except requests.RequestException:
                state = None
            self._state = state
            self._stop.wait(self.period)


class GazeCycles(GazePanel):
    """One episode per reachy-gaze cycle, all with the same instruction."""

    def __init__(self, url: str, task: str, period: float = 0.25) -> None:
        """Label searching and tracking `task`, splitting on each new cycle."""
        self.text = task
        super().__init__(url, period)

    def task(self) -> str | None:
        """The fixed instruction, or None outside searching and tracking."""
        return cycle_label(self._state, self.text)[0]

    def segment(self) -> int:
        """The cycle number, so each cycle is its own episode."""
        return cycle_label(self._state, self.text)[1]
