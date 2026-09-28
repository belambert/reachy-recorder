"""Sources of the language instruction for each tick.

An annotator answers "what is the robot doing right now?" as an instruction,
or None when nothing should be recorded. A change of answer ends an episode.
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

    def close(self) -> None:
        """Nothing to release."""


def gaze_task(state: dict | None) -> str | None:
    """Instruction for one reading of reachy-gaze's control panel."""
    if state is None or not state["enabled"] or not state["detector_ok"]:
        return None
    return f"look at the {state['label']}" if state["locked"] else "look around"


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
        self._task: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def task(self) -> str | None:
        """The instruction from the latest poll."""
        return self._task

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
            self._task = gaze_task(state)
            self._stop.wait(self.period)
