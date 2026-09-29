"""Cut a stream of timed samples into labelled episodes with actions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Step:
    """One tick of the recorder: what the robot saw and where it was."""

    task: str | None  # None when nothing worth recording is happening
    image: np.ndarray
    state: np.ndarray
    segment: int = 0  # a change starts a new episode even if the task doesn't


@dataclass
class Frame:
    """A step paired with its action, ready for the dataset."""

    image: np.ndarray
    state: np.ndarray
    action: np.ndarray


@dataclass
class Episode:
    """Consecutive frames that share one instruction."""

    task: str
    frames: list[Frame] = field(default_factory=list)
    segment: int = 0


class EpisodeBuilder:
    """Turn steps into episodes, labelling each step's action from the next state.

    The robot's commands aren't observable from outside the app that sends
    them, so a step's action is where the robot was one tick later. That makes
    every step wait for its successor, and a step with none is dropped.
    """

    def __init__(
        self, min_len: int, max_len: int, action_idx: list[int] | None = None
    ) -> None:
        """Episodes shorter than `min_len` steps are dropped; `max_len` splits.

        An action is the next state's `action_idx` entries, or all of it.
        """
        self.min_len = min_len
        self.max_len = max_len
        self.action_idx = action_idx
        self.episode: Episode | None = None
        self.pending: Step | None = None

    def push(self, step: Step | None) -> list[Episode]:
        """Add the next tick's step, or None for a gap; return finished episodes."""
        done: list[Episode] = []
        prev, self.pending = self.pending, step

        # a gap leaves the pending step without a successor, so no action
        if step is None or prev is None or prev.task is None:
            return self._close()

        ep = self.episode
        if ep is not None and (ep.task, ep.segment) != (prev.task, prev.segment):
            done += self._close()
        if self.episode is None:
            self.episode = Episode(prev.task, segment=prev.segment)
        action = step.state if self.action_idx is None else step.state[self.action_idx]
        self.episode.frames.append(Frame(prev.image, prev.state, action))

        if len(self.episode.frames) >= self.max_len:
            done += self._close()
        return done

    def flush(self) -> list[Episode]:
        """Finish whatever is in progress, e.g. on shutdown."""
        self.pending = None
        return self._close()

    def _close(self) -> list[Episode]:
        ep, self.episode = self.episode, None
        return [ep] if ep is not None and len(ep.frames) >= self.min_len else []
