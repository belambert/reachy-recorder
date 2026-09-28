"""Write episodes to a LeRobotDataset on a background thread."""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from reachy_recorder.episodes import Episode

ROBOT_TYPE = "reachy_mini"
IMAGE_KEY = "observation.images.head"

logger = logging.getLogger(__name__)


def features(image_shape: tuple[int, int, int], joints: list[str]) -> dict:
    """LeRobot feature spec: one camera, joint state, and next-state action."""
    joint = {"dtype": "float32", "shape": (len(joints),), "names": joints}
    return {
        IMAGE_KEY: {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.state": joint,
        "action": joint,
    }


class DatasetWriter:
    """Save episodes without blocking the recording loop.

    Saving encodes the episode's video, which takes seconds; done inline it
    would leave a hole in the recording every time an episode ended.
    """

    def __init__(
        self, repo_id: str, root: Path | None, fps: int, features: dict
    ) -> None:
        """Open `root` if it already holds a dataset, else create one there."""
        if root is not None and (root / "meta" / "info.json").exists():
            self.ds = LeRobotDataset.resume(repo_id, root=root)
        else:
            self.ds = LeRobotDataset.create(
                repo_id, fps, features, root=root, robot_type=ROBOT_TYPE
            )
        self.saved = 0
        self._queue: queue.Queue[Episode | None] = queue.Queue()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def put(self, ep: Episode) -> None:
        """Queue an episode to be saved."""
        self._queue.put(ep)

    def close(self, push: bool = False, private: bool = True) -> None:
        """Save everything queued, finalize, and optionally push to the Hub."""
        self._queue.put(None)
        self._thread.join()
        self.ds.finalize()
        if push:
            self.ds.push_to_hub(private=private)

    def _drain(self) -> None:
        while (ep := self._queue.get()) is not None:
            for f in ep.frames:
                self.ds.add_frame(
                    {
                        IMAGE_KEY: f.image,
                        "observation.state": f.state,
                        "action": f.action,
                        "task": ep.task,
                    }
                )
            self.ds.save_episode()
            self.saved += 1
            logger.info(
                "saved episode %d: %r, %d frames", self.saved, ep.task, len(ep.frames)
            )
