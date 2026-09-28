"""Watch a Reachy Mini from off-board without commanding it."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image
from reachy_mini import ReachyMini

# Order the daemon reports them in, which follows hardware_config.yaml.
JOINT_NAMES = [
    "body_rotation",
    *(f"stewart_{i}" for i in range(1, 7)),
    "right_antenna",
    "left_antenna",
]


@dataclass
class Sample:
    """The latest frame and joint state at one moment."""

    image: np.ndarray  # RGB, resized
    state: np.ndarray  # float32, in JOINT_NAMES order


class RobotStream:
    """Latest camera frame over WebRTC and joint state over the SDK socket.

    The daemon encodes a separate video stream per viewer, so close the
    control app's camera view while recording: two streams starve each other.
    """

    def __init__(self, host: str, width: int) -> None:
        """Connect to the daemon at `host`; frames are resized to `width`."""
        self.width = width
        # the SDK sends automatic_body_yaw on connect; True matches the
        # default every app starts from
        self.mini = ReachyMini(
            host=host, connection_mode="network", automatic_body_yaw=True
        )
        self._frame: np.ndarray | None = None
        self._frame_at = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._pull, daemon=True)
        self._thread.start()

    def sample(self, max_age: float) -> Sample | None:
        """The current frame and state, or None if the video has stalled."""
        frame = self._frame
        if frame is None or time.monotonic() - self._frame_at > max_age:
            return None
        head, antennas = self.mini.get_current_joint_positions()
        state = np.asarray([*head, *antennas], dtype=np.float32)
        return Sample(to_rgb(frame, self.width), state)

    def image_shape(self) -> tuple[int, int, int]:
        """(height, width, 3) of the images `sample` returns."""
        w, h = self.mini.media.camera.resolution  # type: ignore[union-attr]
        return (round(h * self.width / w), self.width, 3)

    def close(self) -> None:
        """Stop pulling frames and disconnect."""
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.mini.__exit__(None, None, None)

    def _pull(self) -> None:
        # the camera hands over each frame once, so keep the latest here for
        # the sampling loop to read at its own pace
        while not self._stop.is_set():
            if (frame := self.mini.media.get_frame()) is not None:
                self._frame, self._frame_at = frame, time.monotonic()


def to_rgb(bgr: np.ndarray, width: int) -> np.ndarray:
    """A BGR camera frame as RGB, scaled to `width` with its aspect kept."""
    h, w = bgr.shape[:2]
    img = Image.fromarray(bgr[..., ::-1])
    if w != width:
        img = img.resize((width, round(h * width / w)), Image.Resampling.BILINEAR)
    return np.asarray(img)
