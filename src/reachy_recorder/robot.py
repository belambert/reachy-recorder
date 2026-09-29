"""Watch a Reachy Mini from off-board without commanding it."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
from reachy_mini import ReachyMini
from reachy_mini.utils.rotation import Rotation

# Order the daemon reports them in, which follows hardware_config.yaml;
# the antenna order was confirmed by moving one antenna by hand.
JOINT_NAMES = [
    "body_rotation",
    *(f"stewart_{i}" for i in range(1, 7)),
    "right_antenna",
    "left_antenna",
]
# metres and radians, in create_head_pose's convention
POSE_NAMES = ["head_x", "head_y", "head_z", "head_roll", "head_pitch", "head_yaw"]
STATE_NAMES = [*JOINT_NAMES, *POSE_NAMES]
# what an app commands with set_target: head pose, body yaw and antennas
ACTION_NAMES = [*POSE_NAMES, "body_rotation", "right_antenna", "left_antenna"]


@dataclass
class Sample:
    """The latest frame and robot state at one moment."""

    image: np.ndarray  # RGB, resized
    state: np.ndarray  # float32, in STATE_NAMES order


class RobotStream:
    """Latest camera frame over WebRTC; joints and head pose over the SDK socket.

    The daemon encodes a separate video stream per viewer, so quit the control
    app while recording: two streams starve each other.
    """

    def __init__(self, host: str, width: int, video_buffer_ms: int = 200) -> None:
        """Connect to the daemon at `host`; frames are resized to `width`."""
        self.width = width
        set_video_buffer(video_buffer_ms)
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
        pose = pose_to_xyzrpy(self.mini.get_current_head_pose())
        state = np.asarray([*head, *antennas, *pose], dtype=np.float32)
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


def set_video_buffer(ms: int) -> None:
    """Make the SDK's WebRTC client buffer `ms` of video rather than 10 ms.

    Over Wi-Fi, parts of the keyframe the robot sends every 2 s arrive later
    than 10 ms. The jitter buffer drops them, and the decoder then rejects every
    frame until the next keyframe, freezing the video for 2 s.
    """
    # imported here so the rest of this module works without GStreamer
    from reachy_mini.media.webrtc_client_gstreamer import GstWebRTCClient

    # the SDK sets its 10 ms here, whenever a stream arrives
    original = getattr(GstWebRTCClient, "_sdk_configure_webrtcbin", None)
    if original is None:
        original = GstWebRTCClient._configure_webrtcbin
        GstWebRTCClient._sdk_configure_webrtcbin = original

    def configure(client: Any, webrtcsrc: Any) -> None:
        original(client, webrtcsrc)
        for e in client._iterate_gst(webrtcsrc.iterate_recurse()):
            if (f := e.get_factory()) is not None and f.get_name() == "webrtcbin":
                e.set_property("latency", ms)

    GstWebRTCClient._configure_webrtcbin = configure


def to_rgb(bgr: np.ndarray, width: int) -> np.ndarray:
    """A BGR camera frame as RGB, scaled to `width` with its aspect kept."""
    h, w = bgr.shape[:2]
    img = Image.fromarray(bgr[..., ::-1])
    if w != width:
        img = img.resize((width, round(h * width / w)), Image.Resampling.BILINEAR)
    return np.asarray(img)


def pose_to_xyzrpy(pose: np.ndarray) -> np.ndarray:
    """A 4x4 head pose as (x, y, z, roll, pitch, yaw), inverting create_head_pose."""
    rpy = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")
    return np.concatenate([pose[:3, 3], rpy])
