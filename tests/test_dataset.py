import numpy as np
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from reachy_mini.utils import create_head_pose

from reachy_recorder.dataset import IMAGE_KEY, DatasetWriter, features
from reachy_recorder.episodes import Episode, Frame
from reachy_recorder.robot import (
    ACTION_NAMES,
    JOINT_NAMES,
    STATE_NAMES,
    pose_to_xyzrpy,
    state_at,
    to_rgb,
)


def test_to_rgb_swaps_channels_and_keeps_aspect():
    bgr = np.zeros((720, 1280, 3), np.uint8)
    bgr[..., 0] = 255  # blue
    rgb = to_rgb(bgr, 640)
    assert rgb.shape == (360, 640, 3)
    assert rgb[0, 0].tolist() == [0, 0, 255]


def test_state_at_interpolates():
    history = [(1.0, np.array([0.0, 10.0])), (2.0, np.array([1.0, 20.0]))]
    assert state_at(history, 0.5) is None
    assert state_at(history, 1.0).tolist() == [0.0, 10.0]
    assert np.allclose(state_at(history, 1.25), [0.25, 12.5])
    assert state_at(history, 3.0).tolist() == [1.0, 20.0]
    assert state_at([], 1.0) is None


def test_pose_to_xyzrpy_inverts_create_head_pose():
    xyzrpy = [0.01, -0.02, 0.005, 0.1, -0.2, 0.3]
    pose = create_head_pose(*xyzrpy, degrees=False)
    assert np.allclose(pose_to_xyzrpy(pose), xyzrpy)


SHAPE = (36, 64, 3)
FRAMES = [
    Frame(
        np.full(SHAPE, 10 * i, np.uint8),
        np.full(len(STATE_NAMES), i, np.float32),
        np.full(len(ACTION_NAMES), i + 1, np.float32),
    )
    for i in range(5)
]


def test_round_trip(tmp_path):
    shape, frames = SHAPE, FRAMES
    root = tmp_path / "ds"
    feats = features(shape, STATE_NAMES, ACTION_NAMES)
    w = DatasetWriter("test/reachy", root, 10, feats)
    w.put(Episode("look at the cat", frames))
    w.close()

    ds = LeRobotDataset("test/reachy", root=root)
    assert ds.num_episodes == 1 and len(ds) == 5
    item = ds[2]
    assert item["task"] == "look at the cat"
    assert item["observation.state"][0] == 2
    assert item["action"][0] == 3
    assert tuple(item[IMAGE_KEY].shape) == (3, 36, 64)
    assert ds.meta.features["action"]["names"] == ACTION_NAMES

    # reopening the same root appends rather than starting over
    w = DatasetWriter("test/reachy", root, 10, feats)
    w.put(Episode("look around", frames))
    w.close()
    assert LeRobotDataset("test/reachy", root=root).num_episodes == 2


def test_resume_rejects_other_layout(tmp_path):
    root = tmp_path / "ds"
    old = features(SHAPE, JOINT_NAMES, JOINT_NAMES)
    w = DatasetWriter("test/reachy", root, 10, old)
    n = len(JOINT_NAMES)
    w.put(
        Episode(
            "a",
            [
                Frame(f.image, np.zeros(n, np.float32), np.zeros(n, np.float32))
                for f in FRAMES
            ],
        )
    )
    w.close()
    with pytest.raises(ValueError, match="new --root"):
        DatasetWriter(
            "test/reachy", root, 10, features(SHAPE, STATE_NAMES, ACTION_NAMES)
        )
