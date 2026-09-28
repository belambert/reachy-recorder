import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from reachy_recorder.dataset import IMAGE_KEY, DatasetWriter, features
from reachy_recorder.episodes import Episode, Frame
from reachy_recorder.robot import JOINT_NAMES, to_rgb


def test_to_rgb_swaps_channels_and_keeps_aspect():
    bgr = np.zeros((720, 1280, 3), np.uint8)
    bgr[..., 0] = 255  # blue
    rgb = to_rgb(bgr, 640)
    assert rgb.shape == (360, 640, 3)
    assert rgb[0, 0].tolist() == [0, 0, 255]


def test_round_trip(tmp_path):
    shape = (36, 64, 3)
    n = len(JOINT_NAMES)
    frames = [
        Frame(
            np.full(shape, 10 * i, np.uint8),
            np.full(n, i, np.float32),
            np.full(n, i + 1, np.float32),
        )
        for i in range(5)
    ]
    root = tmp_path / "ds"
    w = DatasetWriter("test/reachy", root, 10, features(shape, JOINT_NAMES))
    w.put(Episode("look at the cat", frames))
    w.close()

    ds = LeRobotDataset("test/reachy", root=root)
    assert ds.num_episodes == 1 and len(ds) == 5
    item = ds[2]
    assert item["task"] == "look at the cat"
    assert item["observation.state"][0] == 2
    assert item["action"][0] == 3
    assert tuple(item[IMAGE_KEY].shape) == (3, 36, 64)

    # reopening the same root appends rather than starting over
    w = DatasetWriter("test/reachy", root, 10, features(shape, JOINT_NAMES))
    w.put(Episode("look around", frames))
    w.close()
    assert LeRobotDataset("test/reachy", root=root).num_episodes == 2
