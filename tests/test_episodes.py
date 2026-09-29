import numpy as np

from reachy_recorder.episodes import EpisodeBuilder, Step


def step(task: str | None, i: int, segment: int = 0) -> Step:
    img = np.full((2, 2, 3), i, np.uint8)
    return Step(task, img, np.array([i], np.float32), segment)


def feed(b: EpisodeBuilder, steps: list[Step | None]) -> list:
    return [ep for s in steps for ep in b.push(s)] + b.flush()


def test_action_is_next_state():
    [ep] = feed(EpisodeBuilder(1, 100), [step("a", i) for i in range(4)])
    assert ep.task == "a"
    # the last step has no successor, so it is dropped
    assert [f.state[0] for f in ep.frames] == [0, 1, 2]
    assert [f.action[0] for f in ep.frames] == [1, 2, 3]


def test_task_change_splits_and_keeps_continuity():
    eps = feed(
        EpisodeBuilder(1, 100), [step("a", 0), step("a", 1), step("b", 2), step("b", 3)]
    )
    assert [e.task for e in eps] == ["a", "b"]
    # the last "a" step still gets the first "b" state: no time passed between
    assert [f.action[0] for f in eps[0].frames] == [1, 2]
    assert [f.state[0] for f in eps[1].frames] == [2]


def test_segment_change_splits_same_task():
    eps = feed(
        EpisodeBuilder(1, 100),
        [step("a", 0, 1), step("a", 1, 1), step("a", 2, 2), step("a", 3, 2)],
    )
    assert [[f.state[0] for f in e.frames] for e in eps] == [[0, 1], [2]]
    assert [e.segment for e in eps] == [1, 2]


def test_end_reasons():
    b = EpisodeBuilder(1, 3)
    pushes: list[tuple[Step | None, str]] = [
        (step("a", 0), ""),
        (step("b", 1), ""),
        (step("b", 2), ""),  # ends "a"
        (None, "video stalled"),  # ends "b"
        *((step("c", i), "") for i in range(3, 7)),  # third frame hits max
        (step(None, 7), ""),
        (step(None, 8), ""),  # ends the "c" started at 6
        (step("d", 9, 1), ""),
        (step("d", 10, 2), ""),
        (step("d", 11, 2), ""),  # ends segment 1
    ]
    eps = [ep for s, gap in pushes for ep in b.push(s, gap or "gap")] + b.flush()
    assert [e.end for e in eps] == [
        "task changed",
        "video stalled",
        "max length",
        "nothing to record",
        "new segment",
        "stopped",
    ]


def test_gap_ends_episode_and_drops_orphan():
    eps = feed(
        EpisodeBuilder(1, 100),
        [step("a", 0), step("a", 1), None, step("a", 3), step("a", 4)],
    )
    assert [[f.state[0] for f in e.frames] for e in eps] == [[0], [3]]


def test_none_task_is_not_recorded():
    eps = feed(
        EpisodeBuilder(1, 100),
        [step(None, 0), step(None, 1), step("a", 2), step("a", 3)],
    )
    assert [[f.state[0] for f in e.frames] for e in eps] == [[2]]


def test_short_episodes_dropped():
    assert feed(EpisodeBuilder(3, 100), [step("a", i) for i in range(3)]) == []


def test_max_len_splits():
    eps = feed(EpisodeBuilder(1, 2), [step("a", i) for i in range(6)])
    assert [len(e.frames) for e in eps] == [2, 2, 1]
    assert {e.task for e in eps} == {"a"}


def test_action_idx_selects_from_next_state():
    steps = [
        Step("a", np.zeros((2, 2, 3), np.uint8), np.array([i, 10 * i, 100 * i]))
        for i in range(2)
    ]
    [ep] = feed(EpisodeBuilder(1, 100, [2, 0]), steps)
    assert ep.frames[0].action.tolist() == [100, 1]
