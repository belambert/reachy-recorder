"""Record a Reachy Mini to a LeRobotDataset, whatever app is driving it."""

from __future__ import annotations

import logging
import signal
import threading
import time
from pathlib import Path
from typing import Annotated, Optional

import typer

from reachy_recorder.annotate import Annotator, FixedTask, GazeCycles, GazePanel
from reachy_recorder.dataset import DatasetWriter, features
from reachy_recorder.episodes import EpisodeBuilder, Step
from reachy_recorder.robot import ACTION_NAMES, STATE_NAMES, RobotStream

# LeRobot needs a task on every frame; --gaze-cycles gives them all this one
CYCLE_TASK = "look around"

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)


@app.command()
def record(
    repo_id: Annotated[str, typer.Argument(help="Dataset id, e.g. you/reachy-gaze.")],
    host: str = "reachy-mini.local",
    root: Annotated[
        Optional[Path], typer.Option(help="Dataset directory; resumed if it exists.")
    ] = None,
    task: Annotated[
        Optional[str], typer.Option(help="Label every episode with this.")
    ] = None,
    gaze: Annotated[
        bool, typer.Option(help="Label episodes from reachy-gaze's panel.")
    ] = False,
    gaze_cycles: Annotated[
        bool,
        typer.Option(help="One unlabelled episode per reachy-gaze look-around cycle."),
    ] = False,
    fps: int = 10,
    width: Annotated[int, typer.Option(help="Recorded image width.")] = 640,
    min_seconds: float = 1.0,
    max_seconds: Annotated[
        Optional[float],
        typer.Option(help="Split longer episodes; default 30, 120 with --gaze-cycles."),
    ] = None,
    push: Annotated[bool, typer.Option(help="Push to the Hub when done.")] = False,
    private: bool = True,
) -> None:
    """Record until Ctrl-C, cutting an episode whenever the task changes."""
    # force: an imported library has already configured the root logger
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    if (task is not None) + gaze + gaze_cycles != 1:
        raise typer.BadParameter("pass exactly one of --task, --gaze, --gaze-cycles")
    if max_seconds is None:
        max_seconds = 120.0 if gaze_cycles else 30.0

    url = f"http://{host}:8042/state"
    annotator: Annotator = (
        GazeCycles(url, CYCLE_TASK)
        if gaze_cycles
        else GazePanel(url) if gaze else FixedTask(task or "")
    )
    stream = RobotStream(host, width)
    writer = DatasetWriter(
        repo_id, root, fps, features(stream.image_shape(), STATE_NAMES, ACTION_NAMES)
    )
    builder = EpisodeBuilder(
        round(min_seconds * fps),
        round(max_seconds * fps),
        [STATE_NAMES.index(n) for n in ACTION_NAMES],
    )

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        run(stream, annotator, builder, writer, fps, stop)
    finally:
        for ep in builder.flush():
            writer.put(ep)
        logger.info("finishing: saving queued episodes")
        annotator.close()
        stream.close()
        writer.close(push=push, private=private)
        logger.info("%d episodes saved", writer.saved)


def run(
    stream: RobotStream,
    annotator: Annotator,
    builder: EpisodeBuilder,
    writer: DatasetWriter,
    fps: int,
    stop: threading.Event,
) -> None:
    """Sample at `fps` on absolute deadlines and hand finished episodes over."""
    period = 1.0 / fps
    # older than two ticks means the video stalled; the stream also takes a
    # couple of seconds to settle after connecting
    max_age = 2 * period
    next_tick = time.monotonic()
    while not stop.is_set():
        s = stream.sample(max_age)
        step = (
            None
            if s is None
            else Step(annotator.task(), s.image, s.state, annotator.segment())
        )
        for ep in builder.push(step):
            writer.put(ep)

        next_tick += period
        if (delay := next_tick - time.monotonic()) > 0:
            stop.wait(delay)
        else:
            # fell behind: the missed ticks are a gap, not a burst to catch up
            for ep in builder.push(None):
                writer.put(ep)
            next_tick = time.monotonic()


def main() -> None:
    """Console entry point."""
    app()
