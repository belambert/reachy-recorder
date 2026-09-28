# Reachy Recorder

Records a Reachy Mini to a [LeRobot](https://github.com/huggingface/lerobot)
dataset for training vision-language-action models. It runs on a separate
machine, watches the robot without commanding it, and doesn't care which app is
driving: reachy-gaze, a teleop session, or you moving the head by hand.

## How It Works

The recorder connects to the robot's daemon as a second client. It pulls the
head camera over WebRTC and the joint state over the SDK's socket, and samples
both at a fixed rate (10 Hz by default).

Each sample's **action** is the robot's joint state one tick later. The
commands an app sends aren't visible from outside it, so the recorder uses the
position the robot actually reached next.

An **annotator** supplies the language instruction for each tick, and a change
of instruction ends the episode:

| Annotator | Instructions                                                    |
| --------- | --------------------------------------------------------------- |
| `--task`  | One fixed instruction for the whole session                     |
| `--gaze`  | Read from reachy-gaze's panel: "look at the cat", "look around" |

Nothing is recorded while the annotator has no instruction (e.g. reachy-gaze is
disabled or its detector is down) or while the video has stalled.

## Recording

    uv sync
    uv run reachy-recorder you/reachy-gaze --gaze
    uv run reachy-recorder you/reachy-wave --task "wave the antennas" --root data/wave

Press Ctrl-C to stop; queued episodes are saved before it exits. Recording into
an existing `--root` adds to that dataset. Pass `--push` to upload to the
Hugging Face Hub when done (private by default).

Before recording:

- **Close the camera view in the Reachy Mini control app.** The daemon encodes
  a separate stream for each viewer, and two streams starve each other on the
  robot. The recorder's own stream costs reachy-gaze about 10% of its detection
  rate.
- **Turn off behaviour the camera can't explain.** For reachy-gaze, raise
  `BORED_AFTER` and `DWELL_MEMORY` so the head doesn't leave a target for
  reasons invisible in the image.
- **Expect a couple of seconds' warm-up.** The video stalls briefly after
  connecting; nothing is recorded until it settles.

Connecting sends the SDK's `automatic_body_yaw` setting (on), which is what
apps start from.

## Dataset

| Feature                   | Contents                                               |
| ------------------------- | ------------------------------------------------------ |
| `observation.images.head` | Head camera, RGB, 640 px wide (`--width`), as video    |
| `observation.state`       | Body rotation, six Stewart joints, both antennas (rad) |
| `action`                  | `observation.state` at the next tick                   |
| `task`                    | The annotator's instruction                            |

State is ordered `body_rotation`, `stewart_1`..`stewart_6`, `right_antenna`,
`left_antenna`. The antenna order comes from the SDK's `hardware_config.yaml`
and hasn't been confirmed on a robot yet.

## Status

The recorder hasn't been run against a real robot yet. Before relying on the
data, make a short recording, move one antenna by hand to confirm the antenna
order, and inspect the resulting dataset.

## Checking the Connection

`scripts/probe.py` connects alongside a running app and reports per-second
frame and state rates, plus reachy-gaze's detection rate:

    uv run python scripts/probe.py --host reachy-mini.local --seconds 15

With the control app's camera view closed, it measured a steady 30 fps at 720p
and state at about 48 Hz. With the view open, the second stream stalled within
about 6 s and reachy-gaze's detection rate halved.

If the WebRTC stream becomes a problem, a fallback is to record the 640 px
JPEGs that reachy-gaze already sends to its detector (about 12 Hz), which costs
the robot nothing extra.

## Development

    uv sync
    uv run pytest
    uv run black src tests scripts && uv run isort src tests scripts
