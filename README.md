# Reachy Recorder

Records a Reachy Mini to a [LeRobot](https://github.com/huggingface/lerobot)
dataset for training vision-language-action models. It runs on a separate
machine, watches the robot without commanding it, and doesn't care which app is
driving: reachy-gaze, a teleop session, or you moving the head by hand.

## How It Works

The recorder connects to the robot's daemon as a second client. It pulls the
head camera over WebRTC and the joint state and head pose over the SDK's
socket, and samples them at a fixed rate (10 Hz by default).

Each sample's **action** is where the robot was one tick later: its head pose,
body rotation and antennas. The commands an app sends aren't visible from
outside it, so the recorder uses the position the robot actually reached next.

An **annotator** supplies the language instruction for each tick, and a change
of instruction ends the episode:

| Annotator       | Instructions                                                    |
| --------------- | --------------------------------------------------------------- |
| `--task`        | One fixed instruction for the whole session                     |
| `--gaze`        | Read from reachy-gaze's panel: "look at the cat", "look around" |
| `--gaze-cycles` | Always "look around"; one episode per reachy-gaze cycle         |

Nothing is recorded while the annotator has no instruction (e.g. reachy-gaze is
disabled or its detector is down) or while the video has stalled.

A short video hiccup doesn't end an episode: a tick whose latest camera frame
is up to `--max-stale-seconds` old (0.5 by default) keeps that frame, since the
joint state and head pose are fresh every tick. Dropping the tick instead would
squeeze time, because LeRobot assumes frames are evenly spaced. An older frame
counts as a stall and ends the episode. Each saved episode is logged with why
it ended: `task changed`, `new segment`, `video stalled`, `fell behind`,
`nothing to record`, `max length` or `stopped`.

reachy-gaze runs the head through a look-around cycle: `moving` to a random
pose, `holding` there, `scanning` for something to look at, then `tracking` it
until it gets bored, which starts the next cycle. With `--gaze-cycles`, each
episode is the scanning and tracking of one cycle. The move and hold are left
out, since the camera can't explain them, and a change of the cycle number
reachy-gaze reports always starts a new episode. Episodes can run up to 120 s
by default in this mode.

## Recording

    uv sync
    uv run reachy-recorder you/reachy-gaze --gaze
    uv run reachy-recorder you/reachy-cycles --gaze-cycles --root data/cycles
    uv run reachy-recorder you/reachy-wave --task "wave the antennas" --root data/wave

Press Ctrl-C to stop; queued episodes are saved before it exits. Recording into
an existing `--root` adds to that dataset. Pass `--push` to upload to the
Hugging Face Hub when done (private by default).

Before recording:

- **Quit the Reachy Mini control app.** The daemon encodes a separate stream
  for each viewer, and a second viewer loads the robot's encoder and the Wi-Fi
  enough that the recorder's video barely decodes. The recorder's own stream
  costs reachy-gaze about 10% of its detection rate.
- **Turn off behaviour the camera can't explain.** For reachy-gaze, raise
  `BORED_AFTER` and `DWELL_MEMORY` so the head doesn't leave a target for
  reasons invisible in the image.
- **Expect a few seconds' warm-up.** Nothing is recorded until the video
  arrives.

Connecting sends the SDK's `automatic_body_yaw` setting (on), which is what
apps start from.

## Dataset

| Feature                   | Contents                                                     |
| ------------------------- | ------------------------------------------------------------ |
| `observation.images.head` | Head camera, RGB, 640 px wide (`--width`), as video          |
| `observation.state`       | All nine joints plus the head pose (15 values)               |
| `action`                  | Head pose, body rotation and antennas at the next tick (9)   |
| `task`                    | The annotator's instruction                                  |

State is ordered `body_rotation`, `stewart_1`..`stewart_6`, `right_antenna`,
`left_antenna`, then `head_x`, `head_y`, `head_z`, `head_roll`, `head_pitch`,
`head_yaw`. Joints and angles are in radians and positions in metres; the pose
uses the same convention as the SDK's `create_head_pose` (extrinsic `xyz`
Euler angles). The antenna order was confirmed on a robot by moving the right
antenna by hand.

The head pose is in the world frame: it includes the body's rotation. Turning
the body by hand moves `head_yaw` by exactly as much as `body_rotation`, while
the Stewart joints stay still. `body_rotation` in the action therefore tells
how a turn is split between the body and the head platform.

The action is what an app would pass to `set_target`: `head_x`..`head_yaw`,
`body_rotation`, `right_antenna`, `left_antenna`. Because every action is a
subset of the next tick's state, a different action (such as next-tick joints,
or poses relative to the current one) can be derived later from
`observation.state` alone. The pose and joints arrive in separate daemon
messages, so within a frame they can be up to one daemon update (about 20 ms)
apart.

Recording into an existing `--root` with a different layout fails rather than
mixing the two.

## Status

Recordings from a real robot have been read back and look right, and the
antenna order and head pose frame have been checked by moving the robot by hand.

## Checking the Connection

`scripts/probe.py` connects alongside a running app and reports per-second
frame and state rates, plus reachy-gaze's detection rate:

    uv run python scripts/probe.py --host reachy-mini.local --seconds 15

With the control app's camera view closed, it measured a steady 30 fps at 720p
and state at about 48 Hz. With the view open, the second stream stalled within
about 6 s and reachy-gaze's detection rate halved.

The SDK's WebRTC client buffers only 10 ms of video. Over Wi-Fi, parts of the
keyframe the robot sends every 2 s arrive later than that and are dropped, and
the decoder then rejects every frame until the next keyframe: the video freezes
for 2 s, several times a minute. The recorder raises the buffer to
`--video-buffer-ms` (200 by default), which removed the freezes in testing; 50
and 100 ms also worked but still dropped a few late packets.

The camera image reaches the recorder about 150 ms plus the video buffer (so
about 350 ms by default) later than the joint state for the same moment. Each
frame is recorded with the latest state, so the image lags the state by that
much; the recorder doesn't yet compensate. The delay was measured by lining up
how much the image changes with how fast the head turns, at 50 and 200 ms
buffers.

If the WebRTC stream becomes a problem, a fallback is to record the 640 px
JPEGs that reachy-gaze already sends to its detector (about 12 Hz), which costs
the robot nothing extra.

## Development

    uv sync
    uv run pytest
    uv run black src tests scripts && uv run isort src tests scripts
