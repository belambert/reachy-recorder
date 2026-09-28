"""Check that a second client can watch the robot while an app drives it.

Connects to the daemon over the LAN alongside a running reachy-gaze, never
commands the robot, and reports whether camera frames and pushed state both
arrive, and at what rate. Run from the machine that would host the recorder:

    uv run python scripts/probe.py --host reachy-mini.local
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Any

import numpy as np
import requests
from PIL import Image
from reachy_mini import ReachyMini
from reachy_mini.io.protocol import HeadPoseMsg


def main() -> None:
    """Sample frames, state and the gaze panel for a while, then summarize."""
    args = parse_args()
    panel = f"http://{args.host}:8042/state"

    before = panel_state(panel)
    print(f"gaze panel before: {summarize_panel(before)}")

    # automatic_body_yaw matches what reachy-gaze already set, so connecting
    # changes nothing; the constructor sends it unconditionally.
    with ReachyMini(
        host=args.host, connection_mode="network", automatic_body_yaw=True
    ) as mini:
        print(f"connected: mode={mini.connection_mode}, media={media_kind(mini)}")
        stats = sample(mini, panel, args.seconds, args.out)

    after = panel_state(panel)
    print(f"gaze panel after:  {summarize_panel(after)}")
    report(stats, before, after, args.seconds)


def parse_args() -> argparse.Namespace:
    """Robot address, how long to sample, and where to save a frame."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="reachy-mini.local")
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--out", default="probe_frame.jpg")
    return p.parse_args()


def sample(mini: ReachyMini, panel: str, seconds: float, out: str) -> dict:
    """Timestamp every frame and state push, each on its own thread.

    One polling loop for everything capped both rates at the loop's own speed,
    so each stream is counted where it arrives instead.
    """
    frame_ts: list[float] = []
    state_ts: list[float] = []
    yaws: list[float] = []
    labels: set[str] = set()
    polls: list[tuple[float, dict | None]] = []
    shape = None
    stop = threading.Event()

    # the WS client routes every push through _dispatch; wrapping it counts
    # arrivals exactly rather than sampling for changes
    dispatch = mini.client._dispatch

    def on_msg(msg: Any) -> None:
        dispatch(msg)
        if isinstance(msg, HeadPoseMsg):
            state_ts.append(time.monotonic())
            yaws.append(yaw_of(np.array(msg.head_pose)))

    mini.client._dispatch = on_msg  # type: ignore[method-assign]

    def pull_frames() -> None:
        nonlocal shape
        while not stop.is_set():
            # each non-None read is a new frame: the appsink hands over its
            # one buffered sample, or waits up to 20 ms and returns None
            if (frame := mini.media.get_frame()) is None:
                continue
            frame_ts.append(time.monotonic())
            if shape is None:
                shape = frame.shape
                Image.fromarray(frame[..., ::-1]).save(out)

    t0 = time.monotonic()
    puller = threading.Thread(target=pull_frames, daemon=True)
    puller.start()
    while time.monotonic() - t0 < seconds:
        polls.append((time.monotonic(), s := panel_state(panel)))
        if s is not None and s["label"]:
            labels.add(s["label"])
        time.sleep(0.5)
    stop.set()
    puller.join(timeout=1.0)
    mini.client._dispatch = dispatch  # type: ignore[method-assign]

    return {
        "frames": rate(frame_ts, t0, seconds),
        "states": rate(state_ts, t0, seconds),
        "timeline": timeline(frame_ts, state_ts, polls, t0, seconds),
        "shape": shape,
        "configured": camera_config(mini),
        "yaw_range": (min(yaws), max(yaws)) if yaws else None,
        "labels": labels,
        "out": out,
    }


def rate(ts: list[float], t0: float, seconds: float) -> dict:
    """Arrival count, rate, and gap statistics for one stream.

    The rate is over the whole window, not first-to-last arrival: a stream that
    stalls after two seconds must not read as a healthy 30 Hz.
    """
    gaps = np.diff(ts) * 1000 if len(ts) > 1 else np.array([np.nan])
    return {
        "n": len(ts),
        "hz": len(ts) / seconds,
        "first": ts[0] - t0 if ts else None,
        "last": ts[-1] - t0 if ts else None,
        "gap_ms": (float(np.median(gaps)), float(np.max(gaps))),
    }


def timeline(
    frame_ts: list[float],
    state_ts: list[float],
    polls: list[tuple[float, dict | None]],
    t0: float,
    seconds: float,
) -> list[str]:
    """One row per second, to see what stalls together and when."""
    bins = np.arange(0, int(np.ceil(seconds)) + 1)
    frames = np.histogram(np.subtract(frame_ts, t0), bins)[0]
    states = np.histogram(np.subtract(state_ts, t0), bins)[0]
    rows = []
    for i, (f, st) in enumerate(zip(frames, states)):
        # the panel's own detector fps shows whether the app's vision slowed
        seen = [p for t, p in polls if i <= t - t0 < i + 1]
        panel = " ".join("--" if p is None else f"{p['fps']:>4.1f}" for p in seen)
        rows.append(f"  {i:>3}s  frames {f:>3}  state {st:>3}  app fps {panel}")
    return rows


def camera_config(mini: ReachyMini) -> str:
    """What the SDK asked the WebRTC pipeline for, to compare with what came."""
    cam = mini.media.camera
    res = getattr(cam, "resolution", None)
    fps = getattr(cam, "framerate", None)
    return f"{res} @ {fps} fps"


def report(stats: dict, before: dict | None, after: dict | None, secs: float) -> None:
    """Print rates and a verdict on each thing the recorder depends on."""
    f, s = stats["frames"], stats["states"]
    print()
    print(f"camera: {f['n']} frames, {f['hz']:.1f} fps, shape {stats['shape']}")
    print(f"        configured {stats['configured']}")
    print(f"        gap median {f['gap_ms'][0]:.0f} ms, max {f['gap_ms'][1]:.0f} ms")
    if f["first"] is not None:
        print(f"        first frame at {f['first']:.1f}s -> {stats['out']}")
        print(f"        last frame at {f['last']:.1f}s")
    print(f"state:  {s['n']} pushes, {s['hz']:.1f} Hz")
    print(f"        gap median {s['gap_ms'][0]:.0f} ms, max {s['gap_ms'][1]:.0f} ms")
    if (r := stats["yaw_range"]) is not None:
        # a stationary target leaves this flat; that's not a failure
        print(f"        head yaw ranged {r[0]:.1f}..{r[1]:.1f} deg")
    print(f"labels seen on panel: {sorted(stats['labels']) or 'none'}")
    print()
    print("\n".join(stats["timeline"]))

    # the app has to stay alive and keep tracking through our connection
    app_ok = after is not None and after["enabled"] and after["detector_ok"]
    checks = {
        "camera frames arrive (>= 10 fps)": f["hz"] >= 10,
        "state arrives (>= 20 Hz)": s["hz"] >= 20,
        "gaze app still healthy afterwards": app_ok,
    }
    print()
    for name, ok in checks.items():
        print(f"  [{'x' if ok else ' '}] {name}")
    if before is None:
        print("\nnote: gaze panel unreachable; is reachy-gaze running?")


def panel_state(url: str) -> dict | None:
    """The gaze control panel's state, or None if the app isn't serving it."""
    try:
        r = requests.get(url, timeout=1.0)
        r.raise_for_status()
    except requests.RequestException:
        return None
    return r.json()


def summarize_panel(s: dict | None) -> str:
    """One line on what the gaze app says it is doing."""
    if s is None:
        return "unreachable"
    what = f"locked on {s['label']}" if s["locked"] else "scanning"
    return f"enabled={s['enabled']} detector_ok={s['detector_ok']} {what}"


def media_kind(mini: ReachyMini) -> str:
    """Which camera reader the SDK picked; WebRTC is expected off-robot."""
    cam = mini.media.camera
    return type(cam).__name__ if cam is not None else "none"


def yaw_of(pose: np.ndarray) -> float:
    """Head yaw in degrees, +left, from the forward axis of a 4x4 pose."""
    x, y, _ = pose[:3, 0]
    return float(np.degrees(np.arctan2(y, x)))


if __name__ == "__main__":
    main()
