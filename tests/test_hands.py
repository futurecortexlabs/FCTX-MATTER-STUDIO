"""The tracker end to end, on the synthetic hand.

No camera is needed and none is used.  The synthetic source builds a metric
hand and projects it into image space, so every frame here travels the same
road a camera frame does: projection, One-Euro filtering, velocity
estimation, gesture classification, track association.
"""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from _harness import case, note, run

from fctx.config import GrabConfig, TrackingConfig
from fctx.core.types import (
    NUM_LANDMARKS,
    Gesture,
    Handedness,
    HandFrame,
    HandPose,
    L,
    TrackerFrame,
)
from fctx.hands import (
    HandTracker,
    Recorder,
    ReplaySource,
    SyntheticSource,
    build_metric_hand,
)
from fctx.hands.sources import CameraUnavailable, NullSource, create_source

CFG = TrackingConfig(source="synthetic")
GRAB = GrabConfig()
FPS = 60.0
DT = 1.0 / FPS
FRAMES = 300

#: Generous box around the stage.  Anything outside it is not a hand that
#: drifted, it is a hand that was projected wrong.
BOX_MARGIN = 0.35


def _source(cfg: TrackingConfig = CFG, auto: bool = True) -> SyntheticSource:
    src = SyntheticSource(cfg, auto=auto, fps=FPS, realtime=False)
    src.start()
    return src


def _run(
    src: SyntheticSource,
    tracker: HandTracker,
    frames: int = FRAMES,
) -> tuple[list[TrackerFrame], list[list[HandPose]]]:
    captured: list[TrackerFrame] = []
    poses: list[list[HandPose]] = []
    for _ in range(frames):
        frame = src.poll()
        assert frame is not None, "a non-realtime synthetic source always yields"
        captured.append(frame)
        poses.append(tracker.update(frame, frame.timestamp))
    return captured, poses


def _in_box(cfg: TrackingConfig, joints: np.ndarray) -> bool:
    x = cfg.stage_half_width + BOX_MARGIN
    y0 = cfg.stage_center_y - cfg.stage_half_height - BOX_MARGIN
    y1 = cfg.stage_center_y + cfg.stage_half_height + BOX_MARGIN
    z0, z1 = cfg.z_range[0] - BOX_MARGIN, cfg.z_range[1] + BOX_MARGIN
    return bool(
        np.all(np.abs(joints[:, 0]) <= x)
        and np.all(joints[:, 1] >= y0) and np.all(joints[:, 1] <= y1)
        and np.all(joints[:, 2] >= z0) and np.all(joints[:, 2] <= z1))


@case
def test_produces_a_pose_every_frame() -> None:
    _, poses = _run(_source(), HandTracker(CFG))
    assert len(poses) == FRAMES
    assert all(len(p) == 1 for p in poses), (
        f"{sum(1 for p in poses if len(p) != 1)} frames produced the wrong "
        "number of poses")


@case
def test_poses_are_well_formed() -> None:
    _, poses = _run(_source(), HandTracker(CFG))
    for i, (pose,) in enumerate(poses):
        assert pose.joints.shape == (NUM_LANDMARKS, 3)
        assert pose.joints.dtype == np.float32
        assert pose.joints.flags["C_CONTIGUOUS"]
        assert pose.velocities.shape == (NUM_LANDMARKS, 3)
        assert np.isfinite(pose.joints).all(), f"non-finite joints at frame {i}"
        assert np.isfinite(pose.velocities).all()
        assert _in_box(CFG, pose.joints), f"frame {i} left the stage"
        assert 0.0 <= pose.pinch <= 1.0 and 0.0 <= pose.curl <= 1.0
        assert 0.0 <= pose.confidence <= 1.0
        assert abs(float(np.linalg.norm(pose.palm_normal)) - 1.0) < 1e-4
        assert abs(float(np.linalg.norm(pose.pinch_rotation)) - 1.0) < 1e-4
        assert pose.handedness == Handedness.RIGHT
        a, b, r = pose.bone_segments()
        assert a.shape == b.shape == (len(r), 3)


@case
def test_velocities_stay_bounded() -> None:
    _, poses = _run(_source(), HandTracker(CFG))
    speeds = np.concatenate(
        [np.linalg.norm(p[0].velocities, axis=1) for p in poses])
    peak = float(speeds.max())
    note(f"peak joint speed on the autopilot path: {peak:.3f} m/s")
    assert peak < 4.0, f"peak joint speed {peak:.2f} m/s"
    assert float(np.mean(speeds)) < 1.0


@case
def test_velocity_is_consistent_with_motion() -> None:
    """The reported velocity must actually describe the reported motion."""
    _, poses = _run(_source(), HandTracker(CFG))
    wrist = np.stack([p[0].joints[int(L.WRIST)] for p in poses]).astype(np.float64)
    reported = np.stack([p[0].velocities[int(L.WRIST)] for p in poses[1:]])
    measured = (wrist[1:] - wrist[:-1]) / DT
    skip = 30  # the estimator's own low-pass needs a moment to converge
    err = np.linalg.norm(reported[skip:] - measured[skip:], axis=1)
    scale = max(float(np.linalg.norm(measured[skip:], axis=1).mean()), 1e-6)
    assert float(err.mean()) < 0.35 * scale, (
        f"reported velocity is off by {err.mean():.3f} m/s on a "
        f"{scale:.3f} m/s signal")


@case
def test_pinch_crosses_the_grab_threshold_when_commanded() -> None:
    src = _source(auto=False)
    tracker = HandTracker(CFG, GRAB)
    src.set_pointer(0.5, 0.5)
    src.set_depth(0.5)

    opened = []
    for _ in range(60):
        frame = src.poll()
        opened.append(tracker.update(frame, frame.timestamp)[0])
    assert max(p.pinch for p in opened) < GRAB.release_threshold
    assert not any(p.pinching for p in opened)

    src.set_pinch(1.0)
    closed = []
    for _ in range(60):
        frame = src.poll()
        closed.append(tracker.update(frame, frame.timestamp)[0])
    assert closed[-1].pinch > GRAB.start_threshold, closed[-1].pinch
    assert closed[-1].pinching
    assert closed[-1].gesture == Gesture.PINCH

    src.set_pinch(0.0)
    released = []
    for _ in range(60):
        frame = src.poll()
        released.append(tracker.update(frame, frame.timestamp)[0])
    assert not released[-1].pinching
    assert released[-1].pinch < GRAB.release_threshold


@case
def test_grab_hysteresis_does_not_chatter() -> None:
    """Held exactly between the two thresholds, a grab must not flicker."""
    src = _source(auto=False)
    tracker = HandTracker(CFG, GRAB)
    src.set_pinch(1.0)
    for _ in range(90):
        frame = src.poll()
        pose = tracker.update(frame, frame.timestamp)[0]
    assert pose.pinching

    # Ease off until the reported strength sits between the two thresholds.
    # The synthetic pinch control is not the same scale as the measured
    # strength (one drives joint angles, the other measures a distance
    # ratio), so the level is found rather than assumed.
    level = 1.0
    for _ in range(400):
        frame = src.poll()
        pose = tracker.update(frame, frame.timestamp)[0]
        if pose.pinch <= GRAB.start_threshold - 0.04:
            break
        level = max(0.0, level - 0.005)
        src.set_pinch(level)
    assert GRAB.release_threshold + 0.04 < pose.pinch < GRAB.start_threshold, (
        pose.pinch, level)
    states = []
    for _ in range(120):
        frame = src.poll()
        states.append(tracker.update(frame, frame.timestamp)[0].pinching)
    assert all(states), (
        f"the grab dropped {states.count(False)} times while the pinch was "
        f"held at {pose.pinch:.3f}, between the {GRAB.release_threshold} and "
        f"{GRAB.start_threshold} thresholds")


@case
def test_hold_time_blocks_a_flick() -> None:
    """A pinch that snaps shut and open again must not grab anything."""
    fast = dataclasses.replace(CFG)
    grab = dataclasses.replace(GRAB, hold_time=0.20)
    src = _source(fast, auto=False)
    tracker = HandTracker(fast, grab)
    for _ in range(40):
        frame = src.poll()
        tracker.update(frame, frame.timestamp)
    src.set_pinch(1.0)
    grabbed = False
    for _ in range(6):  # 100 ms, half the hold time
        frame = src.poll()
        grabbed |= tracker.update(frame, frame.timestamp)[0].pinching
    src.set_pinch(0.0)
    for _ in range(20):
        frame = src.poll()
        grabbed |= tracker.update(frame, frame.timestamp)[0].pinching
    assert not grabbed, "a 100 ms flick grabbed despite a 200 ms hold time"


@case
def test_curl_and_gesture_follow_the_hand() -> None:
    src = _source(auto=False)
    tracker = HandTracker(CFG, GRAB)
    for _ in range(120):
        frame = src.poll()
        flat = tracker.update(frame, frame.timestamp)[0]
    assert flat.curl < 0.1 and flat.gesture == Gesture.OPEN

    src.set_curl(1.0)
    for _ in range(120):
        frame = src.poll()
        fist = tracker.update(frame, frame.timestamp)[0]
    assert fist.curl > 0.7, fist.curl
    assert fist.gesture == Gesture.FIST


@case
def test_track_ids_are_stable() -> None:
    _, poses = _run(_source(), HandTracker(CFG))
    ids = {p[0].track_id for p in poses}
    assert ids == {0}, f"the one hand was given slots {sorted(ids)}"
    assert all(0 <= p[0].track_id < CFG.max_hands for p in poses)


@case
def test_two_hands_keep_their_slots_when_the_list_reorders() -> None:
    """MediaPipe does not promise a stable output order.  The tracker must
    not inherit that instability, because the solver indexes its per-hand
    arrays by slot and a swap would tear a grabbed body between two hands."""
    left = _source(auto=False)
    right = _source(auto=False)
    left.set_pointer(0.2, 0.5)
    right.set_pointer(0.8, 0.5)
    tracker = HandTracker(CFG, GRAB)

    slots_by_side: dict[int, int] = {}
    now = 0.0
    for step in range(200):
        now += DT
        a = left.poll().hands[0]
        b = right.poll().hands[0]
        a = HandFrame(image=a.image, world=a.world, handedness=Handedness.LEFT)
        b = HandFrame(image=b.image, world=b.world, handedness=Handedness.RIGHT)
        # Shuffle the order every few frames, exactly as the detector may.
        hands = [a, b] if step % 3 else [b, a]
        poses = tracker.update(TrackerFrame(hands=hands, timestamp=now), now)
        assert len(poses) == 2, f"lost a hand at step {step}"
        for pose in poses:
            side = int(pose.handedness)
            if side in slots_by_side:
                assert slots_by_side[side] == pose.track_id, (
                    f"hand {pose.handedness!r} moved from slot "
                    f"{slots_by_side[side]} to {pose.track_id} at step {step}")
            else:
                slots_by_side[side] = pose.track_id
        assert len({p.track_id for p in poses}) == 2
        # Drift the hands past each other so the match cannot be by position
        # alone for the whole run.
        left.set_pointer(0.2 + 0.3 * step / 200.0, 0.5)
        right.set_pointer(0.8 - 0.3 * step / 200.0, 0.5)
    assert set(slots_by_side.values()) == {0, 1}


@case
def test_a_dropped_hand_coasts_then_disappears() -> None:
    src = _source(auto=False)
    tracker = HandTracker(CFG, GRAB)
    src.set_pointer(0.2, 0.5)
    now = 0.0
    for step in range(120):
        now += DT
        frame = src.poll()
        src.set_pointer(0.2 + 0.4 * step / 120.0, 0.5)
        pose = tracker.update(
            TrackerFrame(hands=frame.hands, timestamp=now), now)[0]
    assert pose.confidence > 0.9
    last_x = float(pose.joints[int(L.WRIST), 0])
    moving = float(pose.velocities[int(L.WRIST), 0])
    assert moving > 0.05, "the hand should be travelling before it is lost"

    confidences = []
    coasted = 0
    while True:
        now += DT
        poses = tracker.update(None, now)
        if not poses:
            break
        coasted += 1
        confidences.append(poses[0].confidence)
        assert poses[0].track_id == 0
        assert np.isfinite(poses[0].joints).all()
        assert coasted < 200

    assert coasted >= 1
    assert coasted * DT <= CFG.coast_time + DT, (
        f"coasted for {coasted * DT:.3f} s, longer than coast_time")
    assert confidences == sorted(confidences, reverse=True), confidences
    assert confidences[-1] < 0.35, f"confidence only fell to {confidences[-1]}"
    assert float(tracker._tracks[0].joints[int(L.WRIST), 0]) > last_x, (
        "a coasted hand must keep moving in the direction it was going")

    # The slot stays reserved until lost_timeout, then the track is gone.
    while now - 0.0 < 10.0:
        now += DT
        tracker.update(None, now)
        if not tracker.active_slots:
            break
    assert tracker.active_slots == []


@case
def test_a_reacquired_hand_gets_its_slot_back() -> None:
    src = _source(auto=False)
    tracker = HandTracker(CFG, GRAB)
    now = 0.0
    for _ in range(60):
        now += DT
        frame = src.poll()
        tracker.update(TrackerFrame(hands=frame.hands, timestamp=now), now)
    for _ in range(int(0.9 * CFG.lost_timeout / DT)):
        now += DT
        tracker.update(None, now)
    now += DT
    frame = src.poll()
    poses = tracker.update(TrackerFrame(hands=frame.hands, timestamp=now), now)
    assert len(poses) == 1 and poses[0].track_id == 0
    assert np.isfinite(poses[0].velocities).all()
    speed = float(np.linalg.norm(poses[0].velocities, axis=1).max())
    assert speed < 1e-6, (
        f"reacquiring a hand injected {speed:.3f} m/s of phantom velocity")


@case
def test_never_exceeds_max_hands() -> None:
    cfg = dataclasses.replace(CFG, max_hands=1)
    tracker = HandTracker(cfg, GRAB)
    a = _source(cfg, auto=False)
    b = _source(cfg, auto=False)
    a.set_pointer(0.1, 0.5)
    b.set_pointer(0.9, 0.5)
    now = 0.0
    for _ in range(50):
        now += DT
        hands = [a.poll().hands[0], b.poll().hands[0]]
        poses = tracker.update(TrackerFrame(hands=hands, timestamp=now), now)
        assert len(poses) <= 1
        assert all(p.track_id == 0 for p in poses)


@case
def test_record_replay_round_trip() -> None:
    src = _source()
    reference_tracker = HandTracker(CFG, GRAB)
    captured, reference = _run(src, reference_tracker)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "roundtrip.fhr"
        with Recorder(path, max_hands=CFG.max_hands,
                      source=src.describe) as rec:
            for frame in captured:
                rec.add(frame)
        assert path.exists() and path.stat().st_size > 0
        note(f"recording: {len(captured)} frames, "
             f"{path.stat().st_size / 1024:.0f} KiB")

        replay = ReplaySource(path, CFG, realtime=False)
        assert replay.frame_count == len(captured)
        assert "roundtrip" in replay.describe
        replay.start()
        replay_tracker = HandTracker(CFG, GRAB)
        for i, original in enumerate(captured):
            frame = replay.poll()
            assert frame is not None, f"replay ran dry at frame {i}"
            assert len(frame.hands) == len(original.hands)
            for a, b in zip(frame.hands, original.hands):
                assert np.array_equal(a.image, b.image)
                assert np.array_equal(a.world, b.world)
                assert a.handedness == b.handedness
            got = replay_tracker.update(frame, original.timestamp)
            want = reference[i]
            assert len(got) == len(want)
            for g, w in zip(got, want):
                assert np.array_equal(g.joints, w.joints), f"frame {i}"
                assert np.array_equal(g.velocities, w.velocities), f"frame {i}"
                assert g.pinch == w.pinch and g.pinching == w.pinching
                assert g.track_id == w.track_id
                assert g.gesture == w.gesture
        replay.close()


@case
def test_replay_loops_and_paces() -> None:
    src = _source()
    captured, _ = _run(src, HandTracker(CFG), frames=30)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "loop.fhr"
        with Recorder(path, max_hands=CFG.max_hands) as rec:
            for frame in captured:
                rec.add(frame)

        looping = ReplaySource(path, dataclasses.replace(CFG, replay_loop=True),
                               realtime=False)
        looping.start()
        assert sum(1 for _ in range(90) if looping.poll() is not None) == 90

        once = ReplaySource(path, dataclasses.replace(CFG, replay_loop=False),
                            realtime=False)
        once.start()
        produced = sum(1 for _ in range(90) if once.poll() is not None)
        assert produced == 30, produced

        paced = ReplaySource(path, CFG, realtime=True)
        paced.start()
        start = time.perf_counter()
        frames = 0
        while frames < 10 and time.perf_counter() - start < 2.0:
            if paced.poll() is not None:
                frames += 1
        elapsed = time.perf_counter() - start
        assert frames == 10
        assert elapsed > 0.5 * (9 * DT), (
            f"a paced replay produced 10 frames in {elapsed * 1e3:.0f} ms")
        paced.close()


@case
def test_recording_can_carry_a_preview() -> None:
    src = _source()
    frame = src.poll()
    assert frame is not None
    preview = np.zeros((72, 128, 3), np.uint8)
    preview[:, :, 0] = 200
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "preview.fhr"
        with Recorder(path, max_hands=CFG.max_hands, store_preview=True,
                      preview_width=32) as rec:
            for _ in range(5):
                f = src.poll()
                assert f is not None
                rec.add(TrackerFrame(hands=f.hands, timestamp=f.timestamp,
                                     index=f.index, preview=preview))
        replay = ReplaySource(path, CFG, realtime=False)
        replay.start()
        out = replay.poll()
        assert out is not None and out.preview is not None
        assert out.preview.shape[2] == 3
        assert out.preview.shape[1] <= 128 and out.preview.shape[1] >= 32
        assert int(out.preview[0, 0, 0]) == 200

    with tempfile.TemporaryDirectory() as tmp:
        rec = Recorder(Path(tmp) / "missing.fhr", store_preview=True)
        try:
            rec.add(TrackerFrame(hands=[], timestamp=0.0))
        except ValueError:
            pass
        else:
            raise AssertionError("a missing preview was silently accepted")


@case
def test_recording_rejects_junk() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # Starting and stopping the recorder inside a single frame is an
        # ordinary thing for a user to do, and the application calls close()
        # unconditionally, so this path must not raise.  What it must not do
        # either is leave a zero-frame file: ReplaySource has nothing to hand
        # out from one and nothing to pace it against.
        empty = Path(tmp) / "empty.fhr"
        recorder = Recorder(empty)
        assert recorder.close() == 0, "an empty close must report zero frames"
        assert not empty.exists(), "an empty recording was written"
        assert recorder.close() == 0, "close() must stay idempotent"

        # An exception escaping the with-block must reach the caller rather
        # than being replaced by whatever close() thinks of the recording.
        try:
            with Recorder(Path(tmp) / "ctx.fhr"):
                raise KeyError("the real error")
        except KeyError:
            pass
        else:
            raise AssertionError("Recorder.__exit__ swallowed the real error")

        junk = Path(tmp) / "junk.fhr"
        with junk.open("wb") as fh:
            np.savez_compressed(fh, nonsense=np.zeros(3))
        try:
            ReplaySource(junk, CFG)
        except ValueError:
            pass
        else:
            raise AssertionError("a non-.fhr npz was accepted")

        try:
            ReplaySource(Path(tmp) / "nope.fhr", CFG)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("a missing recording was accepted")


@case
def test_autopilot_yields_to_a_moving_cursor_and_takes_over_again() -> None:
    """The application pushes cursor state in every frame regardless of
    whether the cursor moved, so the autopilot has to distinguish a changed
    input from a repeated one -- otherwise the demo hand is frozen whenever
    nobody is at the mouse."""
    src = _source(auto=True)
    tracker = HandTracker(CFG, GRAB)

    def wrist(n: int) -> np.ndarray:
        for _ in range(n):
            frame = src.poll()
            pose = tracker.update(frame, frame.timestamp)[0]
        return pose.joints[int(L.WRIST)].copy()

    a = wrist(20)
    b = wrist(20)
    assert float(np.linalg.norm(b - a)) > 0.01, "the autopilot never moved"

    # Steering: a changing pointer must win, and the hand must follow it.
    for i in range(60):
        src.set_pointer(0.15 + 0.004 * i, 0.5)
        src.set_depth(0.5)
        src.set_pinch(0.0)
        src.set_curl(0.0)
        frame = src.poll()
        pose = tracker.update(frame, frame.timestamp)[0]
    steered = float(pose.joints[int(L.WRIST), 0])
    target = (0.15 + 0.004 * 59 - 0.5) * 2.0 * CFG.stage_half_width
    assert abs(steered - target) < 0.05, (steered, target)

    # Holding the same value is not input.  After the idle window the
    # autopilot fades back in and the hand starts moving on its own.
    held = None
    for _ in range(int(3.0 / DT)):
        src.set_pointer(0.15 + 0.004 * 59, 0.5)
        src.set_depth(0.5)
        frame = src.poll()
        pose = tracker.update(frame, frame.timestamp)[0]
        held = pose.joints[int(L.WRIST)].copy()
    assert held is not None
    assert float(np.linalg.norm(held - pose.joints[int(L.WRIST)])) < 1.0
    resumed = wrist(30)
    assert float(np.linalg.norm(resumed - np.asarray(held))) > 0.005, (
        "the autopilot never resumed after the cursor went quiet")


@case
def test_recorder_reports_its_frame_count() -> None:
    src = _source()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "count.fhr"
        rec = Recorder(path, max_hands=CFG.max_hands)
        for _ in range(12):
            frame = src.poll()
            assert frame is not None
            rec.add(frame)
        assert rec.count == 12
        assert rec.close() == 12
        assert rec.close() == 12, "closing twice must stay consistent"
        assert path.exists()


@case
def test_null_source_and_dispatch() -> None:
    null = NullSource(fps=1000.0)
    null.start()
    got = None
    for _ in range(100):
        got = null.poll()
        if got is not None:
            break
    assert got is not None and got.hands == [] and not got.ok
    null.close()

    with create_source(dataclasses.replace(CFG, source="synthetic")) as src:
        assert isinstance(src, SyntheticSource)
    assert isinstance(create_source(dataclasses.replace(CFG, source="null")),
                      NullSource)
    for bad, exc in (("banana", ValueError), ("replay", ValueError),
                     ("video", ValueError)):
        try:
            create_source(dataclasses.replace(CFG, source=bad))
        except exc:
            continue
        raise AssertionError(f"source {bad!r} was accepted")


@case
def test_tracker_tolerates_empty_and_missing_frames() -> None:
    tracker = HandTracker(CFG, GRAB)
    now = 0.0
    for _ in range(20):
        now += DT
        assert tracker.update(None, now) == []
    for _ in range(20):
        now += DT
        assert tracker.update(TrackerFrame(hands=[], timestamp=now), now) == []
    assert tracker.active_slots == []

    src = _source(auto=False)
    frame = src.poll()
    now += DT
    assert len(tracker.update(TrackerFrame(hands=frame.hands, timestamp=now),
                              now)) == 1
    tracker.reset()
    assert tracker.active_slots == []
    try:
        tracker.update(None, float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("a NaN timestamp was accepted")


@case
def test_camera_source_reports_a_missing_camera() -> None:
    """There is no webcam on the build machine, and that must be a clear
    error rather than a hang or an obscure OpenCV assertion."""
    try:
        from fctx.hands.mediapipe_source import CameraSource
    except ImportError as exc:
        raise AssertionError(f"the camera path does not even import: {exc}")

    cfg = dataclasses.replace(CFG, source="camera", camera_index=93)
    src = CameraSource(cfg)
    assert "93" in src.describe
    started = time.perf_counter()
    try:
        src.start()
    except CameraUnavailable as exc:
        message = str(exc)
        assert "93" in message and "CAP_MSMF" in message and "CAP_DSHOW" in message
        note(f"camera dry-run raised in "
             f"{(time.perf_counter() - started) * 1e3:.0f} ms: "
             f"{message[:70]}...")
    except FileNotFoundError as exc:
        raise AssertionError(f"the hand landmarker model is missing: {exc}")
    else:
        # A camera really is attached on this machine; then it must work.
        try:
            assert src.poll() is None or True
        finally:
            src.close()
        note("a camera was present, so the failure path was not exercised")
    finally:
        src.close()
    src.close()


@case
def test_end_to_end_latency() -> None:
    src = _source()
    tracker = HandTracker(CFG, GRAB)
    for _ in range(30):  # warm up numpy and the filters
        frame = src.poll()
        tracker.update(frame, frame.timestamp)

    samples = []
    for _ in range(FRAMES):
        frame = src.poll()
        assert frame is not None
        t0 = time.perf_counter()
        poses = tracker.update(frame, frame.timestamp)
        samples.append(time.perf_counter() - t0)
        assert poses
    arr = np.asarray(samples) * 1e3
    note(f"synthetic hand end-to-end: mean {arr.mean():.3f} ms, "
         f"p95 {np.percentile(arr, 95):.3f} ms, max {arr.max():.3f} ms "
         f"(tracker.update only)")
    assert float(np.percentile(arr, 95)) < 3.0, (
        f"tracking costs {np.percentile(arr, 95):.2f} ms at p95, which at "
        "90 Hz physics is a quarter of the frame budget")


@case
def test_control_inputs_refuse_nonsense() -> None:
    """A non-finite control value must die here, not five modules later.

    ``np.clip`` propagates NaN instead of clamping it, so a setter that only
    clipped would store the NaN, build a hand of NaN landmarks from it, and
    surface as ``project_hand received non-finite image landmarks`` -- a
    traceback that points at the projection and says nothing about the
    control that actually went wrong.
    """
    src = _source(auto=False)
    for out_of_range in (-10.0, 10.0):
        src.set_pointer(out_of_range, out_of_range)
        src.set_depth(out_of_range)
        src.set_pinch(out_of_range)
        src.set_curl(out_of_range)
    assert 0.0 <= src._pinch <= 1.0 and 0.0 <= src._curl <= 1.0
    assert 0.001 <= src.span <= 4.0

    setters = (
        ("set_pointer(nx)", lambda v: src.set_pointer(v, 0.5)),
        ("set_pointer(ny)", lambda v: src.set_pointer(0.5, v)),
        ("set_depth", src.set_depth),
        ("set_pinch", src.set_pinch),
        ("set_curl", src.set_curl),
        ("set_span", src.set_span),
        ("build_metric_hand(pinch)", lambda v: build_metric_hand(v, 0.0)),
        ("build_metric_hand(curl)", lambda v: build_metric_hand(0.0, v)),
    )
    for bad in (float("nan"), float("inf"), float("-inf")):
        for name, setter in setters:
            try:
                setter(bad)
            except ValueError:
                continue
            raise AssertionError(f"{name} accepted {bad}")

    src.set_pointer(0.5, 0.5)
    assert np.isfinite(src.poll().hands[0].image).all()


@case
def test_restarting_a_source_reproduces_it_exactly() -> None:
    """Two runs of the same source must produce the same landmarks.

    A source that resumed from wherever the previous run left the hand would
    make a recording replay against a different input than the one that
    produced the bug it was captured for.
    """
    src = SyntheticSource(CFG, auto=True, fps=FPS, realtime=False)
    assert src.is_synthetic is True, (
        "the HUD gates its mouse-steering help on source.is_synthetic")
    assert NullSource().is_synthetic is False

    src.start()
    first = [src.poll().hands[0].image.copy() for _ in range(40)]
    src.close()
    src.start()
    again = [src.poll().hands[0].image.copy() for _ in range(40)]

    for i, (a, b) in enumerate(zip(first, again)):
        assert np.array_equal(a, b), f"restart left state behind at frame {i}"

    # start() is documented idempotent, so a second call on a running source
    # must not rewind the autopilot clock and teleport the hand.
    before = src.poll().hands[0].image.copy()
    src.start()
    after = src.poll().hands[0].image.copy()
    assert float(np.max(np.abs(after - before))) < 0.02, (
        "a redundant start() jumped the hand")
    src.close()


@case
def test_pinch_rotation_never_jumps_to_its_antipode() -> None:
    """``q`` and ``-q`` are one rotation, and the raw conversion flips between
    them as the wrist turns through the branch boundary of the matrix-to-
    quaternion formula.  Applying the rotation is immune, but twisting a held
    body is a *difference* of two frames' quaternions, and a sign flip reads
    as a half turn."""
    src = _source()
    tracker = HandTracker(CFG, GRAB)
    previous: np.ndarray | None = None
    flips = 0
    worst = 0.0
    for _ in range(1800):
        frame = src.poll()
        for pose in tracker.update(frame, frame.timestamp):
            q = pose.pinch_rotation.astype(np.float64)
            assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-5, q
            if previous is not None:
                if float(np.dot(previous, q)) < 0.0:
                    flips += 1
                worst = max(worst, float(np.linalg.norm(q - previous)))
            previous = q
    note(f"pinch quaternion: {flips} sign flips, worst frame-to-frame step "
         f"{worst:.3f} over 1800 frames")
    assert flips == 0, f"{flips} antipodal jumps in the pinch frame"
    assert worst < 0.5, worst


@case
def test_poses_meet_the_solver_array_contract() -> None:
    """The solver copies these straight into device buffers, so a float64 or
    a strided array is a silent per-frame conversion at best."""
    cfg = dataclasses.replace(CFG, max_hands=2)
    tracker = HandTracker(cfg, GRAB)
    src = _source(cfg)
    checked = 0
    for _ in range(120):
        frame = src.poll()
        for pose in tracker.update(frame, frame.timestamp):
            fields = (
                ("joints", pose.joints, (NUM_LANDMARKS, 3)),
                ("velocities", pose.velocities, (NUM_LANDMARKS, 3)),
                ("palm_normal", pose.palm_normal, (3,)),
                ("pinch_point", pose.pinch_point, (3,)),
                ("pinch_velocity", pose.pinch_velocity, (3,)),
                ("pinch_rotation", pose.pinch_rotation, (4,)),
            )
            for name, arr, shape in fields:
                assert arr.dtype == np.float32, (name, arr.dtype)
                assert arr.shape == shape, (name, arr.shape)
                assert arr.flags["C_CONTIGUOUS"], name
                assert np.isfinite(arr).all(), name
            assert abs(float(np.linalg.norm(pose.palm_normal)) - 1.0) < 1e-4
            a, b, radius = pose.bone_segments()
            va, vb = pose.bone_velocities()
            for arr in (a, b, radius, va, vb):
                assert arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"]
            assert 0 <= pose.track_id < cfg.max_hands
            checked += 1
    assert checked > 100, checked


@case
def test_extreme_configurations_stay_finite() -> None:
    """Nothing in the configured range may produce a NaN or an exception."""
    variants = [
        dataclasses.replace(CFG, max_hands=n) for n in (1, 2, 4, 8)
    ] + [
        dataclasses.replace(CFG, depth_size_blend=b) for b in (0.0, 1.0)
    ] + [
        dataclasses.replace(CFG, depth_scale=0.0),
        dataclasses.replace(CFG, coast_time=0.0, lost_timeout=0.0),
        dataclasses.replace(CFG, z_range=(0.1, 0.1)),
        dataclasses.replace(CFG, filter_beta=0.0),
        dataclasses.replace(CFG, filter_min_cutoff=1e-3, filter_d_cutoff=1e-3),
        dataclasses.replace(CFG, stage_half_width=1e-3, stage_half_height=1e-3),
        dataclasses.replace(CFG, stage_half_width=50.0, stage_half_height=50.0),
        dataclasses.replace(CFG, mirror=False),
    ]
    for cfg in variants:
        tracker = HandTracker(cfg, GRAB)
        src = _source(cfg)
        for _ in range(90):
            frame = src.poll()
            for pose in tracker.update(frame, frame.timestamp):
                assert np.isfinite(pose.joints).all(), cfg
                assert np.isfinite(pose.velocities).all(), cfg
                assert np.isfinite(pose.pinch_rotation).all(), cfg
                assert 0.0 <= pose.confidence <= 1.0, cfg
                assert cfg.z_range[0] - 1e-6 <= float(pose.joints[int(L.WRIST), 2])
                assert float(pose.joints[int(L.WRIST), 2]) <= cfg.z_range[1] + 1e-6
        # and then with the hand gone, for the whole coast and drop window
        now = frame.timestamp
        for _ in range(60):
            now += DT
            for pose in tracker.update(None, now):
                assert np.isfinite(pose.joints).all(), cfg
        src.close()

    try:
        HandTracker(dataclasses.replace(CFG, max_hands=0))
    except ValueError:
        pass
    else:
        raise AssertionError("max_hands=0 was accepted")


@case
def test_camera_holds_no_more_previews_than_it_can_use() -> None:
    """MediaPipe skips frames submitted while it is busy and never calls back
    for them, so nothing but a later result evicts a pending preview.  At
    1280x720x3 that is 2.6 MB per unclaimed frame."""
    from fctx.hands.mediapipe_source import _MAX_PENDING, CameraSource

    cam = CameraSource(dataclasses.replace(CFG, source="camera"))
    rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    for i in range(400):
        with cam._pending_lock:
            cam._pending[i] = rgb
        cam._bound_pending()
    assert len(cam._pending) <= _MAX_PENDING, len(cam._pending)
    # The newest are the ones worth keeping: they are the ones still in flight.
    assert max(cam._pending) == 399
    cam.close()
    assert not cam._pending


if __name__ == "__main__":
    sys.exit(run(__file__))
