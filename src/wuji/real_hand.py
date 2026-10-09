"""Stream retargeted hinge angles to a real right Wuji Hand 2.

Connection follows wuji-mjlab ``scripts/move_wuji_hand.py``: the right hand
is reached at a fixed Zenoh address (auto-connect finds the left hand). Gains
follow the wuji-sdk 2026.9.22 teleop examples. A single servo thread owns the
only joint_command publisher and evaluates the active motion by wall-clock
time, so Python timing hiccups never change step sizes. HINGE_JOINTS order
matches the flat-20 firmware order. Requires ``pip install wuji-sdk==2026.9.22``
(joint feed-forward compensation is on by default from that release).

The hand moves as soon as the streamer starts. Run it supervised with the
hand clear.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Iterable, Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from .joint_map import NUM_HINGES

KP = 5.0
KD = 0.1
EFFORT_LIMIT_A = 1.5
RATE_HZ = 1000.0
_ENABLED = 2  # ext_state: 0=Init 1=Ready 2=Enabled 3=Stopped
# Zenoh data port. 50001 is not what this hand answers on.
RIGHT_HAND_ADDRESS = "192.168.50.111:7447"


def configure_hand(
    hand,
    kp: float = KP,
    kd: float = KD,
    effort_limit: float = EFFORT_LIMIT_A,
    timeout_s: float = 5.0,
) -> None:
    """Check all 20 joints, set effort limit and MIT gains, enable, wait."""
    online = hand.online_joints_count().get()
    if online != NUM_HINGES:
        raise RuntimeError(f"expected 20/20 online joints, got {online}/20")
    hand.effort_limit().set(effort_limit)
    hand.mit_params().set((kp, kd))
    hand.enable()
    # enable() returns before the motors finish enabling.
    sub = hand.joint_diagnostics().subscribe()
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            time.sleep(0.2)
            frame = sub.recv()
            if frame is None or len(frame.joints) != NUM_HINGES:
                continue
            if all(e.status_word.ext_state == _ENABLED for e in frame.joints):
                return
        raise TimeoutError("not all motors reached Enabled")
    finally:
        sub.close()


def read_current_q(hand, nid_to_joint_index, timeout_s: float = 2.0) -> np.ndarray:
    """Newest complete joint_states frame in flat-20 order.

    ``recv()`` returns the oldest queued frame, so the queue is drained first.
    """
    sub = hand.joint_states().subscribe()
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frame = None
            while True:
                newer = sub.recv()
                if newer is None:
                    break
                frame = newer
            if frame is not None and len(frame.joints) == NUM_HINGES:
                q = np.full(NUM_HINGES, np.nan)
                for entry in frame.joints:
                    q[nid_to_joint_index(entry.nid)] = float(entry.position)
                if not np.isnan(q).any():
                    return q
            time.sleep(0.005)
        raise TimeoutError("no complete joint_states frame")
    finally:
        sub.close()


def smooth_keyframes(keyframes: np.ndarray, sigma_frames: float) -> np.ndarray:
    """Zero-phase Gaussian smoothing with the first and last frames kept exact."""
    q = np.asarray(keyframes, dtype=np.float64)
    if sigma_frames <= 0.0 or len(q) < 3:
        return q.copy()
    smooth = gaussian_filter1d(q, sigma_frames, axis=0, mode="nearest")
    u = np.linspace(0.0, 1.0, len(q))[:, None]
    smooth += (1.0 - u) * (q[0] - smooth[0]) + u * (q[-1] - smooth[-1])
    return smooth


class SplineSegment:
    """C2 cubic spline in seconds; zero velocity at the end."""

    def __init__(self, spline: CubicSpline, limits: np.ndarray):
        self._spline = spline
        self._limits = limits
        self.duration = float(spline.x[-1])
        self.end = np.clip(spline(self.duration), limits[:, 0], limits[:, 1])

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if t >= self.duration:
            return self.end, np.zeros(NUM_HINGES)
        q = np.clip(self._spline(t), self._limits[:, 0], self._limits[:, 1])
        return q, self._spline(t, 1)


class TrapezoidSegment:
    """Straight line with constant accel, cruise, then constant decel."""

    def __init__(self, start, goal, accel, peak, ramp, cruise):
        self.start = np.asarray(start, dtype=np.float64)
        self.end = np.asarray(goal, dtype=np.float64)
        self._delta = self.end - self.start
        self._distance = float(np.max(np.abs(self._delta)))
        self._accel, self._peak = accel, peak
        self._ramp, self._cruise = ramp, cruise
        self.duration = 2.0 * ramp + cruise

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if t >= self.duration or self._distance <= 0.0:
            return self.end, np.zeros(NUM_HINGES)
        a, ramp = self._accel, self._ramp
        if t <= ramp:
            s, v = 0.5 * a * t * t, a * t
        elif t <= ramp + self._cruise:
            s = 0.5 * a * ramp * ramp + self._peak * (t - ramp)
            v = self._peak
        else:
            tau = self.duration - t
            s, v = self._distance - 0.5 * a * tau * tau, a * tau
        unit = self._delta / self._distance
        return self.start + unit * s, unit * v


def _trapezoid_phases(
    distance: float, max_speed: float, max_acceleration: float, duration: float
) -> tuple[float, float, float, float]:
    """Acceleration, peak speed, ramp time, cruise time covering ``distance``."""
    t_to_v = max_speed / max_acceleration
    ramp_distance = 0.5 * max_acceleration * t_to_v * t_to_v
    if 2.0 * ramp_distance <= distance:
        t_min = 2.0 * t_to_v + (distance - 2.0 * ramp_distance) / max_speed
    else:
        t_min = 2.0 * math.sqrt(distance / max_acceleration)
    duration = max(float(duration), t_min)
    disc = max(
        (max_acceleration * duration) ** 2 - 4.0 * max_acceleration * distance, 0.0
    )
    peak = min((max_acceleration * duration - math.sqrt(disc)) / 2.0, max_speed)
    if peak <= 1e-12:
        return 0.0, 0.0, 0.0, duration
    ramp = peak / max_acceleration
    cruise = duration - 2.0 * ramp
    if cruise < 0.0:
        accel = 4.0 * distance / duration**2
        return accel, 2.0 * distance / duration, duration / 2.0, 0.0
    return max_acceleration, peak, ramp, cruise


def fit_trapezoid(
    start: np.ndarray,
    goal: np.ndarray,
    duration_s: float,
    limits: np.ndarray,
    max_speed: float,
    max_acceleration: float,
) -> TrapezoidSegment:
    """End-only acceleration from ``start`` (at rest) to ``goal``."""
    start = np.clip(np.asarray(start, dtype=np.float64), limits[:, 0], limits[:, 1])
    goal = np.clip(np.asarray(goal, dtype=np.float64), limits[:, 0], limits[:, 1])
    distance = float(np.max(np.abs(goal - start)))
    if distance <= 1e-12:
        return TrapezoidSegment(start, goal, 0.0, 0.0, 0.0, 0.0)
    phases = _trapezoid_phases(distance, max_speed, max_acceleration, duration_s)
    return TrapezoidSegment(start, goal, *phases)


def fit_clip_spline(
    q0: np.ndarray,
    qd0: np.ndarray,
    keyframes: np.ndarray,
    duration_s: float,
    limits: np.ndarray,
    max_speed: float,
    max_acceleration: float,
    smoothing_sigma: float = 2.0,
) -> SplineSegment:
    """Spline from the current commanded pose and velocity through the clip.

    Keyframes are smoothed, then spread uniformly over ``duration_s``. If the
    clip starts away from ``q0`` a blend knot is prepended. Time is stretched
    globally until dense samples respect the speed and acceleration caps.
    """
    q = np.asarray(keyframes, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != NUM_HINGES or len(q) < 2:
        raise ValueError(f"keyframes must have shape (N, {NUM_HINGES}), N >= 2")
    if not np.isfinite(q).all():
        raise ValueError("keyframes contain non-finite values")
    if duration_s <= 0 or max_speed <= 0 or max_acceleration <= 0:
        raise ValueError("duration, speed, and acceleration must be > 0")
    q0 = np.clip(np.asarray(q0, dtype=np.float64), limits[:, 0], limits[:, 1])
    qd0 = np.asarray(qd0, dtype=np.float64)
    q = smooth_keyframes(np.clip(q, limits[:, 0], limits[:, 1]), smoothing_sigma)
    q = np.clip(q, limits[:, 0], limits[:, 1])

    times = np.linspace(0.0, float(duration_s), len(q))
    gap = float(np.max(np.abs(q[0] - q0)))
    if gap > 1e-3:
        blend = max(0.1, 2.0 * gap / max_speed)
        knots = np.vstack([q0, q])
        times = np.concatenate([[0.0], blend + times])
    else:
        knots = q.copy()
        knots[0] = q0
    zero = np.zeros(NUM_HINGES)
    for _ in range(10):
        spline = CubicSpline(times, knots, axis=0, bc_type=((1, qd0), (1, zero)))
        dense = np.linspace(0.0, times[-1], max(2048, len(times) * 64))
        speed = float(np.max(np.abs(spline(dense, 1))))
        accel = float(np.max(np.abs(spline(dense, 2))))
        ratio = max(speed / max_speed, math.sqrt(accel / max_acceleration))
        if ratio <= 1.0 + 1e-9:
            return SplineSegment(spline, limits)
        times = times * ratio * 1.02
    raise RuntimeError("could not fit a clip within the speed and acceleration caps")


class SmoothHand2Streamer:
    """Servo thread publishing every tick from the active motion.

    ``play`` runs an offline clip as a time-based spline (or trapezoid).
    ``set_target`` tracks a live target while limiting velocity and
    acceleration on every tick.
    """

    def __init__(
        self,
        publisher,
        joint_command_cls,
        q0: np.ndarray,
        limits: np.ndarray,
        rate_hz: float = RATE_HZ,
        max_speed: float = 1.5,
        max_acceleration: float = 8.0,
        smoothing_sigma: float = 2.0,
        velocity_ff: bool | Iterable[int] = False,
        tracker_gain: float = 30.0,
    ):
        if rate_hz <= 0 or max_speed <= 0 or max_acceleration <= 0:
            raise ValueError("rate, speed, and acceleration must be > 0")
        self._publisher = publisher
        self._JointCommand = joint_command_cls
        self.limits = np.asarray(limits, dtype=np.float64).reshape(NUM_HINGES, 2)
        self.rate_hz = float(rate_hz)
        self.max_speed = float(max_speed)
        self.max_acceleration = float(max_acceleration)
        self.smoothing_sigma = float(smoothing_sigma)
        self.tracker_gain = float(tracker_gain)
        if isinstance(velocity_ff, bool):
            self._ff_mask = np.full(NUM_HINGES, 1.0 if velocity_ff else 0.0)
        else:
            self._ff_mask = np.zeros(NUM_HINGES)
            self._ff_mask[list(velocity_ff)] = 1.0
        self.q = np.clip(
            np.asarray(q0, dtype=np.float64), self.limits[:, 0], self.limits[:, 1]
        )
        self.qd = np.zeros(NUM_HINGES)
        self.late_ticks = 0
        self._lock = threading.Lock()
        self._mode = "hold"
        self._segment = None
        self._segment_t0 = 0.0
        self._target = self.q.copy()
        self._last_tick: Optional[float] = None
        self._done = threading.Event()
        self._done.set()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None

    def start(self) -> "SmoothHand2Streamer":
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="wuji-hand-servo", daemon=True
            )
            self._thread.start()
        return self

    def play(
        self,
        keyframes: np.ndarray,
        duration_s: float,
        wait: bool = False,
        smoothing_movement: bool = False,
    ) -> float:
        """Start a clip from the current commanded state; returns its duration.

        ``smoothing_movement`` goes straight to the last keyframe with
        acceleration only on the opening and closing ramps.
        """
        self._raise_if_failed()
        with self._lock:
            if smoothing_movement:
                segment = fit_trapezoid(
                    self.q,
                    np.asarray(keyframes, dtype=np.float64)[-1],
                    duration_s,
                    self.limits,
                    self.max_speed,
                    self.max_acceleration,
                )
            else:
                segment = fit_clip_spline(
                    self.q,
                    self.qd,
                    keyframes,
                    duration_s,
                    self.limits,
                    self.max_speed,
                    self.max_acceleration,
                    self.smoothing_sigma,
                )
            self._segment = segment
            self._segment_t0 = time.monotonic()
            self._mode = "segment"
            self._done.clear()
        if wait:
            self.wait()
        return segment.duration

    def move_to(self, goal: np.ndarray, wait: bool = True) -> float:
        """Rest-to-rest move with end-only acceleration."""
        goal = np.asarray(goal, dtype=np.float64).reshape(1, NUM_HINGES)
        return self.play(
            goal.repeat(2, axis=0), 1e-6, wait=wait, smoothing_movement=True
        )

    def set_target(self, q: np.ndarray) -> None:
        """Track a live target within the speed and acceleration caps."""
        self._raise_if_failed()
        target = np.clip(
            np.asarray(q, dtype=np.float64), self.limits[:, 0], self.limits[:, 1]
        )
        with self._lock:
            self._target = target
            self._mode = "track"
            self._done.set()

    def stop_motion(self) -> None:
        """Brake to the nearest reachable stop without a velocity jump."""
        with self._lock:
            stop = self.q + self.qd * np.abs(self.qd) / (2.0 * self.max_acceleration)
            self._target = np.clip(stop, self.limits[:, 0], self.limits[:, 1])
            self._mode = "track"
            self._done.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        while not self._done.wait(0.05 if timeout is None else timeout):
            self._raise_if_failed()
            if timeout is not None:
                return False
        self._raise_if_failed()
        return True

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Wuji servo thread failed") from self._error

    def _step(self, now: float) -> tuple[np.ndarray, np.ndarray]:
        period = 1.0 / self.rate_hz
        dt = period if self._last_tick is None else now - self._last_tick
        dt = min(max(dt, 0.0), 5.0 * period)
        self._last_tick = now
        if self._mode == "segment":
            t = now - self._segment_t0
            q, qd = self._segment(t)
            if t >= self._segment.duration:
                self._mode = "hold"
                self._done.set()
        elif self._mode == "track" and dt > 0.0:
            error = self._target - self.q
            dist = np.abs(error)
            v_des = np.sign(error) * np.minimum(
                np.minimum(self.max_speed, self.tracker_gain * dist),
                np.sqrt(2.0 * self.max_acceleration * dist),
            )
            accel = np.clip(
                (v_des - self.qd) / dt, -self.max_acceleration, self.max_acceleration
            )
            qd = self.qd + accel * dt
            q = self.q + qd * dt
        else:
            q, qd = self.q, np.zeros(NUM_HINGES) if self._mode == "hold" else self.qd
        q = np.clip(q, self.limits[:, 0], self.limits[:, 1])
        self.q, self.qd = q, np.asarray(qd, dtype=np.float64)
        return self.q.copy(), self.qd.copy()

    def _publish(self, q: np.ndarray, qd: np.ndarray) -> None:
        velocity = qd * self._ff_mask
        self._publisher.send(
            [
                self._JointCommand(float(p), float(v), 0.0)
                for p, v in zip(q, velocity)
            ]
        )

    def _run(self) -> None:
        period = 1.0 / self.rate_hz
        deadline = time.monotonic()
        try:
            while not self._stop.is_set():
                with self._lock:
                    q, qd = self._step(time.monotonic())
                self._publish(q, qd)
                deadline += period
                delay = deadline - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
                elif delay < -period:
                    # Skip missed ticks instead of bursting to catch up.
                    self.late_ticks += 1
                    deadline = time.monotonic()
        except BaseException as exc:
            self._error = exc
            self._done.set()


class RealWujiHand:
    """Enabled right Wuji Hand 2 driven by one ``SmoothHand2Streamer``."""

    def __init__(
        self,
        limits: np.ndarray,
        max_speed: float = 1.5,
        max_acceleration: float = 8.0,
        rate_hz: float = RATE_HZ,
        kp: float = KP,
        kd: float = KD,
        smoothing_sigma: float = 2.0,
        velocity_ff: bool | Iterable[int] = False,
    ):
        try:
            from wuji_sdk import JointCommand, SdkManager, WujiHand2
        except ImportError as exc:
            raise ImportError(
                "Real hand control needs wuji-sdk: pip install wuji-sdk==2026.9.22"
            ) from exc
        self.limits = np.asarray(limits, dtype=np.float64).reshape(NUM_HINGES, 2)
        self._manager = SdkManager.instance()
        self._publisher = None
        self.streamer: Optional[SmoothHand2Streamer] = None
        # Left is auto-discovered. Right answers only at this address.
        self.hand = self._manager.connect(
            address=RIGHT_HAND_ADDRESS, device_name="wuji_hand_2"
        )
        try:
            side = self.hand.handedness().get().lower()
            if side != "right":
                raise RuntimeError(f"connected hand is {side}, expected right")
            configure_hand(self.hand, kp=kp, kd=kd)
            q0 = read_current_q(self.hand, WujiHand2.nid_to_joint_index)
            self._publisher = self.hand.joint_command().publish()
            self.streamer = SmoothHand2Streamer(
                self._publisher,
                JointCommand,
                q0,
                self.limits,
                rate_hz=rate_hz,
                max_speed=max_speed,
                max_acceleration=max_acceleration,
                smoothing_sigma=smoothing_sigma,
                velocity_ff=velocity_ff,
            ).start()
        except BaseException:
            self._shutdown()
            raise
        print(
            f"Real Wuji hand {self.hand.serial_number} (right) enabled, "
            f"kp={kp} kd={kd}, {rate_hz:.0f} Hz"
        )

    @property
    def q(self) -> np.ndarray:
        return self.streamer.q.copy()

    def play(
        self,
        keyframes: np.ndarray,
        duration_s: float,
        wait: bool = False,
        smoothing_movement: bool = False,
    ) -> float:
        return self.streamer.play(
            keyframes, duration_s, wait=wait, smoothing_movement=smoothing_movement
        )

    def wait(self) -> None:
        self.streamer.wait()

    def stop_motion(self) -> None:
        self.streamer.stop_motion()

    def hold(self, seconds: float) -> None:
        """The servo thread keeps publishing the last pose; just wait."""
        time.sleep(seconds)
        self.streamer._raise_if_failed()

    def _shutdown(self) -> None:
        if self.streamer is not None:
            self.streamer.close()
            if self.streamer.late_ticks:
                print(f"Real Wuji hand: {self.streamer.late_ticks} late servo ticks")
            self.streamer = None
        if self._publisher is not None:
            try:
                self._publisher.close()
            except Exception:
                pass
            self._publisher = None
        try:
            self.hand.disable()
        except Exception:
            pass
        finally:
            self._manager.disconnect_all()

    def close(self, open_hand: bool = True) -> None:
        """Open the hand to q=0, then disable the motors and disconnect."""
        if self.streamer is not None and open_hand:
            try:
                self.streamer.move_to(np.zeros(NUM_HINGES))
            except Exception as exc:
                print(f"Real Wuji hand: failed to open before disable: {exc}")
        self._shutdown()
        print("Real Wuji hand disabled")
