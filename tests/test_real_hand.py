"""Tests for the Wuji Hand 2 servo streamer and its motion profiles."""

from __future__ import annotations

import sys
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

from hug.wuji.real_hand import (
    KD,
    KP,
    RIGHT_HAND_ADDRESS,
    RealWujiHand,
    SmoothHand2Streamer,
    fit_clip_spline,
    fit_trapezoid,
    read_current_q,
    smooth_keyframes,
)


LIMITS = np.tile(np.array([-2.0, 2.0]), (20, 1))


class JointCommand:
    def __init__(self, position, velocity, effort):
        self.position = position
        self.velocity = velocity
        self.effort = effort


class Publisher:
    def __init__(self, sent):
        self.sent = sent
        self.closed = False

    def send(self, commands):
        self.sent.append(
            (
                time.monotonic(),
                np.array([c.position for c in commands]),
                np.array([c.velocity for c in commands]),
            )
        )

    def close(self):
        self.closed = True


def _fake_sdk(start_positions=None):
    sent: list = []
    calls: dict[str, object] = {"publishers": []}

    class Subscription:
        def __init__(self, frames):
            self.frames = list(frames)

        def recv(self):
            return self.frames.pop(0) if self.frames else None

        def close(self):
            pass

    class Topic:
        def __init__(self, frames=()):
            self.frames = frames

        def subscribe(self):
            return Subscription(self.frames)

        def publish(self):
            pub = Publisher(sent)
            calls["publishers"].append(pub)
            return pub

    class Value:
        def __init__(self, name, value):
            self.name, self.value = name, value

        def get(self):
            return self.value

        def set(self, value):
            calls[self.name] = value

    entry = types.SimpleNamespace(status_word=types.SimpleNamespace(ext_state=2))
    diagnostics = types.SimpleNamespace(joints=[entry] * 20)
    positions = np.zeros(20) if start_positions is None else start_positions
    states = types.SimpleNamespace(
        joints=[
            types.SimpleNamespace(nid=i, position=float(positions[i]))
            for i in range(20)
        ]
    )

    class Hand:
        serial_number = "FAKE.RIGHT"

        def online_joints_count(self):
            return Value("online", 20)

        def handedness(self):
            return Value("side", "right")

        def effort_limit(self):
            return Value("effort_limit", 0.0)

        def mit_params(self):
            return Value("mit_params", (0.0, 0.0))

        def enable(self):
            calls["enabled"] = True

        def disable(self):
            calls["disabled"] = True

        def joint_diagnostics(self):
            return Topic([diagnostics])

        def joint_states(self):
            return Topic([states])

        def joint_command(self):
            return Topic()

    class Manager:
        @staticmethod
        def instance():
            return Manager()

        def connect(self, *, address, device_name):
            calls["connect"] = (address, device_name)
            return Hand()

        def disconnect_all(self):
            calls["disconnected"] = True

    module = types.ModuleType("wuji_sdk")
    module.JointCommand = JointCommand
    module.SdkManager = Manager
    module.WujiHand2 = type(
        "WujiHand2", (), {"nid_to_joint_index": staticmethod(lambda nid: nid)}
    )
    return module, sent, calls


def _sample(segment, rate=2000.0):
    t = np.arange(0.0, segment.duration + 1.0 / rate, 1.0 / rate)
    q = np.array([segment(ti)[0] for ti in t])
    qd = np.array([segment(ti)[1] for ti in t])
    return t, q, qd


class ProfileTest(unittest.TestCase):
    def test_gaussian_smoothing_keeps_endpoints(self):
        rng = np.random.default_rng(0)
        frames = np.zeros((60, 20))
        frames[:, 0] = np.linspace(0.0, 1.0, 60) + 0.03 * rng.standard_normal(60)
        smooth = smooth_keyframes(frames, 2.0)
        raw_noise = np.mean(np.abs(np.diff(frames[:, 0], n=2)))
        smooth_noise = np.mean(np.abs(np.diff(smooth[:, 0], n=2)))
        self.assertLess(smooth_noise, 0.3 * raw_noise)
        np.testing.assert_allclose(smooth[0], frames[0])
        np.testing.assert_allclose(smooth[-1], frames[-1])

    def test_clip_spline_blends_from_current_state_within_caps(self):
        rng = np.random.default_rng(1)
        frames = np.zeros((60, 20))
        noise = 0.02 * rng.standard_normal(60)
        frames[:, 3] = np.sin(np.linspace(0.0, np.pi, 60)) + noise
        frames[-1, 3] = 0.3
        q0 = np.full(20, 0.2)
        qd0 = np.full(20, 0.5)
        segment = fit_clip_spline(q0, qd0, frames, 0.5, LIMITS, 1.5, 8.0)
        np.testing.assert_allclose(segment(0.0)[0], q0)
        np.testing.assert_allclose(segment(0.0)[1], qd0)
        np.testing.assert_allclose(segment.end, frames[-1], atol=1e-9)
        _, q, qd = _sample(segment)
        self.assertLessEqual(np.max(np.abs(qd)), 1.5 * 1.01)
        acc = np.diff(qd, axis=0) * 2000.0
        self.assertLessEqual(np.max(np.abs(acc)), 8.0 * 1.05)
        self.assertGreater(segment.duration, 0.5)

    def test_trapezoid_accelerates_only_at_ends(self):
        goal = np.zeros(20)
        goal[0] = 1.0
        goal[5] = -0.5
        segment = fit_trapezoid(np.zeros(20), goal, 0.05, LIMITS, 1.5, 8.0)
        _, q, qd = _sample(segment)
        np.testing.assert_allclose(q[0], 0.0)
        np.testing.assert_allclose(q[-1], goal)
        self.assertLessEqual(np.max(np.abs(qd[:, 0])), 1.5 + 1e-9)
        acc = np.diff(qd[:, 0]) * 2000.0
        n = len(acc)
        np.testing.assert_allclose(acc[5 : n // 10], 8.0, atol=1e-6)
        np.testing.assert_allclose(acc[n // 2 - 20 : n // 2 + 20], 0.0, atol=1e-6)
        np.testing.assert_allclose(acc[-n // 10 : -5], -8.0, atol=1e-6)


class StreamerTest(unittest.TestCase):
    def test_tracker_limits_speed_and_acceleration_every_tick(self):
        sent: list = []
        streamer = SmoothHand2Streamer(
            Publisher(sent), JointCommand, np.zeros(20), LIMITS, rate_hz=1000.0
        )
        target = np.zeros(20)
        target[2] = 1.0
        streamer.set_target(target)
        q_hist, qd_hist = [], []
        for i in range(3000):
            q, qd = streamer._step(i * 1e-3)
            q_hist.append(q[2])
            qd_hist.append(qd[2])
        qd_hist = np.array(qd_hist)
        acc = np.diff(qd_hist) / 1e-3
        self.assertLessEqual(np.max(np.abs(qd_hist)), 1.5 + 1e-9)
        self.assertLessEqual(np.max(np.abs(acc)), 8.0 + 1e-6)
        self.assertAlmostEqual(q_hist[-1], 1.0, places=3)
        self.assertLess(max(q_hist), 1.0 + 1e-3)

    def test_velocity_feedforward_only_on_selected_joints(self):
        sent: list = []
        streamer = SmoothHand2Streamer(
            Publisher(sent), JointCommand, np.zeros(20), LIMITS, velocity_ff=[4]
        )
        streamer._publish(np.zeros(20), np.full(20, 0.7))
        velocity = sent[-1][2]
        self.assertAlmostEqual(velocity[4], 0.7)
        self.assertEqual(np.count_nonzero(velocity), 1)

    def test_read_current_q_drains_to_newest_frame(self):
        def frame(value):
            return types.SimpleNamespace(
                joints=[types.SimpleNamespace(nid=i, position=value) for i in range(20)]
            )

        sub = types.SimpleNamespace(frames=[frame(0.1), frame(0.2), frame(0.3)])
        sub.recv = lambda: sub.frames.pop(0) if sub.frames else None
        sub.close = lambda: None
        hand = types.SimpleNamespace(
            joint_states=lambda: types.SimpleNamespace(subscribe=lambda: sub)
        )
        np.testing.assert_allclose(read_current_q(hand, lambda nid: nid), 0.3)


class RealHandTest(unittest.TestCase):
    def test_play_wait_stop_and_close_with_fake_sdk(self):
        start = np.full(20, 0.05)
        sdk, sent, calls = _fake_sdk(start)
        with patch.dict(sys.modules, {"wuji_sdk": sdk}):
            hand = RealWujiHand(LIMITS, max_speed=3.0, max_acceleration=40.0)
            self.assertEqual(calls["connect"], (RIGHT_HAND_ADDRESS, "wuji_hand_2"))
            self.assertEqual(calls["mit_params"], (KP, KD))
            self.assertEqual((KP, KD), (5.0, 0.1))
            self.assertEqual(len(calls["publishers"]), 1)
            np.testing.assert_allclose(hand.q, start)

            frames = np.zeros((10, 20))
            frames[:, 0] = np.linspace(0.05, 0.3, 10)
            duration = hand.play(frames, 0.1)
            hand.wait()
            np.testing.assert_allclose(hand.q[0], 0.3, atol=1e-9)
            self.assertGreaterEqual(duration, 0.1)
            positions = np.array([s[1][0] for s in sent])
            self.assertLess(np.max(np.abs(np.diff(positions))), 3.0 / 1000.0 * 1.5)
            self.assertTrue(np.all(np.array([s[2] for s in sent]) == 0.0))

            far = np.zeros((2, 20))
            far[-1, 1] = 1.5
            hand.play(far, 1.0)
            time.sleep(0.2)
            hand.stop_motion()
            time.sleep(0.4)
            self.assertLess(hand.q[1], 1.4)
            self.assertLess(abs(hand.streamer.qd[1]), 1e-3)

            hand.close(open_hand=False)
            self.assertTrue(calls["publishers"][0].closed)
            self.assertTrue(calls["disabled"])
            self.assertTrue(calls["disconnected"])


if __name__ == "__main__":
    unittest.main()
