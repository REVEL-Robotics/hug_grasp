"""MuJoCo viewer: retarget each MANO skeleton frame onto the Wuji hand."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..models.mano import mano_params_to_animation
from .joint_map import DEFAULT_MJCF_PATH, NUM_HINGES, WRIST_BODY, compose_T
from .retarget import WujiRetargeter, mano_rest_landmarks, wrist_local_landmarks
from .skeleton_viz import aim_camera_at_pair, draw_skeleton_overlay

# Wuji wrist +X points along the thumb. Keep the skeleton wrist just past it.
_SKELETON_THUMB_OFFSET_M = np.array([0.12, 0.0, 0.0], dtype=np.float64)


def _wrist_T_seq(
    T_camera_wrist: np.ndarray,
    T_mano_wuji: np.ndarray,
    n_frames: int,
    pre_offset_m: tuple[float, float],
) -> np.ndarray:
    """Lerp palm translation like Viser; keep predicted wrist rotation fixed."""
    T_cw = np.asarray(T_camera_wrist, dtype=np.float64)
    R = T_cw[:3, :3]
    t_grasp = T_cw[:3, 3]
    offset = np.array([pre_offset_m[0], pre_offset_m[1], 0.0], dtype=np.float64)
    t_pre = t_grasp + R @ offset
    alphas = np.linspace(0.0, 1.0, n_frames)
    Ts = np.zeros((n_frames, 4, 4), dtype=np.float64)
    for i, a in enumerate(alphas):
        t = (1.0 - a) * t_pre + a * t_grasp
        Ts[i] = compose_T(R, t) @ T_mano_wuji
    return Ts


def _mano_joints_seq(
    mano_model,
    mano_params: Optional[torch.Tensor],
    landmarks_cam: np.ndarray,
    n_frames: int,
    pre_offset_m: tuple[float, float],
    joints_seq_cam: Optional[np.ndarray],
    betas: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Camera-frame SNAP joints over time for the overlay."""
    if joints_seq_cam is not None:
        return np.asarray(joints_seq_cam, dtype=np.float64)
    if mano_params is not None:
        device = next(mano_model.buffers()).device
        params = mano_params.to(device).reshape(-1)
        if betas is None:
            shape = torch.zeros(10, device=device)
        else:
            shape = betas.to(device).reshape(-1)[:10]
        _, joints_seq = mano_params_to_animation(
            params,
            shape,
            mano_model,
            n_frames=n_frames,
            pre_offset_m=pre_offset_m,
        )
        return np.asarray(joints_seq, dtype=np.float64)
    # Fallback: hold the final landmarks static.
    lm = np.asarray(landmarks_cam, dtype=np.float64).reshape(1, -1, 3)
    return np.repeat(lm, n_frames, axis=0)


class WujiPreviewSession:
    """Persistent MuJoCo window driven from HUG clicks or a CLI."""

    def __init__(
        self,
        mjcf_path: Optional[Path] = None,
        real_hand: bool = False,
        real_max_speed: float = 1.5,
        real_max_acceleration: float = 8.0,
        real_smoothing_sigma: float = 2.0,
        smoothing_movement: bool = False,
        real_kp: float = 5.0,
        real_kd: float = 0.1,
        real_rate_hz: float = 1000.0,
        real_velocity_ff: bool | tuple[int, ...] = False,
    ):
        self.mjcf_path = Path(mjcf_path) if mjcf_path else DEFAULT_MJCF_PATH
        self.retargeter: Optional[WujiRetargeter] = None
        self.real_hand_enabled = real_hand
        self.real_options = dict(
            max_speed=real_max_speed,
            max_acceleration=real_max_acceleration,
            smoothing_sigma=real_smoothing_sigma,
            kp=real_kp,
            kd=real_kd,
            rate_hz=real_rate_hz,
            velocity_ff=real_velocity_ff,
        )
        self.smoothing_movement = smoothing_movement
        self.real = None
        self.viewer = None
        self._camera_aimed = False
        self._calibrated = False
        self._beta_key: Optional[tuple] = None

    def _ensure(self, mano_model, betas=None) -> WujiRetargeter:
        import mujoco.viewer

        if self.retargeter is None:
            self.retargeter = WujiRetargeter(self.mjcf_path)
        beta_key = None
        if betas is not None:
            if hasattr(betas, "detach"):
                beta_arr = betas.detach().cpu().numpy()
            else:
                beta_arr = np.asarray(betas)
            beta_key = tuple(
                np.round(
                    np.asarray(beta_arr, dtype=np.float64).reshape(-1)[:10], 5
                )
            )
        if not self._calibrated or beta_key != self._beta_key:
            rest = mano_rest_landmarks(mano_model, betas=betas)
            self.retargeter.calibrate_mano_frame(rest)
            self._calibrated = True
            self._beta_key = beta_key
        if self.real_hand_enabled and self.real is None:
            from .real_hand import RealWujiHand

            self.real = RealWujiHand(self.retargeter.limits, **self.real_options)
        if self.viewer is None or not self.viewer.is_running():
            self.viewer = mujoco.viewer.launch_passive(
                self.retargeter.model, self.retargeter.data
            )
            self._camera_aimed = False
        return self.retargeter

    def close(self) -> None:
        if self.real is not None:
            try:
                self.real.close()
            finally:
                self.real = None
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

    def play(
        self,
        mano_model,
        landmarks_cam: np.ndarray,
        T_camera_wrist: np.ndarray,
        t_wrist: np.ndarray,
        duration_s: float = 1.0,
        n_frames: Optional[int] = None,
        pre_offset_m: tuple[float, float] = (0.03, 0.03),
        retarget_every_frame: bool = True,
        joints_seq_cam: Optional[np.ndarray] = None,
        mano_params: Optional[torch.Tensor] = None,
        betas: Optional[torch.Tensor] = None,
        skeleton_offset_m: Optional[np.ndarray] = None,
        no_float: bool = False,
    ) -> np.ndarray:
        """Drive the hand so each frame matches the MANO skeleton.

        Finger IK runs in the canonical wrist frame. The freejoint applies
        ``T_camera_wrist`` once. ``no_float`` holds the wrist at the origin,
        dropping the cartesian approach. Also draws the MANO SNAP skeleton
        beside the hand.

        Returns the end hinge configuration (20,).
        """
        import mujoco

        rt = self._ensure(mano_model, betas)
        n_frames = n_frames or min(120, max(2, int(duration_s * 60)))
        R = np.asarray(T_camera_wrist, dtype=np.float64)[:3, :3]

        joints_seq = _mano_joints_seq(
            mano_model,
            mano_params,
            landmarks_cam,
            n_frames,
            pre_offset_m,
            joints_seq_cam,
            betas=betas,
        )

        if retarget_every_frame:
            qs = np.zeros((len(joints_seq), NUM_HINGES), dtype=np.float64)
            q = np.zeros(NUM_HINGES)
            for i, joints_i in enumerate(joints_seq):
                qs[i] = rt.retarget(
                    wrist_local_landmarks(joints_i, joints_i[0], R),
                    q_init=q,
                )
                q = qs[i]
            q_end = qs[-1]
        else:
            lm_wrist = wrist_local_landmarks(landmarks_cam, t_wrist, R)
            q_end = rt.retarget(lm_wrist, q_init=np.zeros(NUM_HINGES))
            alphas = np.linspace(0.0, 1.0, n_frames)
            qs = alphas[:, None] * q_end[None, :]
            if len(joints_seq) != len(qs):
                idx = np.linspace(0, len(joints_seq) - 1, len(qs)).astype(int)
                joints_seq = joints_seq[idx]

        Ts = _wrist_T_seq(T_camera_wrist, rt.T_mano_wuji, len(qs), pre_offset_m)
        if no_float:
            Ts = Ts.copy()
            Ts[:, :3, 3] = 0.0
        world_offset = (
            np.asarray(skeleton_offset_m, dtype=np.float64).reshape(3)
            if skeleton_offset_m is not None
            else None
        )
        dt = duration_s / max(len(qs) - 1, 1)
        viewer = self.viewer
        if self.real is not None:
            self.real.play(
                qs, duration_s, wait=False, smoothing_movement=self.smoothing_movement
            )
        completed = True
        for i, q in enumerate(qs):
            if viewer is None or not viewer.is_running():
                completed = False
                break
            rt.set_freejoint_T(Ts[i])
            rt.set_hinges(q)
            n_act = rt.model.nu
            rt.data.ctrl[:n_act] = q[:n_act]
            rt.forward()

            wid = mujoco.mj_name2id(rt.model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY)
            palm = rt.data.xpos[wid].copy()
            if world_offset is None:
                wrist_R = rt.data.xmat[wid].reshape(3, 3)
                offset = wrist_R @ _SKELETON_THUMB_OFFSET_M
            else:
                offset = world_offset
            joints_i = joints_seq[i]
            if no_float:
                joints_i = joints_i - joints_i[0]
            skel = joints_i + offset
            draw_skeleton_overlay(viewer, skel)
            if not self._camera_aimed:
                aim_camera_at_pair(viewer, palm, skel.mean(axis=0))
                self._camera_aimed = True
            viewer.sync()
            time.sleep(dt)
        if self.real is not None:
            if completed:
                self.real.wait()
            else:
                self.real.stop_motion()
        return q_end


def play_mano_grasp(
    mano_model,
    landmarks_cam: np.ndarray,
    T_camera_wrist: np.ndarray,
    t_wrist: np.ndarray,
    session: Optional[WujiPreviewSession] = None,
    **kwargs,
) -> tuple[WujiPreviewSession, np.ndarray]:
    session = session or WujiPreviewSession()
    q = session.play(
        mano_model, landmarks_cam, T_camera_wrist, t_wrist, **kwargs
    )
    return session, q
