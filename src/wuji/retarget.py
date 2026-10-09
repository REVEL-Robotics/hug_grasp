"""Vector IK: 21 SNAP landmarks (wrist frame) → 20 Wuji hinge angles.

Matches each phalanx direction in the Wuji wrist frame, the same key-vector
idea as wuji-retargeting, using an analytical MuJoCo Jacobian. Hinge limits
are the Hand 2 Beta 1 MJCF ranges.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from .joint_map import (
    FINGERS,
    HINGE_JOINTS,
    NUM_HINGES,
    WRIST_BODY,
    apply_T,
    compose_T,
    hinge_limits,
    hinge_qpos_addrs,
    invert_T,
    kabsch,
    snap_align_points,
    wrap_mjcf_with_freejoint,
)

# Moving SNAP indices: PIP/MCP, DIP/IP, tip. The knuckle itself is fixed on the palm.
_TASK_INDICES: tuple[int, ...] = tuple(
    idx for spec in FINGERS for idx in spec.snap[1:]
)


def _finger_bones() -> tuple[tuple[int, int], ...]:
    """Parent→child along each finger, knuckle frozen, then three phalanges."""
    bones: list[tuple[int, int]] = []
    for spec in FINGERS:
        parent = spec.snap[0]
        for child in spec.snap[1:]:
            bones.append((parent, child))
            parent = child
    return tuple(bones)


def grasp_dict_to_mano_params(grasp: dict) -> np.ndarray:
    """Rebuild the 99-D vector from a saved Grasp dict."""
    t = np.asarray(grasp["t"], dtype=np.float32).reshape(-1)
    R_6d = np.asarray(grasp["R_6d"], dtype=np.float32).reshape(-1)
    pose_6d = np.asarray(grasp["pose_6d"], dtype=np.float32).reshape(-1)
    return np.concatenate([t, R_6d, pose_6d], axis=0)


def wrist_local_landmarks(
    landmarks_3d: np.ndarray,
    t: np.ndarray,
    R: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Camera-frame SNAP joints → MANO canonical wrist frame.

    Subtracts the wrist translation. When ``R`` is the wrist rotation already
    baked into the joints, it is removed so finger IK stays in the wrist
    frame. The preview freejoint applies that rotation once, on the outside.
    """
    p = np.asarray(landmarks_3d, dtype=np.float64) - np.asarray(
        t, dtype=np.float64
    ).reshape(1, 3)
    if R is not None:
        p = p @ np.asarray(R, dtype=np.float64).reshape(3, 3)
    return p


class WujiRetargeter:
    """MuJoCo FK + bounded least squares for the right Hand 2 Beta 1."""

    def __init__(self, mjcf_path=None):
        import mujoco

        from .joint_map import DEFAULT_MJCF_PATH

        path = mjcf_path or DEFAULT_MJCF_PATH
        xml = wrap_mjcf_with_freejoint(path)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._mujoco = mujoco
        self.qpos_addrs = hinge_qpos_addrs(self.model)
        self.limits = hinge_limits(self.model)
        self._hinge_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in HINGE_JOINTS
            ],
            dtype=np.int32,
        )
        self.free_adr = int(
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "r_wrist_free")
            ]
        )
        self._hinge_dofs = np.array(
            [int(self.model.jnt_dofadr[jid]) for jid in self._hinge_ids],
            dtype=np.int32,
        )
        self._wrist_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY
        )
        self._snap_ids = self._lookup_snap_ids()
        self._bones = _finger_bones()
        self._set_freejoint_identity()
        self.set_hinges(np.zeros(NUM_HINGES))
        mujoco.mj_forward(self.model, self.data)
        self.T_mano_wuji: Optional[np.ndarray] = None
        self._wuji_rest: Optional[np.ndarray] = None
        self._vec_scale = np.ones(21, dtype=np.float64)
        self._cache_q: Optional[np.ndarray] = None
        self._cache: Optional[tuple[np.ndarray, np.ndarray]] = None

    def _lookup_snap_ids(self) -> list[tuple[int, str, int]]:
        mujoco = self._mujoco
        refs: list[tuple[int, str, int]] = []
        for spec in FINGERS:
            for snap_i, kind, name in zip(spec.snap, spec.point_kinds, spec.points):
                obj = (
                    mujoco.mjtObj.mjOBJ_SITE
                    if kind == "site"
                    else mujoco.mjtObj.mjOBJ_BODY
                )
                idx = mujoco.mj_name2id(self.model, obj, name)
                if idx < 0:
                    raise KeyError(f"{kind} {name} missing from MJCF")
                refs.append((snap_i, kind, int(idx)))
        return refs

    def _set_freejoint_identity(self) -> None:
        adr = self.free_adr
        self.data.qpos[adr : adr + 3] = 0.0
        self.data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[:] = 0.0

    def set_hinges(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64).reshape(NUM_HINGES)
        q = np.clip(q, self.limits[:, 0], self.limits[:, 1])
        self.data.qpos[self.qpos_addrs] = q

    def set_freejoint_T(self, T: np.ndarray) -> None:
        from .joint_map import rotmat_to_quat_wxyz

        adr = self.free_adr
        self.data.qpos[adr : adr + 3] = T[:3, 3]
        self.data.qpos[adr + 3 : adr + 7] = rotmat_to_quat_wxyz(T[:3, :3])
        self.data.qvel[:] = 0.0

    def forward(self) -> None:
        self._mujoco.mj_forward(self.model, self.data)

    def _snap_state(self) -> tuple[np.ndarray, np.ndarray]:
        """(21, 3) keypoints and (21, 3, 20) hinge Jacobians, wrist frame."""
        mujoco = self._mujoco
        model, data = self.model, self.data
        pts = np.zeros((21, 3), dtype=np.float64)
        jac = np.zeros((21, 3, NUM_HINGES), dtype=np.float64)
        jp = np.zeros((3, model.nv), dtype=np.float64)
        pts[0] = data.xpos[self._wrist_bid]
        mujoco.mj_jacBody(model, data, jp, None, self._wrist_bid)
        jac[0] = jp[:, self._hinge_dofs]
        for snap_i, kind, idx in self._snap_ids:
            jp.fill(0.0)
            if kind == "site":
                pts[snap_i] = data.site_xpos[idx]
                mujoco.mj_jacSite(model, data, jp, None, idx)
            else:
                pts[snap_i] = data.xpos[idx]
                mujoco.mj_jacBody(model, data, jp, None, idx)
            jac[snap_i] = jp[:, self._hinge_dofs]
        return pts, jac

    def _eval(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=np.float64).reshape(NUM_HINGES)
        if self._cache_q is not None and np.max(np.abs(q - self._cache_q)) <= 1e-12:
            return self._cache  # type: ignore[return-value]
        self._set_freejoint_identity()
        self.set_hinges(q)
        self.forward()
        pts, jac = self._snap_state()
        self._cache_q = np.clip(q, self.limits[:, 0], self.limits[:, 1]).copy()
        self._cache = (pts, jac)
        return pts, jac

    def calibrate_mano_frame(self, mano_rest_landmarks: np.ndarray) -> np.ndarray:
        """Kabsch: Wuji rest keypoints → MANO rest SNAP (wrist frame).

        p_mano = R @ p_wuji + t  stored as 4x4 T_mano_wuji.
        Per-keypoint scales map MANO rest lengths onto the Wuji rest lengths.
        """
        self._set_freejoint_identity()
        self.set_hinges(np.zeros(NUM_HINGES))
        self.forward()
        self._cache_q = None
        src = self._align_points_wuji()
        dst = snap_align_points(np.asarray(mano_rest_landmarks, dtype=np.float64))
        R, t = kabsch(src, dst)
        self.T_mano_wuji = compose_T(R, t)
        rest, _ = self._snap_state()
        self._wuji_rest = rest - rest[0]
        mapped_rest = apply_T(invert_T(self.T_mano_wuji), mano_rest_landmarks)
        mano_len = np.linalg.norm(mapped_rest - mapped_rest[0], axis=1)
        wuji_len = np.linalg.norm(self._wuji_rest, axis=1)
        self._vec_scale[:] = 1.0
        ok = mano_len > 1e-8
        self._vec_scale[ok] = wuji_len[ok] / mano_len[ok]
        self._vec_scale[0] = 0.0
        return self.T_mano_wuji

    def _align_points_wuji(self) -> np.ndarray:
        from .joint_map import align_keypoints_wrist

        return align_keypoints_wrist(self.model, self.data)

    def _targets(self, mapped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Unit phalanx directions and scaled wrist-to-joint positions.

        Directions are the MANO bones expressed in the Wuji wrist frame.
        Positions use the rest length ratio, as in the key-vector optimizer.
        """
        if self._wuji_rest is None:
            raise RuntimeError("call calibrate_mano_frame before retarget")
        directions = np.zeros((21, 3), dtype=np.float64)
        for parent, child in self._bones:
            bone = mapped[child] - mapped[parent]
            length = float(np.linalg.norm(bone))
            if length < 1e-8:
                direction = self._wuji_rest[child] - self._wuji_rest[parent]
                direction = direction / (np.linalg.norm(direction) + 1e-12)
            else:
                direction = bone / length
            directions[child] = direction
        vec = mapped - mapped[0]
        positions = vec * self._vec_scale[:, None]
        return directions, positions

    def retarget(
        self,
        mano_wrist_landmarks: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        pos_weight: float = 0.0,
    ) -> np.ndarray:
        """Match MANO wrist-local SNAP bones with 20 hinge angles.

        Each phalanx direction is matched in the Wuji wrist frame. A positive
        ``pos_weight`` also pulls wrist-to-joint positions, in centimeters.
        """
        if self.T_mano_wuji is None:
            raise RuntimeError("call calibrate_mano_frame before retarget")
        mapped = apply_T(
            invert_T(self.T_mano_wuji),
            np.asarray(mano_wrist_landmarks, dtype=np.float64),
        )
        directions, positions = self._targets(mapped)
        task = np.asarray(_TASK_INDICES, dtype=np.int32)
        bones = self._bones
        pos_w = 100.0 * float(pos_weight)

        q0 = (
            np.zeros(NUM_HINGES, dtype=np.float64)
            if q_init is None
            else np.clip(
                np.asarray(q_init, dtype=np.float64).reshape(NUM_HINGES),
                self.limits[:, 0],
                self.limits[:, 1],
            )
        )

        def _dir_residual(pts: np.ndarray) -> np.ndarray:
            dir_res = np.empty((len(bones), 3), dtype=np.float64)
            for i, (parent, child) in enumerate(bones):
                bone = pts[child] - pts[parent]
                length = float(np.linalg.norm(bone))
                unit = bone / length if length >= 1e-8 else directions[child]
                dir_res[i] = unit - directions[child]
            return dir_res

        def _dir_jacobian(pts: np.ndarray, jac: np.ndarray) -> np.ndarray:
            dir_jac = np.empty((len(bones), 3, NUM_HINGES), dtype=np.float64)
            eye = np.eye(3)
            for i, (parent, child) in enumerate(bones):
                bone = pts[child] - pts[parent]
                length = float(np.linalg.norm(bone))
                if length < 1e-8:
                    dir_jac[i] = 0.0
                    continue
                unit = bone / length
                Jv = jac[child] - jac[parent]
                # d(v/||v||)/dq = (I - uu^T) / ||v|| @ dv/dq
                dir_jac[i] = (eye - np.outer(unit, unit)) @ Jv / length
            return dir_jac.reshape(len(bones) * 3, NUM_HINGES)

        def residual(q: np.ndarray) -> np.ndarray:
            pts, _ = self._eval(q)
            dir_res = _dir_residual(pts).ravel()
            if pos_w == 0.0:
                return dir_res
            pos_res = ((pts[task] - positions[task]) * pos_w).ravel()
            return np.concatenate([dir_res, pos_res])

        def jacobian(q: np.ndarray) -> np.ndarray:
            pts, jac = self._eval(q)
            dir_jac = _dir_jacobian(pts, jac)
            if pos_w == 0.0:
                return dir_jac
            pos_jac = (jac[task] * pos_w).reshape(len(task) * 3, NUM_HINGES)
            return np.concatenate([dir_jac, pos_jac], axis=0)

        res = least_squares(
            residual,
            q0,
            jac=jacobian,
            bounds=(self.limits[:, 0], self.limits[:, 1]),
            method="trf",
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=40,
        )
        q = np.clip(res.x, self.limits[:, 0], self.limits[:, 1])
        self._cache_q = None
        self._set_freejoint_identity()
        self.set_hinges(q)
        self.forward()
        return q.astype(np.float64)


def retarget_landmarks(
    retargeter: WujiRetargeter,
    landmarks_wrist: np.ndarray,
    q_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    return retargeter.retarget(landmarks_wrist, q_init=q_init)


def mano_rest_landmarks(mano_model, device=None, betas=None) -> np.ndarray:
    """(21, 3) SNAP joints at MANO rest (zero axis-angle), wrist-centered."""
    import torch

    if device is None:
        device = next(mano_model.buffers()).device
    pose_coeffs = torch.zeros(1, 48, device=device)
    if betas is None:
        shape = torch.zeros(1, 10, device=device)
    elif torch.is_tensor(betas):
        shape = betas.detach().to(device=device, dtype=torch.float32).reshape(1, 10)
    else:
        shape = torch.as_tensor(betas, device=device, dtype=torch.float32).reshape(1, 10)
    out = mano_model.mano_layer(pose_coeffs, shape)
    return out.joints[0].detach().cpu().numpy()
