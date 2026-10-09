"""Draw SNAP-style skeletons into MuJoCo viewer.user_scn."""

from __future__ import annotations

import numpy as np

from ..utils.visualization_utils import MANO_FINGERTIP_INDICES, MANO_SKELETON_PAIRS
from .joint_map import FINGERS, WRIST_BODY, named_point_world

SKELETON_OFFSET_M = np.array([0.0, 0.3, 0.0], dtype=np.float64)
JOINT_RADIUS = 0.006
TIP_RADIUS = 0.009
BONE_WIDTH = 0.004
JOINT_RGBA = np.array([0.15, 0.75, 1.0, 0.95], dtype=np.float32)
TIP_RGBA = np.array([1.0, 0.25, 0.85, 0.95], dtype=np.float32)
BONE_RGBA = np.array([0.55, 0.85, 1.0, 0.9], dtype=np.float32)

CAM_DISTANCE_M = 1.2
CAM_AZIMUTH_DEG = 90.0
CAM_ELEVATION_DEG = -35.0


def aim_camera_at_pair(viewer, palm_pos: np.ndarray, skel_center: np.ndarray) -> None:
    if viewer is None:
        return
    viewer.cam.lookat[:] = 0.5 * (palm_pos + skel_center)
    viewer.cam.distance = CAM_DISTANCE_M
    viewer.cam.azimuth = CAM_AZIMUTH_DEG
    viewer.cam.elevation = CAM_ELEVATION_DEG


def draw_skeleton_overlay(
    viewer,
    joints_world: np.ndarray,
    pairs=MANO_SKELETON_PAIRS,
    tip_indices=MANO_FINGERTIP_INDICES,
) -> None:
    """Draw joints (spheres) + bones (capsules) into viewer.user_scn."""
    import mujoco

    scn = getattr(viewer, "user_scn", None)
    if scn is None:
        return
    scn.ngeom = 0
    identity = np.eye(3, dtype=np.float64).reshape(9)
    tips = set(tip_indices)
    joints = np.asarray(joints_world, dtype=np.float64).reshape(-1, 3)

    for i, p in enumerate(joints):
        if scn.ngeom >= scn.maxgeom:
            return
        is_tip = i in tips
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([TIP_RADIUS if is_tip else JOINT_RADIUS, 0.0, 0.0]),
            p,
            identity,
            TIP_RGBA if is_tip else JOINT_RGBA,
        )
        scn.ngeom += 1

    for a, b in pairs:
        if scn.ngeom >= scn.maxgeom:
            return
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            identity,
            BONE_RGBA,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            BONE_WIDTH,
            joints[a],
            joints[b],
        )
        scn.ngeom += 1


def wuji_snap_keypoints_world(model, data) -> np.ndarray:
    """(21, 3) SNAP-layout keypoints from Wuji FK (wrist + 5×4 finger points)."""
    import mujoco

    pts = np.zeros((21, 3), dtype=np.float64)
    wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY)
    pts[0] = data.xpos[wid]
    for spec in FINGERS:
        for snap_i, kind, name in zip(spec.snap, spec.point_kinds, spec.points):
            pts[snap_i] = named_point_world(model, data, kind, name)
    return pts
