"""SNAP landmark ↔ Wuji Hand 2 Beta 1 (right) joint / site tables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Vendor MJCF: 20 hinges, r_wrist welded to world, tips as sites.
DEFAULT_MJCF_PATH = Path(
    os.environ.get(
        "WUJI_MJCF",
        "/home/revel/code/wuji-description/hand2/hand2_beta1/body/mjcf/right.xml",
    )
)

NUM_HINGES = 20
FREEJOINT_NAME = "r_wrist_free"
ROOT_BODY = "wuji_root"
WRIST_BODY = "r_wrist"

# Wire / URDF order (thumb → pinky, proximal → distal). Matches adam-moveit.
HINGE_JOINTS: tuple[str, ...] = (
    "r_thumb_cmc_flex",
    "r_thumb_cmc_abd",
    "r_thumb_mcp",
    "r_thumb_ip",
    "r_index_finger_mcp_flex",
    "r_index_finger_mcp_abd",
    "r_index_finger_pip",
    "r_index_finger_dip",
    "r_middle_finger_mcp_flex",
    "r_middle_finger_mcp_abd",
    "r_middle_finger_pip",
    "r_middle_finger_dip",
    "r_ring_finger_mcp_flex",
    "r_ring_finger_mcp_abd",
    "r_ring_finger_pip",
    "r_ring_finger_dip",
    "r_pinky_mcp_flex",
    "r_pinky_mcp_abd",
    "r_pinky_pip",
    "r_pinky_dip",
)


@dataclass(frozen=True)
class FingerSpec:
    """One finger: 4 SNAP landmarks and 4 Wuji FK points (3 bones)."""

    name: str
    snap: tuple[int, int, int, int]  # MCP/CMC, PIP/MCP, DIP/IP, tip
    points: tuple[str, str, str, str]  # body, body, body, site
    point_kinds: tuple[str, str, str, str]


FINGERS: tuple[FingerSpec, ...] = (
    FingerSpec(
        "thumb",
        (1, 2, 3, 4),
        ("r_thumb_proximal", "r_thumb_middle", "r_thumb_distal", "r_thumb_tip"),
        ("body", "body", "body", "site"),
    ),
    FingerSpec(
        "index",
        (5, 6, 7, 8),
        (
            "r_index_finger_proximal",
            "r_index_finger_middle",
            "r_index_finger_distal",
            "r_index_finger_tip",
        ),
        ("body", "body", "body", "site"),
    ),
    FingerSpec(
        "middle",
        (9, 10, 11, 12),
        (
            "r_middle_finger_proximal",
            "r_middle_finger_middle",
            "r_middle_finger_distal",
            "r_middle_finger_tip",
        ),
        ("body", "body", "body", "site"),
    ),
    FingerSpec(
        "ring",
        (13, 14, 15, 16),
        (
            "r_ring_finger_proximal",
            "r_ring_finger_middle",
            "r_ring_finger_distal",
            "r_ring_finger_tip",
        ),
        ("body", "body", "body", "site"),
    ),
    FingerSpec(
        "pinky",
        (17, 18, 19, 20),
        ("r_pinky_proximal", "r_pinky_middle", "r_pinky_distal", "r_pinky_tip"),
        ("body", "body", "body", "site"),
    ),
)

# Palm keypoints used to align MANO wrist ↔ Wuji wrist at rest.
ALIGN_SNAP: tuple[int, ...] = (1, 5, 9, 13, 17, 4, 8, 12, 16, 20)
ALIGN_WUJI: tuple[tuple[str, str], ...] = (
    ("body", "r_thumb_proximal"),
    ("body", "r_index_finger_proximal"),
    ("body", "r_middle_finger_proximal"),
    ("body", "r_ring_finger_proximal"),
    ("body", "r_pinky_proximal"),
    ("site", "r_thumb_tip"),
    ("site", "r_index_finger_tip"),
    ("site", "r_middle_finger_tip"),
    ("site", "r_ring_finger_tip"),
    ("site", "r_pinky_tip"),
)


def wrap_mjcf_with_freejoint(xml_path: Path) -> str:
    """Load vendor MJCF, add a floating root, zero gravity, absolute meshdir."""
    xml_path = Path(xml_path).resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(
            f"Wuji MJCF not found: {xml_path}. Set WUJI_MJCF or install "
            "wuji-description (hand2_beta1 right.xml)."
        )
    xml = xml_path.read_text()
    meshdir = (xml_path.parent / "../meshes/right").resolve()
    xml = xml.replace('meshdir="../meshes/right/"', f'meshdir="{meshdir}/"')
    xml = xml.replace(
        "<option ",
        '<option gravity="0 0 0" ',
        1,
    )
    if "r_wrist_free" not in xml:
        xml = xml.replace(
            '<body name="r_wrist">',
            (
                f'<body name="{ROOT_BODY}">\n'
                f'      <freejoint name="{FREEJOINT_NAME}"/>\n'
                '      <body name="r_wrist">'
            ),
            1,
        )
        xml = xml.replace("</worldbody>", "    </body>\n  </worldbody>", 1)
    return xml


def hinge_limits(model) -> np.ndarray:
    """(20, 2) joint ranges in XML / qpos hinge order."""
    import mujoco

    limits = np.zeros((NUM_HINGES, 2), dtype=np.float64)
    for i, name in enumerate(HINGE_JOINTS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"Joint {name} missing from MJCF")
        limits[i] = model.jnt_range[jid]
    return limits


def hinge_qpos_addrs(model) -> np.ndarray:
    import mujoco

    addrs = np.zeros(NUM_HINGES, dtype=np.int32)
    for i, name in enumerate(HINGE_JOINTS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        addrs[i] = model.jnt_qposadr[jid]
    return addrs


def named_point_world(model, data, kind: str, name: str) -> np.ndarray:
    import mujoco

    if kind == "site":
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise KeyError(f"Site {name} missing")
        return data.site_xpos[sid].copy()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(f"Body {name} missing")
    return data.xpos[bid].copy()


def world_to_wrist(model, data, p_world: np.ndarray) -> np.ndarray:
    import mujoco

    wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY)
    R = data.xmat[wid].reshape(3, 3)
    t = data.xpos[wid]
    return R.T @ (p_world - t)


def finger_keypoints_wrist(model, data) -> np.ndarray:
    """(5, 4, 3) keypoints in the Wuji wrist frame."""
    out = np.zeros((len(FINGERS), 4, 3), dtype=np.float64)
    for f, spec in enumerate(FINGERS):
        for k, (kind, name) in enumerate(zip(spec.point_kinds, spec.points)):
            out[f, k] = world_to_wrist(
                model, data, named_point_world(model, data, kind, name)
            )
    return out


def align_keypoints_wrist(model, data) -> np.ndarray:
    """(N, 3) Wuji rest-alignment points in the wrist frame."""
    pts = []
    for kind, name in ALIGN_WUJI:
        pts.append(
            world_to_wrist(model, data, named_point_world(model, data, kind, name))
        )
    return np.stack(pts, axis=0)


def unit_bone_vectors(pts: np.ndarray) -> np.ndarray:
    """pts (..., 4, 3) → (..., 3, 3) unit MCP→PIP, PIP→DIP, DIP→tip."""
    bones = pts[..., 1:, :] - pts[..., :-1, :]
    n = np.linalg.norm(bones, axis=-1, keepdims=True).clip(min=1e-8)
    return bones / n


def snap_finger_points(landmarks: np.ndarray) -> np.ndarray:
    """(21, 3) SNAP landmarks → (5, 4, 3) finger keypoints."""
    return np.stack([landmarks[list(spec.snap)] for spec in FINGERS], axis=0)


def snap_align_points(landmarks: np.ndarray) -> np.ndarray:
    return np.asarray(landmarks)[list(ALIGN_SNAP)]


def kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform mapping src → dst: p_dst = R @ p_src + t."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = mu_d - R @ mu_s
    return R, t


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    q_xyzw = Rotation.from_matrix(R).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def compose_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv
