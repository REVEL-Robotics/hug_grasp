"""Interactive MANO joint sliders for the capsule hand.

Drives the 15 finger ball joints in ``assets/mano_mujoco/capsule_hand.xml``.
Each joint is an axis-angle (x, y, z), written into MuJoCo as a quaternion.
The wrist stays pinned at the origin.
"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import tyro
from rich.console import Console

console = Console()

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MJCF_PATH = _REPO_ROOT / "assets" / "mano_mujoco" / "capsule_hand.xml"

# MANO kinematic order: index, middle, pinky, ring, thumb; proximal → distal.
_JOINTS: tuple[tuple[str, str], ...] = (
    ("index mcp", "mano_joint_1"),
    ("index pip", "mano_joint_2"),
    ("index dip", "mano_joint_3"),
    ("middle mcp", "mano_joint_4"),
    ("middle pip", "mano_joint_5"),
    ("middle dip", "mano_joint_6"),
    ("pinky mcp", "mano_joint_7"),
    ("pinky pip", "mano_joint_8"),
    ("pinky dip", "mano_joint_9"),
    ("ring mcp", "mano_joint_10"),
    ("ring pip", "mano_joint_11"),
    ("ring dip", "mano_joint_12"),
    ("thumb cmc", "mano_joint_13"),
    ("thumb mcp", "mano_joint_14"),
    ("thumb ip", "mano_joint_15"),
)
_AXES = ("x", "y", "z")
NUM_JOINTS = len(_JOINTS)
NUM_DOFS = NUM_JOINTS * 3
WRIST_BODY = "mano_body_0"
_ANGLE_LIMIT = np.pi


def axis_angles_to_quat_wxyz(aa: np.ndarray) -> np.ndarray:
    """(N, 3) axis-angle radians → (N, 4) MuJoCo quaternions (w, x, y, z)."""
    ang = np.linalg.norm(aa, axis=1)
    quat = np.zeros((aa.shape[0], 4), dtype=np.float64)
    quat[:, 0] = 1.0
    big = ang >= 1e-12
    if np.any(big):
        axis = aa[big] / ang[big, None]
        half = 0.5 * ang[big]
        quat[big, 0] = np.cos(half)
        quat[big, 1:] = axis * np.sin(half)[:, None]
    return quat


def _ball_qpos_addrs(model) -> np.ndarray:
    import mujoco

    addrs = np.zeros(NUM_JOINTS, dtype=np.int32)
    for i, (_, name) in enumerate(_JOINTS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"Joint {name} missing from MJCF")
        addrs[i] = int(model.jnt_qposadr[jid])
    return addrs


def _wrist_qpos(model) -> tuple[np.ndarray, int]:
    """Slide addresses (3,) and ball-quaternion address for the wrist."""
    import mujoco

    slides = []
    for name in ("wrist_tx", "wrist_ty", "wrist_tz"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"Joint {name} missing from MJCF")
        slides.append(int(model.jnt_qposadr[jid]))
    rot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_rot")
    if rot < 0:
        raise KeyError("Joint wrist_rot missing from MJCF")
    return np.asarray(slides, dtype=np.int32), int(model.jnt_qposadr[rot])


def main(
    mjcf: Path = DEFAULT_MJCF_PATH,
) -> None:
    """Open MuJoCo + axis-angle sliders for the MANO capsule hand.

    Args:
        mjcf: Path to ``capsule_hand.xml``.
    """
    import mujoco
    import mujoco.viewer

    mjcf = Path(mjcf).resolve()
    if not mjcf.is_file():
        raise FileNotFoundError(f"MANO MJCF not found: {mjcf}")

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0.0
    q_addrs = _ball_qpos_addrs(model)
    slide_addrs, wrist_quat_adr = _wrist_qpos(model)

    def pin_wrist() -> None:
        data.qpos[slide_addrs] = 0.0
        data.qpos[wrist_quat_adr : wrist_quat_adr + 4] = np.array(
            [1.0, 0.0, 0.0, 0.0]
        )

    pin_wrist()
    for adr in q_addrs:
        data.qpos[adr : adr + 4] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)

    state = {
        "q": np.zeros(NUM_DOFS, dtype=np.float64),
        "lock": threading.Lock(),
        "running": True,
    }

    viewer = mujoco.viewer.launch_passive(model, data)
    wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY)
    viewer.cam.lookat[:] = data.xpos[wid]
    viewer.cam.distance = 0.45
    viewer.cam.azimuth = 90.0
    viewer.cam.elevation = -25.0
    console.print(
        "[cyan]MANO sliders[/cyan]: capsule hand, 15 ball joints × xyz "
        "axis-angle. Close the slider window to exit."
    )

    def apply_and_draw() -> None:
        with state["lock"]:
            q = state["q"].copy()
        q = np.clip(q, -_ANGLE_LIMIT, _ANGLE_LIMIT).reshape(NUM_JOINTS, 3)
        quats = axis_angles_to_quat_wxyz(q)
        pin_wrist()
        for adr, quat in zip(q_addrs, quats):
            data.qpos[adr : adr + 4] = quat
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        viewer.sync()

    root = tk.Tk()
    root.title("MANO capsule hand — joint sliders")
    root.geometry("460x780")

    canvas = tk.Canvas(root)
    scroll = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    frame = ttk.Frame(canvas)
    frame.bind(
        "<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=frame, anchor="nw")
    canvas.configure(yscrollcommand=scroll.set)
    canvas.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")

    def _on_mousewheel(event: tk.Event) -> None:
        canvas.yview_scroll(int(-event.delta / 120) or (-1 if event.num == 4 else 1), "units")

    canvas.bind_all("<Button-4>", lambda _e: canvas.yview_scroll(-3, "units"))
    canvas.bind_all("<Button-5>", lambda _e: canvas.yview_scroll(3, "units"))
    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    scales: list[tk.Scale] = []

    def on_slide(i: int, raw: str) -> None:
        with state["lock"]:
            state["q"][i] = float(raw)

    current_finger = ""
    for j, (label, _name) in enumerate(_JOINTS):
        finger = label.split()[0]
        if finger != current_finger:
            current_finger = finger
            ttk.Label(frame, text=finger, font=("TkDefaultFont", 10, "bold")).pack(
                anchor="w", padx=8, pady=(10, 2)
            )
        for a, axis in enumerate(_AXES):
            i = j * 3 + a
            row = ttk.Frame(frame)
            row.pack(fill="x", padx=8, pady=1)
            ttk.Label(row, text=f"{label} {axis}", width=16).pack(side="left")
            scale = tk.Scale(
                row,
                from_=-_ANGLE_LIMIT,
                to=_ANGLE_LIMIT,
                resolution=0.01,
                orient="horizontal",
                length=260,
                command=lambda v, idx=i: on_slide(idx, v),
            )
            scale.set(0.0)
            scale.pack(side="left", fill="x", expand=True)
            scales.append(scale)

    def reset_all() -> None:
        with state["lock"]:
            state["q"][:] = 0.0
        for s in scales:
            s.set(0.0)

    ttk.Button(frame, text="Reset all to 0", command=reset_all).pack(
        fill="x", padx=8, pady=8
    )

    def tick() -> None:
        if not state["running"] or not viewer.is_running():
            state["running"] = False
            root.quit()
            return
        apply_and_draw()
        root.after(16, tick)

    def on_close() -> None:
        state["running"] = False
        root.quit()

    root.protocol("WM_DELETE_WINDOW", on_close)
    apply_and_draw()
    root.after(16, tick)
    try:
        root.mainloop()
    finally:
        state["running"] = False
        try:
            viewer.close()
        except Exception:
            pass
        time.sleep(0.05)


if __name__ == "__main__":
    tyro.cli(main)
