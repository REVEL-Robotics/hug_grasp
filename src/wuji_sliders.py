"""Interactive Wuji joint sliders + kinematic skeleton overlay (no pkl).

Drives Hand 2 Beta 1 (right) with 20 hinge sliders. The same FK keypoints are
drawn as a SNAP-layout skeleton 0.3 m beside the mesh hand.
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

from .wuji.joint_map import (
    DEFAULT_MJCF_PATH,
    HINGE_JOINTS,
    NUM_HINGES,
    WRIST_BODY,
    hinge_limits,
    hinge_qpos_addrs,
    wrap_mjcf_with_freejoint,
)
from .wuji.skeleton_viz import (
    SKELETON_OFFSET_M,
    aim_camera_at_pair,
    draw_skeleton_overlay,
    wuji_snap_keypoints_world,
)

console = Console()

# Short labels for the slider panel.
_SLIDER_LABELS = (
    "thumb cmc_flex",
    "thumb cmc_abd",
    "thumb mcp",
    "thumb ip",
    "index mcp_flex",
    "index mcp_abd",
    "index pip",
    "index dip",
    "middle mcp_flex",
    "middle mcp_abd",
    "middle pip",
    "middle dip",
    "ring mcp_flex",
    "ring mcp_abd",
    "ring pip",
    "ring dip",
    "pinky mcp_flex",
    "pinky mcp_abd",
    "pinky pip",
    "pinky dip",
)


def main(
    mjcf: Path = DEFAULT_MJCF_PATH,
    skeleton_offset_m: float = 0.3,
) -> None:
    """Open MuJoCo + joint sliders for Wuji right hand.

    Args:
        mjcf: Path to Hand 2 Beta 1 right MJCF (or set WUJI_MJCF).
        skeleton_offset_m: Lateral offset of the FK skeleton beside the mesh.
    """
    import mujoco
    import mujoco.viewer

    xml = wrap_mjcf_with_freejoint(mjcf)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    q_addrs = hinge_qpos_addrs(model)
    limits = hinge_limits(model)
    free_adr = int(
        model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_wrist_free")
        ]
    )
    # Pin palm at the origin.
    data.qpos[free_adr : free_adr + 3] = 0.0
    data.qpos[free_adr + 3 : free_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qpos[q_addrs] = 0.0
    mujoco.mj_forward(model, data)

    offset = np.array([0.0, skeleton_offset_m, 0.0], dtype=np.float64)
    state = {
        "q": np.zeros(NUM_HINGES, dtype=np.float64),
        "lock": threading.Lock(),
        "running": True,
    }

    viewer = mujoco.viewer.launch_passive(model, data)
    console.print(
        "[cyan]Wuji sliders[/cyan]: mesh hand + FK skeleton "
        f"offset {skeleton_offset_m:.2f} m. Close the slider window to exit."
    )

    def apply_and_draw() -> None:
        with state["lock"]:
            q = state["q"].copy()
        data.qpos[q_addrs] = np.clip(q, limits[:, 0], limits[:, 1])
        data.qpos[free_adr : free_adr + 3] = 0.0
        data.qpos[free_adr + 3 : free_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        n_act = model.nu
        data.ctrl[:n_act] = data.qpos[q_addrs][:n_act]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        skel = wuji_snap_keypoints_world(model, data) + offset
        draw_skeleton_overlay(viewer, skel)
        wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY)
        aim_camera_at_pair(viewer, data.xpos[wid].copy(), skel.mean(axis=0))
        viewer.sync()

    # ---- Tk slider panel ----
    root = tk.Tk()
    root.title("Wuji Hand 2 Beta 1 — joint sliders")
    root.geometry("420x720")

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

    scales: list[tk.Scale] = []

    def on_slide(i: int, raw: str) -> None:
        with state["lock"]:
            state["q"][i] = float(raw)

    for i, (label, lo, hi) in enumerate(
        zip(_SLIDER_LABELS, limits[:, 0], limits[:, 1])
    ):
        row = ttk.Frame(frame)
        row.pack(fill="x", padx=8, pady=2)
        ttk.Label(row, text=label, width=18).pack(side="left")
        scale = tk.Scale(
            row,
            from_=float(lo),
            to=float(hi),
            resolution=0.01,
            orient="horizontal",
            length=240,
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
        # Give the viewer thread a moment to exit.
        time.sleep(0.05)


if __name__ == "__main__":
    tyro.cli(main)
