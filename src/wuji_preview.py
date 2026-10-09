"""CLI: retarget a saved HUG grasp pickle onto Wuji Hand 2 Beta 1 in MuJoCo."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from rich.console import Console

from .models.mano import MANO, mano_params_to_animation
from .utils.data_keys import PROJECT_ROOT
from .wuji.joint_map import DEFAULT_MJCF_PATH
from .wuji.preview import WujiPreviewSession
from .wuji.retarget import grasp_dict_to_mano_params

console = Console()

DEFAULT_DATASET = PROJECT_ROOT / "data" / "hug_bench"


def _pred_dir(dataset_path: Path) -> Path:
    nested = dataset_path / "grasp_pred"
    if nested.is_dir():
        return nested
    if dataset_path.is_dir() and dataset_path.name == "grasp_pred":
        return dataset_path
    return nested


def list_grasp_pred_pkls(dataset_path: Path) -> list[Path]:
    pred_dir = _pred_dir(dataset_path)
    if not pred_dir.is_dir():
        return []
    return sorted(pred_dir.rglob("*.pkl"), key=lambda p: p.as_posix())


def _load_grasp(from_pkl: Path) -> dict:
    with open(from_pkl, "rb") as f:
        data = pickle.load(f)
    grasp = data["grasp"] if isinstance(data, dict) and "grasp" in data else data
    if grasp is None:
        raise ValueError(f"{from_pkl} has no grasp")
    if not isinstance(grasp, dict):
        grasp = {
            "t": grasp.t,
            "R_6d": grasp.R_6d,
            "pose_6d": grasp.pose_6d,
            "landmarks_3d": grasp.landmarks_3d,
            "T_camera_wrist": grasp.T_camera_wrist,
        }
    return grasp


def _play_kwargs(
    mano,
    grasp: dict,
    duration: float,
    pre_offset_cm: float,
    retarget_every_frame: bool,
) -> dict:
    device = next(mano.buffers()).device
    mano_params = torch.from_numpy(grasp_dict_to_mano_params(grasp)).to(device)
    off = pre_offset_cm / 100.0
    n_frames = min(120, max(2, int(duration * 60)))
    if "shape" in grasp and grasp["shape"] is not None:
        betas = torch.from_numpy(
            np.asarray(grasp["shape"], dtype=np.float32).reshape(10)
        ).to(device)
    else:
        betas = torch.zeros(10, device=device)
    _, joints_seq = mano_params_to_animation(
        mano_params,
        betas,
        mano,
        n_frames=n_frames,
        pre_offset_m=(off, off),
    )
    return dict(
        duration_s=duration,
        n_frames=n_frames,
        pre_offset_m=(off, off),
        retarget_every_frame=retarget_every_frame,
        joints_seq_cam=joints_seq,
        mano_params=mano_params,
        betas=betas,
        landmarks=np.asarray(grasp["landmarks_3d"]),
        T=np.asarray(grasp["T_camera_wrist"]),
        t=np.asarray(grasp["t"]).reshape(3),
    )


def _play_once(session: WujiPreviewSession, mano, packed: dict) -> bool:
    session.play(
        mano,
        packed["landmarks"],
        packed["T"],
        packed["t"],
        duration_s=packed["duration_s"],
        n_frames=packed["n_frames"],
        pre_offset_m=packed["pre_offset_m"],
        retarget_every_frame=packed["retarget_every_frame"],
        joints_seq_cam=packed["joints_seq_cam"],
        mano_params=packed["mano_params"],
        betas=packed["betas"],
        no_float=packed["no_float"],
    )
    return session.viewer is not None and session.viewer.is_running()


def play_pkls(
    paths: list[Path],
    mjcf: Path,
    duration: float,
    pre_offset_cm: float,
    retarget_every_frame: bool,
    loop: bool,
    pred_dir: Optional[Path] = None,
    real_hand: bool = False,
    real_max_speed: float = 1.5,
    real_max_acceleration: float = 8.0,
    real_smoothing_sigma: float = 2.0,
    real_hold: float = 1.0,
    no_float: bool = False,
    smoothing_movement: bool = False,
    real_kp: float = 5.0,
    real_kd: float = 0.1,
    real_rate_hz: float = 1000.0,
    real_velocity_ff_joints: tuple[int, ...] = (),
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mano = MANO().to(device)
    session = WujiPreviewSession(
        mjcf,
        real_hand=real_hand,
        real_max_speed=real_max_speed,
        real_max_acceleration=real_max_acceleration,
        real_smoothing_sigma=real_smoothing_sigma,
        smoothing_movement=smoothing_movement,
        real_kp=real_kp,
        real_kd=real_kd,
        real_rate_hz=real_rate_hz,
        real_velocity_ff=real_velocity_ff_joints,
    )
    try:
        while True:
            for path in paths:
                name = (
                    path.relative_to(pred_dir).as_posix()
                    if pred_dir is not None
                    else path.name
                )
                console.print(f"[cyan]{name}[/cyan]")
                packed = _play_kwargs(
                    mano,
                    _load_grasp(path),
                    duration,
                    pre_offset_cm,
                    retarget_every_frame,
                )
                packed["no_float"] = no_float
                if not _play_once(session, mano, packed):
                    return
                if session.real is not None:
                    session.real.hold(real_hold)
            if not loop:
                break
    except KeyboardInterrupt:
        console.print("\nstopping")
    finally:
        session.close()


def main(
    from_pkl: Optional[Path] = None,
    dataset_path: Path = DEFAULT_DATASET,
    mjcf: Path = DEFAULT_MJCF_PATH,
    duration: float = 1.0,
    pre_offset_cm: float = 5.0,
    retarget_every_frame: bool = True,
    loop: bool = True,
    real_hand: bool = False,
    real_max_speed: float = 1.5,
    real_max_acceleration: float = 8.0,
    real_smoothing_sigma: float = 2.0,
    real_hold: float = 1.0,
    no_float: bool = False,
    smoothing_movement: bool = False,
    real_kp: float = 5.0,
    real_kd: float = 0.1,
    real_rate_hz: float = 1000.0,
    real_velocity_ff_joints: tuple[int, ...] = (),
) -> None:
    """Play saved grasp_pred pkls on the Wuji right hand.

    With no --from-pkl, plays every pickle under <dataset-path>/grasp_pred
    one after another (prints each filename). --loop repeats the sequence.

    Args:
        from_pkl: Single GraspData pickle. If omitted, play the whole folder.
        dataset_path: Dataset folder whose grasp_pred/ is used (default data/hug_bench).
        mjcf: Wuji Hand 2 Beta 1 right MJCF (or set WUJI_MJCF).
        duration: Playback length in seconds per grasp.
        pre_offset_cm: Wrist back-off during the approach, matching the Viser app.
        retarget_every_frame: IK each MANO frame. If false, lerp hinges from 0
            to the final grasp.
        loop: Repeat (the file, or the whole folder) until the window is closed.
        real_hand: Also execute the motion on a connected real right Wuji Hand 2
            (needs wuji-sdk).
        real_max_speed: Joint speed cap on the real hand, rad/s.
        real_max_acceleration: Joint acceleration cap on the real hand, rad/s^2.
        real_smoothing_sigma: Gaussian smoothing width over retargeted frames
            (zero-phase, 0 disables).
        real_hold: Seconds the real hand holds each final grasp before the next one.
        no_float: Keep the wrist fixed at the origin. Finger motion still plays.
        smoothing_movement: Real hand moves from the first pose to the last in a
            straight line. Acceleration is constant at the start, zero in the
            middle, and constant at the end, within the speed and acceleration caps.
        real_kp: MIT position gain. Try up to ~20 if the hand feels soft.
        real_kd: MIT damping gain.
        real_rate_hz: Servo thread command rate.
        real_velocity_ff_joints: Flat-20 joint indices that also receive the
            planned velocity in JointCommand (A/B test; others send 0).
    """
    real = dict(
        real_hand=real_hand,
        real_max_speed=real_max_speed,
        real_max_acceleration=real_max_acceleration,
        real_smoothing_sigma=real_smoothing_sigma,
        real_hold=real_hold,
        no_float=no_float,
        smoothing_movement=smoothing_movement,
        real_kp=real_kp,
        real_kd=real_kd,
        real_rate_hz=real_rate_hz,
        real_velocity_ff_joints=real_velocity_ff_joints,
    )
    if from_pkl is not None:
        play_pkls(
            [from_pkl],
            mjcf,
            duration,
            pre_offset_cm,
            retarget_every_frame,
            loop,
            **real,
        )
        return
    pred_dir = _pred_dir(dataset_path)
    files = list_grasp_pred_pkls(dataset_path)
    if not files:
        raise FileNotFoundError(
            f"No .pkl files under {pred_dir}. Run the app with --save-pred, "
            "or pass --from-pkl."
        )
    play_pkls(
        files,
        mjcf,
        duration,
        pre_offset_cm,
        retarget_every_frame,
        loop,
        pred_dir=pred_dir,
        **real,
    )


if __name__ == "__main__":
    tyro.cli(main)
