"""Prepare inference pkl(s) from rgb + depth + intrinsics.

Point this at a folder holding RGB/depth image pairs and one shared
intrinsics file. Each pair is center-cropped to a square, resized to
224x224, and written as `{stem}.pkl` beside the inputs.

Expected folder contents:
    *rgb*.{png,jpg}              uint8 RGB, any HxW (one or many)
    matching *depth*.png         uint16, 1mm units, same HxW as its rgb
    *intrinsics*.{txt,npy,json}  shared intrinsics at the rgb resolution
                                 (or a lone .txt/.npy/.json)

Pairing: ``photo_01_rgb.png`` + ``photo_01_depth.png`` → ``photo_01.pkl``.
A single-capture folder (one rgb + one depth) still works the same way.

The pkl lands beside the inputs; point ``--dataset-path`` at the folder to
run the app (it discovers .pkl samples recursively).
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tyro
from rich.console import Console

from .dataloader.data_classes import CameraIntrinsics, GraspData

console = Console()

TARGET_SIZE = 224
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
INTRINSICS_EXTS = (".txt", ".csv", ".npy", ".json")
_RGB_SUFFIX = re.compile(r"[_-]?rgb$", re.IGNORECASE)


def _center_crop_square(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Center-crop to a square along the shorter side. Returns crop + offsets."""
    h, w = img.shape[:2]
    size = min(h, w)
    x_off = (w - size) // 2
    y_off = (h - size) // 2
    return img[y_off : y_off + size, x_off : x_off + size], x_off, y_off


def _adjust_K(K: np.ndarray, x_off: int, y_off: int, scale: float) -> np.ndarray:
    """Shift principal point for an (x_off, y_off) crop then scale by `scale`."""
    K_new = K.copy().astype(np.float64)
    K_new[0, 2] -= x_off
    K_new[1, 2] -= y_off
    K_new[:2, :] *= scale
    return K_new


def _encode_depth_224(depth: np.ndarray) -> bytes:
    """Center-crop + nearest-resize uint16 depth to 224 and PNG-encode."""
    if depth.dtype != np.uint16:
        raise ValueError(f"depth must be uint16, got {depth.dtype}")
    depth_sq, _, _ = _center_crop_square(depth)
    depth_224 = cv2.resize(
        depth_sq, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST
    )
    _, buf = cv2.imencode(".png", depth_224)
    return buf.tobytes()


def prepare_pkl(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    stem: str,
    out_dir: Path,
    object_name: str = "",
    frame_index: int = 0,
) -> Path:
    """Encode one (rgb, depth, K) sample to an inference pkl.

    Center-crops the input to a square (shorter side), resizes to 224x224, and
    adjusts K accordingly. Stores the original-square K as `camera_original`.

    Args:
        rgb: (H, W, 3) uint8 RGB image.
        depth: (H, W) uint16 depth, 1mm units. Must match rgb HxW.
        K: (3, 3) intrinsics at the original RGB resolution.
        stem: Output filename stem.
        out_dir: Directory to write `{stem}.pkl` into (created if missing).
        object_name: Optional string saved in the pkl.
        frame_index: Optional frame index saved in the pkl.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"rgb must be (H,W,3) uint8, got {rgb.shape} {rgb.dtype}")
    if depth.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"rgb {rgb.shape[:2]} != depth {depth.shape[:2]}")

    rgb_sq, x_off, y_off = _center_crop_square(rgb)
    sq_size = rgb_sq.shape[0]
    K_orig = _adjust_K(K, x_off, y_off, scale=1.0)
    K_224 = _adjust_K(K, x_off, y_off, scale=TARGET_SIZE / sq_size)

    rgb_224 = cv2.resize(
        rgb_sq, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA
    )
    rgb_bgr = cv2.cvtColor(rgb_224, cv2.COLOR_RGB2BGR)
    _, img_buf = cv2.imencode(".jpg", rgb_bgr)

    entry = asdict(
        GraspData(
            object_name=object_name,
            frame_index=frame_index,
            grasp_index=0,
            camera=CameraIntrinsics(K=K_224, width=TARGET_SIZE, height=TARGET_SIZE),
            camera_original=CameraIntrinsics(K=K_orig, width=sq_size, height=sq_size),
            grasp=None,
            image=img_buf.tobytes(),
            depth=_encode_depth_224(depth),
            object_mask=b"",
        )
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.pkl"
    tmp = out_path.with_suffix(".tmp.pkl")
    with open(tmp, "wb") as f:
        pickle.dump(entry, f)
    tmp.rename(out_path)
    return out_path


def _load_intrinsics(path: Path) -> np.ndarray:
    """Load a 3x3 K from `fx fy cx cy`, a 3x3 matrix, or .npy/.json.

    Text/csv files may hold four numbers (`fx fy cx cy`) or nine (a flat 3x3).
    JSON may be a bare 3x3 list or a dict with a `K` key.
    """
    if path.suffix == ".npy":
        K = np.asarray(np.load(path), dtype=np.float64)
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
        K = np.asarray(data["K"] if isinstance(data, dict) and "K" in data else data)
    else:
        K = np.loadtxt(path)
    vals = np.asarray(K, dtype=np.float64).ravel()
    if vals.size == 4:
        fx, fy, cx, cy = vals
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    if vals.size == 9:
        return vals.reshape(3, 3)
    raise ValueError(
        f"{path}: expected 4 (fx fy cx cy) or 9 (3x3) numbers, got {vals.size}"
    )


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_depth_uint16(path: Path) -> np.ndarray:
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise IOError(f"Failed to read {path}")
    if d.dtype != np.uint16:
        raise ValueError(f"{path}: expected uint16 depth, got {d.dtype}")
    return d


def _stem_base_from_rgb(rgb_path: Path) -> str:
    """``photo_01_rgb`` → ``photo_01``; otherwise the rgb stem."""
    base = _RGB_SUFFIX.sub("", rgb_path.stem).rstrip("_-")
    return base or rgb_path.stem


def _find_depth_for_rgb(folder: Path, rgb_path: Path) -> Optional[Path]:
    """Locate the depth map paired with an rgb file."""
    base = _stem_base_from_rgb(rgb_path)
    for ext in IMAGE_EXTS:
        for name in (
            f"{base}_depth{ext}",
            f"{base}-depth{ext}",
            f"{base}depth{ext}",
        ):
            cand = folder / name
            if cand.is_file():
                return cand
    # Fallback: unique file containing both the base and "depth".
    hits = [
        p
        for p in sorted(folder.iterdir())
        if p.suffix.lower() in IMAGE_EXTS
        and "depth" in p.stem.lower()
        and base.lower() in p.stem.lower()
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def _find_intrinsics(folder: Path, intrinsics: Optional[Path] = None) -> Path:
    if intrinsics is not None:
        return intrinsics
    hits = [
        p
        for p in sorted(folder.iterdir())
        if p.suffix.lower() in INTRINSICS_EXTS and "intrinsics" in p.stem.lower()
    ]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        txts = [
            p
            for p in sorted(folder.iterdir())
            if p.suffix.lower() in INTRINSICS_EXTS
        ]
        if len(txts) == 1:
            return txts[0]
    raise FileNotFoundError(
        f"Need one shared intrinsics file under {folder} "
        f"(name it `*intrinsics*` or keep a single .txt/.npy/.json); "
        f"found {[p.name for p in hits]}"
    )


def find_rgb_depth_pairs(folder: Path) -> list[tuple[Path, Path, str]]:
    """Return ``(rgb, depth, stem)`` for every rgb/depth pair in ``folder``."""
    rgbs = [
        p
        for p in sorted(folder.iterdir())
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and "rgb" in p.stem.lower()
        and "depth" not in p.stem.lower()
    ]
    if not rgbs:
        # Single-capture fallback: lone non-depth image.
        imgs = [
            p
            for p in sorted(folder.iterdir())
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
            and "depth" not in p.stem.lower()
        ]
        depths = [
            p
            for p in sorted(folder.iterdir())
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
            and "depth" in p.stem.lower()
        ]
        if len(imgs) == 1 and len(depths) == 1:
            return [(imgs[0], depths[0], _stem_base_from_rgb(imgs[0]))]
        raise FileNotFoundError(
            f"No *rgb* images under {folder}. Expected names like "
            f"`photo_01_rgb.png` + `photo_01_depth.png`."
        )

    pairs: list[tuple[Path, Path, str]] = []
    missing: list[str] = []
    for rgb in rgbs:
        depth = _find_depth_for_rgb(folder, rgb)
        if depth is None:
            missing.append(rgb.name)
            continue
        pairs.append((rgb, depth, _stem_base_from_rgb(rgb)))
    if missing:
        raise FileNotFoundError(
            f"No matching *depth* for: {missing}. "
            f"Expected e.g. photo_01_rgb.png ↔ photo_01_depth.png."
        )
    if not pairs:
        raise FileNotFoundError(f"No rgb/depth pairs under {folder}")
    return pairs


def main(
    dataset_path: Path,
    rgb: Optional[Path] = None,
    depth: Optional[Path] = None,
    intrinsics: Optional[Path] = None,
    stem: Optional[str] = None,
    object_name: str = "",
) -> None:
    """Build inference pkl(s) from rgb + depth pairs and a shared intrinsics file.

    Scans ``dataset_path`` for ``*_rgb.*`` / ``*_depth.*`` pairs (or one lone
    rgb+depth), uses one shared intrinsics file, and writes ``{stem}.pkl`` next
    to the inputs.

    Args:
        dataset_path: Capture folder holding rgb/depth images and intrinsics.
        rgb: Optional single RGB path (disables multi-pair scan).
        depth: Optional single depth path (requires ``--rgb``).
        intrinsics: Shared intrinsics path; auto-detected if omitted.
        stem: Output stem for single-pair / explicit ``--rgb`` mode only.
        object_name: Optional string saved in each pkl.
    """
    if not dataset_path.is_dir():
        raise NotADirectoryError(dataset_path)

    intr_path = _find_intrinsics(dataset_path, intrinsics)
    K = _load_intrinsics(intr_path)

    if rgb is not None:
        depth_path = depth or _find_depth_for_rgb(dataset_path, Path(rgb))
        if depth_path is None:
            raise FileNotFoundError(f"No matching depth for {rgb}")
        out_stem = stem or _stem_base_from_rgb(Path(rgb))
        pairs = [(Path(rgb), Path(depth_path), out_stem)]
    else:
        if depth is not None:
            raise ValueError("--depth requires --rgb")
        pairs = find_rgb_depth_pairs(dataset_path)
        if stem is not None and len(pairs) > 1:
            raise ValueError("--stem only applies when preparing a single pair")
        if stem is not None and len(pairs) == 1:
            pairs = [(pairs[0][0], pairs[0][1], stem)]

    console.print(
        f"[cyan]intrinsics[/cyan] {intr_path.name}  "
        f"[cyan]pairs[/cyan] {len(pairs)} under {dataset_path}"
    )
    for i, (rgb_path, depth_path, out_stem) in enumerate(pairs):
        out_path = prepare_pkl(
            _read_rgb(rgb_path),
            _read_depth_uint16(depth_path),
            K,
            out_stem,
            dataset_path,
            object_name=object_name,
            frame_index=i,
        )
        console.print(
            f"[green]wrote {out_path.name}[/green]  "
            f"({rgb_path.name} + {depth_path.name})"
        )


if __name__ == "__main__":
    tyro.cli(main)
