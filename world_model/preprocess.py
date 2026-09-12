import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .config import ProjectPaths, create_project


def _estimate_camera_action(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    h, w = prev_gray.shape
    fx = flow[..., 0]
    fy = flow[..., 1]
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]

    strafe_x = -float(np.median(fx)) * 10.0
    forward_z = -float(np.median(fy)) * 10.0

    left_flow = float(np.median(fx[:, : max(1, w // 3)]))
    right_flow = float(np.median(fx[:, max(1, 2 * w // 3) :]))
    yaw = (right_flow - left_flow) * 5.0

    radial = fx * xs
    zoom = float(np.median(radial)) * 5.0

    action = np.array([strafe_x, forward_z, yaw, zoom], dtype=np.float32)
    return np.clip(action, -1.0, 1.0)


def preprocess_video(
    project: str | Path,
    video: str | Path,
    size: int = 480,
    max_frames: int | None = None,
) -> ProjectPaths:
    paths = create_project(project)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if orig_w <= 0 or orig_h <= 0:
        raise RuntimeError("Invalid video dimensions")

    scale = size / max(orig_w, orig_h)
    target_w = int(orig_w * scale)
    target_h = int(orig_h * scale)
    target_w = target_w - (target_w % 32)
    target_h = target_h - (target_h % 32)
    target_w = max(32, target_w)
    target_h = max(32, target_h)

    frames: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    prev_gray = None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    if max_frames is not None and total is not None:
        total = min(total, max_frames)

    pbar = tqdm(total=total, desc="preprocess")
    while True:
        if max_frames is not None and len(frames) >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            actions.append(np.zeros(5, dtype=np.float32))
        else:
            action_4d = _estimate_camera_action(prev_gray, gray)
            action_5d = np.array([action_4d[0], action_4d[1], action_4d[2], 0.0, action_4d[3]], dtype=np.float32)
            actions.append(action_5d)

        frames.append(rgb)
        prev_gray = gray
        pbar.update(1)

    pbar.close()
    cap.release()

    if len(frames) < 3:
        raise RuntimeError("Need at least 3 frames to train a world model.")

    frames_np = np.stack(frames).astype(np.uint8)
    actions_np = np.stack(actions).astype(np.float32)

    np.save(paths.frames_file, frames_np)
    np.save(paths.actions_file, actions_np)

    meta = {
        "video": str(video),
        "frame_count": int(len(frames_np)),
        "size": [int(target_w), int(target_h)],
        "original_size": [orig_w, orig_h],
        "action_order": ["strafe_x", "forward_z", "yaw", "pitch", "zoom"],
    }
    paths.meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return paths
