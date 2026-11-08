"""Teleop dataset access (data/station/unplug-v1): states, actions and synced frames."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data" / "station" / "unplug-v1"
VID = {"scene": "observation.images.scene", "wrist_l": "observation.images.left_wrist",
       "wrist_r": "observation.images.right_wrist"}


def episode(e):
    df = pd.read_parquet(ROOT / "data" / "chunk-000" / f"episode_{e:06d}.parquet")
    return np.stack(df["observation.state"]).astype(np.float64), np.stack(df["action"]).astype(np.float64)


def frames(e, cam, idx):
    cap = cv2.VideoCapture(str(ROOT / "videos" / "chunk-000" / VID[cam] / f"episode_{e:06d}.mp4"))
    out = {}
    for i in sorted(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, im = cap.read()
        if ok:
            out[i] = im
    return [out.get(i) for i in idx]


def arm(state, side):
    """(joints[6], gripper 0..1) of one arm from a 14-wide state row."""
    o = 0 if side == "left" else 7
    return state[o:o + 6], state[o + 6] / 100.0


def finger_mask(bgr):
    """The fingers in a wrist frame: dark regions touching the bottom edge, lower 60 % of the image."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    dark = ((hsv[..., 2] < 70) & (hsv[..., 1] < 90)).astype(np.uint8)
    dark[: int(bgr.shape[0] * 0.4)] = 0
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark)
    keep = np.zeros_like(dark)
    H = bgr.shape[0]
    for k in range(1, n):
        x, y, w, h, a = stats[k]
        if y + h >= H - 2 and a > 800:
            keep[lab == k] = 1
    return keep
