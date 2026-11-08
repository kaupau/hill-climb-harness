"""Real finger masks per wrist camera and gripper opening, from the teleop videos.

The fingers ride with the camera, so at one gripper opening they cover the same pixels in every frame while the
scene behind them changes: pixels that stay put (low temporal spread) across many frames from different poses are
the fingers. Writes calib/finger_masks.npz: {cam}_{open|closed} masks at 320x180, plus the median frames.
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import teleop as T  # noqa: E402

W, H = 320, 180


def collect(cam, side, lo, hi, step=2, cap=400):
    stack = []
    for e in range(8):
        S, _ = T.episode(e)
        g = np.array([T.arm(s, side)[1] for s in S])
        q = np.array([T.arm(s, side)[0] for s in S])
        moving = np.r_[0, np.linalg.norm(np.diff(q, axis=0), axis=1)] > 0.002  # the background must change
        idx = [i for i in range(0, len(S), step) if lo <= g[i] <= hi and moving[i]]
        if not idx:
            continue
        cap_ = cv2.VideoCapture(str(T.ROOT / "videos" / "chunk-000" / T.VID[cam] / f"episode_{e:06d}.mp4"))
        want, i = set(idx), 0
        while True:
            ok, im = cap_.read()
            if not ok:
                break
            if i in want:
                stack.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (W, H), interpolation=cv2.INTER_AREA))
            i += 1
    stack = np.array(stack[:cap] if len(stack) <= cap else [stack[j] for j in
                                                             np.linspace(0, len(stack) - 1, cap).astype(int)])
    return stack


def mask_from(stack):
    med = np.median(stack, axis=0)
    spread = np.percentile(stack, 80, axis=0) - np.percentile(stack, 20, axis=0)
    m = (spread < 18).astype(np.uint8)
    m[: H // 3] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for k in range(1, n):
        if st[k][1] + st[k][3] >= H - 2 and st[k][4] > 150:
            keep[lab == k] = 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return keep, med.astype(np.uint8), spread


if __name__ == "__main__":
    out, vis = {}, []
    for cam, side in (("wrist_r", "right"), ("wrist_l", "left")):
        for name, lo, hi in (("open", 0.97, 1.01), ("closed", -0.01, 0.03)):
            st = collect(cam, side, lo, hi)
            m, med, spread = mask_from(st)
            out[f"{cam}_{name}"], out[f"{cam}_{name}_median"] = m, med
            print(cam, name, "frames", len(st), "mask px", int(m.sum()))
            v = cv2.cvtColor(med, cv2.COLOR_GRAY2BGR)
            v[m > 0] = (0.4 * v[m > 0] + [0, 0, 150]).astype(np.uint8)
            cv2.putText(v, f"{cam} {name} n={len(st)}", (4, 14), 0, 0.4, (0, 255, 0), 1)
            vis.append(v)
    np.savez_compressed(__file__.rsplit("/", 1)[0] + "/finger_masks.npz", **out)
    cv2.imwrite(__file__.rsplit("/", 1)[0] + "/finger_masks.jpg",
                np.vstack([np.hstack(vis[:2]), np.hstack(vis[2:])]))
