"""Fit each wrist camera's mount and lens to the real one from the finger silhouettes (calib/finger_masks.npz).

Parameters per camera: position in link6 (x, y, z), pitch/yaw/roll from looking straight along the fingers,
focal length and one fisheye term (Kannala-Brandt: r = f (theta + k1 theta^3)), and the gripper scale (how far the
real fingers open at reading 1.0, as a fraction of the URDF's 5 cm per finger).

    .venv/bin/python calib/fit_wrist_cam.py        # writes calib/wrist_cams.json and calib/wrist_fit.jpg
"""
import json
import math
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from station_sim import StationSim  # noqa: E402

W, H = 320, 180                 # mask size
RW, RH, RHFOV = 1280, 720, 150  # wide pinhole render, remapped through the fisheye model
REAL = dict(np.load(HERE / "finger_masks.npz"))
X0 = np.array([0.062, 0.0, 0.046, 0.0, 0.0, 0.0, 92.0, 0.0, 1.0])
NAMES = ["x", "y", "z", "pitch_deg", "yaw_deg", "roll_deg", "f_px320", "k1", "grip_scale"]


def rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def cam_R(pitch, yaw, roll):
    """Camera axes (x right, y up, z back) in link6: looking along +x with link6 z up, then turned."""
    base = np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).astype(float)
    return base @ rot("x", -pitch) @ rot("y", -yaw) @ rot("z", roll)


class Fitter:
    def __init__(self, cam):
        self.cam, self.side = cam, "right" if cam == "wrist_r" else "left"
        self.sim = StationSim(seed=0, randomize=False)
        m = self.sim.m
        self.cid = m.camera(cam).id
        m.cam_fovy[self.cid] = math.degrees(2 * math.atan(math.tan(math.radians(RHFOV / 2)) * RH / RW))
        hous = m.geom(f"{self.side}/cam_housing").id
        m.geom_group[hous] = 5
        self.r = mujoco.Renderer(m, RH, RW)
        self.r.enable_segmentation_rendering()
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[5] = 0
        fingers = [f"{self.side}/link7", f"{self.side}/link8"]
        self.fgeoms = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in fingers}
        self.real = {k: REAL[f"{cam}_{k}"] for k in ("open", "closed")}
        self.roi = {"open": np.ones((H, W), bool), "closed": np.zeros((H, W), bool)}
        self.roi["open"][: H // 3] = False
        self.roi["closed"][int(H * 0.64):] = True
        self.fp = (RW / 2) / math.tan(math.radians(RHFOV / 2))

    def render(self, p, g):
        m, d = self.sim.m, self.sim.d
        x, y, z, pitch, yaw, roll, f, k1, sc = p
        m.cam_pos[self.cid] = [x, y, z]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, cam_R(pitch, yaw, roll).flatten())
        m.cam_quat[self.cid] = q
        d.qpos[self.sim.gq[self.side]] = -0.05 * sc * g
        d.qpos[self.sim.gq8[self.side]] = 0.05 * sc * g
        mujoco.mj_forward(m, d)
        self.r.update_scene(d, camera=self.cam, scene_option=self.opt)
        seg = self.r.render()[..., 0]
        mask = np.isin(seg, list(self.fgeoms)).astype(np.uint8)
        # fisheye remap: target pixel -> angle theta -> wide pinhole pixel
        u, v = np.meshgrid(np.arange(W) + 0.5 - W / 2, np.arange(H) + 0.5 - H / 2)
        r = np.hypot(u, v) / f
        th = r.copy()
        for _ in range(6):
            th -= (th + k1 * th ** 3 - r) / (1 + 3 * k1 * th ** 2)
        th = np.clip(th, 0, math.radians(RHFOV / 2) - 0.01)
        scale = np.where(r > 1e-9, self.fp * np.tan(th) / np.maximum(np.hypot(u, v), 1e-9), self.fp / f)
        mx = (RW / 2 + u * scale).astype(np.float32)
        my = (RH / 2 + v * scale).astype(np.float32)
        return cv2.remap(mask, mx, my, cv2.INTER_NEAREST, borderValue=0)

    def loss(self, p):
        tot = 0.0
        for k, g, w in (("open", 1.0, 2.0), ("closed", 0.0, 1.0)):
            s = self.render(p, g).astype(bool) & self.roi[k]
            rmask = self.real[k].astype(bool) & self.roi[k]
            iou = (s & rmask).sum() / max(1, (s | rmask).sum())
            tot += w * (1 - iou)
        return tot / 3.0

    def fit(self):
        best = None
        for x0 in (X0, X0 + [0.02, 0, 0.01, 5, 0, 0, 10, -0.05, -0.1], X0 + [-0.02, 0, -0.01, -5, 0, 0, -10, 0.05, 0]):
            res = minimize(self.loss, x0, method="Powell",
                           bounds=[(0.0, 0.12), (-0.03, 0.03), (0.0, 0.09), (-30, 30), (-15, 15), (-15, 15),
                                   (50, 200), (-0.3, 0.3), (0.5, 1.2)],
                           options={"maxiter": 3000, "xtol": 1e-3, "ftol": 1e-4})
            if best is None or res.fun < best.fun:
                best = res
        return best


def main():
    out, vis = {}, []
    for cam in ("wrist_r", "wrist_l"):
        fz = Fitter(cam)
        l0 = fz.loss(X0)
        res = fz.fit()
        p = res.x
        ious = {}
        for k, g in (("open", 1.0), ("closed", 0.0)):
            s = fz.render(p, g).astype(bool) & fz.roi[k]
            rm = fz.real[k].astype(bool) & fz.roi[k]
            ious[k] = round(float((s & rm).sum() / max(1, (s | rm).sum())), 3)
            v = cv2.cvtColor(REAL[f"{cam}_{k}_median"], cv2.COLOR_GRAY2BGR)
            v[rm & ~s] = (0, 0, 220)
            v[s & ~rm] = (220, 120, 0)
            v[s & rm] = (0, 200, 0)
            cv2.putText(v, f"{cam} {k} IoU {ious[k]}", (4, 14), 0, 0.4, (255, 255, 255), 1)
            vis.append(v)
        out[cam] = {**{n: round(float(v), 5) for n, v in zip(NAMES, p)}, "iou": ious,
                    "loss_before": round(l0, 4), "loss_after": round(float(res.fun), 4)}
        print(cam, json.dumps(out[cam]))
    (HERE / "wrist_cams.json").write_text(json.dumps(out, indent=1))
    cv2.imwrite(str(HERE / "wrist_fit.jpg"), np.vstack([np.hstack(vis[:2]), np.hstack(vis[2:])]))


if __name__ == "__main__":
    main()
