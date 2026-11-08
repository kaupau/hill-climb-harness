"""Fit the power strip's pose and size in each arm's base frame from the teleop wrist frames.

With the wrist cameras calibrated (calib/wrist_cams.json) and each frame's recorded joints, the camera pose is
known; the strip's black silhouette in frames looking down at the table then fixes where the strip is relative to
that arm. Fitting from the right and the left wrist gives the strip in both base frames, and their difference is
the base-to-base offset.

    .venv/bin/python calib/fit_strip.py      # writes calib/strip_fit.json and calib/strip_fit.jpg
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
sys.path.insert(0, str(HERE))
import build_scene as B  # noqa: E402
import teleop as T  # noqa: E402
from station_sim import StationSim  # noqa: E402

W, H = 320, 180
FINGERS = dict(np.load(HERE / "finger_masks.npz"))


def dark(bgr):
    hsv = cv2.cvtColor(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    return ((hsv[..., 2] < 75) & (hsv[..., 1] < 100)).astype(bool)


def pick_frames(sim, side, per_ep=2):
    """Frames whose camera looks down (forward z < -0.55), gripper open, arm still, from every episode."""
    cam = "wrist_r" if side == "right" else "wrist_l"
    cid = sim.m.camera(cam).id
    out = []
    for e in range(8):
        S, _ = T.episode(e)
        cand = []
        for i in range(0, len(S), 5):
            q, g = T.arm(S[i], side)
            if g < 0.9:
                continue
            if i > 5 and np.linalg.norm(T.arm(S[i], side)[0] - T.arm(S[i - 5], side)[0]) > 0.03:
                continue
            sim.d.qpos[sim.jq[side]] = q
            mujoco.mj_kinematics(sim.m, sim.d)
            mujoco.mj_camlight(sim.m, sim.d)
            fwd = -sim.d.cam_xmat[cid].reshape(3, 3)[:, 2]
            if fwd[2] < -0.55:
                cand.append(i)
        if cand:
            sel = [cand[j] for j in np.linspace(0, len(cand) - 1, min(per_ep, len(cand))).astype(int)]
            ims = T.frames(e, cam, sel)
            out += [(e, i, T.arm(T.episode(e)[0][i], side)[0], im) for i, im in zip(sel, ims) if im is not None]
    return out


class StripFit:
    def __init__(self, side):
        self.side = side
        self.cam = "wrist_r" if side == "right" else "wrist_l"
        self.sim = StationSim(seed=0, randomize=False)
        m = self.sim.m
        for g in range(m.ngeom):  # nothing but the strip and the fingers should render
            b = m.body(m.geom_bodyid[g]).name
            if not (b == "strip" or b.startswith(side + "/")):
                m.geom_group[g] = 5
        m.geom_group[m.geom(f"{side}/cam_housing").id] = 5
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[5] = 0
        self.rsize = (B.WRIST_RENDER_SIZE[0] // 2, B.WRIST_RENDER_SIZE[1] // 2)  # half size: plenty for 320x180
        self.r = mujoco.Renderer(m, self.rsize[1], self.rsize[0])
        self.r.enable_segmentation_rendering()
        self.strip_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.sim.strip}
        mx, my = self.sim._fisheye_maps(self.cam, B.WRIST_CALIB[self.cam], W, H)
        self.maps = (mx / 2, my / 2)
        self.frames = pick_frames(self.sim, side)
        fm = FINGERS[f"{self.cam}_open"].astype(bool)
        self.fmask = cv2.dilate(fm.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        self.real = [dark(im) & ~self.fmask for (_, _, _, im) in self.frames]
        self.base_size = self.sim.m.geom_size[self.sim.m.geom("strip_col").id].copy()
        print(side, "frames", len(self.frames))

    def strip_mask(self, p, q):
        x, y, yaw_deg, L, Wd = p
        m, d = self.sim.m, self.sim.d
        g = m.geom("strip_col").id
        m.geom_size[g] = [L / 2, Wd / 2, self.base_size[2]]
        base = self.sim.base[self.side]
        adr = m.jnt_qposadr[m.joint("strip_free").id]
        d.qpos[adr:adr + 3] = [base[0] + x, base[1] + y, self.base_size[2]]
        a = math.radians(yaw_deg)
        d.qpos[adr + 3:adr + 7] = [math.cos(a / 2), 0, 0, math.sin(a / 2)]
        d.qpos[self.sim.jq[self.side]] = q
        d.qpos[self.sim.gq[self.side]] = -0.05
        d.qpos[self.sim.gq8[self.side]] = 0.05
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)  # camera poses are not part of mj_kinematics
        self.r.update_scene(d, camera=self.cam, scene_option=self.opt)
        seg = self.r.render()[..., 0]
        mk = np.isin(seg, list(self.strip_geoms)).astype(np.uint8)
        return cv2.remap(mk, *self.maps, cv2.INTER_NEAREST, borderValue=0).astype(bool) & ~self.fmask

    def ious(self, p):
        out = []
        for (_, _, q, _), real in zip(self.frames, self.real):
            s = self.strip_mask(p, q)
            u = (s | real).sum()
            out.append((s & real).sum() / u if u else 1.0)
        return np.array(out)

    def loss(self, p):
        iou = self.ious(p)
        return 1 - np.mean(np.sort(iou)[len(iou) // 5:])  # drop the worst fifth (people, cords, a moved strip)


def main():
    result, vis = {}, []
    for side in ("right", "left"):
        f = StripFit(side)
        base = f.sim.base[side]
        cx = B.STRIP_NEAR_X + B.STRIP_L / 2
        p0 = np.array([cx - base[0], B.STRIP_Y - base[1], 0.0, B.STRIP_L, B.STRIP_W])
        best = None
        for dx, dy, dyaw in ((0, 0, 0), (0.05, 0.05, 6)):
            res = minimize(f.loss, p0 + [dx, dy, dyaw, 0, 0], method="Powell",
                           bounds=[(0.2, 0.9), (-0.5, 0.5), (-45, 45), (0.2, 0.45), (0.07, 0.14)],
                           options={"maxiter": 400, "xtol": 2e-3, "ftol": 1e-3})
            print(f"  {side} start {dx:+.2f},{dy:+.2f},{dyaw:+d}: loss {res.fun:.3f} at {np.round(res.x, 3).tolist()}")
            if best is None or res.fun < best.fun:
                best = res
        iou = f.ious(best.x)
        result[side] = {"strip_centre_in_base": [round(float(best.x[0]), 4), round(float(best.x[1]), 4)],
                        "yaw_deg": round(float(best.x[2]), 2), "length": round(float(best.x[3]), 4),
                        "width": round(float(best.x[4]), 4), "iou_median": round(float(np.median(iou)), 3),
                        "frames": len(iou), "loss_start": round(float(f.loss(p0)), 3),
                        "loss": round(float(best.fun), 3)}
        print(side, result[side])
        for k in np.argsort(-iou)[:3].tolist() + np.argsort(iou)[:1].tolist():
            e, i, q, im = f.frames[k]
            v = cv2.resize(im, (W, H))
            s, rm = f.strip_mask(best.x, q), f.real[k]
            v[rm & ~s] = (0, 0, 220)
            v[s & ~rm] = (220, 120, 0)
            v[s & rm] = (0, 200, 0)
            cv2.putText(v, f"{f.cam} ep{e} f{i} IoU {iou[k]:.2f}", (4, 14), 0, 0.4, (255, 255, 255), 1)
            vis.append(v)
    r, l = result["right"], result["left"]
    # base offset: the strip centre seen from each base; same strip, so base_l - base_r = c_r - c_l (yaws ~ equal)
    off = np.subtract(r["strip_centre_in_base"], l["strip_centre_in_base"])
    result["left_base_minus_right_base_xy"] = [round(float(v), 4) for v in off]
    print("left base - right base:", result["left_base_minus_right_base_xy"])
    (HERE / "strip_fit.json").write_text(json.dumps(result, indent=1))
    cv2.imwrite(str(HERE / "strip_fit.jpg"), np.vstack([np.hstack(vis[:4]), np.hstack(vis[4:8])]))


if __name__ == "__main__":
    main()
