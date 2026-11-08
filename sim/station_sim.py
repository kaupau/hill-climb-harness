"""The PAH station in MuJoCo: two Metal arms, the power strip with four plugged USB cables, loose cables, blocks,
balls and a bin. Python API; server.py puts Dream's station HTTP API in front of it.

    from station_sim import StationSim
    sim = StationSim(seed=0)              # builds station.xml if missing, resets with randomized object poses
    sim.pose()["right"]                   # joints (rad), gripper 0 shut..1 open, xyz (m, arm base frame), R
    sim.move("right", {"dxyz": [0, 0, 0.03], "keep_orientation": True})   # same contract as /move
    sim.set_targets("right", q=[...], gripper=1.0); sim.advance(1.0)      # raw joint targets, then step 1 s
    img = sim.render("wrist_r")           # RGB uint8, 720x1280
    sim.checks()                          # task success checks + rubric score

Frames: each arm's base frame is its URDF base_link (x forward, y left, z up) on the table top; the tool point
(TCP) is Dream's, 0.135 m out along link6 x, with R columns (x, y, z) = (jaw closing axis, link6 z, approach).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

import build_scene as B

HERE = Path(__file__).parent
ARMS = ("left", "right")
CAMERAS = ("scene", "wrist_l", "wrist_r")
# station.yaml (~/.dreamlabs on the station): follower home poses, radians
HOME = {"left": [0.1436, 0.0018, 0.0091, -0.0256, 0.2147, -0.03],
        "right": [0.008, 0.0005, -0.0419, 0.1758, -0.003, -0.0792]}
LIMITS_DEG = [(-160, 160), (-180, 0), (0, 180), (-123, 81), (-85, 85), (-145, 145)]
LO = np.radians([l for l, _ in LIMITS_DEG]) + 0.02
HI = np.radians([h for _, h in LIMITS_DEG]) - 0.02
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"]
TOOL_RELABEL = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # Dream's Metal TCP axes in link6
# robots/motor_guard.py on the station
JOINT_BLOCK_DEG, STILL_DEG_S, BLOCK_S = 12.0, 3.0, 0.5
RETAIN_N = (10.0, 15.0)        # axial pull that frees a plug, drawn per plug at reset
INSERT_TOL = dict(axial=0.003, lateral=0.002, vertical=0.002, angle_deg=10.0)


class MoveRefused(ValueError):
    """The move was refused (HTTP 409 on the real station)."""


def _rot_err(Rt, R):
    E = Rt @ R.T
    ang = math.acos(max(-1.0, min(1.0, (np.trace(E) - 1) / 2)))
    if ang < 1e-6:
        return np.zeros(3)
    axis = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]]) / (2 * math.sin(ang))
    return axis * ang


def _aim(R, direction):
    a = R[:, 2] / np.linalg.norm(R[:, 2])
    b = np.asarray(direction, dtype=np.float64)
    b = b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return R if c > 0 else None
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return (np.eye(3) + K + K @ K * (1 / (1 + c))) @ R


def _smooth(a):
    return a * a * (3 - 2 * a)


def _yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])


class StationSim:
    def __init__(self, xml: str | Path | None = None, seed: int | None = 0, randomize: bool = True,
                 strip_held: bool = False):
        """strip_held: weld the strip to the table (tape / the left arm holding it). Free by default: a 1 kg
        strip lifts off the table before a 10-15 N port lets go of a plug pulled straight up."""
        self.strip_held = strip_held
        xml = Path(xml) if xml else HERE / "station.xml"
        if not xml.exists() or xml.stat().st_mtime < (HERE / "build_scene.py").stat().st_mtime:
            B.main(xml)
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self._kd = mujoco.MjData(self.m)  # scratch data for FK/IK
        m = self.m
        self.jq = {a: [m.jnt_qposadr[m.joint(f"{a}/joint{i + 1}").id] for i in range(6)] for a in ARMS}
        self.jv = {a: [m.jnt_dofadr[m.joint(f"{a}/joint{i + 1}").id] for i in range(6)] for a in ARMS}
        self.gq = {a: m.jnt_qposadr[m.joint(f"{a}/joint7").id] for a in ARMS}
        self.gq8 = {a: m.jnt_qposadr[m.joint(f"{a}/joint8").id] for a in ARMS}
        self.act = {a: [m.actuator(f"{a}/a{i + 1}").id for i in range(6)] for a in ARMS}
        self.gact = {a: m.actuator(f"{a}/grip").id for a in ARMS}
        self.tcp = {a: m.site(f"{a}/tcp").id for a in ARMS}
        self.base = {a: np.array(m.body(f"{a}/base_link").pos) for a in ARMS}
        self.strip = m.body("strip").id
        self.port_site = [m.site(f"port{k}").id for k in range(4)]
        self.plug = [m.body(f"plug{k}").id for k in range(4)]
        self.loose = [m.body(f"loose{i}").id for i in range(len(B.LOOSE_CABLES))]
        self.retain_eq = [m.equality(f"retain{k}").id for k in range(4)]
        self.seat_eq = [m.equality(f"seat_loose{i}").id for i in range(len(B.LOOSE_CABLES))]
        self.blocks = [m.body(f"piece{i}").id for i in range(len(B.PIECES))]
        self.piece_half = [B.FOAM_BLOCK / 2 if k == "block" else B.FOAM_CYL_H / 2 for k, _ in B.PIECES]
        self.balls = [(m.body(f"ball{i}_{c}").id, c) for i, c in enumerate(B.BALLS)]
        self.bin = m.body("bin").id
        self.bin_site = m.site("bin_inside").id
        self.arm_geoms = {a: {g for g in range(m.ngeom) if m.geom_contype[g] == 2
                              and m.body(m.geom_bodyid[g]).name.startswith(a + "/")} for a in ARMS}
        self.finger_geoms = {a: {g for g in self.arm_geoms[a] if m.body(m.geom_bodyid[g]).name in
                                 (f"{a}/link7", f"{a}/link8")} for a in ARMS}
        self.table_geom = m.geom("table").id
        self.strip_geom = m.geom("strip_col").id
        self._renderer = None
        self.tick_hook = None  # called every ~33 ms of sim time inside advance() (server.py serves reads there)
        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed, randomize=randomize)

    # ------------------------------------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None, randomize: bool = True, settle_s: float = 1.0):
        """Arms at home with the grippers open, strip and plugs in place, loose objects at random poses."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rng = self.rng if randomize else np.random.default_rng(0)
        jit = (lambda s: rng.uniform(-s, s)) if randomize else (lambda s: 0.0)
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        for a in ARMS:
            for i, adr in enumerate(self.jq[a]):
                d.qpos[adr] = HOME[a][i]
            d.qpos[self.gq[a]], d.qpos[self.gq8[a]] = -B.GRIP_TRAVEL, B.GRIP_TRAVEL
        # strip: a centimetre and a few degrees of jitter
        sx, sy, syaw = jit(0.01), jit(0.01), jit(math.radians(3))
        cx = B.STRIP_NEAR_X + B.STRIP_L / 2 + sx
        self._set_free("strip_free", [cx, B.STRIP_Y + sy, B.STRIP_H / 2], _yaw_quat(syaw))
        mujoco.mj_kinematics(m, d)
        Rs = d.xmat[self.strip].reshape(3, 3)
        for k in range(4):
            seat = d.xpos[self.strip] + Rs @ m.site_pos[self.port_site[k]]
            qs = np.zeros(4)
            mujoco.mju_mulQuat(qs, _yaw_quat(syaw), np.array(B.SEAT_QUAT))
            self._set_free(f"plug{k}_free", seat, qs)
            mode = ["over", "left", "right"][int(rng.integers(3))] if randomize else ["over", "left", "over", "right"][k]
            self._lay_cord(f"plug{k}", mode=mode, jitter=jit)
            d.eq_active[self.retain_eq[k]] = 1
        for i in range(len(self.loose)):
            d.eq_active[self.seat_eq[i]] = 0
        mujoco.mj_kinematics(m, d)
        hold = m.equality("strip_hold").id
        # relpose is body2's pose in body1's frame: here the world's pose in the strip's frame at reset
        inv_q = np.zeros(4)
        mujoco.mju_negQuat(inv_q, d.xquat[self.strip])
        Rinv = np.zeros(9)
        mujoco.mju_quat2Mat(Rinv, inv_q)
        m.eq_data[hold][3:6] = -(Rinv.reshape(3, 3) @ d.xpos[self.strip])
        m.eq_data[hold][6:10] = inv_q
        d.eq_active[hold] = int(self.strip_held)
        placed = [(cx, B.STRIP_Y, 0.2)]  # (x, y, radius) keep-outs; strip first

        def spot(xr, yr, r):
            for _ in range(200):
                x, y = rng.uniform(*xr), rng.uniform(*yr)
                if all(math.hypot(x - px, y - py) > r + pr for px, py, pr in placed):
                    placed.append((x, y, r))
                    return x, y
            placed.append((xr[0], yr[0], r))
            return xr[0], yr[0]

        for i in range(len(self.loose)):
            x, y = spot((0.12, 0.26), (-0.38, 0.05), 0.06)
            self._set_free(f"loose{i}_free", [x, y, B.PLUG_HEAD[B.LOOSE_CABLES[i]][2] / 2 + 0.001],
                           _yaw_quat(math.pi / 2 + rng.uniform(-0.6, 0.6)))
            self._lay_cord(f"loose{i}", mode="curl", jitter=lambda s: rng.uniform(-s, s))
        bx, by = spot((0.5, 0.62), (0.3, 0.45), 0.16)
        self._set_free("bin_free", [bx, by, 0.0], _yaw_quat(jit(0.3)))
        for i in range(len(B.PIECES)):
            x, y = spot((0.18, 0.42), (0.02, 0.42), 0.04)
            self._set_free(f"piece{i}_free", [x, y, self.piece_half[i] + 0.0005], _yaw_quat(rng.uniform(0, math.pi)))
        for i in range(len(B.BALLS)):
            x, y = spot((0.2, 0.55), (-0.05, 0.42), 0.03)
            self._set_free(f"ball{i}_free", [x, y, B.BALL_R + 0.0005], [1, 0, 0, 0])
        self.retain_n = rng.uniform(*RETAIN_N, size=4) if randomize else np.full(4, np.mean(RETAIN_N))
        self.pull_f = np.zeros(4)
        self.peak_pull = np.zeros(4)
        self.released = [False] * 4
        self.removal_order: list[int] = []
        self.inserted = [None] * len(self.loose)
        # rubric v4 (hillclimb_rubric_v4.pdf) bookkeeping: stage passes, the stage-skip DQ, the 3 s free-standing rule
        self.dq = None                 # reason, once a gripper touched a later stage's object too early
        self.stage1_pass_t = None      # first cable out
        self.stage2_pass_t = None      # tower > 4 in, standing free for 3 s
        self._tower_free_since = None
        self.guard = {a: np.zeros(6) for a in ARMS}
        self.state, self.last_error = "idle", None
        self.target = {a: np.array(HOME[a], dtype=np.float64) for a in ARMS}
        self.gtarget = {a: 1.0 for a in ARMS}
        self._traj: dict[str, tuple] = {}
        self.max_contact = {}
        mujoco.mj_forward(m, d)
        self._apply_ctrl()
        self.advance(settle_s, check=False)
        self.t_reset = d.time
        return self.checks()

    def _set_free(self, joint, pos, quat):
        adr = self.m.jnt_qposadr[self.m.joint(joint).id]
        self.d.qpos[adr:adr + 3] = pos
        self.d.qpos[adr + 3:adr + 7] = quat

    def _lay_cord(self, name, mode="curl", jitter=lambda s: 0.0):
        """Initial cord shape, as in the teleop videos. Plugged cords leave the head straight up, then
        'left'/'right': arch sideways down onto the table; 'over': arch back over the top of the strip;
        'curl': a gentle random curve (loose cables, lying flat)."""
        for i in range(B.N_SEG + 1):
            jn = f"{name}_j{i}" if i < B.N_SEG else f"{name}_jfar"
            adr = self.m.jnt_qposadr[self.m.joint(jn).id]
            q = np.array([1.0, 0, 0, 0])
            if mode == "curl":
                q = _yaw_quat(0.12 + jitter(0.12))
            elif mode in ("left", "right") and 1 <= i <= 4:
                g = (-1 if mode == "left" else 1) * (math.radians(30) + jitter(0.1))  # about plug z (= strip x)
                q = np.array([math.cos(g / 2), 0, 0, math.sin(g / 2)])
            elif mode == "over" and 1 <= i <= 4:
                b = math.radians(28) + jitter(0.08)        # about plug y: from straight up to along the strip
                q = np.array([math.cos(b / 2), 0, math.sin(b / 2), 0])
            elif i > 4:
                q = _yaw_quat(jitter(0.12))
            self.d.qpos[adr:adr + 4] = q

    # ---------------------------------------------------------------------------------------- stepping
    def set_targets(self, arm, q=None, gripper=None, duration_s=0.0):
        """Joint targets (rad) and/or gripper (0 shut..1 open), reached along a smoothstep over duration_s."""
        q1 = self.target[arm].copy() if q is None else np.clip(np.asarray(q, dtype=np.float64), LO, HI)
        g1 = self.gtarget[arm] if gripper is None else float(np.clip(gripper, 0, 1))
        if duration_s <= 0:
            self._traj.pop(arm, None)
            self.target[arm], self.gtarget[arm] = q1, g1
        else:
            self._traj[arm] = (self.d.time, duration_s, self.target[arm].copy(), q1, self.gtarget[arm], g1)
        self._apply_ctrl()

    def _apply_ctrl(self):
        for a, (t0, dur, q0, q1, g0, g1) in list(self._traj.items()):
            s = min(1.0, (self.d.time - t0) / dur)
            k = _smooth(s)
            self.target[a], self.gtarget[a] = q0 + (q1 - q0) * k, g0 + (g1 - g0) * k
            if s >= 1.0:
                del self._traj[a]
        for a in ARMS:
            self.d.ctrl[self.act[a]] = self.target[a]
            self.d.ctrl[self.gact[a]] = -B.GRIP_TRAVEL * self.gtarget[a]

    def advance(self, seconds: float, check: bool = True, realtime: float = 0.0):
        """Step the physics `seconds` of sim time. realtime > 0 paces it at that multiple of wall-clock time."""
        n = max(1, int(round(seconds / self.m.opt.timestep)))
        t_wall = time.monotonic()
        for i in range(n):
            self._apply_ctrl()
            mujoco.mj_step(self.m, self.d)
            if i % 2 == 0:
                self._retention()
            if check and i % 10 == 0:
                self._guard(10 * self.m.opt.timestep)
                self._contacts()
                self._insertion()
            if check and i % 100 == 0:
                self._stages()
            if self.tick_hook is not None and i % 33 == 0:
                self.tick_hook()
            if realtime > 0 and i % 20 == 0:
                lag = t_wall + (i + 1) * self.m.opt.timestep / realtime - time.monotonic()
                if lag > 0:
                    time.sleep(lag)

    # ------------------------------------------------------------------------------- plug retention
    def _retention(self):
        m, d = self.m, self.d
        rows = np.nonzero(d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)[0]
        if not len(rows):
            return
        Rs = d.xmat[self.strip].reshape(3, 3)
        out = Rs[:, 2]  # out of the port: straight up from the strip's top face
        for k in range(4):
            eid = self.retain_eq[k]
            if not d.eq_active[eid]:
                continue
            r = rows[d.efc_id[rows] == eid][:3]
            vec = np.zeros(d.nefc)
            vec[r] = d.efc_force[r]
            res = np.zeros(m.nv)
            mujoco.mj_mulJacTVec(m, d, res, vec)
            adr = m.jnt_dofadr[m.body_jntadr[self.plug[k]]]
            f_on_plug = res[adr:adr + 3]
            pull = -float(f_on_plug @ out)
            self.pull_f[k] += 0.1 * (pull - self.pull_f[k])  # ~20 ms low-pass at 2 ms updates
            self.peak_pull[k] = max(self.peak_pull[k], self.pull_f[k])
            if self.pull_f[k] > self.retain_n[k]:
                d.eq_active[eid] = 0
                self.released[k] = True
                self.removal_order.append(k)

    # ------------------------------------------------------------------------------------- motor guard
    def _guard(self, dt):
        """The station's block check: a joint more than 12 deg off its command and not moving, for 0.5 s,
        stops the arm (it holds where it is; the station reports an error)."""
        for a in ARMS:
            q = self.d.qpos[self.jq[a]]
            qd = self.d.qvel[self.jv[a]]
            err = np.abs(self.target[a] - q)
            blocked = (err > math.radians(JOINT_BLOCK_DEG)) & (np.abs(qd) < math.radians(STILL_DEG_S))
            self.guard[a] = np.where(blocked, self.guard[a] + dt, 0.0)
            if np.any(self.guard[a] >= BLOCK_S):
                j = int(np.argmax(self.guard[a]))
                self.state = "error"
                self.last_error = (f"motor guard: {a} {JOINT_NAMES[j]} blocked (commanded "
                                   f"{math.degrees(self.target[a][j]):.0f} deg, at {math.degrees(q[j]):.0f} deg); "
                                   f"the arm stopped")
                self._traj.pop(a, None)
                self.target[a] = q.copy()
                self.guard[a][:] = 0

    def _contacts(self):
        """Largest contact force per (arm, thing) since the last read; and the before-unplug touch rule."""
        m, d = self.m, self.d
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            for a in ARMS:
                ag = self.arm_geoms[a]
                if (g1 in ag) == (g2 in ag):
                    continue
                other = g2 if g1 in ag else g1
                ob = m.body(m.geom_bodyid[other]).name
                what = ("table" if other == self.table_geom else "strip" if other == self.strip_geom else
                        "block" if ob.startswith("piece") else "ball" if ob.startswith("ball") else
                        "plug" if (ob.startswith("plug") or ob.startswith("loose")) else ob)
                part = "finger" if (g1 in self.finger_geoms[a] or g2 in self.finger_geoms[a]) else \
                    m.body(m.geom_bodyid[g1 if g1 in ag else g2]).name.split("/")[1]
                mujoco.mj_contactForce(m, d, i, f6)
                key = f"{a}:{part}->{what}"
                self.max_contact[key] = max(self.max_contact.get(key, 0.0), float(abs(f6[0])))
                if self.dq is None and what == "block" and self.stage1_pass_t is None:
                    self.dq = f"{a} gripper touched a stack piece at t={self.d.time - self.t_reset:.1f}s before a cable was out"
                if self.dq is None and what == "ball" and self.stage2_pass_t is None:
                    self.dq = f"{a} gripper touched a ball at t={self.d.time - self.t_reset:.1f}s before the tower passed"

    def _stages(self):
        t = self.d.time
        if self.stage1_pass_t is None and any(self.released):
            self.stage1_pass_t = t
        if self.stage1_pass_t is not None and self.stage2_pass_t is None:
            tw = self.tower()
            if tw["height_m"] > 4 * 0.0254 and tw["stands"]:
                self._tower_free_since = self._tower_free_since or t
                if t - self._tower_free_since >= 3.0:
                    self.stage2_pass_t = t
            else:
                self._tower_free_since = None

    def take_contacts(self):
        out = {k: round(v, 1) for k, v in sorted(self.max_contact.items()) if v > 0.5}
        self.max_contact = {}
        return out

    # -------------------------------------------------------------------------------- loose insertion
    def _seat_pose(self, k):
        return self.m.site_pos[self.port_site[k]].copy()

    def _port_free(self, k):
        if self.d.eq_active[self.retain_eq[k]]:
            return False
        seat = self.d.xpos[self.strip] + self.d.xmat[self.strip].reshape(3, 3) @ self._seat_pose(k)
        bodies = self.plug + self.loose
        return all(np.linalg.norm(self.d.xpos[b] - seat) > 0.01 for b in bodies)

    def _insertion(self):
        d = self.d
        Rs = d.xmat[self.strip].reshape(3, 3)
        for i, b in enumerate(self.loose):
            if self.inserted[i] is not None:
                continue
            kind = B.LOOSE_CABLES[i]
            rel = Rs.T @ (d.xpos[b] - d.xpos[self.strip])
            Rrel = Rs.T @ d.xmat[b].reshape(3, 3)
            ang = math.degrees(math.acos(max(-1, min(1, -Rrel[2, 0]))))  # plug -x (head) vs strip +z
            for k in range(4):
                if B.PORT_KINDS[k] != kind:
                    continue
                e = rel - self._seat_pose(k)
                if (abs(e[0]) < INSERT_TOL["axial"] and abs(e[1]) < INSERT_TOL["lateral"]
                        and abs(e[2]) < INSERT_TOL["vertical"] and ang < INSERT_TOL["angle_deg"]
                        and self._port_free(k)):
                    eid = self.seat_eq[i]
                    self.m.eq_data[eid][3:6] = self._seat_pose(k)   # relpose: plug in the strip's frame
                    self.m.eq_data[eid][6:10] = B.SEAT_QUAT
                    d.eq_active[eid] = 1
                    self.inserted[i] = k
                    break

    # ------------------------------------------------------------------------------------- kinematics
    def fk(self, arm, q):
        """Tool point and Dream's tool rotation in the arm's base frame for joints q."""
        kd = self._kd
        kd.qpos[:] = self.d.qpos
        kd.qpos[self.jq[arm]] = q
        mujoco.mj_kinematics(self.m, kd)
        s = self.tcp[arm]
        p = kd.site_xpos[s] - self.base[arm]
        R = kd.site_xmat[s].reshape(3, 3) @ TOOL_RELABEL
        return p, R

    def _jac(self, arm, q):
        kd = self._kd
        kd.qpos[:] = self.d.qpos
        kd.qpos[self.jq[arm]] = q
        mujoco.mj_kinematics(self.m, kd)
        mujoco.mj_comPos(self.m, kd)
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, kd, jp, jr, self.tcp[arm])
        return jp[:, self.jv[arm]], jr[:, self.jv[arm]]

    def ik(self, arm, q0, p_target, R_target, iters=150):
        """move_addon.ik: damped least squares on position (+ 0.15-weighted orientation)."""
        q = np.array(q0, dtype=np.float64)
        for _ in range(iters):
            p, R = self.fk(arm, q)
            err = p_target - p
            if R_target is not None:
                err = np.concatenate([err, 0.15 * _rot_err(R_target, R)])
            if np.linalg.norm(err[:3]) < 0.002 and (len(err) == 3 or np.linalg.norm(err[3:]) < 0.01):
                break
            jp, jr = self._jac(arm, q)
            J = jp if R_target is None else np.vstack([jp, 0.15 * jr])
            step = J.T @ np.linalg.solve(J @ J.T + 0.02 ** 2 * np.eye(len(err)), err)
            q = np.clip(q + np.clip(step, -0.1, 0.1), LO, HI)
        p, _ = self.fk(arm, q)
        return q, float(np.linalg.norm(p_target - p))

    # ------------------------------------------------------------------------------------------ reads
    def joints(self, arm):
        return self.d.qpos[self.jq[arm]].copy()

    def gripper(self, arm):
        return float(np.clip(-self.d.qpos[self.gq[arm]] / B.GRIP_TRAVEL, 0, 1))

    def pose(self):
        out = {}
        for a in ARMS:
            q = self.joints(a)
            p, R = self.fk(a, q)
            out[a] = {"joints": [round(float(v), 4) for v in q], "gripper": round(self.gripper(a), 3),
                      "xyz": [round(float(v), 4) for v in p],
                      "R": [[round(float(v), 3) for v in row] for row in R]}
        return out

    def arms(self):
        return {a: {"gripper": self.gripper(a), "commanded_gripper": float(self.gtarget[a]),
                    "gripper_effort": round(float(self.d.actuator_force[self.gact[a]]), 3)} for a in ARMS}

    # ------------------------------------------------------------------------------------------- moves
    def move(self, arm_or_body, body=None, realtime: float = 0.0):
        """move_addon.move, same body and refusals; blocks for the move's sim time."""
        body = dict(arm_or_body if body is None else body, **({} if body is None else {"arm": arm_or_body}))
        name = body.get("arm", "right")
        if name not in ARMS:
            raise MoveRefused(f"move refused: no arm {name!r}")
        # move_addon starts every move from the measured joints, which on the soft real arm sit one gravity sag
        # below the last command, so each move loses a sag. from_target (sim-only proposal) starts from the last
        # commanded joints instead.
        q0 = self.target[name].copy() if body.get("from_target") else self.joints(name)
        g0 = float(self.gtarget[name]) if body.get("from_target") else self.gripper(name)
        q1, residual, target = q0.copy(), 0.0, None
        if body.get("point") is not None and body.get("dxyz") is None:
            body["dxyz"] = [0, 0, 0]
        if body.get("dxyz") is not None:
            dd = np.asarray(body["dxyz"], dtype=np.float64)
            if np.linalg.norm(dd) > 0.12:
                raise MoveRefused("move refused: dxyz over 12 cm")
            p, R = self.fk(name, q0)  # from_target: in commanded space; the sag rides along roughly constant
            if body.get("point") is not None:
                R = _aim(R, body["point"])
                if R is None:
                    raise MoveRefused("move refused: cannot aim straight backwards")
                body["keep_orientation"] = True
            q1, residual = self.ik(name, q0, p + dd, R)
            if residual < 0.01:
                rot = np.linalg.norm(_rot_err(R, self.fk(name, q1)[1]))
                if rot > 0.15 and body.get("keep_orientation"):
                    raise MoveRefused(f"move refused: could not hold that orientation (off {math.degrees(rot):.0f} deg)")
            if residual > 0.005 and not body.get("keep_orientation"):
                q1, residual = self.ik(name, q0, p + dd, None)
            target = p + dd
            if residual > 0.01:
                raise MoveRefused(f"move refused: IK could not reach it (off by {residual * 100:.1f} cm)")
        for k, v in (body.get("joints") or {}).items():
            q1[int(k)] = float(v)
        for k, v in (body.get("djoints") or {}).items():
            q1[int(k)] += float(v)
        q1 = np.clip(q1, LO, HI)
        if np.max(np.abs(q1 - q0)) > 1.2:
            raise MoveRefused(f"move refused: a joint would change by {np.max(np.abs(q1 - q0)):.2f} rad (> 1.2)")
        g1 = g0 if body.get("gripper") is None else min(max(float(body["gripper"]), 0.0), 1.0)
        dur = min(max(float(body.get("duration_s", 1.5)), 0.3), 6.0)
        self.take_contacts()
        # like the real addon: the command starts from where the arm is (sag included), unless from_target
        self.target[name], self.gtarget[name] = q0.copy(), g0
        self.set_targets(name, q1, g1, dur)
        self.advance(dur + 0.3, realtime=realtime)
        if target is not None and not body.get("no_overshoot"):  # aim past the sag, up to twice
            for _ in range(2):
                miss = target - self.fk(name, self.joints(name))[0]
                if np.linalg.norm(miss) < 0.004:
                    break
                q2, _ = self.ik(name, q1, self.fk(name, q1)[0] + miss, None)
                self.set_targets(name, q2, g1, 0.3)
                self.advance(0.6, realtime=realtime)
                q1 = q2
        out = {"ok": True, "ik_residual_m": round(residual, 4), "pose": self.pose()[name],
               "contacts_N": self.take_contacts()}
        if self.state == "error":
            out["station_error"] = self.last_error
        return out

    def home(self, duration_s=3.0, realtime: float = 0.0):
        for a in ARMS:
            self.target[a] = self.joints(a)
            self.set_targets(a, HOME[a], None, duration_s)
        self.advance(duration_s + 0.3, realtime=realtime)
        self.state, self.last_error = "idle", None

    # ----------------------------------------------------------------------------------------- render
    def render(self, camera: str, width: int = 1280, height: int = 720, show_sites=False):
        """RGB frame. Wrist cameras with a fitted lens (calib/wrist_cams.json) are rendered as a wide pinhole and
        remapped through the real camera's fisheye model."""
        lens = B.WRIST_CALIB.get(camera)
        rw, rh = B.WRIST_RENDER_SIZE if lens else (width, height)
        key = (rw, rh)
        self._renderers = getattr(self, "_renderers", {})
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.m, rh, rw)
        opt = mujoco.MjvOption()
        if show_sites:
            opt.sitegroup[4] = 1
        r = self._renderers[key]
        r.update_scene(self.d, camera=camera, scene_option=opt)
        img = r.render()
        if not lens:
            return img
        return cv2.remap(img, *self._fisheye_maps(camera, lens, width, height), cv2.INTER_LINEAR)

    def _fisheye_maps(self, camera, lens, width, height):
        self._maps = getattr(self, "_maps", {})
        key = (camera, width, height)
        if key not in self._maps:
            rw, rh = B.WRIST_RENDER_SIZE
            fp = (rw / 2) / math.tan(math.radians(B.WRIST_RENDER_HFOV / 2))
            f, k1 = lens["f_px320"] * width / 320.0, lens["k1"]
            u, v = np.meshgrid(np.arange(width) + 0.5 - width / 2, np.arange(height) + 0.5 - height / 2)
            rad = np.hypot(u, v)
            r = rad / f
            th = r.copy()
            for _ in range(8):
                th -= (th + k1 * th ** 3 - r) / (1 + 3 * k1 * th ** 2)
            th = np.clip(th, 0, math.radians(B.WRIST_RENDER_HFOV / 2) - 0.01)
            scale = np.where(rad > 1e-9, fp * np.tan(th) / np.maximum(rad, 1e-9), fp / f)
            self._maps[key] = ((rw / 2 + u * scale).astype(np.float32), (rh / 2 + v * scale).astype(np.float32))
        return self._maps[key]

    # ----------------------------------------------------------------------------------------- checks
    def checks(self):
        d, m = self.d, self.m
        Rs = d.xmat[self.strip].reshape(3, 3)
        in_port, out = [], []
        for k in range(4):
            seat = d.xpos[self.strip] + Rs @ self._seat_pose(k)
            dist = float(np.linalg.norm(d.xpos[self.plug[k]] - seat))
            (in_port if (d.eq_active[self.retain_eq[k]] or dist < 0.005) else out).append(k)
        right_to_left = all(a > b for a, b in zip(self.removal_order, self.removal_order[1:])) and \
            (not self.removal_order or self.removal_order[0] == max(in_port + self.removal_order))
        tower = self.tower()
        balls = self.balls_in_bin()
        unplugged = len(out)
        inserted = sum(i is not None for i in self.inserted)
        tower_in = tower["height_m"] / 0.0254
        # rubric v4: +10 first cable, +5 each further; +20 per loose cable plugged in (up to 4); stage 2 passed:
        # +20 and +3 per full inch of tower height at the end; +5 per ball of our colour (orange) in the bin.
        # A DQ scores 0; a black 8-ball in the bin loses the match.
        passed2 = self.stage2_pass_t is not None
        pts = {"unplug": (10 + 5 * (unplugged - 1)) if unplugged else 0, "plug_in": 20 * min(inserted, 4),
               "tower": (20 + 3 * int(tower_in)) if passed2 else 0, "balls": 5 * balls["orange"]}
        total = 0 if self.dq else sum(pts.values())
        t0 = getattr(self, "t_reset", 0.0)
        since = (lambda x: None if x is None else round(x - t0, 1))
        return {"plugs_in_ports": in_port, "plugs_removed": unplugged, "removal_order": list(self.removal_order),
                "removed_right_to_left": right_to_left,
                "retention_N": [round(float(v), 1) for v in self.retain_n],
                "peak_pull_N": [round(float(v), 1) for v in self.peak_pull],
                "loose_inserted": [None if p is None else int(p) for p in self.inserted],
                "tower": tower, "balls_in_bin": balls,
                "stage1_passed_at_s": since(self.stage1_pass_t), "stage2_passed_at_s": since(self.stage2_pass_t),
                "elapsed_s": round(self.d.time - t0, 1), "over_5_min": self.d.time - t0 > 300,
                "dq": self.dq, "loses": bool(self.dq) or balls["black"] > 0,
                # MuJoCo resets the state when the physics blows up: an episode with this set is not a valid score
                "physics_unstable": int(self.d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) > 0,
                "black_ball_in_bin_loses": balls["black"] > 0,
                "score": dict(pts, total=total)}

    def tower(self):
        """Blocks connected to the table through block-on-block contacts, none touching an arm."""
        m, d = self.m, self.d
        bid = {b: i for i, b in enumerate(self.blocks)}
        adj = {i: set() for i in range(len(self.blocks))}
        on_table, held = set(), set()
        all_arm = set().union(*self.arm_geoms.values())
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 in bid and b2 in bid:
                adj[bid[b1]].add(bid[b2]); adj[bid[b2]].add(bid[b1])
            for g, b in ((c.geom2, b1), (c.geom1, b2)):
                if b in bid and g == self.table_geom:
                    on_table.add(bid[b])
                if b in bid and g in all_arm:
                    held.add(bid[b])
        best = {"height_m": 0.0, "blocks": 0, "stands": True}
        seen = set()
        for s in on_table:
            if s in seen:
                continue
            comp, stack = set(), [s]
            while stack:
                x = stack.pop()
                if x not in comp:
                    comp.add(x); stack.extend(adj[x])
            seen |= comp
            top = max(d.xpos[self.blocks[i]][2] + self.piece_half[i] for i in comp)
            levels = len({round(d.xpos[self.blocks[i]][2] / 0.05) for i in comp})
            if top > best["height_m"]:
                best = {"height_m": round(float(top), 4), "blocks": levels, "stands": not (comp & held)}
        best["height_in"] = round(best["height_m"] / 0.0254, 2)
        return best

    def balls_in_bin(self):
        d = self.d
        Rb = d.site_xmat[self.bin_site].reshape(3, 3)
        half = self.m.site_size[self.bin_site]
        out = {"orange": 0, "white": 0, "black": 0}
        for b, c in self.balls:
            rel = Rb.T @ (d.xpos[b] - d.site_xpos[self.bin_site])
            if abs(rel[0]) < half[0] and abs(rel[1]) < half[1] and abs(rel[2]) < half[2] + B.BALL_R:
                out[c] += 1
        return out
