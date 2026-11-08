"""Direct arm moves on Dream's station, for our grasp primitive. Imported by pah_judge/sitecustomize.py.

  GET  /pose                      per arm: joints (rad), gripper, tool position xyz (m, arm base frame), tool rotation
  POST /move {arm, ...}           one smooth joint-space move, only while no Dream run is going:
        dxyz: [dx, dy, dz] m      move the tool by this much in the arm's base frame, keeping its orientation
        joints: {index: rad}      set these joints (absolute); 5 = wrist_roll, 4 = wrist_yaw, 3 = wrist_flex
        djoints: {index: rad}     nudge these joints
        gripper: 0..1             opening (0 shut, 1 open)
        duration_s: float         default 1.5
Limits: a move may not change any joint by more than 1.2 rad or the tool by more than 12 cm; joint limits clip.
"""
import math
import threading

import numpy as np

LIMITS_DEG = [(-160, 160), (-180, 0), (0, 180), (-123, 81), (-85, 85), (-145, 145)]
LO = np.radians([l for l, _ in LIMITS_DEG]) + 0.02
HI = np.radians([h for _, h in LIMITS_DEG]) - 0.02
_busy = threading.Lock()


def _fk(kin, q):
    model = getattr(kin, "_model", None)
    if model is not None and hasattr(model, "fk"):
        T = np.asarray(model.fk(np.asarray(q[: kin._dof], dtype=np.float64), kin._side), dtype=np.float64)
        return T[:3, 3], T[:3, :3]
    return np.asarray(kin.ee_position(np.asarray(q, dtype=np.float64))), None


def _rot_err(Rt, R):
    E = Rt @ R.T
    ang = math.acos(max(-1.0, min(1.0, (np.trace(E) - 1) / 2)))
    if ang < 1e-6:
        return np.zeros(3)
    axis = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]]) / (2 * math.sin(ang))
    return axis * ang


def ik(kin, q0, p_target, R_target, iters=150):
    """Damped least squares on position (+ orientation when the model gives it), finite-difference Jacobian."""
    q = np.array(q0, dtype=np.float64)
    for _ in range(iters):
        p, R = _fk(kin, q)
        err = p_target - p
        if R is not None and R_target is not None:
            err = np.concatenate([err, 0.15 * _rot_err(R_target, R)])
        if np.linalg.norm(err[:3]) < 0.002 and (len(err) == 3 or np.linalg.norm(err[3:]) < 0.01):
            break
        J = np.zeros((len(err), 6))
        for j in range(6):
            dq = np.zeros(6)
            dq[j] = 1e-4
            p2, R2 = _fk(kin, q + dq)
            e2 = p_target - p2
            if len(err) > 3:
                e2 = np.concatenate([e2, 0.15 * _rot_err(R_target, R2)])
            J[:, j] = (err - e2) / 1e-4
        lam = 0.02
        step = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(len(err)), err)
        q = np.clip(q + np.clip(step, -0.1, 0.1), LO, HI)
    p, _ = _fk(kin, q)
    return q, float(np.linalg.norm(p_target - p))


def _aim(R, direction):
    """R rotated by the smallest turn that points its approach axis (column 2) along `direction`."""
    a = R[:, 2] / np.linalg.norm(R[:, 2])
    b = np.asarray(direction, dtype=np.float64)
    b = b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return R if c > 0 else None
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return (np.eye(3) + K + K @ K * (1 / (1 + c))) @ R


def pose(controller):
    out = {}
    for name, arm in controller._arms.items():
        st = arm.read()
        q = np.asarray(st.positions_rad, dtype=np.float64)
        p, R = _fk(controller._kinematics[name], q)
        out[name] = {"joints": [round(float(v), 4) for v in q], "gripper": round(float(st.gripper), 3),
                     "xyz": [round(float(v), 4) for v in p],
                     "R": None if R is None else [[round(float(v), 3) for v in row] for row in R]}
    return out


def move(controller, body):
    from robots import types
    if controller._run_task is not None or controller.state not in ("idle", "error"):
        raise ValueError(f"move refused: station is {controller.state}; stop the run first")
    name = body.get("arm", "right")
    arm, kin = controller._arms[name], controller._kinematics[name]
    if not _busy.acquire(blocking=False):
        raise ValueError("move refused: another move is in progress")
    try:
        st = arm.read()
        q0 = np.asarray(st.positions_rad, dtype=np.float64)
        g0 = min(max(float(st.gripper), 0.0), 1.0)
        q1, residual = q0.copy(), 0.0
        if body.get("point") is not None and body.get("dxyz") is None:
            body["dxyz"] = [0, 0, 0]
        if body.get("dxyz") is not None:
            d = np.asarray(body["dxyz"], dtype=np.float64)
            if np.linalg.norm(d) > 0.12:
                raise ValueError("move refused: dxyz over 12 cm")
            p, R = _fk(kin, q0)
            if body.get("point") is not None and R is not None:
                R = _aim(R, body["point"])
                body["keep_orientation"] = True
            q1, residual = ik(kin, q0, p + d, R)
            if R is not None and residual < 0.01:
                rot = np.linalg.norm(_rot_err(R, _fk(kin, q1)[1]))
                if rot > 0.15 and body.get("keep_orientation"):
                    raise ValueError(f"move refused: could not hold that orientation (off {math.degrees(rot):.0f} deg)")
            if residual > 0.005 and not body.get("keep_orientation"):
                q1, residual = ik(kin, q0, p + d, None)  # position only: the wrist may tilt
            target = p + d
            if residual > 0.01:
                raise ValueError(f"move refused: IK could not reach it (off by {residual * 100:.1f} cm)")
        for k, v in (body.get("joints") or {}).items():
            q1[int(k)] = float(v)
        for k, v in (body.get("djoints") or {}).items():
            q1[int(k)] += float(v)
        q1 = np.clip(q1, LO, HI)
        if np.max(np.abs(q1 - q0)) > 1.2:
            raise ValueError(f"move refused: a joint would change by {np.max(np.abs(q1 - q0)):.2f} rad (> 1.2)")
        g1 = g0 if body.get("gripper") is None else min(max(float(body["gripper"]), 0.0), 1.0)
        dur = min(max(float(body.get("duration_s", 1.5)), 0.3), 6.0)
        steps = max(10, int(dur * 30))
        import time
        t0 = time.monotonic()
        for i in range(1, steps + 1):
            a = i / steps
            a = a * a * (3 - 2 * a)  # smoothstep: no jolt at the start or end
            arm.write(types.ArmCommand(positions_rad=q0 + (q1 - q0) * a, gripper=g0 + (g1 - g0) * a))
            time.sleep(max(0.0, t0 + i * dur / steps - time.monotonic()))
        time.sleep(0.3)
        if body.get("dxyz") is not None and not body.get("no_overshoot"):  # the arm sags under gravity: aim past the miss, up to twice
            for _ in range(2):
                q_now = np.asarray(arm.read().positions_rad, dtype=np.float64)
                miss = target - _fk(kin, q_now)[0]
                if np.linalg.norm(miss) < 0.004:
                    break
                q2, _ = ik(kin, q1, _fk(kin, q1)[0] + miss, None)
                for i in range(1, 11):
                    arm.write(types.ArmCommand(positions_rad=q1 + (q2 - q1) * i / 10, gripper=g1))
                    time.sleep(0.03)
                q1 = q2
                time.sleep(0.3)
        after = pose(controller)[name]
        return {"ok": True, "ik_residual_m": round(residual, 4), "pose": after}
    finally:
        _busy.release()
