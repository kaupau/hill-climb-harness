"""Joint stiffness from the teleop data: in still moments, (commanded - actual) = offset + gravity torque / kp,
regressed per joint over many poses. Gravity torques come from the URDF masses (MuJoCo, zero velocity).
Writes calib/gains.json: per arm kp and the constant offsets (rad)."""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
from sag import still_samples  # noqa: E402
from station_sim import StationSim  # noqa: E402


def main():
    sim = StationSim(seed=0, randomize=False)
    m, d = sim.m, sim._kd
    out = {}
    for side in ("right", "left"):
        rows = still_samples(side, n=400)
        tau, err = [], []
        for e, i, a, q in rows:
            d.qpos[:] = sim.d.qpos
            d.qpos[sim.jq[side]] = q
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            tau.append(d.qfrc_bias[sim.jv[side]].copy())   # what the motors must hold against gravity
            err.append(a - q)
        tau, err = np.array(tau), np.array(err)
        kp, off, r2 = [], [], []
        for j in range(6):
            X = np.c_[np.ones(len(tau)), tau[:, j]]
            coef, *_ = np.linalg.lstsq(X, err[:, j], rcond=None)
            pred = X @ coef
            ss = ((err[:, j] - err[:, j].mean()) ** 2).sum()
            r2.append(1 - ((err[:, j] - pred) ** 2).sum() / ss if ss > 0 else 0)
            off.append(coef[0])
            kp.append(1 / coef[1] if coef[1] > 1e-6 else None)
        out[side] = {"samples": len(rows), "kp_Nm_per_rad": [None if k is None else round(float(k), 1) for k in kp],
                     "offset_deg": [round(float(np.degrees(o)), 2) for o in off],
                     "r2": [round(float(v), 2) for v in r2],
                     "gravity_torque_range_Nm": np.ptp(tau, 0).round(2).tolist()}
        print(side, json.dumps(out[side]))
    (HERE / "gains.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
