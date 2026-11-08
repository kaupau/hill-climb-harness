"""Real steady-state tracking error (commanded leader joints - follower joints) in still moments of the teleop
data, against the sim's with the same commands. Prints per-joint degrees and the tool-point error in mm."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import teleop as T  # noqa: E402
from station_sim import StationSim  # noqa: E402


def still_samples(side, n=60):
    rows = []
    for e in range(8):
        S, A = T.episode(e)
        o = 0 if side == "left" else 7
        q, a = S[:, o:o + 6], A[:, o:o + 6]
        for i in range(15, len(S), 10):
            if np.abs(a[i - 15:i + 1] - a[i]).max() < 0.004 and np.abs(q[i - 15:i + 1] - q[i]).max() < 0.004:
                rows.append((e, i, a[i], q[i]))
    sel = np.linspace(0, len(rows) - 1, min(n, len(rows))).astype(int)
    return [rows[k] for k in sel]


def main():
    sim = StationSim(seed=0, randomize=False)
    for side in ("right", "left"):
        rows = still_samples(side)
        real_err, sim_err, real_mm, sim_mm = [], [], [], []
        for e, i, a, q in rows:
            real_err.append(a - q)
            pa, _ = sim.fk(side, a)
            pq, _ = sim.fk(side, q)
            real_mm.append(np.linalg.norm(pa - pq) * 1000)
            sim.d.qpos[sim.jq[side]] = a
            sim.d.qvel[:] = 0
            sim.set_targets(side, a, None, 0)
            sim.advance(1.2, check=False)
            qs = sim.joints(side)
            sim_err.append(a - qs)
            sim_mm.append(np.linalg.norm(sim.fk(side, a)[0] - sim.fk(side, qs)[0]) * 1000)
        print(f"{side}: {len(rows)} still moments")
        print("  real  cmd-actual deg, mean:", np.degrees(np.mean(real_err, 0)).round(2),
              f" tool error mm median {np.median(real_mm):.1f} p90 {np.percentile(real_mm, 90):.1f}")
        print("  sim   cmd-actual deg, mean:", np.degrees(np.mean(sim_err, 0)).round(2),
              f" tool error mm median {np.median(sim_mm):.1f} p90 {np.percentile(sim_mm, 90):.1f}")


if __name__ == "__main__":
    main()
