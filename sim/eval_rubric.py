"""Rubric eval: score a policy on 5-minute sim episodes with the Hill Climb rubric (v4), over several seeds.

    .venv/bin/python eval_rubric.py --policy scripted --seeds 0 1 2 3 4
    # anything that talks HTTP (conductor, teammate harness, grasp.py scripts): the sim server must be running and
    # the command gets STATION in its environment. Dream-1 + conductor on the sim:
    #   .venv/bin/python server.py --port 8191 &  doppler run -p typesafe -c dev -- dream/run_dream_station.sh &
    .venv/bin/python eval_rubric.py --policy external --sim http://127.0.0.1:8191 --station http://127.0.0.1:8291 \
        --seeds 0 1 2 --cmd "doppler run -p nous -c dev -- ../harness/.venv/bin/python ../conductor/conductor.py --live --budget 290"

Writes out/eval/<policy>_<timestamp>.json: per episode the score breakdown, stage pass times, DQ / 8-ball, and the
mean. The score is the rubric's (station_sim.checks): unplug 10 + 5 each further, loose cable plugged in 20 each,
tower > 4 in standing free 3 s: 20 + 3 per full inch, 5 per orange ball in the bin; DQ = 0.
"""
import argparse
import json
import os
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EPISODE_S = 300.0


def scripted_episode(seed, held=False, from_target=False):
    """Baseline: grasp.py's top-down pinch on each plug, right to left, dropping each plug clear of the strip."""
    import demo_unplug as D
    from station_sim import StationSim
    sim = StationSim(seed=seed, strip_held=held)
    for plug in (3, 2, 1, 0):
        if sim.d.time - sim.t_reset > EPISODE_S - 20:
            break
        r = D.attempt(sim, plug, "along", verbose=False, from_target=from_target)
        try:  # let go off to the side, then back up
            sim.move("right", {"dxyz": [-0.06, -0.06, 0.04], "duration_s": 1.5, "from_target": from_target})
        except Exception:  # noqa: BLE001 - a refused move just leaves the arm where it is
            pass
        sim.move("right", {"gripper": 1.0, "duration_s": 0.6})
        print(f"  seed {seed} plug {plug}: {'out' if r.get('unplugged') else 'stuck'} "
              f"(pull {r.get('peak_pull_N')}/{r.get('retention_N')} N) t={sim.d.time - sim.t_reset:.0f}s")
    rest = EPISODE_S - (sim.d.time - sim.t_reset)
    if rest > 0:
        sim.home()
        sim.advance(max(0.0, EPISODE_S - (sim.d.time - sim.t_reset)), check=True)
    return sim.checks()


def _http(url, route, body=None):
    req = urllib.request.Request(url + route, data=None if body is None else json.dumps(body).encode(),
                                 method="GET" if body is None else "POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def external_episode(seed, sim_url, station_url, cmd):
    _http(sim_url, "/reset", {"seed": seed})
    t0 = time.monotonic()
    env = dict(os.environ, STATION=station_url)
    p = subprocess.Popen(shlex.split(cmd), env=env, cwd=HERE)
    try:
        p.wait(timeout=EPISODE_S + 30)
    except subprocess.TimeoutExpired:
        p.terminate()
    try:
        _http(station_url, "/stop", {})
    except Exception:  # noqa: BLE001
        pass
    wall = time.monotonic() - t0
    chk = _http(sim_url, "/checks")
    chk["wall_s"] = round(wall, 1)
    return chk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["scripted", "external"], default="scripted")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--held", action="store_true", help="strip held down (tape / left arm)")
    ap.add_argument("--from-target", action="store_true", help="scripted moves start from the last command")
    ap.add_argument("--sim", default="http://127.0.0.1:8191")
    ap.add_argument("--station", default="http://127.0.0.1:8291")
    ap.add_argument("--cmd", default=None)
    a = ap.parse_args()
    eps = []
    for seed in a.seeds:
        print(f"== episode seed {seed}")
        chk = scripted_episode(seed, a.held, a.from_target) if a.policy == "scripted" else \
            external_episode(seed, a.sim, a.station, a.cmd)
        eps.append({"seed": seed, "score": chk["score"], "plugs_removed": chk["plugs_removed"],
                    "removed_right_to_left": chk["removed_right_to_left"],
                    "stage1_passed_at_s": chk["stage1_passed_at_s"], "stage2_passed_at_s": chk["stage2_passed_at_s"],
                    "dq": chk["dq"], "loses": chk["loses"], "peak_pull_N": chk["peak_pull_N"],
                    "physics_unstable": chk.get("physics_unstable", False),
                    "retention_N": chk["retention_N"]})
        print(f"   score {chk['score']['total']}  {chk['score']}  dq {chk['dq']}"
              f"{'  ** PHYSICS UNSTABLE: not a valid score' if chk.get('physics_unstable') else ''}")
    valid = [e for e in eps if not e["physics_unstable"]] or eps
    tot = [e["score"]["total"] for e in valid]
    summary = {"policy": a.policy, "cmd": a.cmd, "held": a.held, "from_target": a.from_target, "episodes": eps,
               "mean_score": round(float(np.mean(tot)), 1), "min": min(tot), "max": max(tot),
               "mean_plugs_removed": round(float(np.mean([e["plugs_removed"] for e in valid])), 2),
               "unstable_episodes": [e["seed"] for e in eps if e["physics_unstable"]],
               "losses": sum(e["loses"] for e in eps)}
    out = HERE / "out" / "eval" / f"{a.policy}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nmean score {summary['mean_score']} (min {summary['min']}, max {summary['max']}), "
          f"plugs out {summary['mean_plugs_removed']}/4, losses {summary['losses']}  -> {out}")


if __name__ == "__main__":
    main()
