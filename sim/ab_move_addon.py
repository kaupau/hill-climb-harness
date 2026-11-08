"""A/B test of the station's move_addon (stock vs patched) through Dream's real station agent on the sim.

    .venv/bin/python server.py --port 8191 [--strip-held] &
    [SIM_MOVE_ADDON=stock] doppler run -p typesafe -c dev -- dream/run_dream_station.sh &
    .venv/bin/python ab_move_addon.py --label patched

A grasp.py-style top-down unplug of the rightmost plug, every step a POST /move to the agent (:8291): joint moves
to a hover over the plug (jaws closing along the strip), dxyz descents with keep_orientation until the fingertips
are ~14 mm above the strip, close (gripper 0), pull up 5 cm. Reports the hand's height, tilt and gripper after
each step and the sim's verdict.
"""
import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
TIP = 0.032


def call(url, route, body=None):
    req = urllib.request.Request(url + route, data=None if body is None else json.dumps(body).encode(),
                                 method="GET" if body is None else "POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="http://127.0.0.1:8291")
    ap.add_argument("--sim", default="http://127.0.0.1:8191")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    import demo_unplug as D
    from station_sim import StationSim
    ref = StationSim(seed=0)  # the same scene the server reset to: for the plug's position and the hover IK
    Rs = ref.d.xmat[ref.strip].reshape(3, 3)
    seat = ref.d.xpos[ref.strip] + Rs @ ref.m.site_pos[ref.port_site[3]] - ref.base["right"]
    top = seat[2]
    rows = []

    def step(body, what):
        out = call(a.agent, "/move", dict(body, arm="right"))
        p = call(a.agent, "/pose")["right"]
        tilt = math.degrees(math.acos(max(-1, min(1, -np.array(p["R"])[2, 2]))))
        row = {"step": what, "tool_z_mm": round(p["xyz"][2] * 1000), "tip_above_strip_mm": round((p["xyz"][2] - TIP - top) * 1000),
               "tilt_deg": round(tilt, 1), "gripper": p["gripper"], "error": out.get("error")}
        rows.append(row)
        print(f"  {what:24s} tool z {row['tool_z_mm']:4d} mm  tip {row['tip_above_strip_mm']:+4d} mm over strip  "
              f"tilt {row['tilt_deg']:5.1f} deg  gripper {p['gripper']:.2f}{'  ERR ' + out['error'] if out.get('error') else ''}")
        return p

    call(a.sim, "/reset", {"seed": 0})
    call(a.agent, "/home", {})
    hover = np.array([seat[0], seat[1], top + 0.014 + TIP + 0.12])
    q, _ = ref.ik("right", ref.joints("right"), hover, D.tool_R("along", Rs), iters=400)
    q0 = np.array(call(a.agent, "/pose")["right"]["joints"])
    n = int(np.ceil(np.max(np.abs(q - q0)) / 1.1))
    step({"gripper": 0.45, "duration_s": 0.8}, "open")
    for i in range(1, n + 1):
        step({"joints": {str(j): float(v) for j, v in enumerate(q0 + (q - q0) * i / n)}, "duration_s": 2}, f"to hover {i}/{n}")
    goal = top + 0.014 + TIP
    for k in range(5):
        z = call(a.agent, "/pose")["right"]["xyz"][2]
        dz = max(goal - z, -0.04)
        if abs(dz) < 0.003:
            break
        step({"dxyz": [0, 0, dz], "keep_orientation": True, "duration_s": 1.2}, f"down {abs(dz) * 100:.1f} cm")
    step({"gripper": 0.0, "duration_s": 1.0}, "close")
    step({"dxyz": [0, 0, 0.05], "keep_orientation": True, "duration_s": 1.5}, "pull up 5 cm")
    chk = call(a.sim, "/checks")
    res = {"label": a.label, "unplugged": 3 not in chk["plugs_in_ports"], "peak_pull_N": chk["peak_pull_N"][3],
           "retention_N": chk["retention_N"][3], "steps": rows}
    print(f"-> {a.label}: {'UNPLUGGED' if res['unplugged'] else 'still in'}  peak pull {res['peak_pull_N']} of "
          f"{res['retention_N']} N")
    out = HERE / "out" / "ab" / f"{a.label or 'run'}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
