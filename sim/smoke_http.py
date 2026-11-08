"""Smoke test of the HTTP stand-in with our own clients: raw routes, then conductor/grasp.py commands.

    .venv/bin/python server.py --port 8191 --speed 3 &
    doppler run -p typesafe -c dev -- .venv/bin/python smoke_http.py --station http://127.0.0.1:8191

grasp.py is run from a copy in scratch/grasp_copy (it saves frames next to itself; conductor/ is not ours).
`find`/`center` need TYPESAFE_API_KEY (Jev); without it they are skipped.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
GRASP = HERE.parent / "conductor" / "grasp.py"
HARNESS_PY = HERE.parent / "harness" / ".venv" / "bin" / "python"


def get(st, route):
    with urllib.request.urlopen(st + route, timeout=30) as r:
        return r.read()


def post(st, route, body):
    req = urllib.request.Request(st + route, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="http://127.0.0.1:8191")
    st = ap.parse_args().station
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

    print(post(st, "/reset", {"seed": 0})[0], "reset")
    s = json.loads(get(st, "/status"))
    check("GET /status", s["state"] in ("idle", "moving") and "last_error" in s, s["state"])
    for cam in ("scene", "wrist_l", "wrist_r"):
        jpg = get(st, f"/frame/{cam}.jpg")
        check(f"GET /frame/{cam}.jpg", jpg[:2] == b"\xff\xd8" and len(jpg) > 10000, f"{len(jpg)} bytes")
    a = json.loads(get(st, "/arms"))
    check("GET /arms", set(a) == {"left", "right"} and 0 <= a["right"]["gripper"] <= 1, a["right"])
    p = json.loads(get(st, "/pose"))["right"]
    check("GET /pose", len(p["joints"]) == 6 and len(p["R"]) == 3, p["xyz"])
    code, r = post(st, "/start", {"prompt": "unplug"})
    check("POST /start refused (no policy)", code == 409, r.get("error", "")[:40])
    code, r = post(st, "/move", {"arm": "right", "dxyz": [0, 0, -0.3]})
    check("POST /move over 12 cm refused", code == 409, r.get("error"))
    code, r = post(st, "/move", {"arm": "right", "djoints": {"0": 2.0}})
    check("POST /move > 1.2 rad refused", code == 409, r.get("error"))
    # put the right hand over the rightmost plug, looking down, jaws along the strip (as demo_unplug does)
    sys.path.insert(0, str(HERE))
    from station_sim import StationSim
    import demo_unplug as D
    sim = StationSim(seed=0)
    Rs = sim.d.xmat[sim.strip].reshape(3, 3)
    seat = sim.d.xpos[sim.strip] + Rs @ sim.m.site_pos[sim.port_site[3]] - sim.base["right"]
    q, _ = sim.ik("right", sim.joints("right"), seat + [0, 0, 0.20], D.tool_R("along", Rs), iters=400)
    q0 = np.array(p["joints"])
    n = int(np.ceil(np.max(np.abs(q - q0)) / 1.1))  # /move takes at most 1.2 rad per joint per move
    for i in range(1, n + 1):
        code, r = post(st, "/move", {"arm": "right", "joints": {str(j): float(v) for j, v in
                                                                 enumerate(q0 + (q - q0) * i / n)}, "duration_s": 2})
    check("POST /move joints (to hover)", code == 200, r["pose"]["xyz"] if code == 200 else r)
    z0 = json.loads(get(st, "/pose"))["right"]["xyz"][2]
    code, r = post(st, "/move", {"arm": "right", "dxyz": [0, 0, -0.02], "keep_orientation": True})
    # with the real arm's sag (teleop-fitted gains) a 2 cm move from the measured pose lands up to a sag lower
    drop = z0 - r["pose"]["xyz"][2] if code == 200 else None
    check("POST /move dxyz keep_orientation (2 cm down)", code == 200 and 0.01 < drop < 0.05,
          f"moved down {drop * 100:.1f} cm" if code == 200 else r)

    # conductor/grasp.py against the sim, from a copy
    work = HERE / "scratch" / "grasp_copy"
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy(GRASP, work / "grasp.py")
    env = dict(os.environ, STATION=st, PLUG_Z=str(round(seat[2] + 0.010 + 0.032, 3)))
    cmds = ["down 0.03", "down 0.03", "grip", "pull 0.05", "open"]
    if os.environ.get("TYPESAFE_API_KEY"):
        cmds = ["find", "center"] + cmds
    else:
        print("SKIP  grasp.py find/center (no TYPESAFE_API_KEY)")
    for c in cmds:
        out = subprocess.run([str(HARNESS_PY), "grasp.py", *c.split()], cwd=work, env=env, capture_output=True,
                             text=True, timeout=300)
        last = (out.stdout + out.stderr).strip().splitlines()
        check(f"grasp.py {c}", out.returncode == 0, " | ".join(last[-2:]))
    chk = json.loads(get(st, "/checks"))
    print("checks:", json.dumps({k: chk[k] for k in ("plugs_in_ports", "peak_pull_N", "score")}))
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
