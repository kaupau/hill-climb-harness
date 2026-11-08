"""Scripted unplug in sim: grasp.py's top-down pinch on a plug standing in the strip's top-face USB row.

    .venv/bin/python demo_unplug.py                       # rightmost plug, both jaw rolls, report + frames
    .venv/bin/python demo_unplug.py --plug 2 --style along --seed 3

Sequence (all StationSim.move calls, the same contract as the station's POST /move, saved as `moves`):
open -> hover above the plug with the approach pointing straight down and the wrist rolled so the jaws close
`along` the strip (one finger in front of the plug head, one behind it) or `across` it (fingers in the gaps to
the neighbours) -> descend until the fingertips are `tip_above_mm` above the strip's top face -> close ->
pull straight up.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import build_scene as B
from station_sim import MoveRefused, StationSim

HERE = Path(__file__).parent
TIP = 0.032           # fingertips beyond Dream's TCP along the approach axis (link6 x 0.167 vs 0.135)
FINGER_TIP_MM = 9.5   # finger thickness along the closing axis at the tip (URDF link7 mesh); 14 mm 1 cm back


def room(style, plug):
    """Free space each finger has to go down into, beside the plug head, mm (geometry only)."""
    L, W, H = B.PLUG_HEAD[B.PORT_KINDS[plug]]
    if style == "along":   # in front (toward the strip's near edge); behind is open top face
        return {"front_mm": round((B.PORT_INSET - H / 2) * 1000, 1)}
    ws = [B.PLUG_HEAD[k][1] for k in B.PORT_KINDS]
    gaps = [None if not 0 <= nb < 4 else round((B.PORT_PITCH - (ws[plug] + ws[nb]) / 2) * 1000, 1)
            for nb in (plug - 1, plug + 1)]
    return {"gap_to_left_mm": gaps[0], "gap_to_right_mm": gaps[1]}


def tool_R(style, strip_R):
    z = np.array([0.0, 0.0, -1.0])
    x = strip_R[:, 0] if style == "along" else strip_R[:, 1]
    x = x - z * (x @ z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def attempt(sim, plug=3, style="along", opening=0.45, tip_above_mm=14.0, frames=None, tag="", verbose=True,
            from_target=False):
    arm = "right"
    log, moves = [], []
    res = {"plug": plug, "kind": B.PORT_KINDS[plug], "style": style, "opening": opening,
           "finger_tip_thickness_mm": FINGER_TIP_MM, **room(style, plug)}

    def snap(name):
        if frames:
            Path(frames).mkdir(parents=True, exist_ok=True)
            for cam in ("wrist_r", "scene"):
                cv2.imwrite(str(Path(frames) / f"{tag}{name}_{cam}.jpg"),
                            cv2.cvtColor(sim.render(cam), cv2.COLOR_RGB2BGR))

    def mv(body, what):
        if from_target:
            body = dict(body, from_target=True)
        moves.append(dict(body, arm=arm))
        try:
            out = sim.move(arm, dict(body))
        except MoveRefused as e:
            log.append(f"{what}: REFUSED {e}")
            if verbose:
                print(f"  {what}: REFUSED {e}")
            return None
        c = out["contacts_N"]
        log.append(f"{what}: tool {out['pose']['xyz']} gripper {out['pose']['gripper']:.2f} contacts {c}")
        if verbose:
            print(f"  {what}: tool {np.round(out['pose']['xyz'], 3).tolist()} gripper {out['pose']['gripper']:.2f}"
                  f"{'  contacts ' + json.dumps(c) if c else ''}"
                  f"{'  ** ' + out['station_error'] if out.get('station_error') else ''}")
        return out

    d, m = sim.d, sim.m
    Rs = d.xmat[sim.strip].reshape(3, 3)
    seat = d.xpos[sim.strip] + Rs @ m.site_pos[sim.port_site[plug]] - sim.base[arm]
    top_z = seat[2]
    R = tool_R(style, Rs)
    tcp_grasp = np.array([seat[0], seat[1], top_z + tip_above_mm / 1000 + TIP])
    hover = tcp_grasp + np.array([0, 0, 0.10])
    best = None
    for flip in (1, -1):  # the jaw axis either way round: take the one the wrist reaches with less turning
        Rf = R @ np.diag([flip, flip, 1])
        q, r = sim.ik(arm, sim.joints(arm), hover, Rf, iters=400)
        cost = r * 100 + np.abs(q - sim.joints(arm)).sum()
        if best is None or cost < best[0]:
            best = (cost, q, r)
    _, q_hover, r = best
    if r > 0.01:
        res.update(ok=False, why=f"IK cannot reach the hover pose (off {r * 100:.1f} cm)", log=log, moves=moves)
        return res
    mv({"gripper": opening, "duration_s": 0.8}, "open")
    q0 = sim.joints(arm)
    n = int(np.ceil(np.max(np.abs(q_hover - q0)) / 1.1))
    for i in range(1, n + 1):
        mv({"joints": {str(j): float(v) for j, v in enumerate(q0 + (q_hover - q0) * i / n)}, "duration_s": 2.0},
           f"to hover {i}/{n}")
    # the last few millimetres over the plug in Cartesian space (grasp.py's `center` does this with the camera)
    miss = hover - np.array(sim.pose()[arm]["xyz"])
    if np.linalg.norm(miss) > 0.003:
        mv({"dxyz": miss.tolist(), "point": [0, 0, -1], "duration_s": 0.8}, "centre over the plug")
    snap("1_hover")
    # every move re-aims the approach straight down (`point`): keep_orientation holds the measured, sagged
    # orientation, and on the real arm's soft joints the tilt grew 19 -> 26 deg over a descent, putting a fingertip
    # ~1 cm low onto the strip
    # descend in closed-loop steps on the measured pose, without /move's aim-past-the-sag correction: on the real
    # arm's soft shoulder that correction lands ~7-10 mm low (it measures before the arm has settled), which drives
    # the fingertips into the strip
    for i in range(6):
        dz = float((tcp_grasp - np.array(sim.pose()[arm]["xyz"]))[2])
        if abs(dz) < 0.002:
            break
        step = max(dz * (1.0 if dz > -0.03 else 0.8), -0.05)
        mv({"dxyz": [0, 0, step], "point": [0, 0, -1], "duration_s": 1.0, "no_overshoot": True},
           f"down {abs(step) * 100:.1f} cm")
        sim.advance(0.5)
    for i in range(4):  # re-centre in xy: the descent sags sideways (grasp.py re-centres with the camera here)
        e = tcp_grasp - np.array(sim.pose()[arm]["xyz"])
        if np.linalg.norm(e[:2]) < 0.0015:
            break
        mv({"dxyz": [float(e[0]) * 1.3, float(e[1]) * 1.3, 0.0], "point": [0, 0, -1], "duration_s": 0.6,
            "no_overshoot": True}, f"re-centre {i + 1}")
    snap("2_straddle")
    tool = np.array(sim.pose()[arm]["xyz"])
    res["tip_above_top_mm"] = round((tool[2] - TIP - top_z) * 1000, 1)
    res["xy_miss_mm"] = round(float(np.linalg.norm(tool[:2] - tcp_grasp[:2])) * 1000, 1)
    # close inside a zero dxyz move: /move's sag correction then holds the height while the jaws shut
    mv({"dxyz": [0, 0, 0], "point": [0, 0, -1], "gripper": 0.0, "duration_s": 1.0}, "close")
    sim.advance(0.4)
    g = sim.gripper(arm)
    res["gripper_after_close"] = round(g, 3)
    res["holding"] = g > 0.04
    snap("3_closed")
    # sag-corrected, like grasp.py's pull: with the real arm's soft shoulder a plain 5 cm command lifts a few mm
    mv({"dxyz": [0, 0, 0.05], "point": [0, 0, -1], "duration_s": 1.5}, "pull up 5 cm")
    sim.advance(0.3)
    snap("4_pulled")
    chk = sim.checks()
    res["unplugged"] = plug not in chk["plugs_in_ports"]
    res["neighbours_unplugged"] = [k for k in range(4) if k != plug and k not in chk["plugs_in_ports"]]
    res["peak_pull_N"] = chk["peak_pull_N"][plug]
    res["retention_N"] = chk["retention_N"][plug]
    res["still_holding"] = sim.gripper(arm) > 0.04
    res["station_error"] = sim.last_error
    res["ok"] = bool(res["unplugged"]) and sim.last_error is None
    res["log"], res["moves"] = log, moves
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plug", type=int, default=3, help="port index, 0 = leftmost seen from the robot")
    ap.add_argument("--style", nargs="*", default=["along", "across"])
    ap.add_argument("--opening", type=float, default=0.45)
    ap.add_argument("--tip-above-mm", type=float, default=14.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--held", nargs="*", type=int, default=[0, 1], help="strip free (0) / held down (1)")
    ap.add_argument("--from-target", action="store_true", help="moves start from the last command (proposed fix)")
    ap.add_argument("--frames", default=str(HERE / "out" / "demo"))
    ap.add_argument("--json", default=str(HERE / "out" / "demo_unplug.json"))
    args = ap.parse_args()
    results = []
    for held, style in [(h, st) for h in args.held for st in args.style]:
        sim = StationSim(seed=args.seed, strip_held=bool(held))
        print(f"\n== strip {'HELD DOWN' if held else 'free'}; plug {args.plug} ({B.PORT_KINDS[args.plug]}), "
              f"jaws close {style} the strip, "
              f"retention {sim.retain_n[args.plug]:.1f} N, room {room(style, args.plug)}")
        r = attempt(sim, args.plug, style, args.opening, args.tip_above_mm, frames=args.frames, tag=f"{'held' if held else 'free'}_{style}_", from_target=args.from_target)
        r["strip_held"] = bool(held)
        r["strip_lifted_mm"] = round(float(sim.d.xpos[sim.strip][2] - B.STRIP_H / 2) * 1000, 1)
        results.append(r)
        print(f"  -> {'UNPLUGGED' if r.get('unplugged') else 'failed'}: gripper {r.get('gripper_after_close')}, "
              f"peak pull {r.get('peak_pull_N')} N of {r.get('retention_N')}, still holding "
              f"{r.get('still_holding')}, tip {r.get('tip_above_top_mm')} mm above the top, "
              f"xy miss {r.get('xy_miss_mm')} mm, strip lifted {r.get('strip_lifted_mm')} mm, error {r.get('station_error') or r.get('why')}")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(results, indent=1))
    print("\nwrote", args.json)


if __name__ == "__main__":
    main()
