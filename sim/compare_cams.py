"""Sim cameras next to the real ones: the three cameras with both arms at home (real: a harness run's first
frames), and the right wrist looking down at the USB row (real: conductor/grasp_frames/160458_find.jpg).

    .venv/bin/python compare_cams.py          # writes out/compare_cams.jpg
"""
from pathlib import Path

import cv2
import numpy as np

import demo_unplug as D
from station_sim import StationSim

HERE = Path(__file__).parent
ROOT = HERE.parent
REAL_HOME = ROOT / "teammate-harness" / "runs" / "20261107T233613Z"
REAL_DOWN = ROOT / "conductor" / "grasp_frames" / "160458_find.jpg"
W, H = 640, 360


def label(img, text):
    img = cv2.resize(img, (W, H))
    cv2.rectangle(img, (0, 0), (W, 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    sim = StationSim(seed=0)
    rows = []
    for cam in ("scene", "wrist_l", "wrist_r"):
        real = cv2.imread(str(REAL_HOME / f"0001_unplug_1_a1_{cam}.jpg"))
        s = cv2.cvtColor(sim.render(cam), cv2.COLOR_RGB2BGR)
        rows.append(np.hstack([label(real, f"REAL {cam} (arms at home)"), label(s, f"SIM {cam} (arms at home)")]))
    # right wrist pointing down over the USB row, ~10 cm above the plugs (as in grasp.py's find)
    Rs = sim.d.xmat[sim.strip].reshape(3, 3)
    seat = sim.d.xpos[sim.strip] + Rs @ sim.m.site_pos[sim.port_site[2]] - sim.base["right"]
    q, _ = sim.ik("right", sim.joints("right"), seat + [-0.06, 0, 0.15], D.tool_R("across", Rs), iters=400)
    sim.set_targets("right", q, 1.0, 0)
    sim.d.qpos[sim.jq["right"]] = q
    sim.advance(1.0)
    real = cv2.imread(str(REAL_DOWN))
    s = cv2.cvtColor(sim.render("wrist_r"), cv2.COLOR_RGB2BGR)
    rows.append(np.hstack([label(real, "REAL wrist_r looking down (grasp.py find)"),
                           label(s, "SIM wrist_r looking down over the USB row")]))
    out = HERE / "out" / "compare_cams.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", out)


if __name__ == "__main__":
    main()
