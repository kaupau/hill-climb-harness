"""Loaded by Python at start-up when PYTHONPATH has this folder (run_dream_station.sh sets it).
1. Swaps the Metal body's arm opener for the sim driver (sim_arm.SimArm): no CAN bus, no motors.
2. Adds the station's own pah_judge routes (pah_judge_routes.py, a copy of ~/pah_judge/sitecustomize.py on the
   station): GET /frame/<cam>.jpg, /arms, /pose and POST /move (move_addon.py, identical to station_addons/)."""
import dataclasses
import os
import sys

try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from robots import bodies, followers
    import sim_arm

    def _open(cfg, name):
        return followers.FollowerArm(name, sim_arm.SimArm(cfg, name), home_rad=cfg.home)

    bodies.BODIES["metal"] = dataclasses.replace(bodies.METAL, open_arm=_open, open_arm_read_only=_open)
    bodies.METAL = bodies.BODIES["metal"]
except Exception as e:  # noqa: BLE001
    print(f"sim addon: metal body not replaced: {e}", file=sys.stderr)

# SIM_MOVE_ADDON=stock serves the station's unpatched move_addon (move_addon_stock.py) for A/B tests
if os.environ.get("SIM_MOVE_ADDON") == "stock":
    import move_addon_stock  # noqa: E402
    sys.modules["move_addon"] = move_addon_stock

import pah_judge_routes  # noqa: E402,F401
