"""A robots.followers.Driver whose arm is the MuJoCo sim (server.py's /sim/state and /sim/command).

Dream's station agent loads it through addons/sitecustomize.py in place of the Metal arm's Damiao/CAN driver, so
everything above the driver (Dream-1, Dream's IK, the control loop, the station API) is the station's own code.
SIM_URL (default http://127.0.0.1:8191) is the physics server.
"""
import os
import threading
import time

import numpy as np
import requests

from robots import followers

URL = os.environ.get("SIM_URL", "http://127.0.0.1:8191")
_lock = threading.Lock()
_cache = {"t": 0.0, "state": None}
_http = requests.Session()


def _state(max_age_s=0.004):
    """One /sim/state shared by both arms' reads within a few ms."""
    with _lock:
        if _cache["state"] is None or time.monotonic() - _cache["t"] > max_age_s:
            _cache["state"] = _http.get(URL + "/sim/state", timeout=5).json()
            _cache["t"] = time.monotonic()
        return _cache["state"]


class SimArm(followers.Driver):
    drops_when_limp = False

    def __init__(self, cfg, side):
        self._side = side if side in ("left", "right") else "right"
        _state(0)

    def read(self):
        s = _state()[self._side]
        return followers.Reading(np.array(s["positions_rad"], dtype=np.float64),
                                 np.array(s["velocities_rad_s"], dtype=np.float64), float(s["gripper"]),
                                 gripper_effort=float(s["gripper_effort"]))

    def write(self, positions_rad, gripper):
        _http.post(URL + "/sim/command", json={"arm": self._side, "positions_rad": [float(v) for v in positions_rad],
                                              "gripper": float(gripper)}, timeout=5)

    def disengage(self):
        return False

    def close(self):
        pass

    def torque_off(self):
        pass

    def gripper_deg(self):
        return None
