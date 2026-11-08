"""A camera source for Dream's station agent (`cameras[i].source`): frames rendered by the MuJoCo sim.
cv2.VideoCapture shape: read() -> (ok, frame_bgr), release(). SIM_URL as in sim_arm.py."""
import os
import time

import cv2
import numpy as np
import requests

URL = os.environ.get("SIM_URL", "http://127.0.0.1:8191")


class SimCamera:
    def __init__(self, cfg):
        self.name, self.period = cfg.name, 1.0 / float(cfg.fps or 10)
        self._next = 0.0
        self._http = requests.Session()

    def read(self):
        wait = self._next - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._next = time.monotonic() + self.period
        try:
            r = self._http.get(f"{URL}/frame/{self.name}.jpg", timeout=5)
            r.raise_for_status()
        except requests.RequestException:
            return False, None
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        return img is not None, img

    def release(self):
        self._http.close()
