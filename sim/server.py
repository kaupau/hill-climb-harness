"""Dream's station HTTP API in front of the MuJoCo station, so our harnesses can point at the sim.

    .venv/bin/python server.py                    # http://127.0.0.1:8191, real-time (8091 = real station tunnel)
    .venv/bin/python server.py --port 8191 --speed 3 --seed 4
    STATION=http://127.0.0.1:8191 ../harness/.venv/bin/python ../conductor/grasp.py pull 0.03

Routes (same shapes as the station + our pah_judge addons):
  GET  /status                 the controller's status dict (state, last_error, arms, cameras, ...) + "sim"
  GET  /frame/<scene|wrist_l|wrist_r>.jpg   1280x720 JPEG
  GET  /arms                   per arm: gripper (0 shut..1 open), commanded_gripper, gripper_effort
  GET  /pose                   per arm: joints, gripper, xyz (arm base frame), R (Dream's tool axes)
  POST /move                   station_addons/move_addon.py contract: arm, dxyz | point | joints | djoints,
                               gripper, duration_s, keep_orientation, no_overshoot; 409 when refused
  POST /home, /stop, /reset    /reset takes optional {"seed": int, "randomize": bool}
  POST /start, /prompt         409: there is no policy in the sim
  GET  /checks                 sim only: task success checks and rubric score
  GET  /sim/state              sim only: per arm joints (rad) + gripper, sim time (for dream_driver.py)
  POST /sim/command            sim only: {arm, positions_rad[6], gripper} as a joint target, applied at once
Never touches hardware.
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from station_sim import CAMERAS, MoveRefused, StationSim

MAX_BODY = 64 * 1024


class _Abort(Exception):
    pass


class SimStation:
    """Owns the StationSim on one thread; HTTP handlers hand it jobs."""

    def __init__(self, seed=0, speed=1.0, idle=True, strip_held=False):
        self.seed, self.speed, self.idle, self.strip_held = seed, speed, idle, strip_held
        self.jobs: queue.Queue = queue.Queue()
        self.busy = None          # name of the long job running (move/home/reset)
        self.stop_flag = False
        self.later: list = []     # long jobs the tick hook picked up while idle, for the main loop
        self.t0 = time.monotonic()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.ready.wait(60)

    # every call goes through here; quick=True jobs may run inside a long job's stepping
    def call(self, fn, quick, timeout=120):
        box = {"done": threading.Event()}
        self.jobs.put((fn, quick, box))
        if not box["done"].wait(timeout):
            raise TimeoutError
        if "exc" in box:
            raise box["exc"]
        return box["out"]

    def _do(self, fn, box):
        try:
            box["out"] = fn()
        except BaseException as e:  # noqa: BLE001 - handed back to the HTTP thread
            box["exc"] = e
        box["done"].set()

    def _hook(self):
        """Inside a long job: serve reads, refuse a second long job, honour /stop."""
        pending = []
        while True:
            try:
                fn, quick, box = self.jobs.get_nowait()
            except queue.Empty:
                break
            if quick:
                self._do(fn, box)
            else:
                pending.append((fn, quick, box))
        for job in pending:
            if self.busy is None:  # idle stepping: the main loop runs it next
                self.later.append(job)
            elif getattr(job[0], "is_stop", False):
                self.stop_flag = True
                self._do(job[0], job[2])
            else:
                job[2]["exc"] = MoveRefused(f"move refused: another {self.busy} is in progress")
                job[2]["done"].set()
        if self.stop_flag:
            self.stop_flag = False
            raise _Abort

    def _run(self):
        self.sim = StationSim(seed=self.seed, strip_held=self.strip_held)
        self.sim.tick_hook = self._hook
        self.ready.set()
        # physics keeps pace with the wall clock (x speed) between jobs, however busy the queue is: a client
        # streaming commands and frames (Dream's station agent) must not starve the stepping
        clock = (time.monotonic(), self.sim.d.time)
        while True:
            try:
                fn, quick, box = self.later.pop(0) if self.later else self.jobs.get(timeout=0.002)
            except queue.Empty:
                fn = None
            if fn is not None and (quick or getattr(fn, "is_stop", False)):
                self._do(fn, box)
            elif fn is not None:
                self.busy = getattr(fn, "job_name", "move")
                try:
                    self._do(fn, box)
                finally:
                    self.busy = None
                clock = (time.monotonic(), self.sim.d.time)
            if self.idle:
                debt = (time.monotonic() - clock[0]) * (self.speed or 1.0) - (self.sim.d.time - clock[1])
                if debt > 0.25:  # too far behind (slow machine): drop the backlog rather than run flat out
                    clock = (time.monotonic(), self.sim.d.time)
                    debt = 0.02
                if debt > 0.002:
                    try:
                        self.sim.advance(min(debt, 0.02))
                    except _Abort:
                        pass

    # ------------------------------------------------------------------------------------- route bodies
    def status(self):
        s = self.sim
        return {"state": "moving" if self.busy and s.state != "error" else s.state, "prompt": None,
                "checkpoint": None, "action_space": None, "steps": 0, "run": 0, "chunks": 0, "stalls": 0,
                "clamps": 0, "max_ee_speed_requested_m_s": 0.0, "max_ee_speed_m_s": 0.6,
                "arms": {"left": True, "right": True}, "cameras": {c: 0.0 for c in CAMERAS},
                "exposure": {}, "capture_fps": {c: 30.0 for c in CAMERAS}, "session_id": None,
                "last_error": s.last_error, "timing": {}, "started_at": None,
                "uptime_s": round(time.monotonic() - self.t0, 1), "run_elapsed_s": None,
                "sim": {"time_s": round(s.d.time, 2), "speed": self.speed, "busy": self.busy,
                        "plugs_in_ports": s.checks()["plugs_in_ports"]}}

    def sim_state(self):
        s = self.sim
        return {"time_s": s.d.time, **{a: {"positions_rad": s.joints(a).tolist(),
                                          "velocities_rad_s": s.d.qvel[s.jv[a]].tolist(), "gripper": s.gripper(a),
                                          "commanded_gripper": float(s.gtarget[a]),
                                          "gripper_effort": float(s.d.actuator_force[s.gact[a]])}
                                      for a in ("left", "right")}}

    def sim_command(self, body):
        a = body["arm"]
        if a not in ("left", "right"):
            raise ValueError(f"no arm {a!r}")
        self.sim.set_targets(a, body.get("positions_rad"), body.get("gripper"), 0)
        return {"ok": True}

    def frame(self, cam):
        img = self.sim.render(cam)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

    def move(self, body):
        def f():
            try:
                return self.sim.move(body, realtime=self.speed)
            except _Abort:
                return {"ok": False, "stopped": True, "pose": self.sim.pose()[body.get("arm", "right")]}
        f.job_name = "move"
        return self.call(f, quick=False)

    def home(self):
        def f():
            try:
                self.sim.home(realtime=self.speed)
            except _Abort:
                pass
            return self.status()
        f.job_name = "home"
        return self.call(f, quick=False)

    def stop(self):
        def f():
            for a in ("left", "right"):
                self.sim.set_targets(a, self.sim.joints(a), None, 0)
            if self.sim.state != "error":
                self.sim.state = "idle"
            return self.status()
        f.is_stop = True
        return self.call(f, quick=False)

    def reset(self, body):
        def f():
            self.sim.reset(seed=body.get("seed"), randomize=body.get("randomize", True))
            return self.status()
        f.job_name = "reset"
        return self.call(f, quick=False)


def make_handler(st: SimStation):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def _reply(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ValueError("body too large")
            raw = self.rfile.read(n) if n else b""
            if not raw.strip():
                return {}
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            return body

        def do_GET(self):
            route = self.path.split("?", 1)[0]
            try:
                if route == "/status":
                    return self._reply(200, st.call(st.status, quick=True))
                if route == "/arms":
                    return self._reply(200, st.call(st.sim.arms, quick=True))
                if route == "/pose":
                    return self._reply(200, st.call(st.sim.pose, quick=True))
                if route == "/checks":
                    return self._reply(200, st.call(st.sim.checks, quick=True))
                if route == "/sim/state":
                    return self._reply(200, st.call(st.sim_state, quick=True))
                m = re.match(r"^/frame/([A-Za-z0-9_]+)\.jpg$", route)
                if m:
                    if m.group(1) not in CAMERAS:
                        return self._reply(404, {"error": f"no frame from camera {m.group(1)!r}"})
                    body = st.call(lambda: st.frame(m.group(1)), quick=True)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self._reply(404, {"error": f"no such route: GET {self.path}"})
            except Exception as e:  # noqa: BLE001
                self._reply(500, {"error": f"{type(e).__name__}: {e}"})

        def do_POST(self):
            route = self.path.split("?", 1)[0]
            try:
                body = self._read_json()
                if route == "/move":
                    return self._reply(200, st.move(body))
                if route == "/sim/command":
                    return self._reply(200, st.call(lambda: st.sim_command(body), quick=True))
                if route == "/home":
                    return self._reply(200, st.home())
                if route == "/stop":
                    return self._reply(200, st.stop())
                if route == "/reset":
                    return self._reply(200, st.reset(body))
                if route in ("/start", "/prompt"):
                    if route == "/prompt" and "prompt" not in body:
                        raise ValueError("body must be a JSON object with a 'prompt' key")
                    return self._reply(409, {"error": "no policy in sim: Dream-1 does not run here; drive the "
                                                      "arms with POST /move (or add a scripted policy)"})
                self._reply(404, {"error": f"no such route: POST {route}"})
            except MoveRefused as e:
                self._reply(409, {"error": str(e)})
            except (ValueError, json.JSONDecodeError) as e:
                self._reply(400, {"error": str(e)})
            except TimeoutError:
                self._reply(504, {"error": "the sim did not answer in time"})
            except Exception as e:  # noqa: BLE001
                self._reply(500, {"error": f"{type(e).__name__}: {e}"})

    return H


def serve(port=8191, host="127.0.0.1", seed=0, speed=1.0, idle=True, strip_held=False):
    st = SimStation(seed=seed, speed=speed, idle=idle, strip_held=strip_held)
    srv = ThreadingHTTPServer((host, port), make_handler(st))
    return st, srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8191)  # 8091 here is the ssh tunnel to the real station
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0, help="sim seconds per wall second for moves; 0 = flat out")
    ap.add_argument("--no-idle", action="store_true", help="freeze physics between requests")
    ap.add_argument("--strip-held", action="store_true", help="weld the strip to the table (tape / left arm)")
    a = ap.parse_args()
    st, srv = serve(a.port, a.host, a.seed, a.speed, not a.no_idle, a.strip_held)
    print(f"sim station on http://{a.host}:{a.port} (speed {a.speed}x, seed {a.seed})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
