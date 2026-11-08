# PAH station sim (MuJoCo)

> Copied from `makermods-pah/sim`. The calibration scripts (`calib/`) read the teleop dataset at `../data/station/unplug-v1`, and `smoke_http.py` / `compare_cams.py` expect `../conductor/grasp.py`, `../harness/.venv` and `../teammate-harness/runs`: they run as-is only in that layout. The sim, server, Dream agent, demo and eval need only this folder.


The MakerMods PAH Hill Climb station in MuJoCo: two MakerMods Metal arms (the station's own URDF and meshes),
the black power strip with a 4-port USB row (USB-C, USB-A, USB-C, USB-A) on its **top face** near the robot end,
four plugged orange cables, two loose cables, three holed foam blocks, two foam cylinders, seven ping-pong balls
(4 orange, 1 white, 2 black) and the white basket, plus the three 16:9 cameras (`scene`, `wrist_l`, `wrist_r`).

Three ways in:

| what | how | port |
|---|---|---|
| Python API | `from station_sim import StationSim` | – |
| Stand-in for the station API (no policy) | `.venv/bin/python server.py` | 8191 |
| **Dream's real station agent + Dream-1 on the sim** | `server.py`, then `dream/run_dream_station.sh` | 8291 |

**Port 8091 on this Mac is the ssh tunnel to the real station.** Nothing here uses it; point clients at 8191 or
8291 (`STATION=http://127.0.0.1:8291`).

## Setup

```bash
cd ~/Documents/Work/hackathon/makermods-pah/sim
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python mujoco numpy pillow opencv-python-headless requests pyyaml scipy
uv venv --python 3.12 .venv-dream && uv pip install --python .venv-dream/bin/python "dreamlabs==1.0.37" "dreamlabs-client==1.1.3" requests opencv-python-headless
```

Both venvs already exist. `.venv-dream` pins the dreamlabs versions the station runs.

## Python API

```python
from station_sim import StationSim
sim = StationSim(seed=0)                  # strip_held=True welds the strip to the table
sim.reset(seed=3)                         # new random object poses (strip +-1 cm / 3 deg, cord routing, objects)
sim.pose()["right"]                       # joints, gripper (0 shut..1 open), xyz (arm base frame), R (Dream's tool axes)
sim.move("right", {"dxyz": [0, 0, -0.02], "keep_orientation": True})   # exactly move_addon.py's /move
sim.set_targets("right", q=[...], gripper=0.3, duration_s=1.0); sim.advance(1.0)   # raw joint targets
sim.ik("right", q0, p, R)                 # move_addon's DLS IK on the sim's FK
img = sim.render("wrist_r")               # RGB, 720x1280
sim.checks()                              # task checks + rubric score (below)
```

Frames: world origin is on the table top midway between the arm bases, with x forward, y left, z up. Each arm's
base frame is its URDF `base_link`, standing on the table at y = ±0.30. The tool point is Dream's: 0.135 m out
along link6 x, with R columns = (jaw closing axis, link6 z, approach), the same as `/pose` on the station.

## HTTP stand-in (`server.py`)

`GET /status`, `/frame/<scene|wrist_l|wrist_r>.jpg`, `/arms`, `/pose`; `POST /move` (move_addon contract and its
refusals as 409), `/home`, `/stop` (also aborts a running move), `/reset` (`{"seed", "randomize"}`).
`POST /start` and `/prompt` return 409 "no policy in sim". Sim-only routes: `GET /checks`, `GET /sim/state`,
`POST /sim/command`.

Physics keeps pace with the wall clock (`--speed 2` runs it twice as fast). Reads are served while a move runs.

Smoke test, with conductor/grasp.py run from a copy in `scratch/grasp_copy` (it saves frames next to itself):

```bash
.venv/bin/python server.py --port 8191 &
doppler run -p typesafe -c dev -- .venv/bin/python smoke_http.py --station http://127.0.0.1:8191
```

All routes passed, including `grasp.py find` (Jev boxed a plug in the sim frame), `center`, `down`, `grip`,
`pull` and `open`. Two findings:

- `grasp.py eagle` from home is refused, for a 1.78 rad joint change. So is `down` from the folded home pose (the
  IK cannot reach it), the same as the real addon would.
- `grasp.py center` moved the hand away from the plug. Its pixel-to-metre Jacobian (`JX`, `JY`) and `AIM` pixel
  were measured on the real camera; the sim wrist camera's mount is a guess. Re-run `grasp.py calibrate` against
  the sim before servoing there.

## Dream-1 in the loop (`dream/`)

This is Dream's own station agent (`python -m robots.station`, dreamlabs 1.0.37) with `body: metal`. A
`sitecustomize` swaps only the Metal arm's motor driver for `dream/sim_arm.py`, which reads and writes the sim over
`/sim/state` and `/sim/command`, and the three cameras for `dream/sim_camera.py`. Everything above the driver is
the station's code: Dream-1, Dream's Metal IK (`eef` head), the control loop, its speed caps and the station API.
The station's route addon (`~/pah_judge/sitecustomize.py`, copied to `dream/addons/pah_judge_routes.py`, with the
same `move_addon.py`) adds `/frame`, `/arms`, `/pose` and `/move`. So the harness, conductor and grasp.py run
unchanged against 8291, including `/start` and `/prompt` with Dream-1.

```bash
.venv/bin/python server.py --port 8191 &                                   # physics + frames
doppler run -p typesafe -c dev -- ./dream/run_dream_station.sh &           # agent on :8291 (DREAM_API_KEY)
curl -X POST localhost:8291/start -H 'Content-Type: application/json' \
  -d '{"prompt": "The power strip is unplugged from the wall and has no power; the orange cords are insulated low-voltage USB cables, not bare wires. Unplug the rightmost orange USB cable from the power strip.", "checkpoint": "dreamlabs/dream-1-beta"}'
STATION=http://127.0.0.1:8291 ...                                          # any of our clients
```

The Metal station has no default checkpoint, so `/start` must name `dreamlabs/dream-1-beta`. The config is
`dream/station_sim.yaml`: the station's `~/.dreamlabs/station.yaml` with its `live` section removed (never reuse
the real robot id), cameras at 10 fps and `stale_after_s` 0.5.

First run (`out/dream_run_montage.jpg`): Dream-1 reached the USB row, opened and closed on the plugs and pulled
plug 1 with up to 7.4 N (retention 10–15 N), but unplugged nothing in 60 s. Inference round trip was about 0.65 s
(p50), with no stalls and with the EE speed cap clamping now and then. Expect a domain gap: the renders are flat,
with no room behind the table.

## Scripted unplug (`demo_unplug.py`)

grasp.py's top-down pinch, run only through `/move` calls (saved in `out/demo_unplug.json` → `moves`):

1. Open to 0.45.
2. Hover over the plug looking straight down, with the wrist rolled so the jaws close **along** the strip (one finger
   in front of the head, one behind) or **across** it.
3. Descend in closed-loop steps on the measured pose, aiming the fingertips 14 mm above the strip's top face. Every
   move passes `point: [0,0,-1]`, which re-aims the approach straight down.
4. Close inside a `dxyz: [0,0,0]` move.
5. Pull straight up 5 cm, sag-corrected as grasp.py does.

Rightmost plug (USB-A), seed 0, retention 10.4 N, with the **teleop-fitted gains** (real-arm sag, ~37 mm at the
tool) and no-slip contacts. `--from-target` makes the sim's own /move start from the last command, as the patched `move_addon` does.

| /move | strip | result | what happened |
|---|---|---|---|
| stock | free or held | failed (4 of 4) | the hand ratchets down onto the strip top and lands 10–20 mm off the plug; peak pull 0.1–8 N |
| **from last command** | **held** | **unplugged** (along and across) | fingertips 11–14 mm above the top, within 1.5–3 mm of centre, plug held in the jaws (10.5 N) |
| from last command | free | failed | the grip holds, but the pull lifts the 1 kg strip 23 mm instead of freeing the plug |

Things the sim showed on the way, which are worth checking on the real arms:

- **Two bugs in `move_addon`, confirmed with the station's own code** (`ab_move_addon.py`, `scratch/ab_ratchet.py`:
  the real addon file in Dream's real station agent on the sim). Both are fixed in `station_addons/move_addon.py`.
  - *Each move starts from the measured joints*, which on the soft real arm (shoulder ~100 N·m/rad, fitted from
    teleop) sit one sag below the command, so every move loses a sag. Four +1 cm `no_overshoot` lifts moved the
    hand **down 76 mm** with the stock addon and up 41 mm with the patched one. `grasp.py`'s `down` and `pull` keep
    the aim-past-the-sag correction, which hides most of this; `advance` (front approach) uses `no_overshoot` and
    is exposed to it.
  - *The gripper restarts at its measured opening*, so any move after a grip that doesn't name `gripper` (such as
    `pull`) re-commands the opening the jaws already have and drops the squeeze: commanded gripper 0.12 after
    `close` + `pull` with the stock addon, 0.0 with the patched one.
  - The fix starts each move from the station's `commanded_rad` / `commanded_gripper` (what the motors hold),
    falling back to the measured joints when the arm is more than 0.12 rad off its command.
  - It did **not** by itself make the full grasp.py-style unplug succeed in the sim: in both versions the plug
    slipped out of the jaws at ~7.3 N against 10.4 N retention. What limits it there is grip force and pad friction,
    which are guesses (25 N, μ 1.1) and should be measured on the real gripper.
- **keep_orientation drifts.** It holds the measured, sagged orientation, so over one descent the gripper tilted
  from 19° to 26° off vertical, which puts a fingertip ~1 cm low. Use `point: [0,0,-1]` for top-down moves.
- **Strip weight against retention.** A free 1 kg strip (guessed weight) lifts before a 10–15 N port lets go. Have
  the left arm press the strip down, or check whether the real strip is heavier or taped down. `strip_held=True`
  models a held strip.
- **The cords are the obstacle.** Cords leave the heads straight up and arch back over the strip, right where the
  rear finger goes down. With the tips at 1 mm above the top, the rear finger landed on the neighbours' cords and the
  strip, and the coupled jaws pushed the plug from one side only. Grasp the upper half of the head (tips 8–10 mm up).
- An earlier version of this sim modelled the USB row on the end face with a horizontal top/bottom pinch. That was
  geometrically impossible: the gripper's rail is ±8.5 cm along the closing axis. It is moot now that the ports
  are on top (see scratch/demo_unplug_endface_old.py).

## Rubric checks and the eval (`sim.checks()`, `GET /checks`, `eval_rubric.py`)

Scored by `hillclimb_rubric_v4.pdf`:

- **plugs_in_ports / plugs_removed / removal_order / removed_right_to_left.** A plug is out once its retention let
  go and it is more than 5 mm from its seat. `peak_pull_N` and `retention_N` are reported per port.
- **loose_inserted.** A loose cable's plug counts as inserted when it sits within 3 mm axial / 2 mm lateral / 10°
  of a free port of its kind, pointing down into it; it is then held by a seat weld.
- **stage1_passed_at_s / stage2_passed_at_s.** Stage 1 passes at the first cable out. Stage 2 passes when a tower
  over 4 in has stood with no arm touching it for 3 s.
- **tower.** The tallest stack connected to the table: height in m and in, and whether it stands on its own.
- **balls_in_bin.** Counted by colour. `loses` is set for a black 8-ball in the bin, or for a DQ.
- **dq.** A stage skip: a gripper touched a stack piece before stage 1 passed, or a ball before stage 2 passed.
- **score.** Unplug +10 for the first cable and +5 for each further one; +20 per loose cable plugged in (up to 4);
  if stage 2 passed, +20 and +3 per full inch of tower height at the end; +5 per orange ball in the bin. A DQ
  scores 0. The LiveKit and Dimensional bonuses are not scored here.

`eval_rubric.py` runs 5-minute episodes over several seeds and writes the mean rubric score, plugs out, stage pass
times and losses to `out/eval/`. `--policy scripted` is the in-process baseline. `--policy external --cmd ...`
launches any HTTP client against the sim, such as the conductor or the teammate's runner with Dream-1 on :8291, and
scores the episode from `/checks`. This lets plans, judge models and primitives be compared on the same seeds
without the arms. Episodes where the physics blew up (MuJoCo reset itself) are flagged `physics_unstable` and
left out of the mean.

Baseline (scripted top-down pinch on all four plugs, right to left, unplug only, 5-min episodes, seeds 0–2,
teleop-fitted gains):

| /move | strip | mean score | plugs out (of 4) | notes |
|---|---|---|---|---|
| stock | free | 0 | 0 | peak pulls 0.4–6.6 N against 11–15 N retention |
| stock | held | 3.3 | 0.33 | peak pulls 1.3–8.7 N; one plug came loose later, during homing |
| from last command | free | 6.7 | 0.67 | the strip lifts before most ports let go |
| from last command | held | **25** | **4** | seeds 0 and 1, all four out in ~95 s; seed 2 went physics-unstable (not counted) |

## Fidelity: measured vs guessed

Read off the station (read-only copies are in `assets/`):

- The URDF (`metal_with_gripper.urdf`) and its meshes (from the station's makermodslab frontend).
- Home poses and gripper ends (`station_configs/dreamlabs_station.yaml`).
- Dream's TCP (0.135 m along link6 x) and its Metal tool axes (`dream_ik/specs.py`).
- Joint limits and efforts.
- The motor-guard rule: a joint more than 12° off its command, not moving, for 0.5 s stops the arm and reports an
  error (`robots/motor_guard.py`).
- The /move contract (move_addon.py, verbatim logic).

Fitted to the teleop recordings (`data/station/unplug-v1`: 8 episodes, 7,708 frames of joints, commands and the
three cameras; scripts in `calib/`):

- **Wrist cameras: mount and fisheye lens** (`calib/fit_wrist_cam.py` → `calib/wrist_cams.json`). The fingers ride
  with the camera, so the pixels that stay still across hundreds of frames at one gripper opening are the fingers.
  The camera pose, focal length and one fisheye term were fitted to those silhouettes, giving finger IoU 0.94
  (right) and 0.95 (left) with the gripper open; `calib/wrist_fit.jpg` shows the overlay. Both cameras sit about
  7.5–8.7 cm along link6 and 6.4–6.6 cm above the finger plane, tilted 11–15° down, with f ≈ 333–369 px at 1280
  wide. The real gripper opens 1.02–1.06× the URDF travel. The sim renders a 150° pinhole and remaps it through
  the fitted lens model.
- **Joint stiffness** (`calib/fit_gains.py` → `calib/gains.json`). In still moments, command minus actual equals an
  offset plus gravity torque over kp. The shoulder fit is clean (R² 0.91 and 0.95) at about 100 N·m/rad on both
  arms; wrist pitch is about 37. The elbow shows a constant 4.6–6.4° leader/follower offset rather than sag. With
  these gains the sim's tool error in those still moments is 36 mm median on the right arm (real: 37) and 20 mm on
  the left (real: 13). `calib/sag.py` checks it.
- **Strip pose: tried, not adopted** (`calib/fit_strip.py`). The strip moved between episodes, and dark-pixel
  segmentation also picks up the other arm and people's clothes, so the fit reached only IoU 0.15–0.36. Its hint of
  a 0.57 m base spacing is too weak to replace 0.60.

Guessed, and worth measuring on the real station:

- Arm spacing: 0.60 m (fits of the overhead frame gave 0.60–0.63).
- Strip: 28×11×3.5 cm, near end at x = 0.33 m, centre line 14 cm right of the middle, 1.0 kg.
- USB row 22 mm in from the end, 22 mm pitch, slots long across the strip.
- Plug heads: USB-A 16.5×8.5×20 mm, USB-C 12.5×7×20 mm. Retention 10–15 N, drawn per plug at each reset.
- Cords: 25 cm, 5 mm thick.
- Foam: blocks are 5 cm cubes with a 2.2 cm square hole; cylinders are 4.5 cm × 6 cm.
- Basket: 19×26×8.5 cm inside.
- Gains of the joints the teleop data can't identify: pan, elbow and roll. Joint damping. Grip force: 25 N per
  finger.
- The overhead camera: 0.6 m up and 68° down with a 106° FOV, pinhole, from a PnP fit of the grippers in one real
  frame. `compare_cams.py` puts sim next to real (`out/compare_cams.jpg`).

Not modelled:

- Arm-to-arm and self collisions. The arm geoms only collide with the scene, because the folded home pose overlaps
  the mesh hulls.
- Plug wiggle: only the axial pull frees a plug.
- The overhead camera's lens distortion, the room behind the table, and cable stiffness memory.

## Files

- `build_scene.py`: writes `station.xml`. `station_sim.py` rebuilds it when it is older than `build_scene.py`.
- `station_sim.py`: the Python API, retention, motor guard, insertion and checks.
- `server.py`: the HTTP stand-in.
- `smoke_http.py`: the route and grasp.py smoke test.
- `demo_unplug.py`: the scripted unplug.
- `compare_cams.py`: sim versus real frames.
- `dream/`: the station agent on the sim: `sim_arm.py`, `sim_camera.py`, `addons/`, `station_sim.yaml` and
  `run_dream_station.sh`.
- `out/`: results and frames. `scratch/`: debugging scripts.
