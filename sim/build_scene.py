"""Write station.xml: the MakerMods PAH station as a MuJoCo scene.

    .venv/bin/python build_scene.py            # writes station.xml next to this file

World frame: origin on the table top midway between the two arm bases, x forward (away from the robot),
y to the robot's left, z up. Each arm's base frame is the URDF base_link (x forward, y left, z up), mounted
on the table top at (0, +-ARM_HALF_SEP, 0). Everything that is a guess (not read off the station) is a named
constant below; README.md lists which is which.
"""
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
URDF = HERE / "assets" / "metal_with_gripper.urdf"

# ---- measured / read from the station
GRIP_TRAVEL = 0.05          # URDF joint7: 0 = shut, -0.05 = open (per finger)
JOINT_EFFORT = [30, 100, 30, 10, 10, 10]   # URDF effort limits, N.m

# ---- estimated from the real camera frames (see README "What is a guess")
ARM_HALF_SEP = 0.30         # base-to-base 0.60 m (overhead-frame fits: 0.60-0.63)
STRIP_L, STRIP_W, STRIP_H = 0.28, 0.11, 0.035
STRIP_NEAR_X = 0.33         # x of the strip's USB end face (nearer the robot)
STRIP_Y = -0.14            # strip centre line: between the middle and the right arm
PORT_INSET = 0.022          # USB row on the TOP face, this far in from the near end (Kaushik: ports on top)
SEAT_QUAT = (0.7071068, 0, 0.7071068, 0)  # plug frame in the strip frame: head (plug -x) points up
PORT_PITCH = 0.022          # port spacing across the end face
PORT_KINDS = ["C", "A", "C", "A"]  # left to right as seen from the robot (index 0 = leftmost)
# plug head (the overmould outside the port): out along the port axis, across, up
PLUG_HEAD = {"A": (0.020, 0.0165, 0.0085), "C": (0.020, 0.0125, 0.0070)}
CORD_R, SEG_L, N_SEG = 0.0025, 0.025, 10   # ~25 cm braided cords (teleop videos)
LOOSE_CABLES = ["C", "A"]
# tower pieces, all foam (Kaushik: 3 blocks with holes, 2 cylinders); sizes are guesses from the videos
FOAM_BLOCK, FOAM_HOLE = 0.05, 0.022           # cube edge, square stand-in for the round through-hole
FOAM_CYL_R, FOAM_CYL_H = 0.0225, 0.06
PIECES = [("block", "0.95 0.45 0.7 1"), ("block", "0.25 0.6 0.95 1"), ("block", "0.3 0.8 0.85 1"),
          ("cyl", "1.0 0.85 0.2 1"), ("cyl", "0.95 0.6 0.2 1")]
BALL_R = 0.02
BALLS = ["orange"] * 4 + ["white"] + ["black"] * 2
BIN_IN = (0.19, 0.26, 0.085)  # white plastic basket, interior x, y, height (guess)
GRIP_FORCE_N = 25.0           # max squeeze per finger

# cameras: hfov guesses (the real ones are wide, 1280x720)
WRIST_HFOV, SCENE_HFOV = 120.0, 106.0
WRIST_CAM_POS = (0.062, 0.0, 0.046)   # fallback guess, in link6; calib/wrist_cams.json replaces it
WRIST_CAM_PITCH = 0.0                # fallback guess, degrees down toward the fingers
# fitted to the real wrist cameras from the teleop videos' finger silhouettes (calib/fit_wrist_cam.py): mount pose,
# fisheye lens (rendered as a 150 deg pinhole, remapped in StationSim.render)
WRIST_CALIB_PATH = HERE / "calib" / "wrist_cams.json"
WRIST_CALIB = json.loads(WRIST_CALIB_PATH.read_text()) if WRIST_CALIB_PATH.exists() else {}
WRIST_RENDER_HFOV, WRIST_RENDER_SIZE = 150.0, (1600, 900)
SCENE_CAM_POS = (0.196, 0.0, 0.599)
SCENE_CAM_LOOKAT = (0.196 + 0.218, -0.01, 0.599 - 0.976)

ORANGE = "1.0 0.42 0.22 1"
COLORS = {"orange": "1.0 0.45 0.12 1", "white": "0.95 0.95 0.95 1", "black": "0.05 0.05 0.05 1"}
BLOCK_COLORS = ["0.9 0.2 0.2 1", "0.2 0.5 0.9 1", "0.95 0.8 0.2 1", "0.3 0.75 0.35 1", "0.9 0.45 0.7 1",
                "0.6 0.4 0.85 1"]


def _f(v):
    return " ".join(f"{x:.6g}" for x in v)


def rpy_quat(r, p, y):
    """URDF rpy (fixed X, then Y, then Z) as a MuJoCo quaternion (w x y z)."""
    cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2),
                              math.cos(y / 2), math.sin(y / 2))
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def fovy(hfov):
    return math.degrees(2 * math.atan(math.tan(math.radians(hfov) / 2) * 9 / 16))


def look_xyaxes(pos, target, up=(0, 0, 1)):
    f = np.subtract(target, pos)
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return _f(r) + " " + _f(u)


def wrist_cam_R(pitch, yaw, roll):
    """Wrist camera axes (x right, y up, z back) in link6: looking along +x with link6 z up, then turned
    (calib/fit_wrist_cam.py uses the same)."""
    def rot(axis, deg):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
                "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]
    base = np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).astype(float)
    return base @ rot("x", -pitch) @ rot("y", -yaw) @ rot("z", roll)


def sub(parent, tag, **kw):
    return ET.SubElement(parent, tag, {k.rstrip("_"): (v if isinstance(v, str) else _f(v) if hasattr(v, "__len__")
                                                        else f"{v:.6g}") for k, v in kw.items()})


def urdf_links():
    root = ET.parse(URDF).getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    joints = root.findall("joint")
    return links, joints


def add_arm(world, side, y):
    """The Metal arm as a body tree, names prefixed '<side>/'. Mesh hulls collide with the scene, never with
    their own arm (the folded home pose overlaps hulls)."""
    links, joints = urdf_links()
    p = f"{side}/"
    base = sub(world, "body", name=p + "base_link", pos=(0, y, 0))
    bodies = {"base_link": base}
    arm_rgba = "0.13 0.13 0.14 1"

    def add_geoms(body, lname):
        if lname in ("link7", "link8"):
            mesh = lname
            sub(body, "geom", name=f"{p}{lname}_vis", type="mesh", mesh=mesh, class_="visual", rgba="0.08 0.08 0.08 1")
            sub(body, "geom", name=f"{p}{lname}_col", type="mesh", mesh=mesh, class_="arm_col")
            # rubber pad on the inner face, front 5 cm: closes the 5.4 mm gap the bare fingers leave when shut
            sx = -1 if lname == "link7" else 1
            sub(body, "geom", name=f"{p}{lname}_pad", type="box", size=(0.0013, 0.006, 0.025),
                pos=(sx * 0.0014, 0, 0.064), class_="pad")
            return
        if lname == "gripper_base":
            sub(body, "geom", name=f"{p}gripper_base_vis", type="mesh", mesh="gripper_base", class_="visual",
                rgba="0.1 0.1 0.11 1")
            return
        sub(body, "geom", name=f"{p}{lname}_vis", type="mesh", mesh=lname, class_="visual", rgba=arm_rgba)
        sub(body, "geom", name=f"{p}{lname}_col", type="mesh", mesh=lname, class_="arm_col")

    add_geoms(base, "base_link")
    for j in joints:
        parent, child = j.find("parent").get("link"), j.find("child").get("link")
        if parent == "world":
            continue
        o = j.find("origin")
        xyz = [float(v) for v in o.get("xyz").split()]
        rpy = [float(v) for v in o.get("rpy").split()]
        b = sub(bodies[parent], "body", name=p + child, pos=xyz, quat=rpy_quat(*rpy))
        bodies[child] = b
        inert = links[child].find("inertial")
        io = inert.find("origin")
        I = inert.find("inertia")
        sub(b, "inertial", pos=[float(v) for v in io.get("xyz").split()],
            mass=float(inert.find("mass").get("value")),
            fullinertia=[float(I.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")])
        jt = j.get("type")
        if jt in ("revolute", "prismatic"):
            lim = j.find("limit")
            lo, hi = float(lim.get("lower")), float(lim.get("upper"))
            sub(b, "joint", name=p + j.get("name"), type="hinge" if jt == "revolute" else "slide",
                axis=[float(v) for v in j.find("axis").get("xyz").split()], range=(lo, hi),
                class_="arm_joint" if jt == "revolute" else "finger")
        add_geoms(b, child)
        if child == "link6":
            # the gripper's rail and body as boxes (the mesh hull would fill the space between the rail ends)
            sub(b, "geom", name=p + "rail_col", type="box", size=(0.0085, 0.085, 0.033), pos=(0.0665, 0, 0.0015),
                class_="arm_col")
            sub(b, "geom", name=p + "rail_front_col", type="box", size=(0.005, 0.085, 0.011), pos=(0.080, 0, 0),
                class_="arm_col")
            sub(b, "geom", name=p + "gbody_col", type="box", size=(0.0205, 0.0128, 0.0254), pos=(0.0395, 0, 0),
                class_="arm_col")
            # wrist camera: a small housing on top of the gripper, looking along the fingers (link6 +x)
            cam = "wrist_l" if side == "left" else "wrist_r"
            c = WRIST_CALIB.get(cam)
            if c:
                pos = (c["x"], c["y"], c["z"])
                Rc = wrist_cam_R(c["pitch_deg"], c["yaw_deg"], c["roll_deg"])
                right, up, fy = Rc[:, 0], Rc[:, 1], fovy(WRIST_RENDER_HFOV)
            else:
                pos, a = WRIST_CAM_POS, math.radians(WRIST_CAM_PITCH)
                fwd = np.array([math.cos(a), 0, -math.sin(a)])
                up = np.array([math.sin(a), 0, math.cos(a)])
                right, fy = np.cross(fwd, up), fovy(WRIST_HFOV)
            sub(b, "geom", name=p + "cam_housing", type="box", size=(0.012, 0.016, 0.008),
                pos=(pos[0] - 0.014, pos[1], pos[2]), class_="visual", rgba="0.05 0.05 0.05 1")
            sub(b, "camera", name=cam, pos=pos, xyaxes=_f(right) + " " + _f(up), fovy=fy)
            sub(b, "site", name=p + "tcp", pos=(0.135, 0, 0), size=(0.004,), rgba="1 0 1 0.6", group=4)
            sub(b, "site", name=p + "tip", pos=(0.167, 0, 0), size=(0.003,), rgba="0 1 1 0.6", group=4)
    return bodies


def add_strip(world):
    cx = STRIP_NEAR_X + STRIP_L / 2
    strip = sub(world, "body", name="strip", pos=(cx, STRIP_Y, STRIP_H / 2))
    sub(strip, "joint", name="strip_free", type="free")
    sub(strip, "geom", name="strip_col", type="box", size=(STRIP_L / 2, STRIP_W / 2, STRIP_H / 2),
        rgba="0.04 0.04 0.045 1", mass=1.0, class_="env", friction=(0.8, 0.01, 0.001))
    # outlets on top (visual), the USB panel on the top face at the near end
    for i in range(4):
        for row in (-1, 1):
            sub(strip, "geom", type="box", size=(0.009, 0.012, 0.0006),
                pos=(-STRIP_L / 2 + 0.085 + i * 0.045, row * 0.026, STRIP_H / 2), class_="visual",
                rgba="0.12 0.12 0.13 1")
    px = -STRIP_L / 2 + PORT_INSET
    top = STRIP_H / 2
    sub(strip, "geom", type="box", size=(0.014, STRIP_W / 2 - 0.008, 0.0004),
        pos=(px, 0, top + 0.0002), class_="visual", rgba="0.1 0.1 0.11 1")
    for k in range(4):
        y = (1.5 - k) * PORT_PITCH
        kind = PORT_KINDS[k]
        w, h = (0.0135, 0.0055) if kind == "A" else (0.0090, 0.0035)
        # slot: long side across the strip (y), thin side along it (x)
        sub(strip, "geom", type="box", size=(h / 2, w / 2, 0.0006), pos=(px, y, top + 0.0004),
            class_="visual", rgba="0.6 0.6 0.62 1" if kind == "A" else "0.02 0.02 0.02 1")
        # seat of port k: where a plugged plug's body origin sits (strip frame), head pointing up
        sub(strip, "site", name=f"port{k}", pos=(px, y, top), quat=SEAT_QUAT, size=(0.002,), rgba="0 1 0 0.5",
            group=4)
    return strip


def add_cable(world, name, kind, pos, yaw, plugged):
    """A plug body (origin on the port face, head out along its -x) with a cord of capsules from its back end.
    The far end gets a second, smaller connector."""
    L, W, H = PLUG_HEAD[kind]
    q = SEAT_QUAT if yaw is None else rpy_quat(0, 0, yaw)
    plug = sub(world, "body", name=name, pos=pos, quat=q)
    sub(plug, "joint", name=name + "_free", type="free")
    # head: from 0.5 mm off the face to L out; a ribbed strain relief behind it
    sub(plug, "geom", name=name + "_head", type="box", size=(L / 2, W / 2, H / 2), pos=(-0.0005 - L / 2, 0, 0),
        rgba="0.04 0.04 0.04 1", mass=0.006, class_="plug")
    sub(plug, "geom", name=name + "_relief", type="capsule", size=(0.0032, 0.004), pos=(-L - 0.004, 0, 0),
        quat=rpy_quat(0, math.pi / 2, 0), rgba="0.06 0.06 0.06 1", mass=0.001, class_="plug")
    tw, th = (0.012, 0.0045) if kind == "A" else (0.0083, 0.0025)
    sub(plug, "geom", name=name + "_tongue", type="box", size=(0.0055, tw / 2, th / 2), pos=(0.0055, 0, 0),
        rgba="0.75 0.75 0.78 1", class_="visual")
    sub(plug, "site", name=name + "_grip", pos=(-0.0005 - L / 2, 0, 0), size=(0.002,), group=4)
    parent, x0 = plug, -L - 0.008
    for i in range(N_SEG):
        seg = sub(parent, "body", name=f"{name}_s{i}", pos=(x0, 0, 0))
        sub(seg, "joint", name=f"{name}_j{i}", type="ball", class_="cord")
        sub(seg, "geom", name=f"{name}_c{i}", type="capsule", fromto=(0, 0, 0, -SEG_L, 0, 0), size=(CORD_R,),
            rgba=ORANGE, mass=0.003, class_="cordgeom")
        parent, x0 = seg, -SEG_L
    end = sub(parent, "body", name=name + "_far", pos=(x0, 0, 0))
    sub(end, "joint", name=name + "_jfar", type="ball", class_="cord")
    sub(end, "geom", name=name + "_farhead", type="box", size=(0.012, 0.006, 0.003), pos=(-0.012, 0, 0),
        rgba="0.04 0.04 0.04 1", mass=0.003, class_="plug")
    return plug


def build():
    root = ET.Element("mujoco", model="makermods_pah_station")
    sub(root, "compiler", angle="radian", meshdir=str(HERE / "assets" / "meshes"), autolimits="true")
    sub(root, "option", timestep=0.001, integrator="implicitfast", cone="elliptic", impratio=10,
        noslip_iterations=4)  # no-slip pass: soft contacts otherwise creep under a held pull
    sz = sub(root, "size", memory="64M")
    vis = sub(root, "visual")
    sub(vis, "global", offwidth=WRIST_RENDER_SIZE[0], offheight=WRIST_RENDER_SIZE[1])
    sub(vis, "quality", shadowsize=4096, offsamples=4)
    sub(vis, "headlight", ambient=(0.32, 0.32, 0.32), diffuse=(0.3, 0.3, 0.3), specular=(0.05, 0.05, 0.05))
    sub(vis, "map", znear=0.005)

    d = sub(root, "default")
    dc = sub(d, "default", class_="visual")
    sub(dc, "geom", contype=0, conaffinity=0, group=1, mass=0)
    # arm geoms: contype 2 / conaffinity 1 -> touch the scene (contype 1), never another arm geom
    dc = sub(d, "default", class_="arm_col")
    sub(dc, "geom", contype=2, conaffinity=1, group=3, mass=0, friction=(0.6, 0.005, 0.0001))
    dc = sub(d, "default", class_="pad")
    sub(dc, "geom", contype=2, conaffinity=1, group=3, mass=0, friction=(1.1, 0.01, 0.001), condim=6,
        solref=(0.004, 1), solimp=(0.95, 0.99, 0.001), rgba="0.15 0.15 0.15 1")
    dc = sub(d, "default", class_="env")
    sub(dc, "geom", contype=1, conaffinity=3, group=0, friction=(0.7, 0.005, 0.0001))
    dc = sub(d, "default", class_="plug")
    sub(dc, "geom", contype=1, conaffinity=3, group=0, friction=(0.9, 0.01, 0.001), condim=4)
    dc = sub(d, "default", class_="cordgeom")
    sub(dc, "geom", contype=1, conaffinity=3, group=0, friction=(0.8, 0.005, 0.0001))
    dc = sub(d, "default", class_="cord")
    sub(dc, "joint", damping=0.004, stiffness=0.004, armature=2e-5)
    dc = sub(d, "default", class_="foam")
    sub(dc, "geom", contype=1, conaffinity=3, group=0, friction=(1.0, 0.01, 0.001), solref=(0.012, 1),
        solimp=(0.8, 0.95, 0.004))
    dc = sub(d, "default", class_="arm_joint")
    sub(dc, "joint", damping=1.0, armature=0.02, frictionloss=0.05)
    dc = sub(d, "default", class_="finger")
    sub(dc, "joint", damping=5.0, armature=0.01)

    asset = sub(root, "asset")
    for mname in ("base_link", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_base", "link7",
                  "link8"):
        sub(asset, "mesh", name=mname, file=f"{mname}.STL")
    sub(asset, "texture", name="sky", type="skybox", builtin="gradient", rgb1=(0.85, 0.86, 0.88),
        rgb2=(0.55, 0.56, 0.6), width=256, height=256)
    sub(asset, "texture", name="wood", type="2d", builtin="flat", rgb1=(0.74, 0.72, 0.68), rgb2=(0.70, 0.68, 0.64),
         width=512, height=512)
    sub(asset, "material", name="table", texture="wood", texrepeat=(3, 3), reflectance=0.0)
    sub(asset, "texture", name="floor", type="2d", builtin="checker", rgb1=(0.45, 0.45, 0.47), rgb2=(0.4, 0.4, 0.42),
        width=256, height=256)
    sub(asset, "material", name="floor", texture="floor", texrepeat=(8, 8))

    world = sub(root, "worldbody")
    sub(world, "light", name="ceiling", pos=(0.3, 0, 2.2), dir=(0, 0, -1), diffuse=(0.55, 0.55, 0.55),
        castshadow="true")
    sub(world, "light", name="fill", pos=(1.5, 0.8, 1.6), dir=(-0.6, -0.3, -0.7), diffuse=(0.25, 0.25, 0.25),
        castshadow="false")
    sub(world, "geom", name="floor", type="plane", size=(4, 4, 0.1), pos=(0, 0, -0.74), material="floor",
        class_="visual")
    # table: 0.9 m deep from 15 cm behind the arm bases, 1.8 m wide; top at z = 0
    sub(world, "geom", name="table", type="box", size=(0.45, 0.9, 0.02), pos=(0.3, 0, -0.02), material="table",
        class_="env", friction=(0.6, 0.005, 0.0001))
    sub(world, "camera", name="scene", pos=SCENE_CAM_POS, xyaxes=look_xyaxes(SCENE_CAM_POS, SCENE_CAM_LOOKAT),
        fovy=fovy(SCENE_HFOV))
    # arm mounts (the clamp plates under the bases)
    for side, y in (("left", ARM_HALF_SEP), ("right", -ARM_HALF_SEP)):
        sub(world, "geom", name=f"{side}/mount", type="box", size=(0.07, 0.07, 0.003), pos=(0, y, 0.003),
            class_="visual", rgba="0.1 0.1 0.1 1")
        add_arm(world, side, y)

    add_strip(world)
    for k in range(4):
        y = STRIP_Y + (1.5 - k) * PORT_PITCH
        add_cable(world, f"plug{k}", PORT_KINDS[k], (STRIP_NEAR_X + PORT_INSET, y, STRIP_H), None, True)
    for i, kind in enumerate(LOOSE_CABLES):
        add_cable(world, f"loose{i}", kind, (0.25, -0.3 - 0.06 * i, 0.01), math.pi * 0.9, False)

    for i, (kind, rgba) in enumerate(PIECES):
        if kind == "block":  # a square tube: four foam walls around a vertical hole
            h = FOAM_BLOCK / 2
            b = sub(world, "body", name=f"piece{i}", pos=(0.3, 0.1 + 0.06 * i, h))
            sub(b, "joint", name=f"piece{i}_free", type="free")
            w = (FOAM_BLOCK - FOAM_HOLE) / 4
            for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                hx, hy = (w, h) if sx else (h - 2 * w, w)
                sub(b, "geom", name=f"piece{i}_w{sx}{sy}", type="box", size=(hx, hy, h),
                    pos=(sx * (h - w), sy * (h - w), 0), rgba=rgba, mass=0.001, class_="foam")
        else:
            b = sub(world, "body", name=f"piece{i}", pos=(0.3, 0.1 + 0.06 * i, FOAM_CYL_H / 2))
            sub(b, "joint", name=f"piece{i}_free", type="free")
            sub(b, "geom", name=f"piece{i}", type="cylinder", size=(FOAM_CYL_R, FOAM_CYL_H / 2), rgba=rgba,
                mass=0.003, class_="foam")
    for i, c in enumerate(BALLS):
        b = sub(world, "body", name=f"ball{i}_{c}", pos=(0.4, 0.05 * i, BALL_R))
        sub(b, "joint", name=f"ball{i}_free", type="free", damping=0.0002)
        sub(b, "geom", name=f"ball{i}_{c}", type="sphere", size=(BALL_R,), rgba=COLORS[c], mass=0.0027,
            class_="env", friction=(0.5, 0.002, 0.0002), solref=(0.004, 0.6))
    # bin: the white plastic basket, an open box, free so it can be pushed
    ix, iy, ih = BIN_IN
    t = 0.004
    b = sub(world, "body", name="bin", pos=(0.55, 0.25, 0))
    sub(b, "joint", name="bin_free", type="free")
    sub(b, "geom", name="bin_floor", type="box", size=(ix / 2 + t, iy / 2 + t, t / 2), pos=(0, 0, t / 2),
        rgba="0.93 0.93 0.93 1", mass=0.12, class_="env")
    for sx, sy, hx, hy in ((1, 0, t / 2, iy / 2 + t), (-1, 0, t / 2, iy / 2 + t), (0, 1, ix / 2 + t, t / 2),
                           (0, -1, ix / 2 + t, t / 2)):
        sub(b, "geom", type="box", size=(hx, hy, ih / 2), pos=(sx * (ix / 2 + t / 2), sy * (iy / 2 + t / 2),
                                                              t + ih / 2),
            rgba="0.93 0.93 0.93 1", mass=0.04, class_="env")
    sub(b, "site", name="bin_inside", pos=(0, 0, t + ih / 2), size=(ix / 2, iy / 2, ih / 2), type="box",
        rgba="0 0 1 0.1", group=4)

    # retention: each plugged plug welded to the strip at its port; the sim releases it on an axial pull
    eq = sub(root, "equality")
    for k in range(4):
        sub(eq, "weld", name=f"retain{k}", body1="strip", body2=f"plug{k}", solref=(-40000, -80),
            solimp=(0.99, 0.999, 0.001))  # absolute stiffness: 15 N stretches it ~0.4 mm
    # the strip held flat on the table (tape, or the left arm pressing it): off unless StationSim(strip_held=True)
    sub(eq, "weld", name="strip_hold", body1="strip", active="false", solref=(0.01, 1))
    for i in range(len(LOOSE_CABLES)):
        sub(eq, "weld", name=f"seat_loose{i}", body1="strip", body2=f"loose{i}", active="false",
            solref=(-40000, -80), solimp=(0.99, 0.999, 0.001))
    for side in ("left", "right"):
        sub(eq, "joint", joint1=f"{side}/joint8", joint2=f"{side}/joint7", polycoef=(0, -1, 0, 0, 0),
            solref=(0.004, 1))

    act = sub(root, "actuator")
    # shoulder and wrist pitch from the teleop data (calib/fit_gains.py: still moments, error vs gravity torque,
    # R^2 0.91-0.95 at the shoulder); elbow unidentifiable there (a constant leader-follower offset), set like the
    # shoulder; pan and roll carry no gravity load
    kp = [180, 100, 100, 40, 20, 30]
    for side in ("left", "right"):
        for i in range(6):
            sub(act, "position", name=f"{side}/a{i + 1}", joint=f"{side}/joint{i + 1}", kp=kp[i],
                kv=kp[i] * 0.06, forcerange=(-JOINT_EFFORT[i], JOINT_EFFORT[i]))
        sub(act, "position", name=f"{side}/grip", joint=f"{side}/joint7", kp=3000, kv=40,
            forcerange=(-GRIP_FORCE_N, GRIP_FORCE_N), ctrlrange=(-GRIP_TRAVEL, 0))

    exclude = sub(root, "contact")
    return root


def main(out=HERE / "station.xml"):
    root = build()
    ET.indent(root)
    Path(out).write_text(ET.tostring(root, encoding="unicode"))
    return out


if __name__ == "__main__":
    print("wrote", main())
