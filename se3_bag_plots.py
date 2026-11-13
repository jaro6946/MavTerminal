#!/usr/bin/env python3
"""se3_bag_plots.py -- one SE3 companion session (ROS 2 bag + node trace) as a
stacked dashboard on one time axis.

What a session folder holds (grazer17's companion, "SE3 Controller Bags"):

    <bag>/<bag>_0.mcap        /fmu/out/* (what PX4 published) and /fmu/in/*
                              (every setpoint the SE3 node streamed)
    <bag>/logs/se3_trace.csv  one row per SE3 control tick: commanded attitude,
                              wrench, reference and position (only while SE3
                              itself is flying -- PX4 phases have no rows)
    <bag>/logs/node.log ...   the node's console (step transitions, errors)

Panels, top to bottom, all sharing the time axis (seconds from bag start):

    0  phase strip   SE3 step (trace), who flies (from vehicle_control_mode),
                     armed, landed
    1  position      E/N/U actual vs SE3 reference (trace), |error|
    2  ground drift  odometry position from its first sample + heading: a
                     landed, disarmed vehicle cannot move, so a slope here is
                     the ESTIMATE drifting, not the vehicle
    3  attitude      roll / pitch / yaw actual vs SE3 command, and the
                     attitude-error angle the node's 60 deg recovery gate uses
    4  wrench        thrust (N) with the hover line, torque x/y/z (normalised)
                     with the +-1 saturation band
    5  motors / RC   actuator outputs, RC channels and RSSI
    6  link health   rate of each stream (Hz), timesync round trip
    7  EKF / failsafe flags   on/off rows

Reading the bag
---------------
The MCAP file is scanned record by record here rather than through rosbags'
Reader: a recorder that was killed (not SIGINT'ed) leaves no footer/summary,
and the Reader refuses such a file ("File end magic is invalid") even though
every complete chunk in it is fine -- the 2026-11-11 11-53-57 bag is one.  The
linear scan reads both kinds.  rosbag2 writes uncompressed chunks; zstd/lz4
chunks are decompressed when the libraries are importable.  Message types are
registered from the schemas embedded in the bag, so nothing needs px4_msgs
built or ROS sourced -- only the pure-python `rosbags` in bag_video/pylib.

Clocks
------
Every bag message carries the companion's receive time (log_time, ns) and the
trace's `wall` column is the same companion clock (time.time() in the node).
Both run on the RB5's dead-RTC 2023 date, which is irrelevant: only
differences are used.  The alignment is CHECKED, not assumed: each trace row
names the odometry sample it used (t_odom_us = that message's timestamp), and
the bag's log_time for that same message must sit just before the row's wall
time.  The median and spread of that lag are reported in the figure header.

Frames
------
PX4 topics are NED world / FRD body, quaternions scalar-first [w,x,y,z].  The
trace's cmd_q is rotorpy's convention, ENU world / FLU body, scalar-last
[x,y,z,w].  Commanded attitude is converted to NED/FRD so the angles read like
any PX4 log (yaw from north, clockwise).  The attitude-error angle is frame
independent once both are in the same frame; it is computed exactly as the
node's _att_gate_tripped does (against odometry q), so the plotted line is
the number the recovery gate saw.

Acronyms: SE3 = special Euclidean group (the geometric controller), EKF =
extended Kalman filter, NED/ENU = north-east-down / east-north-up, FRD/FLU =
forward-right-down / forward-left-up body axes, RC = radio control, RSSI =
received signal strength, MCAP = the bag container format, CDR = the ROS 2
wire encoding.
"""
import csv
import dataclasses
import glob
import os
import re
import struct
import sys

import numpy as np

_PYLIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bag_video", "pylib")
if os.path.isdir(_PYLIB) and _PYLIB not in sys.path:
    sys.path.append(_PYLIB)       # append: the venv's numpy must win (see bag_video)

from ulog_common import (C_ARMED, C_BAD, C_GRID, C_INK, C_MUTED, C_SURFACE,
                         mode_color, nav_state_name)

__all__ = ["BagSession", "load_session", "list_bags", "handoffs",
           "build_dashboard", "PANELS", "HOVER_THRUST_N"]

# Grazer 17's hover thrust as the node reports it (m g, 1.7 kg): "hover thrust
# 0.500 <-> 16.68 N".  Overridden by the node.log line when present.
HOVER_THRUST_N = 16.68
ATT_GATE_DEG = 60.0           # node default max_att_err_deg

# --- colour -----------------------------------------------------------------
C_ACT = "#20222b"             # near-black: what the vehicle did (as ulog_commands)
C_CMD = "#d81b60"             # magenta: what SE3 commanded
AXIS_COL = {"E": "#2a78d6", "N": "#d2691e", "U": "#4a3aa7",
            "x": "#2a78d6", "y": "#d2691e", "z": "#4a3aa7"}
C_SE3 = "#e08a00"             # orange: SE3 flying (direct thrust/torque)
C_PX4_OFFB = "#2a78d6"        # blue: offboard, but PX4's own position loop
C_HOVER = "#1e9e5a"
MOTOR_COL = ["#2a78d6", "#d2691e", "#1e9e5a", "#8a3fb5"]


# ===========================================================================
# MCAP scan
# ===========================================================================

def _str(buf, p):
    n = struct.unpack_from("<I", buf, p)[0]
    return buf[p + 4:p + 4 + n].decode("utf-8", "replace"), p + 4 + n


def _records(buf, pos, end):
    while pos + 9 <= end:
        op = buf[pos]
        ln = struct.unpack_from("<Q", buf, pos + 1)[0]
        if pos + 9 + ln > end:
            return                      # truncated tail (killed recorder)
        yield op, pos + 9, pos + 9 + ln
        pos += 9 + ln


def _decompress(comp, data, size):
    if comp == "":
        return data
    if comp == "zstd":
        try:
            from backports import zstd               # bag_video/pylib
            return zstd.decompress(data)
        except ImportError:
            import zstandard
            return zstandard.ZstdDecompressor().decompress(data, max_output_size=size)
    if comp == "lz4":
        import lz4.frame
        return lz4.frame.decompress(data)
    raise ValueError(f"unsupported MCAP chunk compression {comp!r}")


def _scan_mcap(path):
    """(schemas {id: (name, text)}, channels {id: (topic, schema_id)},
    messages [(channel_id, log_time_ns, bytes)], truncated?)"""
    with open(path, "rb") as f:
        buf = f.read()
    schemas, chans, msgs = {}, {}, []
    truncated = buf[-8:] != b"\x89MCAP0\r\n"

    def handle(b, op, a, e):
        if op == 3:                                   # Schema
            sid = struct.unpack_from("<H", b, a)[0]
            name, p = _str(b, a + 2)
            _enc, p = _str(b, p)
            n = struct.unpack_from("<I", b, p)[0]
            schemas[sid] = (name, b[p + 4:p + 4 + n].decode("utf-8", "replace"))
        elif op == 4:                                 # Channel
            cid, sid = struct.unpack_from("<HH", b, a)
            topic, _p = _str(b, a + 4)
            chans[cid] = (topic, sid)
        elif op == 5:                                 # Message
            cid = struct.unpack_from("<H", b, a)[0]
            lt = struct.unpack_from("<Q", b, a + 6)[0]
            msgs.append((cid, lt, b[a + 22:e]))
        elif op == 6:                                 # Chunk
            p = a + 28
            comp, p = _str(b, p)
            n = struct.unpack_from("<Q", b, p)[0]
            usize = struct.unpack_from("<Q", b, a + 16)[0]
            inner = _decompress(comp, b[p + 8:p + 8 + n], usize)
            for r in _records(inner, 0, len(inner)):
                handle(inner, *r)

    for r in _records(buf, 8, len(buf)):
        handle(buf, *r)
    return schemas, chans, msgs, truncated


def _typestore(schemas):
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    store = get_typestore(Stores.ROS2_JAZZY)
    types = {}
    for name, text in schemas.values():
        # ros2msg schema = the type, then "=====\nMSG: pkg/msg/Nested" sections
        parts = re.split(r"^=+\s*\nMSG: (\S+)\s*\n", text, flags=re.M)
        types.update(get_types_from_msg(parts[0], name))
        for i in range(1, len(parts) - 1, 2):
            types.update(get_types_from_msg(parts[i + 1], parts[i]))
    store.register(types)
    return store


# ===========================================================================
# Session
# ===========================================================================

# Topic (without /fmu/out/) -> fields to keep.  None = every bool field.
WANT = {
    "vehicle_odometry": ("timestamp", "position", "q", "velocity"),
    "vehicle_attitude": ("q",),
    "vehicle_status_v4": ("arming_state", "nav_state"),
    "vehicle_status": ("arming_state", "nav_state"),
    "vehicle_control_mode": ("flag_armed", "flag_control_offboard_enabled",
                             "flag_control_attitude_enabled",
                             "flag_control_rates_enabled",
                             "flag_control_position_enabled",
                             "flag_control_velocity_enabled"),
    "vehicle_land_detected": ("landed", "maybe_landed", "ground_contact"),
    "actuator_outputs": ("output", "noutputs"),
    "input_rc": ("values", "rssi", "rc_lost", "channel_count"),
    "timesync_status": ("round_trip_time", "estimated_offset"),
    "estimator_status_flags": None,
    "failsafe_flags": None,
}


@dataclasses.dataclass
class BagSession:
    path: str
    name: str
    t0_ns: int
    duration_s: float
    truncated: bool
    topics: dict            # short name -> {"t": s, field: array}
    rates: dict             # full topic -> array of receive times (s)
    trace: dict = None      # column -> array, "t" in s; None if no trace
    trace_note: str = ""
    align: str = ""         # alignment check, for the header
    node_log: str = ""
    hover_n: float = HOVER_THRUST_N


def list_bags(date_folder):
    """Bag folders (with an .mcap) in a date folder, sorted."""
    out = []
    for d in sorted(glob.glob(os.path.join(date_folder, "*"))):
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.mcap")):
            out.append(d)
    return out


def _short(topic):
    return topic.rsplit("/", 1)[-1]


def load_session(bag_dir):
    mcaps = sorted(glob.glob(os.path.join(bag_dir, "*.mcap")))
    if not mcaps:
        raise FileNotFoundError(f"no .mcap in {bag_dir}")
    schemas, chans, msgs, truncated = {}, {}, [], False
    for m in mcaps:                    # split bags: _0, _1, ... share schemas
        s, c, ms, tr = _scan_mcap(m)
        schemas.update(s)
        chans.update(c)
        msgs += ms
        truncated |= tr
    if not msgs:
        raise ValueError(f"{os.path.basename(bag_dir)}: no messages")
    msgs.sort(key=lambda r: r[1])
    store = _typestore(schemas)
    t0 = msgs[0][1]

    rates, rows = {}, {}
    for cid, lt, data in msgs:
        topic, sid = chans[cid]
        rates.setdefault(topic, []).append(lt)
        short = _short(topic)
        if not topic.startswith("/fmu/out/") or short not in WANT:
            continue
        try:
            m = store.deserialize_cdr(data, schemas[sid][0])
        except Exception:
            continue
        fields = WANT[short]
        if fields is None:
            fields = [f.name for f in dataclasses.fields(m)
                      if isinstance(getattr(m, f.name), (bool, np.bool_))]
        r = rows.setdefault(short, {"t": []})
        r["t"].append((lt - t0) / 1e9)
        for f in fields:
            r.setdefault(f, []).append(getattr(m, f, np.nan))
    topics = {}
    for short, r in rows.items():
        topics[short] = {k: np.asarray(v) if k == "t" else _arr(v) for k, v in r.items()}
    if "vehicle_status_v4" in topics and "vehicle_status" not in topics:
        topics["vehicle_status"] = topics["vehicle_status_v4"]
    rates = {k: (np.asarray(v, float) - t0) / 1e9 for k, v in rates.items()}

    sess = BagSession(path=bag_dir, name=os.path.basename(bag_dir.rstrip("/")),
                      t0_ns=t0, duration_s=(msgs[-1][1] - t0) / 1e9,
                      truncated=truncated, topics=topics, rates=rates)
    _attach_trace(sess)
    _read_node_log(sess)
    return sess


def _arr(v):
    try:
        return np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return np.asarray(v, dtype=object)


def _attach_trace(sess):
    path = os.path.join(sess.path, "logs", "se3_trace.csv")
    if not os.path.isfile(path):
        sess.trace_note = "no se3_trace.csv for this bag"
        return
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError as e:
        sess.trace_note = f"se3_trace.csv unreadable: {e}"
        return
    if not rows:
        sess.trace_note = "se3_trace.csv is empty"
        return
    tr = {}
    for k in rows[0]:
        if k == "phase":
            tr[k] = np.asarray([r[k] for r in rows], dtype=object)
        else:
            tr[k] = np.asarray([_num(r[k]) for r in rows], dtype=float)
    tr["t"] = (tr["wall"] * 1e9 - sess.t0_ns) / 1e9
    sess.trace = tr

    # Alignment check: the odometry sample each row used vs its bag log_time.
    od = sess.topics.get("vehicle_odometry")
    if od is not None and "timestamp" in od and "t_odom_us" in tr:
        ts = od["timestamp"]
        order = np.argsort(ts)
        tss, tls = ts[order], od["t"][order]
        i = np.searchsorted(tss, tr["t_odom_us"])
        i = np.clip(i, 0, len(tss) - 1)
        hit = tss[i] == tr["t_odom_us"]
        if hit.any():
            lag = (tr["t"][hit] - tls[i][hit]) * 1e3
            sess.align = (f"trace↔bag clock check: {hit.sum()}/{hit.size} rows matched, "
                          f"row written {np.median(lag):.1f} ms after its odometry arrived "
                          f"(5–95%: {np.percentile(lag, 5):.1f}…{np.percentile(lag, 95):.1f} ms)")
        else:
            sess.align = "trace↔bag clock check: no odometry sample matched"


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def _read_node_log(sess):
    p = os.path.join(sess.path, "logs", "node.log")
    if os.path.isfile(p):
        try:
            sess.node_log = open(p, errors="replace").read()
        except OSError:
            return
        m = re.search(r"hover thrust [0-9.]+ .*?<-> ([0-9.]+) N", sess.node_log)
        if m:
            sess.hover_n = float(m.group(1))


# ===========================================================================
# Derived signals
# ===========================================================================

def _q_wxyz_to_R(w, x, y, z):
    """Rotation matrices (n,3,3), body -> world, from scalar-first quaternions."""
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.empty((w.size, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


# ENU<->NED and FLU<->FRD are both their own inverse.
M_ENU_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
M_FLU_FRD = np.diag([1.0, -1.0, -1.0])


def _euler_from_R(R):
    """ZYX roll, pitch, yaw (deg) of body->world matrices (PX4 convention)."""
    roll = np.degrees(np.arctan2(R[:, 2, 1], R[:, 2, 2]))
    pitch = np.degrees(np.arcsin(np.clip(-R[:, 2, 0], -1, 1)))
    yaw = np.degrees(np.arctan2(R[:, 1, 0], R[:, 0, 0]))
    return roll, pitch, yaw


def _px4_R(topic):
    q = topic["q"]
    return _q_wxyz_to_R(q[:, 0], q[:, 1], q[:, 2], q[:, 3])


def _trace_R_ned(tr):
    """Trace cmd_q (ENU/FLU, [x,y,z,w]) -> NED/FRD rotation matrices."""
    R_enu = _q_wxyz_to_R(tr["qw"], tr["qx"], tr["qy"], tr["qz"])
    return M_ENU_NED @ R_enu @ M_FLU_FRD


def _angle_between(Ra, Rb):
    """Rotation angle (deg) between two stacks of rotation matrices."""
    tr = np.einsum("nij,nij->n", Ra, Rb)          # trace(Ra^T Rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def _unwrap_deg(y):
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    out = y.copy()
    out[ok] = np.degrees(np.unwrap(np.radians(y[ok])))
    return out


def _nearest(t_src, t_dst):
    i = np.searchsorted(t_src, t_dst)
    i = np.clip(i, 1, len(t_src) - 1)
    left = t_src[i - 1]
    return np.where(np.abs(t_dst - left) <= np.abs(t_src[i] - t_dst), i - 1, i)


def _spans(t, ok):
    """[(t_start, t_end)] runs where ok is True; a run lasts until the sample
    that turns it off (or the last sample)."""
    t = np.asarray(t, float)
    ok = np.asarray(ok, bool)
    if t.size == 0:
        return []
    edges = np.flatnonzero(np.diff(ok.astype(int))) + 1
    starts = np.concatenate([[0], edges])
    out = []
    for i, s0 in enumerate(starts):
        if not ok[s0]:
            continue
        e = starts[i + 1] if i + 1 < len(starts) else len(t) - 1
        if t[e] > t[s0]:
            out.append((float(t[s0]), float(t[e])))
    return out


def _segments(t, labels, max_gap=None):
    """[(t0, t1, label)] runs of equal label; a time gap > max_gap ends a run."""
    out = []
    if len(t) == 0:
        return out
    s, lab = t[0], labels[0]
    for k in range(1, len(t)):
        gap = max_gap is not None and t[k] - t[k - 1] > max_gap
        if labels[k] != lab or gap:
            out.append((s, t[k - 1] if gap else t[k], lab))
            s, lab = t[k], labels[k]
    out.append((s, t[-1], lab))
    return [x for x in out if x[1] > x[0]]


def who_flies(sess):
    """[(t0, t1, kind, label)], kind in {'se3', 'px4offb', 'px4'}, from
    vehicle_control_mode (+ nav_state for the PX4 mode name).

    In offboard with the attitude loop OFF, PX4 is taking thrust + torque
    straight from the companion: SE3 is flying.  In offboard with position
    control ON, the companion is only sending PX4 a position setpoint."""
    cm = sess.topics.get("vehicle_control_mode")
    if cm is None:
        return []
    st = sess.topics.get("vehicle_status")
    labels = []
    for k in range(len(cm["t"])):
        offb = bool(cm.get("flag_control_offboard_enabled", [0])[k])
        att = bool(cm.get("flag_control_attitude_enabled", [1])[k])
        if offb and not att:
            labels.append(("se3", "SE3 (thrust+torque)"))
        elif offb:
            labels.append(("px4offb", "PX4 offboard position"))
        else:
            name = "PX4"
            if st is not None and len(st["t"]):
                j = min(np.searchsorted(st["t"], cm["t"][k]), len(st["t"]) - 1)
                name = "PX4 " + nav_state_name(st["nav_state"][j])
            labels.append(("px4", name))
    segs = _segments(cm["t"], labels)
    return [(a, b, lab[0], lab[1]) for a, b, lab in segs]


def handoffs(sess):
    """[(t, 'PX4→SE3' | 'SE3→PX4')] every change into or out of SE3 control."""
    out, prev = [], None
    for a, _b, kind, _lab in who_flies(sess):
        is_se3 = kind == "se3"
        if prev is not None and is_se3 != prev:
            out.append((a, "PX4→SE3" if is_se3 else "SE3→PX4"))
        prev = is_se3
    return out


# ===========================================================================
# Figure
# ===========================================================================

# (key, title, height units).  The figure stacks them in this order.
PANELS = [
    ("strip", "phase", 1.5),
    ("pos", "1  Position: actual vs SE3 reference", 2.4),
    ("drift", "2  Ground drift (odometry from first sample) + heading", 2.4),
    ("att_rp", "3  Roll / pitch: actual vs SE3 command", 2.2),
    ("att_y", "3  Yaw (unwrapped): actual vs SE3 command", 2.0),
    ("att_err", "3  Attitude error (node's recovery-gate metric)", 1.6),
    ("thrust", "4  Thrust", 1.7),
    ("torque", "4  Torque command (normalised)", 1.8),
    ("motors", "5  Motor outputs", 1.8),
    ("rc", "5  RC", 1.6),
    ("link", "6  Link health: stream rates", 2.2),
    ("ekf", "7  EKF status flags", 4.2),
    ("failsafe", "7  Failsafe flags (ever set)", 1.6),
]

LINK_STREAMS = [
    # (topic, label, color, expected Hz or None)
    ("/fmu/in/offboard_control_mode", "in: offboard heartbeat", "#20222b", 100),
    ("/fmu/in/trajectory_setpoint", "in: trajectory setpoint", "#d81b60", None),
    ("/fmu/in/vehicle_thrust_setpoint", "in: thrust setpoint", "#1e9e5a", None),
    ("/fmu/in/vehicle_torque_setpoint", "in: torque setpoint", "#8a3fb5", None),
    ("/fmu/out/vehicle_odometry", "out: odometry", "#2a78d6", None),
    ("/fmu/out/vehicle_attitude", "out: attitude", "#d2691e", None),
]

EKF_FLAGS = [
    # (field, label, bad-when-True?)
    ("cs_tilt_align", "tilt aligned", False),
    ("cs_yaw_align", "yaw aligned", False),
    ("cs_mag_hdg", "mag heading fused", False),
    ("cs_mag_3d", "mag 3D fused", False),
    ("cs_mag_field_disturbed", "mag field disturbed", True),
    ("cs_mag_heading_consistent", "mag heading consistent", False),
    ("cs_gnss_pos", "GNSS pos fused", False),
    ("cs_gnss_vel", "GNSS vel fused", False),
    ("cs_gnss_yaw", "GNSS yaw fused", False),
    ("cs_gps_hgt", "GNSS height fused", False),
    ("cs_baro_hgt", "baro height fused", False),
    ("cs_rng_hgt", "range height fused", False),
    ("cs_vehicle_at_rest", "at rest", False),
    ("cs_in_air", "in air", False),
    ("cs_inertial_dead_reckoning", "inertial dead reckoning", True),
    ("cs_mag_fault", "mag fault", True),
    ("fs_bad_hdg", "bad heading", True),
]


def _no_data(ax, msg):
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
            color=C_MUTED, fontsize=9)


def _style(ax, title, ylabel=""):
    ax.set_facecolor(C_SURFACE)
    ax.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors=C_INK)
    ax.set_title(title, loc="left", fontsize=9, color=C_INK, pad=3)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)


def _rows(ax, rows, t_end):
    """Horizontal on/off rows.  rows = [(label, [(t0,t1)], color)]."""
    for i, (label, spans, color) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.broken_barh([(a, b - a) for a, b in spans], (y - 0.38, 0.76),
                       facecolors=color, edgecolors="none")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=6.5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(False)
    ax.grid(True, axis="x", color=C_GRID, lw=0.6)


def _shade_se3(ax, flies):
    for a, b, kind, _ in flies:
        if kind == "se3":
            ax.axvspan(a, b, color=C_SE3, alpha=0.07, lw=0, zorder=0)


def _bool_spans(t, v):
    v = np.asarray(v, float)
    return _spans(t, np.nan_to_num(v) > 0.5)


def _draw_strip(ax, sess, flies, t_end):
    rows = []
    tr = sess.trace
    if tr is not None:
        segs = _segments(tr["t"], list(tr["phase"]), max_gap=0.5)
        rows.append(("SE3 step (trace)", segs, None))
    rows.append(("who flies", flies, None))
    st = sess.topics.get("vehicle_status")
    armed = (_spans(st["t"], np.asarray(st["arming_state"], float) == 2)
             if st is not None else [])
    rows.append(("armed", armed, C_BAD))
    ld = sess.topics.get("vehicle_land_detected")
    rows.append(("landed", _bool_spans(ld["t"], ld["landed"]) if ld else [], C_ARMED))

    n = len(rows)
    for i, (label, items, color) in enumerate(rows):
        y = n - 1 - i
        for it in items:
            a, b = it[0], it[1]
            if label == "who flies":
                kind, text = it[2], it[3]
                c = {"se3": C_SE3, "px4offb": C_PX4_OFFB}.get(kind, C_ARMED)
            elif label.startswith("SE3 step"):
                text = it[2]
                c = C_BAD if "RECOVERY" in text else C_SE3
            else:
                text, c = "", color
            ax.broken_barh([(a, b - a)], (y - 0.4, 0.8), facecolors=c,
                           edgecolors="white", lw=0.5, alpha=0.85)
            if text and (b - a) > (0.009 * len(text) + 0.01) * t_end:
                ax.text((a + b) / 2, y, text, ha="center", va="center",
                        fontsize=6.5, color="white", clip_on=True)
    if tr is None:
        ax.text(0.995, 0.97, sess.trace_note, transform=ax.transAxes, ha="right",
                va="top", fontsize=7, color=C_MUTED)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=7)
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(True, axis="x", color=C_GRID, lw=0.6)


def build_dashboard(sess, window=None, panels=None, width_in=14.0):
    """The stacked dashboard as a matplotlib Figure (NOT pyplot: nothing to
    close, no global registry to leak -- see logGraph's figure-leak notes).

    window=(t0, t1) zooms every panel to that span (handoff view).  panels =
    subset of PANELS keys."""
    from matplotlib.figure import Figure

    keys = [p for p in PANELS if panels is None or p[0] in panels]
    total = sum(h for _, _, h in keys)
    unit = 0.62                                  # inches per height unit
    fig = Figure(figsize=(width_in, total * unit + 1.2), facecolor=C_SURFACE)
    top_in, bot_in = 0.95, 0.55
    gs = fig.add_gridspec(len(keys), 1, height_ratios=[h for _, _, h in keys],
                          left=0.115, right=0.90, hspace=0.62,
                          top=1 - top_in / fig.get_figheight(),
                          bottom=bot_in / fig.get_figheight())
    axes, first = {}, None
    for i, (key, title, _h) in enumerate(keys):
        ax = fig.add_subplot(gs[i, 0], sharex=first)
        first = first or ax
        _style(ax, title)
        axes[key] = ax
    # The trace can outlive the bag: a killed recorder loses its last chunk
    # while the node keeps writing (11-53-57: bag 72.7 s, trace to 81 s).
    t_end = max(sess.duration_s, 1e-3)
    tr = sess.trace
    if tr is not None and np.isfinite(tr["t"]).any():
        t_end = max(t_end, float(np.nanmax(tr["t"])))
    flies = who_flies(sess)

    hdr = (f"{sess.name}   ·   {sess.duration_s:.1f} s"
           + ("   ·   recorder killed: read up to the last complete chunk"
              if sess.truncated else ""))
    fig.text(0.115, 1 - 0.30 / fig.get_figheight(), "SE3 session", fontsize=13,
             fontweight="bold", color=C_INK)
    fig.text(0.115, 1 - 0.52 / fig.get_figheight(), hdr, fontsize=8, color=C_MUTED)
    fig.text(0.115, 1 - 0.70 / fig.get_figheight(), sess.align or sess.trace_note,
             fontsize=8, color=C_MUTED)

    bag_end = sess.duration_s if t_end > sess.duration_s + 0.5 else None
    for key, ax in axes.items():
        if key != "strip":
            _shade_se3(ax, flies)
        if bag_end is not None:
            ax.axvspan(bag_end, t_end, color=C_MUTED, alpha=0.10, lw=0, zorder=0)
            ax.axvline(bag_end, color=C_MUTED, lw=0.9, ls="--")
        try:
            DRAW[key](ax, sess, flies, t_end)
        except Exception as e:                    # one bad panel must not blank the rest
            _no_data(ax, f"could not draw: {type(e).__name__}: {e}")
    if bag_end is not None:
        a0 = axes[keys[0][0]]
        a0.text(bag_end, 1.02, "bag ends (recorder killed) — trace only after this ▸",
                transform=a0.get_xaxis_transform(), ha="right", va="bottom",
                fontsize=7, color=C_MUTED)

    last = axes[keys[-1][0]]
    last.set_xlabel("time from bag start (s)   ·   orange shading = SE3 flying "
                    "(offboard thrust + torque)", fontsize=8)
    for ax in axes.values():
        if ax is not last:
            ax.tick_params(labelbottom=False)
    if window is not None:
        first.set_xlim(*window)
        for key, ax in axes.items():
            if key not in ("strip", "ekf", "failsafe", "rc"):
                _rescale_y(ax, window)
        for t, lab in handoffs(sess):
            if window[0] <= t <= window[1]:
                for ax in axes.values():
                    ax.axvline(t, color=C_INK, lw=0.9, ls=":", zorder=5)
                axes[keys[0][0]].text(t, 1.02, lab, transform=axes[keys[0][0]].get_xaxis_transform(),
                                      ha="center", va="bottom", fontsize=8, color=C_INK)
    else:
        first.set_xlim(0, t_end)
    fig._page_height = int(fig.get_figheight() * fig.dpi)
    return fig


def _rescale_y(ax, window):
    lo, hi = np.inf, -np.inf
    for ln in ax.get_lines():
        x, y = ln.get_xdata(), ln.get_ydata()
        try:
            x = np.asarray(x, float)
            y = np.asarray(y, float)
        except (TypeError, ValueError):
            continue
        if x.size != y.size or x.size < 2:
            continue                                  # axhline etc.
        m = (x >= window[0]) & (x <= window[1]) & np.isfinite(y)
        if m.any():
            lo, hi = min(lo, y[m].min()), max(hi, y[m].max())
    if np.isfinite(lo):
        pad = max((hi - lo) * 0.08, 1e-3)
        ax.set_ylim(lo - pad, hi + pad)


# --- panels ---------------------------------------------------------------

def _p_strip(ax, sess, flies, t_end):
    _draw_strip(ax, sess, flies, t_end)


def _p_pos(ax, sess, flies, t_end):
    tr = sess.trace
    if tr is None:
        _no_data(ax, sess.trace_note)
        return
    t = tr["t"]
    tb = _break(t, 0.5)
    for i, ax_name in enumerate("ENU"):
        c = AXIS_COL[ax_name]
        ax.plot(t, _gap(t, tr[f"pos_r{'xyz'[i]}"], 0.5), color=c, lw=1.3, label=f"{ax_name} actual")
        ax.plot(t, _gap(t, tr[f"ref_r{'xyz'[i]}"], 0.5), color=c, lw=1.1, ls="--",
                label=f"{ax_name} reference")
    err = np.sqrt(sum((tr[f"pos_r{a}"] - tr[f"ref_r{a}"]) ** 2 for a in "xyz"))
    tw = ax.twinx()
    tw.plot(t, _gap(t, err, 0.5), color=C_INK, lw=0.9, alpha=0.6, label="|error|")
    tw.set_ylabel("|error| (m)", fontsize=8)
    tw.tick_params(labelsize=8)
    tw.spines["top"].set_visible(False)
    ax.set_ylabel("m from trace anchor\n(ENU)", fontsize=8)
    _legend(ax, ncol=6, extra=tw)
    del tb


def _p_drift(ax, sess, flies, t_end):
    od = sess.topics.get("vehicle_odometry")
    if od is None or "position" not in od:
        _no_data(ax, "no vehicle_odometry in this bag")
        return
    t, p = od["t"], od["position"]
    ok = np.all(np.isfinite(p), axis=1)
    t, p = t[ok], p[ok]
    p = p - p[0]
    east, north, up = p[:, 1], p[:, 0], -p[:, 2]
    for name, y in (("E", east), ("N", north), ("U", up)):
        ax.plot(t, y, color=AXIS_COL[name], lw=1.2, label=f"{name} (m)")
    R = _px4_R({"q": od["q"][ok]})
    yaw = _unwrap_deg(_euler_from_R(R)[2])
    tw = ax.twinx()
    tw.plot(t, yaw, color=C_INK, lw=0.9, ls="-", alpha=0.55, label="heading (deg)")
    tw.set_ylabel("heading (deg, unwrapped)", fontsize=8)
    tw.tick_params(labelsize=8)
    tw.spines["top"].set_visible(False)
    ax.set_ylabel("m from first\nsample", fontsize=8)
    # Drift rate while landed AND disarmed: the vehicle cannot move, so any
    # motion is the estimate.
    ld = sess.topics.get("vehicle_land_detected")
    st = sess.topics.get("vehicle_status")
    if ld is not None and st is not None:
        landed = np.nan_to_num(np.asarray(ld["landed"], float))[_nearest(ld["t"], t)] > 0.5
        disarmed = np.asarray(st["arming_state"], float)[_nearest(st["t"], t)] != 2
        still = landed & disarmed
        if still.sum() > 20 and t[still][-1] - t[still][0] > 5:
            ts = t[still]
            dur = ts[-1] - ts[0]
            horiz = np.hypot(np.ptp(east[still]), np.ptp(north[still]))
            dyaw = np.ptp(yaw[still])
            ax.text(0.005, 0.03, f"landed + disarmed {dur:.0f} s: horizontal spread "
                    f"{horiz:.2f} m, vertical {np.ptp(up[still]):.2f} m, heading span "
                    f"{dyaw:.1f}°", transform=ax.transAxes, fontsize=7.5,
                    va="bottom", color=C_BAD if horiz > 0.3 or dyaw > 10 else C_MUTED,
                    bbox=dict(boxstyle="round,pad=0.25", fc=C_SURFACE, ec="none", alpha=0.9),
                    zorder=6)
    _legend(ax, ncol=4, extra=tw)


def _att(sess):
    """(t_act, R_act NED) from vehicle_attitude, else odometry."""
    for name in ("vehicle_attitude", "vehicle_odometry"):
        d = sess.topics.get(name)
        if d is not None and "q" in d and len(d["t"]):
            ok = np.all(np.isfinite(d["q"]), axis=1)
            return d["t"][ok], _px4_R({"q": d["q"][ok]}), name
    return None


def _p_att_rp(ax, sess, flies, t_end):
    a = _att(sess)
    if a is None:
        _no_data(ax, "no attitude in this bag")
        return
    t, R, src = a
    r, p, _y = _euler_from_R(R)
    ax.plot(t, r, color=AXIS_COL["x"], lw=1.2, label="roll actual")
    ax.plot(t, p, color=AXIS_COL["y"], lw=1.2, label="pitch actual")
    tr = sess.trace
    if tr is not None:
        rc, pc, _ = _euler_from_R(_trace_R_ned(tr))
        ax.plot(tr["t"], _gap(tr["t"], rc, 0.5), color=AXIS_COL["x"], ls="--", lw=1.1,
                label="roll SE3 cmd")
        ax.plot(tr["t"], _gap(tr["t"], pc, 0.5), color=AXIS_COL["y"], ls="--", lw=1.1,
                label="pitch SE3 cmd")
    else:
        ax.text(0.995, 0.95, "SE3 command: " + sess.trace_note, transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color=C_MUTED)
    ax.set_ylabel("deg", fontsize=8)
    _legend(ax, ncol=4)


def _p_att_y(ax, sess, flies, t_end):
    a = _att(sess)
    if a is None:
        _no_data(ax, "no attitude in this bag")
        return
    t, R, src = a
    yaw = _unwrap_deg(_euler_from_R(R)[2])
    ax.plot(t, yaw, color=C_ACT, lw=1.2, label="yaw actual")
    tr = sess.trace
    if tr is not None:
        yc = _euler_from_R(_trace_R_ned(tr))[2]
        # Unwrap the command onto the same branch as the actual at its first tick.
        yc = _unwrap_deg(yc)
        j = _nearest(t, tr["t"][:1])[0]
        yc = yc + 360.0 * np.round((yaw[j] - yc[0]) / 360.0)
        ax.plot(tr["t"], _gap(tr["t"], yc, 0.5), color=C_CMD, ls="--", lw=1.1,
                label="yaw SE3 cmd")
    ax.set_ylabel("deg (from north,\nclockwise)", fontsize=8)
    _legend(ax, ncol=2)


def _p_att_err(ax, sess, flies, t_end):
    tr = sess.trace
    if tr is None:
        _no_data(ax, sess.trace_note)
        return
    od = sess.topics.get("vehicle_odometry")
    if od is None:
        _no_data(ax, "no vehicle_odometry in this bag")
        return
    # The node compared cmd_q with the odometry sample of the same tick (t_odom_us).
    ts = od["timestamp"]
    order = np.argsort(ts)
    i = np.clip(np.searchsorted(ts[order], tr["t_odom_us"]), 0, len(ts) - 1)
    j = order[i]
    R_act = _px4_R({"q": od["q"][j]})
    err = _angle_between(R_act, _trace_R_ned(tr))
    ax.plot(tr["t"], _gap(tr["t"], err, 0.5), color=C_INK, lw=1.2, label="attitude error")
    ax.axhline(ATT_GATE_DEG, color=C_BAD, lw=1.0, ls="--")
    ax.text(0.005, ATT_GATE_DEG, f" {ATT_GATE_DEG:.0f}° recovery gate (0.3 s)",
            transform=ax.get_yaxis_transform(), va="bottom", fontsize=7, color=C_BAD)
    over = err > ATT_GATE_DEG
    if over.any():
        k = int(np.flatnonzero(over)[0])
        ax.annotate(f"first > gate: {err[k]:.0f}° at {tr['t'][k]:.1f} s",
                    (tr["t"][k], err[k]), xytext=(-8, -2), textcoords="offset points",
                    ha="right", va="top", fontsize=7.5, color=C_BAD)
    ax.set_ylim(0, max(185.0, np.nanmax(err) * 1.05 if np.isfinite(err).any() else 185))
    ax.set_ylabel("deg", fontsize=8)


def _p_thrust(ax, sess, flies, t_end):
    tr = sess.trace
    if tr is None:
        _no_data(ax, sess.trace_note)
        return
    ax.plot(tr["t"], _gap(tr["t"], tr["thrust_n"], 0.5), color=C_CMD, lw=1.3,
            label="SE3 thrust command")
    ax.axhline(sess.hover_n, color=C_HOVER, lw=1.0, ls="--")
    ax.text(0.005, sess.hover_n, f" hover {sess.hover_n:.2f} N (m·g)",
            transform=ax.get_yaxis_transform(), va="bottom", fontsize=7, color=C_HOVER)
    ax.axhline(0, color=C_MUTED, lw=0.8)
    neg = np.asarray(tr["thrust_n"], float) < 0
    if neg.any():
        k = int(np.flatnonzero(neg)[0])
        ax.annotate("thrust < 0: SE3 asked for a DOWNWARD force\n(Lee law -> inverted attitude)",
                    (tr["t"][k], 0), xytext=(-6, -4), textcoords="offset points",
                    ha="right", va="top", fontsize=7.5, color=C_BAD)
    ax.set_ylabel("N", fontsize=8)


def _p_torque(ax, sess, flies, t_end):
    tr = sess.trace
    if tr is None:
        _no_data(ax, sess.trace_note)
        return
    for k in "xyz":
        ax.plot(tr["t"], _gap(tr["t"], tr[f"tau_{k}"], 0.5), color=AXIS_COL[k], lw=1.1,
                label=f"τ{k} (FRD)")
    for s in (1, -1):
        ax.axhspan(s * 0.95, s * 1.1, color=C_BAD, alpha=0.10, lw=0)
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("−1…1\n(±1 = saturated)", fontsize=8)
    _legend(ax, ncol=3)


def _p_motors(ax, sess, flies, t_end):
    ao = sess.topics.get("actuator_outputs")
    if ao is None or "output" not in ao:
        _no_data(ax, "no actuator_outputs in this bag")
        return
    n = int(np.nanmax(ao["noutputs"])) if "noutputs" in ao else 4
    n = max(1, min(n, 8))
    out = ao["output"]
    # Unused outputs sit at one value the whole session; keep the first 4 (a
    # quad's motors) and any other output that actually moved.
    show = [i for i in range(n) if i < 4 or np.nanmax(out[:, i]) - np.nanmin(out[:, i]) > 0]
    for i in show:
        ax.plot(ao["t"], ao["output"][:, i], lw=1.0, color=MOTOR_COL[i % 4],
                label=f"output {i + 1}")
    ax.set_ylabel("raw output\n(PWM/DShot units)", fontsize=8)
    if np.nanmax(np.abs(out[:, :n])) == 0:
        ax.text(0.5, 0.75, "all outputs 0 the whole session (disarmed)",
                transform=ax.transAxes, ha="center", fontsize=8, color=C_MUTED)
    _legend(ax, ncol=len(show))


def _p_rc(ax, sess, flies, t_end):
    rc = sess.topics.get("input_rc")
    if rc is None or "values" not in rc:
        _no_data(ax, "no input_rc in this bag")
        return
    v = rc["values"].astype(float)
    nch = int(np.nanmax(rc.get("channel_count", [v.shape[1]])))
    cols = ["#2a78d6", "#d2691e", "#1e9e5a", "#8a3fb5", "#20222b", "#d81b60",
            "#00838f", "#8c8c85"]
    for i in range(min(nch, 8)):
        if np.nanmax(v[:, i]) - np.nanmin(v[:, i]) > 30 or i < 4:
            ax.plot(rc["t"], v[:, i], lw=0.9, color=cols[i], label=f"ch{i + 1}")
    if "rc_lost" in rc:
        for a, b in _bool_spans(rc["t"], rc["rc_lost"]):
            ax.axvspan(a, b, color=C_BAD, alpha=0.2, lw=0)
    ax.set_ylabel("µs", fontsize=8)
    _legend(ax, ncol=8)


def _rate(times, t_end, bin_s=0.5):
    edges = np.arange(0, t_end + bin_s, bin_s)
    if edges.size < 2:
        return None, None
    h, _ = np.histogram(times, edges)
    return 0.5 * (edges[:-1] + edges[1:]), h / bin_s


def _p_link(ax, sess, flies, t_end):
    any_ = False
    for topic, label, color, _exp in LINK_STREAMS:
        ts = sess.rates.get(topic)
        if ts is None or ts.size == 0:
            continue
        c, r = _rate(ts, t_end)
        if c is None:
            continue
        ax.plot(c, r, color=color, lw=1.0, label=label, drawstyle="steps-mid")
        any_ = True
    if not any_:
        _no_data(ax, "no /fmu streams in this bag")
        return
    ts = sess.topics.get("timesync_status")
    if ts is not None and "round_trip_time" in ts:
        tw = ax.twinx()
        tw.plot(ts["t"], np.asarray(ts["round_trip_time"], float) / 1000.0,
                color=C_MUTED, lw=0.9, ls=":", label="timesync RTT (ms)")
        tw.set_ylabel("RTT (ms)", fontsize=8)
        tw.tick_params(labelsize=8)
        tw.spines["top"].set_visible(False)
        _legend(ax, ncol=4, extra=tw)
    else:
        _legend(ax, ncol=4)
    ax.set_ylabel("Hz (0.5 s bins)", fontsize=8)
    ax.set_ylim(bottom=0)


def _p_ekf(ax, sess, flies, t_end):
    ef = sess.topics.get("estimator_status_flags")
    if ef is None:
        _no_data(ax, "no estimator_status_flags in this bag")
        return
    rows = []
    for f, label, bad in EKF_FLAGS:
        if f in ef:
            rows.append((label + (" ⚠" if bad else ""), _bool_spans(ef["t"], ef[f]),
                         C_BAD if bad else "#2a78d6"))
    # Any fault-status (fs_*) flag that was ever set and isn't listed already.
    listed = {f for f, _, _ in EKF_FLAGS}
    for f in sorted(ef):
        if f.startswith("fs_") and f not in listed and np.nansum(np.asarray(ef[f], float)) > 0:
            rows.append((f + " ⚠", _bool_spans(ef["t"], ef[f]), C_BAD))
    _rows(ax, rows, t_end)


def _p_failsafe(ax, sess, flies, t_end):
    ff = sess.topics.get("failsafe_flags")
    if ff is None:
        _no_data(ax, "no failsafe_flags in this bag")
        return
    rows = []
    for f in sorted(ff):
        if f == "t" or f.startswith("mode_req"):
            continue
        v = np.asarray(ff[f], float)
        if np.nansum(v) > 0:
            rows.append((f, _bool_spans(ff["t"], v), C_BAD))
    if not rows:
        _no_data(ax, "no failsafe flag was set in this bag")
        return
    _rows(ax, rows, t_end)


DRAW = {"strip": _p_strip, "pos": _p_pos, "drift": _p_drift,
        "att_rp": _p_att_rp, "att_y": _p_att_y, "att_err": _p_att_err,
        "thrust": _p_thrust, "torque": _p_torque, "motors": _p_motors,
        "rc": _p_rc, "link": _p_link, "ekf": _p_ekf, "failsafe": _p_failsafe}


# --- small helpers ----------------------------------------------------------

def _gap(t, y, max_gap):
    """y with NaN at the first sample after a time gap > max_gap, so a line is
    not drawn across a stretch with no trace rows (PX4 was flying then)."""
    y = np.asarray(y, float).copy()
    if y.size > 1:
        y[1:][np.diff(np.asarray(t, float)) > max_gap] = np.nan
    return y


def _break(t, max_gap):
    return np.flatnonzero(np.diff(np.asarray(t, float)) > max_gap)


def _legend(ax, ncol=4, extra=None):
    h, l = ax.get_legend_handles_labels()
    if extra is not None:
        h2, l2 = extra.get_legend_handles_labels()
        h, l = h + h2, l + l2
    if h:
        ax.legend(h, l, loc="upper right", fontsize=7, ncol=ncol, frameon=True,
                  framealpha=0.85, borderpad=0.3, handlelength=1.8)
