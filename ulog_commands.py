#!/usr/bin/env python3
"""ulog_commands.py -- what the vehicle was TOLD to do, and how fast it went.

Two figures, placed right after the flight path because they answer its two
obvious follow-up questions -- "what was the controller asking for along that
track" and "how fast was it moving":

  Commands  -- four stacked panels on one time axis:
    1. throttle  -- collective thrust command, 0..1 (optionally per-motor output)
    2. roll      -- commanded vs actual angle, degrees
    3. pitch     -- commanded vs actual angle, degrees
    4. yaw       -- commanded vs actual heading, degrees
    Each attitude panel also carries, OFF by default on a right-hand axis, the
    normalised torque command the mixer actually received on that axis.

  Speed     -- two panels:
    1. speed magnitude, no direction
    2. velocity in the HEADING frame -- forward / right / up (see below)

Which "command"
---------------
PX4 (the flight stack) has a command at every layer of its cascade, and they are
not interchangeable:

    position -> velocity -> ATTITUDE (q_d) -> rates -> TORQUE + THRUST -> motors

The angle panels plot the ATTITUDE setpoint because it has the same units as the
thing it commands, so the dashed and solid lines are directly comparable and the
gap between them IS the tracking error.  The torque command is one layer lower --
normalised -1..1 and not an angle -- so it lives on its own axis, for the
question the angle cannot answer: was the controller pushing as hard as it could
(saturating) while the angle still lagged?

Angles come from the quaternions (`q_d`, `q`), not from
`vehicle_attitude_setpoint.roll_body/pitch_body/yaw_body`: those Euler fields
were dropped from newer PX4 firmware (the HITL board logs carry only `q_d`),
and reading a field that is present in one firmware and absent in another is
exactly how a panel goes silently blank.

Throttle is the collective thrust command, `-vehicle_thrust_setpoint.xyz[2]`.
The x and y components are not plotted: a multirotor can only push along its own
body z axis, so body-frame x/y thrust is identically zero (checked on
landing_test_with_new_cover_4: 269 k samples, all 0.0).  It moves sideways by
TILTING that push, which is what the roll and pitch panels show.

The heading frame
-----------------
Panel 2 of the speed figure resolves velocity along the vehicle's own heading:

    forward  -- horizontal, along the nose     (what PITCHING produces)
    right    -- horizontal, 90 deg clockwise   (what ROLLING produces)
    up       -- world vertical

It follows the heading (yaw) but NOT the tilt.  The body x/y axes cannot be kept
while z points at world up -- once the vehicle tilts, body x is no longer
perpendicular to world up -- and projecting onto the tilted axes instead would
leak a climb into "forward" (20 deg of pitch puts 34% of the climb rate there).
Levelling the frame keeps the three components independent.  It is also the
frame PX4's position-mode sticks command in.

Signs follow PX4's FRD body frame, so each speed shares its sign with the angle
that causes it: nose down (negative pitch) gives positive forward speed, a
positive roll (right wing down) gives positive right speed.  NB the opposite of
rotorpy's {V} frame, which is X forward, Y LEFT, Z up.

Acronyms: NED = North-East-Down, FRD = Forward-Right-Down (the body frame),
GPS = Global Positioning System, HITL = Hardware In The Loop.
"""
import os
import warnings

import numpy as np

from ulog_common import (C_GRID, C_INK, C_MUTED, C_SURFACE, INST_COLORS,
                         PlotCtx, Series, _clean, _get, _rescale, _style_axis,
                         _time_min, add_mouse_navigation, armed_spans,
                         check_panel, draw_armed, draw_mode_changes,
                         duration_min, field, has_topic, mode_changes,
                         mode_key, nav_hint, resample_to,
                         style_time_axis)

COMMAND_TOPICS = [
    "vehicle_thrust_setpoint",      # collective throttle
    "vehicle_torque_setpoint",      # what the mixer was asked for, per axis
    "vehicle_attitude_setpoint",    # q_d, and the throttle fallback
    "vehicle_attitude",             # q, the actual attitude
    "actuator_motors",              # per-motor output, 0..1
    "actuator_armed",
    "vehicle_status",               # flight-mode overlay
]

SPEED_TOPICS = [
    "vehicle_local_position",           # the estimated velocity
    "vehicle_local_position_setpoint",  # the commanded velocity
    "vehicle_gps_position",             # the receiver's own speed
    "actuator_armed",
    "vehicle_status",
]

# --- colour -----------------------------------------------------------------
# The same two roles as the flight path: magenta is COMMANDED, near-black is what
# the vehicle actually DID.  Keeping that mapping lets the eye carry "magenta =
# asked for" from the 3D plot straight into these panels.
C_CMD = "#d81b60"       # magenta -- commanded (ulog_path.C_SETPOINT)
C_ACT = "#20222b"       # near-black -- actual / estimated
C_TORQUE = "#2a78d6"    # blue -- the lower-layer torque command, right axis
C_GROUND = "#1baf7a"    # aqua -- horizontal speed, a component of the total
C_GPS = "#8d6e63"       # brown -- raw receiver (ulog_path.C_GPS)
# Heading-frame components.  One hue per AXIS, actual solid and commanded dashed
# in the same hue, so a component and its command read as one pair.
C_FWD = "#2a78d6"       # blue
C_RIGHT = "#d2691e"     # orange
C_UP = "#4a3aa7"        # violet

AXES = (("roll", 0), ("pitch", 1), ("yaw", 2))


# --- attitude maths -----------------------------------------------------------

def _euler_deg(ulog, topic, prefix):
    """(t_min, roll, pitch, yaw) in degrees from a quaternion field.

    PX4 stores quaternions Hamilton [w, x, y, z] (q[0] is the SCALAR part --
    note this is the opposite order to scipy's [x, y, z, w]), rotating FRD body
    into NED world.  The angles are the standard ZYX (yaw-pitch-roll) Tait-Bryan
    set, the same convention PX4's own Eulerf uses, so they match what QGC
    (QGroundControl) shows.
    """
    d = _get(ulog, topic)
    if d is None or f"{prefix}[0]" not in d.data:
        return None
    t = _time_min(ulog, d)
    w, x, y, z = (np.asarray(d.data[f"{prefix}[{i}]"], float) for i in range(4))
    ok = np.isfinite(t) & np.isfinite(w) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    t, w, x, y, z = t[ok], w[ok], x[ok], y[ok], z[ok]
    if t.size == 0:
        return None
    order = np.argsort(t, kind="stable")
    t, w, x, y, z = t[order], w[order], x[order], y[order], z[order]
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    # clip: rounding can push the argument a hair past +-1 and arcsin returns NaN.
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return t, roll, pitch, _break_wraps(yaw)


def _break_wraps(yaw_deg):
    """Blank the sample where heading wraps across +-180.

    Heading is drawn wrapped to +-180 rather than unwrapped: an unwrapped yaw
    walks off to +-1000 degrees over a flight with a few turns, and "1260
    degrees" is not a heading anyone can read.  But a wrapped series joined
    point-to-point draws a full-height vertical line at every wrap -- a 360
    degree spin the vehicle never made -- so the line is broken there instead.
    """
    y = np.asarray(yaw_deg, float).copy()
    if y.size > 1:
        y[1:][np.abs(np.diff(y)) > 180.0] = np.nan
    return y


def _throttle(ulog):
    """(t_min, collective throttle 0..1, source label) or None.

    vehicle_thrust_setpoint is what the allocator consumed; the attitude
    setpoint's thrust_body is the fallback for firmware/logs without it.  Both
    are FRD, so the upward push is -z.
    """
    for topic, name in (("vehicle_thrust_setpoint", "xyz[2]"),
                        ("vehicle_attitude_setpoint", "thrust_body[2]")):
        t, z = field(ulog, topic, name)
        if z.size:
            return t, -z, f"{topic}.{name}"
    return None


# --- series ---------------------------------------------------------------------

def _command_series(ulog, ctx):
    series = []

    thr = _throttle(ulog)
    if thr is not None:
        t, y, src = thr
        series.append(Series(src, "throttle command", t, y, "thr", C_CMD,
                             lw=1.6, visible=True))
    else:
        ctx.note("no vehicle_thrust_setpoint or attitude-setpoint thrust -- "
                 "no throttle command to plot")

    # Per-motor output: OFF by default, but it is the fastest way to see a motor
    # pinned at 1.0 -- saturation that the collective command averages away.
    d = _get(ulog, "actuator_motors")
    if d is not None:
        tm = _time_min(ulog, d)
        for i in range(12):
            key = f"control[{i}]"
            if key not in d.data:
                continue
            t, y = _clean(tm, d.data[key])
            if y.size:          # unused channels are all-NaN and drop out here
                series.append(Series(f"actuator_motors.{key}", f"motor {i + 1} output",
                                     t, y, "thr", INST_COLORS[i % len(INST_COLORS)],
                                     lw=1.0, alpha=0.8, visible=False))

    cmd = _euler_deg(ulog, "vehicle_attitude_setpoint", "q_d")
    act = _euler_deg(ulog, "vehicle_attitude", "q")
    if cmd is None:
        ctx.note("no vehicle_attitude_setpoint.q_d -- no attitude command to plot")
    if act is None:
        ctx.note("no vehicle_attitude.q -- no actual attitude to compare against")
    for name, k in AXES:
        # Actual first so the dashed command draws on top of it: where they
        # coincide you still see the command, and the gap is the error.
        if act is not None:
            series.append(Series(f"vehicle_attitude.{name}", f"{name} actual",
                                 act[0], act[k + 1], name, C_ACT, lw=1.4,
                                 visible=True, zorder=3))
        if cmd is not None:
            series.append(Series(f"vehicle_attitude_setpoint.{name}",
                                 f"{name} command", cmd[0], cmd[k + 1], name,
                                 C_CMD, ls="--", lw=1.4, visible=True, zorder=4))
        t, y = field(ulog, "vehicle_torque_setpoint", f"xyz[{k}]")
        if y.size:
            series.append(Series(f"vehicle_torque_setpoint.xyz[{k}]",
                                 f"{name} torque cmd", t, y, f"{name}_tq",
                                 C_TORQUE, lw=1.0, alpha=0.8, visible=False))
    return series


def _local_velocity(ulog, topic):
    """(t_min, vN, vE, vD) from a local-position topic, or None.

    Samples flagged invalid become NaN -- an invalid velocity is not a speed, so
    it must draw as a break, not as a value.  (The setpoint topic has no validity
    flags; its NaNs are the modes without a velocity loop.)
    """
    d = _get(ulog, topic)
    if d is None or "vx" not in d.data:
        return None
    t = _time_min(ulog, d)
    vn, ve, vd = (np.asarray(d.data[k], float).copy() for k in ("vx", "vy", "vz"))
    for flag, comps in (("v_xy_valid", (vn, ve)), ("v_z_valid", (vd,))):
        if flag in d.data:
            bad = np.asarray(d.data[flag], float) < 0.5
            for c in comps:
                c[bad] = np.nan
    return t, vn, ve, vd


def _yaw_rad(ulog):
    """(t_min, yaw) from vehicle_attitude, UNWRAPPED, in radians.  None if absent.

    Unwrapped because it is about to be interpolated onto the velocity
    timestamps: interpolating straight across a +-pi wrap averages +179 deg and
    -179 deg into 0 -- pointing the frame backwards for one sample and flipping
    the sign of forward and right in a spike that looks like a real manoeuvre.
    """
    d = _get(ulog, "vehicle_attitude")
    if d is None or "q[0]" not in d.data:
        return None
    t = _time_min(ulog, d)
    w, x, y, z = (np.asarray(d.data[f"q[{i}]"], float) for i in range(4))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    t, yaw = _clean(t, yaw)
    if yaw.size < 2:
        return None
    return t, np.unwrap(yaw)


def _to_heading_frame(vn, ve, vd, yaw):
    """NED velocity -> (forward, right, up) in the levelled heading frame.

    A rotation about world z by -yaw, then down flipped to up:

        forward =  vN cos(yaw) + vE sin(yaw)   -- component along the nose
        right   = -vN sin(yaw) + vE cos(yaw)   -- component along the right wing
        up      = -vD                          -- untouched by heading

    Check: yaw = 0 (nose north) gives forward = vN, right = vE -- NED's own x/y.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return vn * c + ve * s, -vn * s + ve * c, -np.asarray(vd, float)


def _heading_series(ulog, ctx):
    """Panel 2: forward / right / up, actual solid and commanded dashed.

    BOTH are rotated by the ACTUAL heading.  Rotating the command by the
    commanded heading instead would put any yaw-tracking error into the gap
    between the lines, and the gap should be velocity error alone.
    """
    yaw = _yaw_rad(ulog)
    if yaw is None:
        ctx.note("no vehicle_attitude.q -- no heading, so no heading-frame velocity")
        return []
    series = []
    for topic, tag, ls, lw, vis in (
            ("vehicle_local_position", "", "-", 1.5, True),
            ("vehicle_local_position_setpoint", " command", "--", 1.3, True)):
        v = _local_velocity(ulog, topic)
        if v is None:
            continue
        t, vn, ve, vd = v
        fwd, right, up = _to_heading_frame(vn, ve, vd,
                                           resample_to(t, yaw[0], yaw[1]))
        for key, label, y, col in (("forward", "forward", fwd, C_FWD),
                                   ("right", "right", right, C_RIGHT),
                                   ("up", "up", up, C_UP)):
            if not np.isfinite(y).any():
                continue
            # _clean drops NaN samples, which is right for the actual velocity
            # (brief invalid flags) but would BRIDGE the setpoint's long NaN
            # stretches -- so the command keeps its NaNs and draws its gaps.
            tt, yy = (_clean(t, y) if not tag else (t, y))
            series.append(Series(f"{topic}.{key}", label + tag, tt, yy, "hdg",
                                 col, ls=ls, lw=lw, visible=vis,
                                 zorder=4 if tag else 3))
    return series


def _speed_series(ulog, ctx):
    series = []

    v = _local_velocity(ulog, "vehicle_local_position")
    if v is not None:
        t, vx, vy, vz = v
        t3, s3 = _clean(t, np.sqrt(vx ** 2 + vy ** 2 + vz ** 2))
        tg, sg = _clean(t, np.hypot(vx, vy))
        if s3.size:
            series.append(Series("vehicle_local_position.|v|", "speed (3D)",
                                 t3, s3, "spd", C_ACT, lw=1.8, visible=True,
                                 zorder=4))
        if sg.size:
            series.append(Series("vehicle_local_position.|vxy|",
                                 "ground speed (horizontal)", tg, sg, "spd",
                                 C_GROUND, lw=1.3, visible=True, zorder=3))
    else:
        ctx.note("no vehicle_local_position velocity -- no estimated speed")

    # Commanded speed.  Only finite in the modes that run a velocity loop
    # (27% of landing_test_with_new_cover_4); the NaN stretches are the manual
    # and attitude-controlled parts and are left as gaps, not bridged.
    d = _get(ulog, "vehicle_local_position_setpoint")
    if d is not None and "vx" in d.data:
        t = _time_min(ulog, d)
        vx, vy, vz = (np.asarray(d.data[k], float) for k in ("vx", "vy", "vz"))
        s = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        if np.isfinite(s).any():
            series.append(Series("vehicle_local_position_setpoint.|v|",
                                 "commanded speed (3D)", t, s, "spd", C_CMD,
                                 ls="--", lw=1.4, visible=True, zorder=5))

    # The receiver's own Doppler speed: independent of the EKF (Extended Kalman
    # Filter), so it is the witness when the estimate looks wrong.  Off by
    # default -- it duplicates the estimate on a healthy log.
    t, y = field(ulog, "vehicle_gps_position", "vel_m_s")
    if y.size:
        series.append(Series("vehicle_gps_position.vel_m_s", "GPS speed (raw)",
                             t, y, "spd", C_GPS, lw=1.1, visible=False))
    return series


# --- shared figure plumbing -------------------------------------------------

def _break_gaps(t, y, factor=8.0):
    """Insert a NaN wherever the topic stopped publishing for a while.

    matplotlib joins two samples however far apart they are, so a logging hole
    is drawn as a straight line -- a throttle that holds perfectly flat, a yaw
    that ramps smoothly -- that no sample ever recorded.  Measured on a HITL
    FC_log.ulg: a 13.5 min hole bridged into a 13.5 min "steady hover".

    The threshold is relative to the topic's own median spacing because rates
    differ by 100x between topics here (attitude ~200 Hz, GPS speed ~10 Hz);
    the 1 s floor keeps a jittery fast topic from shattering into dots.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if t.size < 3:
        return t, y
    dt = np.diff(t)
    pos = dt[dt > 0]
    if pos.size == 0:
        return t, y
    gap = np.flatnonzero(dt > max(float(np.median(pos)) * factor, 1.0 / 60.0))
    if gap.size == 0:
        return t, y
    return (np.insert(t, gap + 1, 0.5 * (t[gap] + t[gap + 1])),
            np.insert(y, gap + 1, np.nan))


def _plot_series(series, axis_of):
    for s in series:
        s.t, s.y = _break_gaps(s.t, s.y)
        (line,) = axis_of[s.group].plot(
            s.t, s.y, color=s.color, ls=s.ls, lw=s.lw, label=s.label,
            drawstyle=s.drawstyle, alpha=s.alpha,
            zorder=s.zorder if s.zorder is not None else 3)
        line.set_visible(s.visible)
        s.line = line


def _no_data(ax, msg):
    """Say the log lacks it -- an empty gridded panel reads as 'all zero'."""
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
            color=C_MUTED, fontsize=9)


def _header(fig, left, title, ulog, path, detail):
    fig.text(left, 0.955 if detail else 0.94, title, color=C_INK, fontsize=13,
             fontweight="bold", ha="left")
    who = f"{os.path.basename(path)}   |   " if path else ""
    fig.text(left, 0.925 if detail else 0.895,
             f"{who}{duration_min(ulog):.1f} min   |   {detail}",
             color=C_MUTED, fontsize=9, ha="left")


# --- the figures --------------------------------------------------------------

def build_commands(ulog, ctx=None, path=""):
    """Throttle / roll / pitch / yaw commands.  Same signature as every builder."""
    import matplotlib.pyplot as plt

    ctx = ctx or PlotCtx()
    if not any(has_topic(ulog, t) for t in COMMAND_TOPICS[:5]):
        ctx.note("no thrust, torque or attitude topics in this log -- nothing to plot")
        return None
    series = _command_series(ulog, ctx)
    if not series:
        return None

    fig = plt.figure(figsize=(15, 11), facecolor=C_SURFACE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(
            f"logGraph commands - {os.path.basename(path)}")

    # Same left/width as the other stacked plots, so the time axes line up when
    # the browser shows them one above the other.
    left, width = 0.260, 0.655
    rects = [(0.720, 0.165), (0.515, 0.165), (0.310, 0.165), (0.105, 0.165)]
    ax_thr, ax_roll, ax_pitch, ax_yaw = [
        fig.add_axes([left, b, width, h], facecolor=C_SURFACE) for b, h in rects]
    for a in (ax_thr, ax_roll, ax_pitch):
        a.sharex(ax_yaw)
    main = [ax_thr, ax_roll, ax_pitch, ax_yaw]
    twins = {}
    for name, a in (("roll", ax_roll), ("pitch", ax_pitch), ("yaw", ax_yaw)):
        tw = a.twinx()
        tw.set_facecolor("none")
        twins[name] = tw

    axis_of = {"thr": ax_thr, "roll": ax_roll, "pitch": ax_pitch, "yaw": ax_yaw,
               "roll_tq": twins["roll"], "pitch_tq": twins["pitch"],
               "yaw_tq": twins["yaw"]}

    armed_art = []
    for a in main:
        armed_art += draw_armed(a, armed_spans(ulog))

    _plot_series(series, axis_of)

    for name, a in (("thr", ax_thr), ("roll", ax_roll), ("pitch", ax_pitch),
                    ("yaw", ax_yaw)):
        if not any(s.group == name for s in series):
            _no_data(a, "not in this log")

    # --- axis furniture -----------------------------------------------------
    for a in main[:-1]:
        style_time_axis(a, label=False)
    style_time_axis(ax_yaw)
    ax_thr.set_ylabel("throttle command\n(0..1, collective)", fontsize=9)
    ax_roll.set_ylabel("roll (deg)", fontsize=9)
    ax_pitch.set_ylabel("pitch (deg)", fontsize=9)
    ax_yaw.set_ylabel("yaw / heading (deg)", fontsize=9)
    for a in main:
        _style_axis(a, C_INK)
    for name, tw in twins.items():
        tw.set_ylabel("torque cmd (-1..1)", fontsize=8)
        _style_axis(tw, C_TORQUE)
        tw.spines["top"].set_visible(False)

    _header(fig, left, "Commands: throttle / roll / pitch / yaw", ulog, path,
            "dashed magenta = commanded, solid black = actual; "
            "torque command on the right axis (off by default)")

    mode_art, mode_codes = draw_mode_changes(
        main, mode_changes(ulog), text_ax=ax_thr,
        min_gap=max(duration_min(ulog), 1.0) * 0.035)
    mode_art += mode_key(fig, left + width, 0.018, mode_codes)

    def refresh():
        for group, a in axis_of.items():
            _rescale(a, [s.line for s in series if s.group == group])
        # A hidden twin still draws its tick labels, which reads as "the torque
        # is plotted and happens to be flat".  Show the right axis only while
        # something is on it.
        for name, tw in twins.items():
            on = any(s.line.get_visible() for s in series
                     if s.group == f"{name}_tq")
            tw.yaxis.set_visible(on)

    extra = []
    if mode_art:
        extra.append(("mode changes", mode_art, True))
    if armed_art:
        extra.append(("armed (shaded)", armed_art, True))

    h = min(0.86, 0.035 * (len(series) + len(extra) + 7) + 0.05)
    check_panel(fig, [0.012, 0.89 - h, 0.155, h], series,
                [("thr", "ALL throttle"), ("roll", "ALL roll"),
                 ("pitch", "ALL pitch"), ("yaw", "ALL yaw"),
                 ("roll_tq", ""), ("pitch_tq", ""), ("yaw_tq", "")],
                extra=extra, on_change=refresh)
    refresh()
    add_mouse_navigation(fig, main + list(twins.values()),
                         page_scroll=ctx.page_scroll, on_view=refresh)
    fig.text(left, 0.045, nav_hint(ctx.page_scroll), color=C_MUTED,
             fontsize=8, ha="left")
    return fig


def build_speed(ulog, ctx=None, path=""):
    """Speed magnitude (no direction).  Same signature as every builder."""
    import matplotlib.pyplot as plt

    ctx = ctx or PlotCtx()
    series = _speed_series(ulog, ctx)
    hdg = _heading_series(ulog, ctx)
    if not series and not hdg:
        return None
    series += hdg

    fig = plt.figure(figsize=(15, 10), facecolor=C_SURFACE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(
            f"logGraph speed - {os.path.basename(path)}")

    left, width = 0.260, 0.655
    ax = fig.add_axes([left, 0.535, width, 0.345], facecolor=C_SURFACE)
    ax_h = fig.add_axes([left, 0.115, width, 0.345], facecolor=C_SURFACE)
    ax.sharex(ax_h)
    armed_art = draw_armed(ax, armed_spans(ulog)) + draw_armed(ax_h, armed_spans(ulog))
    # Zero is the reference on a signed axis: above it is forward/right/up.
    ax_h.axhline(0.0, color=C_MUTED, lw=0.9, zorder=1)
    _plot_series(series, {"spd": ax, "hdg": ax_h})
    if not hdg:
        _no_data(ax_h, "no attitude in this log -- heading frame unavailable")

    style_time_axis(ax, label=False)
    style_time_axis(ax_h)
    ax.set_ylabel("speed (m/s)", fontsize=9)
    ax_h.set_ylabel("heading-frame velocity (m/s)\n+forward  +right  +up",
                    fontsize=9)
    ax_h.set_title("velocity along the nose / right wing / world up  --  "
                   "follows heading, not tilt; solid = actual, dashed = commanded",
                   fontsize=8, color=C_MUTED, loc="left")
    _style_axis(ax, C_INK)
    _style_axis(ax_h, C_INK)

    # The headline number, and WHEN, so it can be found on the other plots.
    top = next((s for s in series if s.id == "vehicle_local_position.|v|"), None)
    if top is not None and top.y.size:
        i = int(np.nanargmax(top.y))
        detail = f"max {top.y[i]:.2f} m/s (3D) at {top.t[i]:.2f} min"
    else:
        detail = "no estimated speed in this log"
    _header(fig, left, "Speed", ulog, path, detail)

    mode_art, mode_codes = draw_mode_changes(
        [ax, ax_h], mode_changes(ulog), text_ax=ax,
        min_gap=max(duration_min(ulog), 1.0) * 0.035)
    mode_art += mode_key(fig, left + width, 0.015, mode_codes)

    def refresh():
        _rescale(ax, [s.line for s in series if s.group == "spd"])
        _rescale(ax_h, [s.line for s in series if s.group == "hdg"])
        # Speed is non-negative; a rescale that pads below zero invents a region
        # no series can ever occupy.
        lo, hi = ax.get_ylim()
        ax.set_ylim(max(lo, 0.0), hi)

    extra = []
    if mode_art:
        extra.append(("mode changes", mode_art, True))
    if armed_art:
        extra.append(("armed (shaded)", armed_art, True))

    h = min(0.80, 0.035 * (len(series) + len(extra) + 4) + 0.05)
    check_panel(fig, [0.012, 0.89 - h, 0.155, h], series,
                [("spd", "ALL speed"), ("hdg", "ALL heading frame")],
                extra=extra, on_change=refresh)
    refresh()
    add_mouse_navigation(fig, [ax, ax_h], page_scroll=ctx.page_scroll,
                         on_view=refresh)
    fig.text(left, 0.045, nav_hint(ctx.page_scroll), color=C_MUTED,
             fontsize=8, ha="left")
    return fig


# --- motors and vibration -----------------------------------------------------
#
# There is no measured rotor speed on this airframe: `esc_status` is never
# published because the ESC (Electronic Speed Controller) telemetry line is not
# wired.  The closest thing the log DOES carry is PX4's own onboard gyro FFT
# (Fast Fourier Transform), `sensor_gyro_fft` -- run on the RAW gyro at ~2 kHz,
# publishing up to three spectral peaks per axis inside
# [IMU_GYRO_FFT_MIN, IMU_GYRO_FFT_MAX].  A spinning rotor shakes the frame at its
# rotation frequency, so those peaks are an indirect, un-labelled rotor-speed
# witness: which motor a peak belongs to is not known, only that something in
# the airframe is vibrating there.
#
# Why not a spectrogram of the logged gyro instead: the only continuous gyro
# stream in these logs is `sensor_combined` at ~188-200 Hz, so its spectrum stops
# at ~94-100 Hz (Nyquist) -- and the rotor line sits right about there (~100 Hz on
# landing_test_with_new_cover_4).  Computed and checked: it is broadband smear,
# with the rotor line aliased away.  The onboard FFT sees the raw rate.

MOTORS_TOPICS = [
    "actuator_motors",      # per-motor command, 0..1
    "sensor_gyro_fft",      # onboard FFT peaks -- the vibration witness
    "actuator_armed",
    "vehicle_status",
]

FFT_AXES = (("x", "o"), ("y", "s"), ("z", "^"))
FFT_CMAP = "viridis"        # SNR: brighter = stronger peak, as on the path plot


def _motor_series(ulog, ctx):
    """Per-motor commands, BLANKED while disarmed.

    actuator_motors keeps publishing while disarmed, but its value then is not
    what reaches the ESCs -- the output driver sends its disarmed value instead.
    Plotted as-is it is actively misleading: on a HITL FC_log.ulg all four
    motors "hold 0.763" for 13.5 min on the ground (the last command before
    disarm, republished at 10 Hz), which reads as a long hover.  Blanked as NaN
    rather than dropped so the line BREAKS instead of bridging the ground time.
    """
    series = []
    d = _get(ulog, "actuator_motors")
    if d is None:
        return series
    tm = _time_min(ulog, d)
    order = np.argsort(tm, kind="stable")
    tm = tm[order]
    spans = armed_spans(ulog)
    armed = np.zeros(tm.size, bool)
    for a, b in spans:
        armed |= (tm >= a) & (tm <= b)
    if not spans:
        armed[:] = True     # no armed topic: nothing to blank against
    elif (~armed).any():
        ctx.note(f"motor commands blanked while disarmed "
                 f"({100 * (~armed).mean():.0f}% of samples) -- the ESCs get the "
                 f"disarmed value then, not actuator_motors")
    cols = []
    for i in range(12):
        key = f"control[{i}]"
        if key not in d.data:
            continue
        y = np.asarray(d.data[key], float)[order]
        if not np.isfinite(y).any():    # unused outputs are all-NaN
            continue
        y = y.copy()
        y[~armed] = np.nan
        t = tm
        cols.append(y)
        series.append(Series(f"actuator_motors.{key}", f"motor {i + 1}", t, y,
                             "mot", INST_COLORS[i % len(INST_COLORS)], lw=1.0,
                             alpha=0.85, visible=True))
    if len(cols) > 1:
        # The mean is the collective; the spread of the motors about it is the
        # attitude control (roll/pitch/yaw differential).  Off by default -- it
        # hides behind four lines -- but it is the one to hold against the FFT.
        # All-NaN columns (every disarmed instant) make nanmean warn; NaN is
        # exactly the answer wanted there, so the warning is silenced, not fixed.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t, y = tm, np.nanmean(np.vstack(cols), axis=0)
        series.append(Series("actuator_motors.mean", "mean (collective)", t, y,
                             "mot", C_ACT, lw=1.6, visible=False, zorder=5))
    return series


def _fft_peaks(ulog):
    """{axis: (t_min, freq_hz, snr)} with all three peak slots pooled.

    PX4 writes NaN (or 0) into a slot with no peak above IMU_GYRO_FFT_SNR, so
    only finite, positive frequencies are kept.  The three slots are pooled per
    axis because their ORDER carries no identity -- slot 0 is simply the
    strongest peak at that instant, and which physical source that is can swap
    from one sample to the next.
    """
    d = _get(ulog, "sensor_gyro_fft")
    if d is None:
        return {}
    t = _time_min(ulog, d)
    out = {}
    for ax, _ in FFT_AXES:
        ts, fs, ss = [], [], []
        for k in range(3):
            fk = f"peak_frequencies_{ax}[{k}]"
            if fk not in d.data:
                continue
            f = np.asarray(d.data[fk], float)
            s = np.asarray(d.data.get(f"peak_snr_{ax}[{k}]", np.full_like(f, np.nan)), float)
            ok = np.isfinite(f) & (f > 0) & np.isfinite(t)
            ts.append(t[ok]); fs.append(f[ok]); ss.append(s[ok])
        if ts and sum(a.size for a in ts):
            out[ax] = tuple(np.concatenate(a) for a in (ts, fs, ss))
    return out


def build_motors(ulog, ctx=None, path=""):
    """Motor commands over the onboard gyro FFT peaks.  Same signature as every
    builder."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    ctx = ctx or PlotCtx()
    series = _motor_series(ulog, ctx)
    peaks = _fft_peaks(ulog)
    if not series and not peaks:
        ctx.note("no actuator_motors or sensor_gyro_fft in this log -- nothing to plot")
        return None
    if not series:
        ctx.note("no actuator_motors -- no motor commands to plot")
    if not peaks:
        ctx.note("no sensor_gyro_fft peaks in this log (IMU_GYRO_FFT_EN=0, or a "
                 "HITL log with no real gyro) -- no vibration spectrum")

    params = getattr(ulog, "initial_parameters", {}) or {}
    f_lo = float(params.get("IMU_GYRO_FFT_MIN", np.nan))
    f_hi = float(params.get("IMU_GYRO_FFT_MAX", np.nan))
    snr_min = float(params.get("IMU_GYRO_FFT_SNR", 10.0))

    fig = plt.figure(figsize=(15, 10), facecolor=C_SURFACE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(
            f"logGraph motors - {os.path.basename(path)}")

    left, width = 0.260, 0.655
    ax_m = fig.add_axes([left, 0.535, width, 0.345], facecolor=C_SURFACE)
    ax_f = fig.add_axes([left, 0.115, width, 0.345], facecolor=C_SURFACE)
    ax_m.sharex(ax_f)
    cax = fig.add_axes([left + width + 0.012, 0.115, 0.010, 0.345])

    armed_art = draw_armed(ax_m, armed_spans(ulog)) + draw_armed(ax_f, armed_spans(ulog))
    _plot_series(series, {"mot": ax_m})
    if not series:
        _no_data(ax_m, "no actuator_motors in this log")

    # --- FFT peaks: frequency against time, coloured by SNR -------------------
    # A scatter, not a line: consecutive samples are not the same physical
    # source (see _fft_peaks), and joining them would draw a zig-zag between two
    # rotor bands that is not a frequency sweep.
    all_snr = np.concatenate([p[2] for p in peaks.values()]) if peaks else np.array([])
    all_snr = all_snr[np.isfinite(all_snr)]
    snr_hi = float(np.percentile(all_snr, 99)) if all_snr.size else snr_min * 3
    norm = Normalize(vmin=snr_min, vmax=max(snr_hi, snr_min + 1.0))
    for ax_name, marker in FFT_AXES:
        if ax_name not in peaks:
            continue
        t, f, s = peaks[ax_name]
        # Draw weakest first so the strong peaks land on top where they overlap.
        order = np.argsort(np.nan_to_num(s, nan=0.0))
        sc = ax_f.scatter(t[order], f[order], c=s[order], cmap=FFT_CMAP,
                          norm=norm, s=5, marker=marker, linewidths=0,
                          zorder=3)
        # Hand the scatter to the checkbox panel as a Series; it only needs
        # set_visible/get_visible, which a PathCollection has.
        ser = Series(f"sensor_gyro_fft.peak_frequencies_{ax_name}",
                     f"gyro {ax_name} peaks", t, f, "fft", C_MUTED, visible=True)
        ser.line = sc
        series.append(ser)
    if not peaks:
        _no_data(ax_f, "no sensor_gyro_fft in this log")

    # The FFT's own search window: nothing outside it CAN appear, so an empty
    # band above f_hi means "not looked for", not "not there".
    win_art = []
    for f_edge in (f_lo, f_hi):
        if np.isfinite(f_edge):
            win_art.append(ax_f.axhline(f_edge, color=C_MUTED, lw=0.9, ls=":",
                                        zorder=2))
    if np.isfinite(f_lo) and np.isfinite(f_hi):
        win_art.append(ax_f.text(
            0.995, f_hi, f"IMU_GYRO_FFT_MAX {f_hi:g} Hz ", ha="right", va="bottom",
            transform=ax_f.get_yaxis_transform(), fontsize=7, color=C_MUTED))
        win_art.append(ax_f.text(
            0.995, f_lo, f"IMU_GYRO_FFT_MIN {f_lo:g} Hz ", ha="right", va="top",
            transform=ax_f.get_yaxis_transform(), fontsize=7, color=C_MUTED))
        pad = 0.06 * (f_hi - f_lo)
        ax_f.set_ylim(f_lo - pad, f_hi + pad)

    if peaks:
        sm = plt.cm.ScalarMappable(cmap=FFT_CMAP, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(f"peak SNR (published only above {snr_min:g})",
                     fontsize=8, color=C_MUTED)
        cb.ax.tick_params(colors=C_MUTED, labelsize=7)
        cb.outline.set_edgecolor(C_GRID)
    else:
        cax.set_visible(False)      # a colour key for no points is noise

    # --- furniture ------------------------------------------------------------
    style_time_axis(ax_m, label=False)
    style_time_axis(ax_f)
    ax_m.set_ylabel("motor command (0..1)", fontsize=9)
    ax_f.set_ylabel("gyro vibration peak (Hz)", fontsize=9)
    ax_m.set_title("actuator_motors -- what each ESC was sent (blank = disarmed)",
                   fontsize=8,
                   color=C_MUTED, loc="left")
    ax_f.set_title("onboard gyro FFT peaks (sensor_gyro_fft) -- an indirect "
                   "rotor-speed witness; peaks are not labelled by motor",
                   fontsize=8, color=C_MUTED, loc="left")
    _style_axis(ax_m, C_INK)
    _style_axis(ax_f, C_INK)

    if peaks:
        fl = np.concatenate([p[1] for p in peaks.values()])
        tt = np.concatenate([p[0] for p in peaks.values()])
        arm = armed_spans(ulog)
        in_arm = np.zeros(tt.size, bool)
        for a, b in arm:
            in_arm |= (tt >= a) & (tt <= b)
        med = float(np.median(fl[in_arm])) if in_arm.any() else float(np.median(fl))
        detail = (f"armed median peak {med:.0f} Hz (= {med * 60:.0f} rev/min if it "
                  f"is the rotor rotation line)")
        # Peaks with the motors STOPPED cannot be rotors, so they are the
        # floor under the rotor reading and belong in the headline.  Measured on
        # landing_test_with_new_cover_4: a steady 97 Hz peak for 10 min on the
        # ground, 5 Hz from the 102 Hz in-flight line.
        if arm and (~in_arm).any():
            gnd = float(np.median(fl[~in_arm]))
            detail += f"   |   disarmed median {gnd:.0f} Hz -- motors stopped, NOT rotor"
    else:
        detail = "no onboard FFT -- motor commands only"
    _header(fig, left, "Motors / vibration spectrum", ulog, path, detail)

    mode_art, mode_codes = draw_mode_changes(
        [ax_m, ax_f], mode_changes(ulog), text_ax=ax_m,
        min_gap=max(duration_min(ulog), 1.0) * 0.035)
    mode_art += mode_key(fig, left + width, 0.015, mode_codes)

    def refresh():
        # Only the command axis rescales.  The FFT axis is pinned to the search
        # window: fitting it to the visible peaks would stretch a single 40 Hz
        # outlier into half the panel and hide the rotor bands.
        _rescale(ax_m, [s.line for s in series if s.group == "mot"])

    extra = []
    if win_art:
        extra.append(("FFT search window", win_art, True))
    if mode_art:
        extra.append(("mode changes", mode_art, True))
    if armed_art:
        extra.append(("armed (shaded)", armed_art, True))

    h = min(0.80, 0.035 * (len(series) + len(extra) + 4) + 0.05)
    check_panel(fig, [0.012, 0.89 - h, 0.155, h], series,
                [("mot", "ALL motors"), ("fft", "ALL gyro axes")],
                extra=extra, on_change=refresh)
    refresh()
    add_mouse_navigation(fig, [ax_m, ax_f], page_scroll=ctx.page_scroll,
                         on_view=refresh)
    fig.text(left, 0.045, nav_hint(ctx.page_scroll), color=C_MUTED,
             fontsize=8, ha="left")
    return fig
