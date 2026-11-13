#!/usr/bin/env python3
"""ulog_attitude.py -- attitude over time: roll / pitch / yaw / tilt.

Four stacked panels on one time axis:

  1. roll   -- actual (solid black) vs PX4's attitude setpoint (dashed magenta)
  2. pitch  -- same
  3. yaw    -- same, UNWRAPPED, so a slow turn through north is a smooth line
               instead of a +-180 saw-tooth.  The setpoint is put on the same
               360-degree branch as the actual at every sample, so the gap
               between them is always the true (shortest) heading error.
  4. tilt   -- angle of the body z axis from vertical, one number for "was it
               upright": 30 and 60 degree reference lines.  A flip is this line
               crossing 90.

Background: armed (grey, as every logGraph plot), OFFBOARD (light magenta),
and -- hatched orange -- SE3 DIRECT CONTROL: offboard with PX4's attitude loop
OFF (vehicle_control_mode.flag_control_attitude_enabled = 0), i.e. the
companion's SE3 controller sending thrust + torque straight to the mixer.
PX4 computes NO attitude setpoint then, so the dashed line is deliberately
blanked inside those spans and the span is labelled: a missing setpoint there
is the design, not a logging hole.  SE3's own commanded attitude is in the
companion's se3_trace.csv (logGraph Bags tab, SE3 dashboard).

How this differs from "Commands": that plot leads with the throttle command
and the mixer's torques; this one is attitude only, adds tilt and the control-
mode shading, and unwraps heading.

Acronyms: PX4 = the flight stack, SE3 = the companion's geometric controller,
FRD/NED = forward-right-down body / north-east-down world axes.
"""
import os

import numpy as np

from ulog_common import (C_BAD, C_INK, C_MUTED, C_SURFACE, PlotCtx, Series,
                         _get, _rescale, _style_axis, _time_min,
                         add_mouse_navigation, armed_spans, check_panel,
                         draw_armed, draw_mode_changes, duration_min, field,
                         has_topic, mode_changes, mode_key, nav_hint,
                         spans_from_bool, style_time_axis)
from ulog_commands import (C_ACT, C_CMD, _header, _no_data, _plot_series)

ATTITUDE_TOPICS = [
    "vehicle_attitude",              # q, actual
    "vehicle_attitude_setpoint",     # q_d, PX4's command
    "vehicle_control_mode",          # offboard / attitude-loop flags
    "vehicle_status",                # nav_state (OFFBOARD), mode overlay
    "actuator_armed",
]

C_OFFB = "#d81b60"      # offboard shading (magenta, the mode colour, very light)
C_SE3 = "#e08a00"       # SE3 direct control (orange, as the Bags-tab dashboard)
TILT_LINES = (30.0, 60.0)


def _quat(ulog, topic, prefix):
    """(t_min, w, x, y, z) sorted and finite, or None."""
    d = _get(ulog, topic)
    if d is None or f"{prefix}[0]" not in d.data:
        return None
    t = _time_min(ulog, d)
    q = [np.asarray(d.data[f"{prefix}[{i}]"], float) for i in range(4)]
    ok = np.isfinite(t)
    for c in q:
        ok &= np.isfinite(c)
    if not ok.any():
        return None
    o = np.argsort(t[ok], kind="stable")
    return (t[ok][o],) + tuple(c[ok][o] for c in q)


def _euler(w, x, y, z):
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    # Tilt: angle between body z and world z = arccos(R[2,2]).
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0)))
    return roll, pitch, yaw, tilt


def _unwrap(y):
    return np.degrees(np.unwrap(np.radians(y)))


def _se3_spans(ulog):
    """Offboard with PX4's attitude loop off = SE3 thrust/torque direct."""
    t, off = field(ulog, "vehicle_control_mode", "flag_control_offboard_enabled")
    t2, att = field(ulog, "vehicle_control_mode", "flag_control_attitude_enabled")
    if off.size == 0 or att.size != off.size:
        return []
    return spans_from_bool(t, (off > 0.5) & (att < 0.5))


def _offboard_spans(ulog):
    t, nav = field(ulog, "vehicle_status", "nav_state")
    if nav.size == 0:
        return []
    return spans_from_bool(t, nav == 14)


def _mask_spans(t, y, spans):
    y = np.asarray(y, float).copy()
    for a, b in spans:
        y[(t >= a) & (t <= b)] = np.nan
    return y


def _series(ulog, ctx, se3):
    out = []
    act = _quat(ulog, "vehicle_attitude", "q")
    sp = _quat(ulog, "vehicle_attitude_setpoint", "q_d")
    if act is None:
        ctx.note("no vehicle_attitude.q -- no attitude to plot")
        return out
    ta = act[0]
    r, p, y, tilt = _euler(*act[1:])
    yu = _unwrap(y)
    for name, val in (("roll", r), ("pitch", p), ("yaw", yu)):
        out.append(Series(f"vehicle_attitude.{name}", f"{name} actual", ta, val,
                          name, C_ACT, lw=1.3, visible=True, zorder=3))
    out.append(Series("vehicle_attitude.tilt", "tilt actual", ta, tilt, "tilt",
                      C_ACT, lw=1.3, visible=True, zorder=3))
    if sp is None:
        ctx.note("no vehicle_attitude_setpoint.q_d -- no PX4 attitude setpoint")
        return out
    ts = sp[0]
    rs, ps, ys, tilts = _euler(*sp[1:])
    # Yaw setpoint on the actual's branch: actual (interpolated) + wrapped error.
    ya = np.interp(ts, ta, yu)
    ys = ya + (ys - ya + 180.0) % 360.0 - 180.0
    for name, val in (("roll", rs), ("pitch", ps), ("yaw", ys), ("tilt", tilts)):
        out.append(Series(f"vehicle_attitude_setpoint.{name}", f"{name} setpoint",
                          ts, _mask_spans(ts, val, se3), name, C_CMD, ls="--",
                          lw=1.2, visible=True, zorder=4))
    return out


def build_attitude(ulog, ctx=None, path=""):
    import matplotlib.pyplot as plt

    ctx = ctx or PlotCtx()
    if not has_topic(ulog, "vehicle_attitude"):
        ctx.note("no vehicle_attitude in this log -- nothing to plot")
        return None
    se3 = _se3_spans(ulog)
    series = _series(ulog, ctx, se3)
    if not series:
        return None

    fig = plt.figure(figsize=(15, 10.5), facecolor=C_SURFACE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(
            f"logGraph attitude - {os.path.basename(path)}")
    left, width = 0.260, 0.655                 # same column as the other plots
    rects = [(0.690, 0.185), (0.475, 0.185), (0.260, 0.185), (0.090, 0.140)]
    axs = [fig.add_axes([left, b, width, h], facecolor=C_SURFACE) for b, h in rects]
    for a in axs[:-1]:
        a.sharex(axs[-1])
    ax_r, ax_p, ax_y, ax_t = axs
    axis_of = {"roll": ax_r, "pitch": ax_p, "yaw": ax_y, "tilt": ax_t}

    armed_art = []
    for a in axs:
        armed_art += draw_armed(a, armed_spans(ulog))
    offb_art, se3_art = [], []
    for a in axs:
        for t0, t1 in _offboard_spans(ulog):
            offb_art.append(a.axvspan(t0, t1, color=C_OFFB, alpha=0.06, lw=0, zorder=0))
        for t0, t1 in se3:
            se3_art.append(a.axvspan(t0, t1, facecolor="none", edgecolor=C_SE3,
                                     hatch="///", alpha=0.35, lw=0, zorder=0))
    # Label each SE3 span once, on the roll panel.
    dur = max(duration_min(ulog), 1e-6)
    for t0, t1 in se3:
        if (t1 - t0) > 0.04 * dur:
            se3_art.append(ax_r.text((t0 + t1) / 2, 0.97,
                                     "SE3 direct thrust+torque:\nno PX4 attitude setpoint",
                                     transform=ax_r.get_xaxis_transform(), ha="center",
                                     va="top", fontsize=7.5, color=C_SE3,
                                     fontweight="bold", clip_on=True))

    _plot_series(series, axis_of)
    for name, a in axis_of.items():
        if not any(s.group == name for s in series):
            _no_data(a, "not in this log")

    tilt_art = []
    for lim in TILT_LINES:
        tilt_art.append(ax_t.axhline(lim, color=C_BAD if lim >= 60 else C_MUTED,
                                     lw=0.9, ls="--", zorder=2))
        tilt_art.append(ax_t.text(1.002, lim, f"{lim:.0f}°", color=C_MUTED, fontsize=7,
                                  transform=ax_t.get_yaxis_transform(), va="center"))

    for a in axs[:-1]:
        style_time_axis(a, label=False)
    style_time_axis(ax_t)
    ax_r.set_ylabel("roll (deg)", fontsize=9)
    ax_p.set_ylabel("pitch (deg)", fontsize=9)
    ax_y.set_ylabel("yaw (deg, unwrapped,\nfrom north, clockwise)", fontsize=9)
    ax_t.set_ylabel("tilt from vertical\n(deg)", fontsize=9)
    for a in axs:
        _style_axis(a, C_INK)

    n_se3 = len(se3)
    _header(fig, left, "Attitude: roll / pitch / yaw / tilt", ulog, path,
            "black = actual, dashed = PX4 setpoint · hatched orange = SE3 direct "
            f"control ({n_se3} span{'s' if n_se3 != 1 else ''}, no PX4 setpoint)")

    mode_art, mode_codes = draw_mode_changes(
        axs, mode_changes(ulog), text_ax=ax_r, min_gap=max(dur, 1.0) * 0.035)
    mode_art += mode_key(fig, left + width, 0.018, mode_codes)

    def refresh():
        for group, a in axis_of.items():
            _rescale(a, [s.line for s in series if s.group == group])
        lo, hi = ax_t.get_ylim()
        ax_t.set_ylim(0, max(hi, 65.0))       # keep the 60 deg line in view

    extra = []
    if mode_art:
        extra.append(("mode changes", mode_art, True))
    if armed_art:
        extra.append(("armed (shaded)", armed_art, True))
    if offb_art:
        extra.append(("OFFBOARD (shaded)", offb_art, True))
    if se3_art:
        extra.append(("SE3 direct (hatched)", se3_art, True))
    extra.append(("tilt 30/60° lines", tilt_art, True))

    h = min(0.86, 0.035 * (len(series) + len(extra) + 6) + 0.05)
    check_panel(fig, [0.012, 0.89 - h, 0.155, h], series,
                [("roll", "ALL roll"), ("pitch", "ALL pitch"), ("yaw", "ALL yaw"),
                 ("tilt", "ALL tilt")],
                extra=extra, on_change=refresh)
    refresh()
    add_mouse_navigation(fig, axs, page_scroll=ctx.page_scroll, on_view=refresh)
    fig.text(left, 0.035, nav_hint(ctx.page_scroll), color=C_MUTED, fontsize=8,
             ha="left")
    return fig
