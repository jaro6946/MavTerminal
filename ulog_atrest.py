#!/usr/bin/env python3
"""ulog_atrest.py -- the land detector's "vehicle at rest" decision.

One figure: the at-rest decision, with every input that sets it drawn
against the limit it is compared to:
      1. body angular rate |w|                vs 3 deg/s
      2. gyro vibration metric (selected IMU)  vs 0.02
      3. accel vibration metric (selected IMU) vs 1.2
      4. which trigger fired, the 1 s hold, "landed", the logged result, and
         what each EKF instance received
    "Not at rest" is not a fault, but it switches OFF things that are only safe
    on a still vehicle: the EKF's GPS drift checks (EKF2_REQ_HDRIFT/VDRIFT, so
    the origin can latch on a still-settling GPS), its zero-velocity update,
    and in-run gyro bias learning.  That is why it sits beside the faults.

How PX4 decides "at rest" (LandDetector.cpp, v1.14 lab source)
--------------------------------------------------------------
    moving   = |w| > 3 deg/s                    (vehicle_angular_velocity)
            OR gyro_vibration_metric  > 0.02    (vehicle_imu_status of the
            OR accel_vibration_metric > 1.2      gyro sensor_selection picked)
    _at_rest = no "moving" sample in the last 1 s
    at_rest  = landed AND _at_rest              (vehicle_land_detected.at_rest)
The three limits are compile-time constants, not parameters.

The recomputation here uses LOGGED samples.  The land detector sees every
vehicle_angular_velocity sample (hundreds of Hz); the log carries a decimated
copy, so a brief rate spike can be missed.  The header reports how often the
recomputed answer agrees with the logged one, so a disagreement is visible
rather than silent.

Acronyms: IMU = inertial measurement unit, EKF = extended Kalman filter,
GPS = Global Positioning System.
"""
import os

import numpy as np

from ulog_common import (C_ARMED, C_BAD, C_INK, C_MUTED, C_SURFACE, PlotCtx,
                         Series, _clean, _get, _style_axis,
                         _time_min, add_mouse_navigation, armed_spans,
                         check_panel, draw_armed, draw_band_rows,
                         draw_mode_changes, duration_min, has_topic,
                         inst_color, mode_changes, mode_key, nav_hint,
                         spans_from_bool, style_time_axis, window_values)

AT_REST_TOPICS = [
    "vehicle_angular_velocity",  # the |w| trigger
    "vehicle_imu_status",        # the two vibration triggers
    "sensor_selection",          # which IMU's status the land detector reads
    "vehicle_land_detected",     # landed, and the logged at_rest result
    "estimator_status_flags",    # cs_vehicle_at_rest: what each EKF received
    "actuator_armed",
    "vehicle_status",
]

# LandDetector.cpp:111 / :214-215 -- compile-time constants, not parameters.
RATE_LIMIT_DEG_S = 3.0
GYRO_VIBE_LIMIT = 0.02
ACCEL_VIBE_LIMIT = 1.2
HOLD_S = 1.0                    # LandDetector.cpp:256  "> 1_s" since last move

C_LIMIT = C_BAD                 # the limit line: crossing it is the event
C_RATE = "#20222b"              # near-black: one signal, no IMU identity
C_MOVING = "#d2691e"            # orange: the land detector's "moving"
C_REST = "#1baf7a"              # aqua: at rest
C_LANDED = "#8a7fb5"            # violet

# Layout, in inches -- same scheme as ulog_cpu: stacked bottom-up, then divided
# by the figure height, so a band with more rows grows the figure instead of
# squashing the trace panels.
GAP_IN = 0.48
TOP_IN = 0.95
BOTTOM_IN = 1.35
BAND_ROW_IN = 0.40
BAND_PAD_IN = 0.30
BAND_MIN_IN = 1.00
PAGE_PX_PER_IN = 78
LEFT, WIDTH = 0.260, 0.655


# --- data ---------------------------------------------------------------------

def _imu_instances(ulog):
    return sorted(d.multi_id for d in ulog.data_list
                  if d.name == "vehicle_imu_status")


def _imu_field(ulog, m, name):
    d = _get(ulog, "vehicle_imu_status", m)
    if d is None or name not in d.data:
        return np.array([]), np.array([])
    return _clean(_time_min(ulog, d), d.data[name])


def _imu_gyro_id(ulog, m):
    d = _get(ulog, "vehicle_imu_status", m)
    if d is None or "gyro_device_id" not in d.data:
        return None
    ids = np.asarray(d.data["gyro_device_id"])
    ids = ids[ids != 0]
    return int(ids[-1]) if ids.size else None


def _selected_imu_epochs(ulog, ctx):
    """[(t0, t1, imu_index)]: which vehicle_imu_status the land detector read.

    It follows sensor_selection.gyro_device_id (LandDetector::UpdateVehicleAtRest)
    and finds the imu_status instance carrying that gyro.  NOT the EKF primary:
    the two can differ, and the sensor_selection gyro is the one that matters
    here."""
    dur = duration_min(ulog) or 0.0
    imus = _imu_instances(ulog)
    if not imus:
        return []
    by_id = {_imu_gyro_id(ulog, m): m for m in imus}
    d = _get(ulog, "sensor_selection")
    if d is None or "gyro_device_id" not in d.data:
        ctx.note("no sensor_selection in this log -- assuming the land detector "
                 f"read IMU {imus[0]}")
        return [(0.0, dur, imus[0])]
    t = _time_min(ulog, d)
    ids = np.asarray(d.data["gyro_device_id"])
    out = []
    for i in range(t.size):
        m = by_id.get(int(ids[i]))
        if m is None or (out and out[-1][2] == m):
            continue
        if out:                              # close the previous epoch here
            out[-1] = (out[-1][0], float(t[i]), out[-1][2])
        out.append((float(t[i]) if out else 0.0, dur, m))
    if not out:
        ctx.note("sensor_selection's gyro matches no vehicle_imu_status -- "
                 f"assuming IMU {imus[0]}")
        return [(0.0, dur, imus[0])]
    return out


def _selected_series(ulog, epochs, name):
    """(t, y) of `name` taken from whichever IMU was selected at each moment."""
    ts, ys = [], []
    for t0, t1, m in epochs:
        t, y = _imu_field(ulog, m, name)
        k = (t >= t0) & (t <= t1)
        ts.append(t[k])
        ys.append(y[k])
    if not ts:
        return np.array([]), np.array([])
    return _clean(np.concatenate(ts), np.concatenate(ys))


def _rate_deg_s(ulog):
    d = _get(ulog, "vehicle_angular_velocity")
    if d is None or "xyz[0]" not in d.data:
        return np.array([]), np.array([])
    w = np.sqrt(sum(np.asarray(d.data[f"xyz[{i}]"], float) ** 2 for i in range(3)))
    return _clean(_time_min(ulog, d), np.degrees(w))


def _held_spans(t_trig, hold_min, t_end):
    """Spans covered by 'a trigger fired within the last `hold_min`'.

    Each trigger sample opens [t, t + hold]; overlapping windows merge.  This is
    exactly the land detector's `elapsed since last move > 1 s` read backwards."""
    t_trig = np.sort(np.asarray(t_trig, float))
    if t_trig.size == 0:
        return []
    brk = np.flatnonzero(np.diff(t_trig) > hold_min)
    starts = np.r_[t_trig[0], t_trig[brk + 1]]
    ends = np.r_[t_trig[brk], t_trig[-1]] + hold_min
    return [(float(a), float(min(b, t_end))) for a, b in zip(starts, ends)]


def _spans_mask(t, spans):
    m = np.zeros(t.size, bool)
    for a, b in spans:
        m |= (t >= a) & (t <= b)
    return m


# --- shared figure furniture -------------------------------------------------

def _log_rescale(ax, lines, limit):
    """Fit a log axis to the visible lines in the time window, always keeping
    the limit on screen: the limit is what the panel is FOR, and a window where
    every value sits far below it should show that distance, not hide it."""
    v = window_values(ax, lines, positive_only=True)
    vals = np.r_[v, limit] if v.size else np.array([limit])
    lo, hi = float(np.percentile(vals, 0.5)), float(vals.max())
    lo = min(lo, limit)
    ax.set_ylim(lo / 1.6, hi * 1.6)


def _limit_line(ax, y, text):
    ln = ax.axhline(y, color=C_LIMIT, lw=1.1, ls="--", alpha=0.85, zorder=4)
    tx = ax.text(0.995, y, f"{text} ", transform=ax.get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=7, color=C_LIMIT, zorder=5)
    return [ln, tx]


def _layout(fig_h, panels, band_in):
    """{key: (bottom, height)} in figure fractions, band at the bottom."""
    rects, bottom = {}, BOTTOM_IN
    rects["band"] = (bottom / fig_h, band_in / fig_h)
    bottom += band_in + GAP_IN
    for key, h in reversed(panels):
        rects[key] = (bottom / fig_h, h / fig_h)
        bottom += h + GAP_IN
    return rects


def _header(fig, fig_h, title, ulog, path, detail):
    fig.text(LEFT, 1.0 - 0.35 / fig_h, title, color=C_INK, fontsize=13,
             fontweight="bold", ha="left")
    who = f"{os.path.basename(path)}   |   " if path else ""
    fig.text(LEFT, 1.0 - 0.62 / fig_h,
             f"{who}{duration_min(ulog):.1f} min   |   {detail}",
             color=C_MUTED, fontsize=9, ha="left")


def _finish(fig, fig_h, ulog, ctx, series, axes, ax_band, rects, groups,
            anchors, armed_art, refresh, mode_text_ax, limit_art=()):
    dur = duration_min(ulog) or 1.0
    mode_art, codes = draw_mode_changes(axes + [ax_band], mode_changes(ulog),
                                        text_ax=mode_text_ax, min_gap=dur * 0.035)
    mode_art += mode_key(fig, LEFT + WIDTH, 0.10 / fig_h, codes)
    extra = []
    if limit_art:
        extra.append(("limits (dashed red)", list(limit_art), True))
    if mode_art:
        extra.append(("mode changes", mode_art, True))
    if armed_art:
        extra.append(("armed (shaded)", armed_art, True))
    top_key = next(iter(anchors.values()))      # the topmost panel
    cb_top = rects[top_key][0] + rects[top_key][1]
    cb_bot = rects["band"][0]

    def _anchor(key):
        b, ph = rects[key]
        return (b + ph / 2 - cb_bot) / (cb_top - cb_bot)

    check_panel(fig, [0.012, cb_bot, 0.155, cb_top - cb_bot], series, groups,
                extra=extra, on_change=refresh,
                anchors={g: _anchor(k) for g, k in anchors.items()})
    ax_band.set_xlim(0.0, dur)
    refresh()
    add_mouse_navigation(fig, axes + [ax_band], page_scroll=ctx.page_scroll,
                         fixed_y=[ax_band], on_view=refresh)
    fig.text(LEFT, 0.32 / fig_h, nav_hint(ctx.page_scroll), color=C_MUTED,
             fontsize=8, ha="left")
    fig._page_height = int(round(fig_h * PAGE_PX_PER_IN))


def _new_fig(fig_h, path, what):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, fig_h), facecolor=C_SURFACE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(
            f"logGraph {what} - {os.path.basename(path)}")
    return fig


def _plot(ax, s):
    (line,) = ax.plot(s.t, s.y, color=s.color, ls=s.ls, lw=s.lw, label=s.label,
                      alpha=s.alpha, zorder=s.zorder if s.zorder is not None else 3)
    line.set_visible(s.visible)
    s.line = line


# --- the figure -------------------------------------------------------------

def build_at_rest(ulog, ctx=None, path=""):
    """The land detector's at-rest decision and the three inputs that set it."""
    ctx = ctx or PlotCtx()
    if not has_topic(ulog, "vehicle_land_detected"):
        ctx.note("no vehicle_land_detected in this log -- no at-rest decision")
        return None

    dur = duration_min(ulog) or 1.0
    hold = HOLD_S / 60.0
    epochs = _selected_imu_epochs(ulog, ctx)
    imus = _imu_instances(ulog)

    t_w, w = _rate_deg_s(ulog)
    t_g, g = _selected_series(ulog, epochs, "gyro_vibration_metric")
    t_a, a = _selected_series(ulog, epochs, "accel_vibration_metric")

    series = []
    if t_w.size:
        series.append(Series("vehicle_angular_velocity.|xyz|", "|w| (deg/s)",
                             t_w, w, "rate", C_RATE, lw=1.0, visible=True))
    else:
        ctx.note("no vehicle_angular_velocity in this log -- rate trigger not shown")
    sel_set = {m for _a, _b, m in epochs}
    for name, group, lab in (("gyro_vibration_metric", "gvib", "gyro"),
                             ("accel_vibration_metric", "avib", "accel")):
        tt, yy = (t_g, g) if group == "gvib" else (t_a, a)
        if tt.size:
            series.append(Series(f"selected IMU {name}",
                                 f"{lab}: IMU the detector reads",
                                 tt, yy, group, C_INK, lw=1.6, visible=True,
                                 zorder=4))
        # The others, faint and off by default: a reader asking "was it the
        # whole airframe or one sensor" needs them, the decision does not.
        for m in imus:
            t, y = _imu_field(ulog, m, name)
            if t.size:
                series.append(Series(f"vehicle_imu_status[{m}].{name}",
                                     f"{lab}: IMU {m}"
                                     + (" (selected)" if m in sel_set else ""),
                                     t, y, group, inst_color(m), lw=0.9,
                                     alpha=0.7, visible=False))

    # --- the decision, recomputed --------------------------------------------
    trig_w = t_w[w > RATE_LIMIT_DEG_S] if t_w.size else np.array([])
    trig_g = t_g[g > GYRO_VIBE_LIMIT] if t_g.size else np.array([])
    trig_a = t_a[a > ACCEL_VIBE_LIMIT] if t_a.size else np.array([])
    moving = _held_spans(np.r_[trig_w, trig_g, trig_a], hold, dur)

    ld = _get(ulog, "vehicle_land_detected")
    t_ld = _time_min(ulog, ld)
    landed = np.asarray(ld.data.get("landed", np.zeros(t_ld.size)), float) > 0.5
    logged_rest = np.asarray(ld.data.get("at_rest", np.zeros(t_ld.size)), float) > 0.5
    landed_spans = spans_from_bool(t_ld, landed)
    rest_spans = spans_from_bool(t_ld, logged_rest)

    # Agreement, sampled on a 0.1 s grid so a long quiet stretch and a busy one
    # weigh by duration, not by how often vehicle_land_detected published.
    grid = np.arange(0.0, dur, 0.1 / 60.0)
    idx = np.clip(np.searchsorted(t_ld, grid, side="right") - 1, 0, t_ld.size - 1)
    rec = landed[idx] & ~_spans_mask(grid, moving)
    agree = float(np.mean(rec == logged_rest[idx])) if grid.size else np.nan

    rows = [
        (f"|w| > {RATE_LIMIT_DEG_S:g} deg/s", spans_from_bool(t_w, w > RATE_LIMIT_DEG_S)
         if t_w.size else [], C_MOVING),
        (f"gyro vibration > {GYRO_VIBE_LIMIT:g}",
         spans_from_bool(t_g, g > GYRO_VIBE_LIMIT) if t_g.size else [], C_MOVING),
        (f"accel vibration > {ACCEL_VIBE_LIMIT:g}",
         spans_from_bool(t_a, a > ACCEL_VIBE_LIMIT) if t_a.size else [], C_MOVING),
        (f"MOVING: any trigger in last {HOLD_S:g} s (recomputed)", moving, C_BAD),
        ("landed (land detector)", landed_spans, C_LANDED),
        ("AT REST = landed AND not moving (logged)", rest_spans, C_REST),
    ]
    lanes, inst = [], []
    for i in range(4):
        d = _get(ulog, "estimator_status_flags", i)
        if d is None or "cs_vehicle_at_rest" not in d.data:
            continue
        t = _time_min(ulog, d)
        lanes.append((spans_from_bool(t, np.asarray(d.data["cs_vehicle_at_rest"],
                                                    float) > 0.5), inst_color(i)))
        inst.append(str(i))
    if lanes:
        rows.append((f"EKF {'/'.join(inst)} received at rest (cs_vehicle_at_rest)",
                     lanes, C_MUTED))
    rows.append(("armed", armed_spans(ulog), C_ARMED))

    panels = [("rate", 1.9), ("gvib", 1.9), ("avib", 1.9)]
    band_in = max(len(rows) * BAND_ROW_IN + BAND_PAD_IN + (0.3 if lanes else 0),
                  BAND_MIN_IN)
    fig_h = TOP_IN + sum(h for _k, h in panels) + GAP_IN * len(panels) + band_in + BOTTOM_IN
    fig = _new_fig(fig_h, path, "at rest")
    rects = _layout(fig_h, panels, band_in)
    ax_w, ax_g, ax_a, ax_band = [fig.add_axes([LEFT, rects[k][0], WIDTH, rects[k][1]],
                                              facecolor=C_SURFACE)
                                 for k in ("rate", "gvib", "avib", "band")]
    for ax in (ax_w, ax_g, ax_a):
        ax.sharex(ax_band)
        ax.set_yscale("log")

    armed_art = []
    for ax in (ax_w, ax_g, ax_a, ax_band):
        armed_art += draw_armed(ax, armed_spans(ulog))
    axis_of = {"rate": ax_w, "gvib": ax_g, "avib": ax_a}
    for s in series:
        _plot(axis_of[s.group], s)
    lim_art = (_limit_line(ax_w, RATE_LIMIT_DEG_S, f"limit {RATE_LIMIT_DEG_S:g} deg/s")
               + _limit_line(ax_g, GYRO_VIBE_LIMIT, f"limit {GYRO_VIBE_LIMIT:g}")
               + _limit_line(ax_a, ACCEL_VIBE_LIMIT, f"limit {ACCEL_VIBE_LIMIT:g}"))
    draw_band_rows(ax_band, rows, ylabel="decision", min_width=dur * 0.003,
                   track=True)

    for ax in (ax_w, ax_g, ax_a):
        style_time_axis(ax, label=False)
        _style_axis(ax, C_INK)
    style_time_axis(ax_band)
    ax_w.set_ylabel("body rate |w|\n(deg/s, log)", fontsize=9)
    ax_g.set_ylabel("gyro vibration\nmetric (log)", fontsize=9)
    ax_a.set_ylabel("accel vibration\nmetric (log)", fontsize=9)
    ax_w.set_title("trigger 1: vehicle_angular_velocity |xyz| > "
                   f"{RATE_LIMIT_DEG_S:g} deg/s  (log is decimated -- the detector "
                   "sees every sample)", fontsize=8, color=C_MUTED, loc="left")
    ax_g.set_title("trigger 2: gyro_vibration_metric of the IMU whose gyro "
                   f"sensor_selection picked > {GYRO_VIBE_LIMIT:g}",
                   fontsize=8, color=C_MUTED, loc="left")
    ax_a.set_title("trigger 3: accel_vibration_metric of the same IMU > "
                   f"{ACCEL_VIBE_LIMIT:g}   ->  any trigger = moving for "
                   f"{HOLD_S:g} s", fontsize=8, color=C_MUTED, loc="left")

    rest_frac = float(np.mean(logged_rest[idx][landed[idx]])) if landed[idx].any() else np.nan
    first = next((a0 for a0, _b in rest_spans), None)
    detail = (f"at rest {100 * rest_frac:.0f}% of landed time   |   first at rest "
              f"{'never' if first is None else f'{first:.2f} min'}   |   "
              f"recomputed vs logged agree {100 * agree:.1f}%")
    _header(fig, fig_h, "Vehicle at rest (land detector)", ulog, path, detail)
    if np.isfinite(agree) and agree < 0.97:
        ctx.note(f"recomputed at-rest agrees with the logged one only "
                 f"{100 * agree:.1f}% of the time -- the log's decimated "
                 f"angular velocity probably misses rate spikes the detector saw")

    def refresh():
        _log_rescale(ax_w, [s.line for s in series if s.group == "rate"],
                     RATE_LIMIT_DEG_S)
        _log_rescale(ax_g, [s.line for s in series if s.group == "gvib"],
                     GYRO_VIBE_LIMIT)
        _log_rescale(ax_a, [s.line for s in series if s.group == "avib"],
                     ACCEL_VIBE_LIMIT)

    _finish(fig, fig_h, ulog, ctx, series, [ax_w, ax_g, ax_a], ax_band, rects,
            [("rate", "ALL rate"), ("gvib", "ALL gyro vibration"),
             ("avib", "ALL accel vibration")],
            {"rate": "rate", "gvib": "gvib", "avib": "avib"}, armed_art,
            refresh, ax_w, limit_art=lim_art)
    return fig
