#!/usr/bin/env python3
"""ulog_graph.py -- plot a PX4 .ulg flight log as a time series you can toggle.

Companion to ulog_diag.py: that one tells you WHY the vehicle misbehaved in text,
this one lets you LOOK at a signal over time.  Built for thermal work -- the
default figure overlays, on one time axis:

  * every temperature channel the log carries (auto-discovered), in degC
  * GPS satellite count (and fix_type), a plain count
  * the RATE of change of one temperature channel, in degC/min

Each series has a checkbox; the panel on the left doubles as the legend (labels
are colored to match their line).  Armed stretches of the flight are shaded, so
"the IMU spiked at 30 min" immediately reads as "the IMU spiked mid-flight".

Navigation (no toolbar mode to arm first):
    wheel        zoom the time axis about the cursor
    ctrl+wheel   zoom the value axes    shift+wheel   zoom both
    drag         pan                    double-click  reset to the whole flight

Usage:
    ulog_graph.py <log.ulg>                       # interactive window
    ulog_graph.py <log.ulg> --list                # what channels are in there?
    ulog_graph.py <log.ulg> --save out.png        # headless render
    ulog_graph.py <log.ulg> --smooth 60 --abs     # tune the rate series
    ulog_graph.py <log.ulg> --add battery_status.voltage_v

Acronyms: FC = flight controller, IMU = inertial measurement unit,
GPS = global positioning system, ULog = PX4's binary log format.
Requires pyulog + matplotlib + numpy (all in the agc_CTOL_SE3-rotopy venv).
"""
import argparse
import os
import re
import sys

try:
    import numpy as np
    from pyulog import ULog
except ImportError as e:  # pragma: no cover - dependency guard
    sys.exit(f"needs numpy + pyulog: {e}  (pip install pyulog)")


# --- what we pull out of the log -------------------------------------------
# Only these topics are parsed.  pyulog's message_name_filter_list matters a lot
# here: a 137 MB / 43 min log loads in ~3 s filtered, versus reading every one of
# its ~100 topics.  Any topic whose fields include "temp" contributes temperature
# channels automatically, so a firmware that adds a new sensor needs no edit here.
TEMP_TOPICS = [
    "sensor_accel", "sensor_gyro", "sensor_baro", "sensor_mag",
    "vehicle_imu_status", "vehicle_air_data", "battery_status", "esc_status",
]
# vehicle_gps_position is the fused/primary GPS report (~10 Hz in PX4 1.14);
# sensor_gps is the raw driver output and is often logged at only 1 Hz.  Prefer
# the former for resolution, fall back so HITL/SITL logs still work.
GPS_TOPICS = ["vehicle_gps_position", "sensor_gps"]
ARMED_TOPIC = "actuator_armed"

# The cleanest channel to differentiate: vehicle_imu_status publishes the driver's
# *averaged* temperature (quantum ~1e-5 degC) whereas the raw sensor_* channels are
# quantized as coarsely as 0.125 degC, which a derivative turns into pure hash.
DEFAULT_RATE_SRC = "vehicle_imu_status[0].temperature_accel"

# --- color -----------------------------------------------------------------
# Temperatures share one graded warm family because they share the degC axis --
# they read as one group rather than 11 unrelated things.  Satellites and the
# rate series get contrasting categorical hues, and each y-axis label is painted
# to match its series so a line's axis is never ambiguous.  Because ~11 warm
# tones are not distinguishable by hue alone, the sensor family is also encoded
# in the line style (solid/dashed/dotted) -- identity is never color-only.
C_SATS = "#2a78d6"   # blue
C_FIX = "#4a3aa7"    # violet
C_RATE = "#1baf7a"   # aqua
C_ADD = "#e87ba4"    # magenta
C_ARMED = "#8c8c85"  # neutral -- background shading, never a data color
C_INK = "#0b0b0b"
C_MUTED = "#6f6e6a"
C_SURFACE = "#fcfcfb"
C_GRID = "#e4e3df"

# Line style per sensor family (the secondary encoding for the warm ramp).
FAMILY_STYLE = {"imu": "-", "baro": "--", "mag": ":", "other": "-."}


def _family(topic):
    if topic in ("sensor_accel", "sensor_gyro", "vehicle_imu_status"):
        return "imu"
    if topic in ("sensor_baro", "vehicle_air_data"):
        return "baro"
    if topic == "sensor_mag":
        return "mag"
    return "other"


class Series:
    """One toggleable line: its own time base, its own axis group.

    Every series carries its OWN t vector rather than sharing one x array.  The
    log's sample rates differ by 16x (temperatures ~1 Hz, GPS ~10 Hz, air data
    ~16 Hz), and resampling onto a common grid would invent satellite-count
    transitions that never happened.
    """

    def __init__(self, sid, label, t_min, y, group, color, ls="-", visible=False,
                 drawstyle="default", lw=2.0):
        self.id = sid              # canonical "topic[i].field", used by --rate-src
        self.label = label         # short form shown in the checkbox panel
        self.t = t_min             # minutes since log start
        self.y = y
        self.group = group         # temp | sats | rate | add
        self.color = color
        self.ls = ls
        self.visible = visible
        self.drawstyle = drawstyle
        self.lw = lw
        self.line = None


def _short(topic, mid, field):
    """Compact channel name for the checkbox panel."""
    t = topic.replace("vehicle_", "").replace("sensor_", "")
    f = field.replace("temperature_", "").replace("temperature", "")
    f = f.replace("baro_temp_celcius", "baro").replace("_celcius", "")
    name = f"{t}[{mid}]" if mid else t
    return f"{name} {f}".strip()


def _get(ulog, topic, mid=0):
    for d in ulog.data_list:
        if d.name == topic and d.multi_id == mid:
            return d
    return None


def _time_min(ulog, dataset):
    """Timestamps as minutes since the log's start.

    43 minutes of flight on a seconds axis means reading four-digit tick labels;
    minutes is how you'd actually describe an event ("half an hour in")."""
    t0 = getattr(ulog, "start_timestamp", 0) or 0
    return (np.asarray(dataset.data["timestamp"], dtype=float) - t0) / 6e7


def _clean(t, y):
    """Drop NaN samples (several PX4 temperature fields are published but never
    filled, so they arrive as all-NaN) and enforce monotonic time."""
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size > 1:                      # a logged topic can emit out-of-order
        order = np.argsort(t, kind="stable")
        t, y = t[order], y[order]
    return t, y


def sliding_slope(t_s, y, window_s):
    """Least-squares slope of y vs t over a centered window, per sample.

    Why not np.gradient: the raw sensor temperatures are quantized to as much as
    0.125 degC at 1 Hz, so a two-point difference is a +-56 degC/min square wave
    that buries the real signal (a 31 s fit gives +-27 degC/min on the same data
    while keeping the genuine ~50 degC/min cooling events).

    Why not scipy.signal.savgol_filter(..., deriv=1, delta=dt): savgol assumes a
    UNIFORM sample spacing, and this log's dt wanders 0.898..1.088 s -- roughly a
    10% error on the result.  The closed-form normal equation below uses the real
    timestamps, so non-uniform spacing is handled exactly:

        slope = (n*Sum(ty) - Sum(t)*Sum(y)) / (n*Sum(t^2) - Sum(t)^2)

    Each window's five sums come from prefix sums, so this is O(n) rather than
    O(n * window) -- and, more usefully, it is one expression you can check.
    """
    n = t_s.size
    if n < 2:
        return np.full(n, np.nan)
    # Prefix sums, padded with a leading 0 so a window [a, b) is just c[b] - c[a].
    cs_1 = np.arange(n + 1, dtype=float)
    cs_t = np.concatenate(([0.0], np.cumsum(t_s)))
    cs_y = np.concatenate(([0.0], np.cumsum(y)))
    cs_ty = np.concatenate(([0.0], np.cumsum(t_s * y)))
    cs_tt = np.concatenate(([0.0], np.cumsum(t_s * t_s)))

    half = window_s / 2.0
    a = np.searchsorted(t_s, t_s - half, side="left")
    b = np.searchsorted(t_s, t_s + half, side="right")

    cnt = cs_1[b] - cs_1[a]
    st = cs_t[b] - cs_t[a]
    sy = cs_y[b] - cs_y[a]
    sty = cs_ty[b] - cs_ty[a]
    stt = cs_tt[b] - cs_tt[a]

    den = cnt * stt - st * st
    num = cnt * sty - st * sy
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(np.abs(den) > 1e-12, num / den, np.nan)
    return slope


# --- channel discovery ------------------------------------------------------

def discover_temps(ulog):
    """Every (topic, multi_id, field) whose name mentions temperature.

    Returns (live, dead) -- dead ones are channels the firmware publishes but
    never populates, and there are four of them in a typical mRo ControlZero log
    (battery_status.temperature, sensor_gyro[1], sensor_mag[1],
    vehicle_imu_status[1].temperature_gyro).  They are hidden rather than shown
    greyed, but reported once so their absence is never a mystery.
    """
    live, dead = [], []
    for d in sorted(ulog.data_list, key=lambda x: (x.name, x.multi_id)):
        if d.name not in TEMP_TOPICS:
            continue
        for field in sorted(d.data):
            f = field.lower()
            # "contains temp" over-matches: vehicle_air_data.temperature_source is
            # an enum naming WHICH sensor supplied the reading, not a temperature,
            # and plotting it on a degC axis is meaningless.  Same shape of problem
            # for any *_valid / *_count companion field a future topic adds.
            if "temp" not in f or f.endswith(("_source", "_valid", "_count", "_id")):
                continue
            t, y = _clean(_time_min(ulog, d), d.data[field])
            sid = f"{d.name}[{d.multi_id}].{field}"
            if y.size == 0:
                dead.append(sid)
            else:
                live.append((sid, _short(d.name, d.multi_id, field),
                             _family(d.name), t, y))
    return live, dead


def parse_ref(ref):
    """Split 'topic[i].field' (or 'topic.field') into (topic, multi_id, field)."""
    m = re.match(r"^([A-Za-z0-9_]+)(?:\[(\d+)\])?\.([A-Za-z0-9_\[\]]+)$", ref.strip())
    if not m:
        raise ValueError(f"expected topic[i].field, got '{ref}'")
    return m.group(1), int(m.group(2) or 0), m.group(3)


def build_series(ulog, smooth_s, use_abs, rate_src, adds):
    """Assemble every plottable series, in panel order."""
    import matplotlib

    live, dead = discover_temps(ulog)
    series, notes = [], []
    if dead:
        notes.append(f"skipped {len(dead)} channel(s) with no data: "
                     + ", ".join(dead))
    if not live:
        notes.append("no temperature channels in this log")

    # Single-hue orange ramp, light->dark across the group.  Starts at 0.40 so
    # even the lightest step stays legible on a near-white surface, and stays in
    # ORANGE rather than running on into red: a dark red temperature line beside
    # the aqua rate line is the classic red/green pair that deuteranopes cannot
    # separate, and the rate series is the one you most need to read against the
    # temperature it came from.
    n = max(len(live), 1)
    ramp = matplotlib.colormaps["Oranges"](np.linspace(0.40, 0.95, n))

    # Which temperature drives the rate series?  Prefer the requested/default
    # averaged IMU channel; fall back to whatever the log actually has.
    ids = [s[0] for s in live]
    src = rate_src if rate_src in ids else (DEFAULT_RATE_SRC if DEFAULT_RATE_SRC in ids
                                            else (ids[0] if ids else None))
    if rate_src and rate_src not in ids:
        notes.append(f"--rate-src '{rate_src}' not in this log; using '{src}'")

    for i, (sid, label, fam, t, y) in enumerate(live):
        series.append(Series(sid, label, t, y, "temp",
                             tuple(ramp[i]), FAMILY_STYLE[fam],
                             visible=(sid == src)))

    # --- rate of change, derived from the primary temperature ---------------
    if src is not None:
        t, y = next((s[3], s[4]) for s in live if s[0] == src)
        # slope wants seconds (t is minutes for the x axis); *60 -> degC/min.
        rate = sliding_slope(t * 60.0, y, smooth_s) * 60.0
        if use_abs:
            rate = np.abs(rate)
        lbl = ("|dT/dt|" if use_abs else "dT/dt") + f" {_short(*parse_ref(src))}"
        series.append(Series("__rate__", lbl, t, rate, "rate", C_RATE,
                             visible=True))

    # --- GPS ----------------------------------------------------------------
    gps = next((_get(ulog, name) for name in GPS_TOPICS if _get(ulog, name)), None)
    if gps is None:
        notes.append("no GPS topic in this log (HITL/SITL?) -- satellites omitted")
    else:
        tg = _time_min(ulog, gps)
        if "satellites_used" in gps.data:
            t, y = _clean(tg, gps.data["satellites_used"])
            # Thinner than the analog series on purpose: this is a 10 Hz step
            # signal, and at 2 px a 0.1 s dropout draws a full-height bar that
            # reads as a sustained outage.  Real number for this log: only 0.8%
            # of armed samples sit at zero satellites.
            series.append(Series(f"{gps.name}.satellites_used", "satellites",
                                 t, y, "sats", C_SATS, visible=True,
                                 drawstyle="steps-post", lw=1.4))
        if "fix_type" in gps.data:
            t, y = _clean(tg, gps.data["fix_type"])
            # Sat count alone can't tell you the fix was usable -- 32 satellites
            # with fix_type 0 is a very different story from 32 with a 3D fix.
            series.append(Series(f"{gps.name}.fix_type", "fix_type", t, y,
                                 "sats", C_FIX, ls="--", drawstyle="steps-post",
                                 lw=1.4))

    # --- ad-hoc channels ----------------------------------------------------
    for ref in adds:
        try:
            topic, mid, field = parse_ref(ref)
        except ValueError as e:
            notes.append(str(e))
            continue
        d = _get(ulog, topic, mid)
        if d is None or field not in d.data:
            notes.append(f"--add '{ref}' not found in this log")
            continue
        t, y = _clean(_time_min(ulog, d), d.data[field])
        if y.size == 0:
            notes.append(f"--add '{ref}' has no data")
            continue
        series.append(Series(ref, ref, t, y, "add", C_ADD, ls="-"))

    return series, notes


def armed_spans(ulog):
    """[(t_start_min, t_end_min), ...] where actuator_armed.armed was true."""
    d = _get(ulog, ARMED_TOPIC)
    if d is None or "armed" not in d.data:
        return []
    t = _time_min(ulog, d)
    a = np.asarray(d.data["armed"], dtype=float) > 0.5
    spans, start = [], None
    for i in range(len(a)):
        if a[i] and start is None:
            start = t[i]
        elif not a[i] and start is not None:
            spans.append((start, t[i]))
            start = None
    if start is not None:
        spans.append((start, t[-1]))
    return spans


# --- rendering --------------------------------------------------------------

def _rescale(ax, lines, pad=0.06):
    """Fit an axis to only its VISIBLE lines (matplotlib's autoscale counts
    hidden artists, so a toggled-off 80 degC channel would keep the axis stretched)."""
    vals = [ln.get_ydata() for ln in lines if ln.get_visible()]
    vals = [v[np.isfinite(v)] for v in vals]
    vals = [v for v in vals if v.size]
    if not vals:
        return
    lo = min(float(v.min()) for v in vals)
    hi = max(float(v.max()) for v in vals)
    if hi == lo:
        lo, hi = lo - 1, hi + 1
    m = (hi - lo) * pad
    ax.set_ylim(lo - m, hi + m)


def _style_axis(ax, color):
    ax.tick_params(axis="y", colors=color, labelsize=8)
    ax.yaxis.label.set_color(color)


def add_mouse_navigation(fig, axes):
    """Wheel to zoom, drag to pan, double-click to reset -- without having to
    arm a mode on the toolbar first.

    Everything here operates on ALL the stacked y-axes at once.  They are twinx
    siblings, so they already share one x-axis (changing it on any of them moves
    every series together), but each keeps its own y scale -- so a pan has to be
    applied per axis, as a fraction of that axis's own range, or the curves would
    slide apart from each other.

    Modifiers on the wheel: none = time only (what you want 95% of the time,
    and it leaves the vertical scales you set by toggling series alone),
    ctrl = the value axes only, shift = both.
    """
    home = [(a, a.get_xlim(), a.get_ylim()) for a in axes]
    drag = {}

    def _live(event):
        """Ignore events over the checkbox panel, and stand down while a toolbar
        mode (pan/zoom) holds the canvas widget lock -- otherwise both would act
        on the same drag and the view would move twice as fast."""
        return event.inaxes in axes and not fig.canvas.widgetlock.locked()

    def on_scroll(event):
        if not _live(event):
            return
        # One notch = 15%.  event.step is +1 up / -1 down (fractional on
        # high-resolution trackpads, which this handles for free).
        scale = 1.15 ** (-event.step)
        key = event.key or ""
        do_x = key not in ("control", "ctrl+")
        do_y = "control" in key or "shift" in key

        if do_x:
            # Zoom about the cursor, not the axis centre, so the sample under the
            # pointer stays put.  x data coords are identical on every twin.
            x0, x1 = axes[0].get_xlim()
            xc = event.xdata
            axes[0].set_xlim(xc - (xc - x0) * scale, xc + (x1 - xc) * scale)
        if do_y:
            for a in axes:
                # The cursor's y in THIS axis's data coords -- the twins have
                # different scales, so each needs its own inverse transform.
                _, yc = a.transData.inverted().transform((event.x, event.y))
                y0, y1 = a.get_ylim()
                a.set_ylim(yc - (yc - y0) * scale, yc + (y1 - yc) * scale)
        fig.canvas.draw_idle()

    def on_press(event):
        if event.button != 1 or not _live(event):
            return
        if event.dblclick:                      # back to the full flight
            for a, xl, yl in home:
                a.set_xlim(xl)
                a.set_ylim(yl)
            drag.clear()
            fig.canvas.draw_idle()
            return
        # Freeze the press-time transforms: the scale doesn't change during a
        # pan, so converting the pixel delta with these stays exact and avoids
        # the feedback drift you get from re-reading the transform each motion.
        drag["at"] = (event.x, event.y)
        drag["axes"] = [(a, a.get_xlim(), a.get_ylim(),
                         a.transData.inverted()) for a in axes]

    def on_motion(event):
        if not drag or event.x is None:
            return
        px, py = drag["at"]
        for a, (x0, x1), (y0, y1), inv in drag["axes"]:
            xa, ya = inv.transform((px, py))
            xb, yb = inv.transform((event.x, event.y))
            dx, dy = xa - xb, ya - yb
            a.set_xlim(x0 + dx, x1 + dx)
            a.set_ylim(y0 + dy, y1 + dy)
        fig.canvas.draw_idle()

    def on_release(event):
        drag.clear()

    for name, fn in (("scroll_event", on_scroll), ("button_press_event", on_press),
                     ("motion_notify_event", on_motion),
                     ("button_release_event", on_release)):
        fig.canvas.mpl_connect(name, fn)


def render(ulog, series, path, notes, use_abs, smooth_s):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import CheckButtons

    fig = plt.figure(figsize=(15, 8), facecolor=C_SURFACE)
    if fig.canvas.manager is not None:          # None under the headless Agg backend
        fig.canvas.manager.set_window_title(f"logGraph - {os.path.basename(path)}")
    ax = fig.add_axes([0.20, 0.10, 0.53, 0.79], facecolor=C_SURFACE)

    # Several y scales, because 20-80 degC, 0-32 satellites and +-30 degC/min
    # cannot share one.  This is a deliberate exception to the usual "never
    # dual-axis" rule: it's a diagnostic overlay whose whole point is time
    # correlation, and every series can be switched off, which is the real
    # mitigation.  ALL of them are stacked on the RIGHT, innermost-first in the
    # order temperature / count / rate / added, so the scales read as one ladder
    # instead of one lone axis on the left that's easy to mis-attribute.
    ax_cnt = ax.twinx()
    ax_cnt.spines["right"].set_position(("outward", 56))
    ax_rate = ax.twinx()
    ax_rate.spines["right"].set_position(("outward", 112))
    ax_add = ax.twinx()
    ax_add.spines["right"].set_position(("outward", 172))
    ax_add.set_visible(False)
    for a in (ax_cnt, ax_rate, ax_add):
        a.set_facecolor("none")
        a.spines["left"].set_visible(False)
    # AFTER the twins: twinx() ends by calling self.yaxis.tick_left() on its
    # parent, so moving the base axis to the right any earlier gets silently
    # undone and the temperature ticks disappear.
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.spines["left"].set_visible(False)

    axis_of = {"temp": ax, "sats": ax_cnt, "rate": ax_rate, "add": ax_add}

    # Armed stretches, behind everything.
    spans = armed_spans(ulog)
    span_art = [ax.axvspan(a, b, color=C_ARMED, alpha=0.13, lw=0, zorder=0)
                for a, b in spans]

    if any(s.group == "rate" for s in series):
        ax_rate.axhline(0.0, color=C_MUTED, lw=1, ls=":", alpha=0.6, zorder=1)

    for s in series:
        # Step signals sit below the analog ones so a busy satellite trace never
        # hides the temperature curve you are trying to read it against.
        (line,) = axis_of[s.group].plot(
            s.t, s.y, color=s.color, ls=s.ls, lw=s.lw, label=s.label,
            drawstyle=s.drawstyle, zorder=2 if s.group == "sats" else 3,
        )
        line.set_visible(s.visible)
        s.line = line

    ax.set_xlabel("time since log start (minutes)", color=C_MUTED, fontsize=9)
    ax.set_ylabel("temperature (degC)", fontsize=9)
    ax_cnt.set_ylabel("satellites / fix_type", fontsize=9)
    ax_rate.set_ylabel(("|dT/dt|" if use_abs else "dT/dt") + " (degC/min)", fontsize=9)
    ax_add.set_ylabel("added channel", fontsize=9)
    _style_axis(ax, "#a33603")
    _style_axis(ax_cnt, C_SATS)
    _style_axis(ax_rate, C_RATE)
    _style_axis(ax_add, C_ADD)

    ax.grid(True, color=C_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top",):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", colors=C_MUTED, labelsize=8)

    dur = ulog.last_timestamp - ulog.start_timestamp
    temps = [s for s in series if s.group == "temp"]
    peak = max((float(np.nanmax(s.y)) for s in temps), default=float("nan"))
    fig.text(0.20, 0.955, os.path.basename(path), color=C_INK, fontsize=13,
             fontweight="bold", ha="left")
    fig.text(0.20, 0.925,
             f"{dur/6e7:.1f} min   |   {len(temps)} temperature channel(s), "
             f"peak {peak:.1f} degC   |   rate window {smooth_s:.0f} s",
             color=C_MUTED, fontsize=9, ha="left")

    # --- checkbox panel; it doubles as the legend --------------------------
    # Labels are painted in their line's color, so identity is carried without a
    # separate legend box eating plot area.
    entries = []          # (label, series_or_None, members)
    groups = [("temp", "ALL temperatures"), ("sats", "ALL gps"),
              ("rate", None), ("add", "ALL added")]
    for gname, master in groups:
        members = [s for s in series if s.group == gname]
        if not members:
            continue
        if master and len(members) > 1:
            entries.append((master, None, gname))
        for s in members:
            entries.append(("   " + s.label if master and len(members) > 1
                            else s.label, s, None))
    if span_art:
        entries.append(("armed (shaded)", None, "__armed__"))

    labels = [e[0] for e in entries]
    states = [bool(e[1].visible) if e[1] is not None
              else (True if e[2] == "__armed__" else False) for e in entries]
    h = min(0.86, 0.035 * len(labels) + 0.05)
    ax_cb = fig.add_axes([0.015, 0.89 - h, 0.19, h], facecolor=C_SURFACE)
    ax_cb.set_title("series", fontsize=9, color=C_MUTED, loc="left")
    for spine in ax_cb.spines.values():
        spine.set_visible(False)
    cb = CheckButtons(ax_cb, labels, states)
    for txt, e in zip(cb.labels, entries):
        txt.set_fontsize(8)
        txt.set_color(e[1].color if e[1] is not None else C_MUTED)

    guard = {"busy": False}

    def refresh():
        _rescale(ax, [s.line for s in series if s.group == "temp"])
        _rescale(ax_cnt, [s.line for s in series if s.group == "sats"])
        _rescale(ax_rate, [s.line for s in series if s.group == "rate"])
        add_lines = [s.line for s in series if s.group == "add"]
        ax_add.set_visible(any(ln.get_visible() for ln in add_lines))
        _rescale(ax_add, add_lines)
        fig.canvas.draw_idle()

    def on_click(label):
        if guard["busy"]:
            return
        i = labels.index(label)
        state = cb.get_status()[i]
        _, s, group = entries[i]
        if s is not None:
            s.line.set_visible(state)
        elif group == "__armed__":
            for art in span_art:
                art.set_visible(state)
        else:
            # Group master: drive the members' lines AND their check marks.  The
            # guard stops set_active's callback from re-entering this handler.
            guard["busy"] = True
            try:
                for j, (_, ms, _) in enumerate(entries):
                    if ms is not None and ms.group == group:
                        ms.line.set_visible(state)
                        if cb.get_status()[j] != state:
                            cb.set_active(j)
            finally:
                guard["busy"] = False
        refresh()

    cb.on_clicked(on_click)
    # A matplotlib widget whose only reference is a local gets garbage-collected
    # and silently stops responding -- keep it alive on the figure.
    fig._checkbuttons = cb
    refresh()
    # After refresh(), so "home" is the fitted view rather than the raw one.
    add_mouse_navigation(fig, [ax, ax_cnt, ax_rate, ax_add])
    fig.text(0.20, 0.02, "wheel: zoom time  ·  ctrl+wheel: zoom values  ·  "
                         "drag: pan  ·  double-click: reset",
             color=C_MUTED, fontsize=8, ha="left")

    for n in notes:
        print(f"  note: {n}")
    return fig


# --- entry points -----------------------------------------------------------

def list_channels(path):
    """Print what the log carries, without opening a window."""
    ulog = ULog(path, message_name_filter_list=TEMP_TOPICS + GPS_TOPICS + [ARMED_TOPIC])
    live, dead = discover_temps(ulog)
    print(f"=== {os.path.basename(path)} "
          f"({(ulog.last_timestamp - ulog.start_timestamp)/6e7:.1f} min) ===")
    print("temperature channels:")
    for sid, label, fam, t, y in live:
        print(f"  {sid:48s} n={y.size:6d}  {y.min():7.2f}..{y.max():7.2f} degC"
              f"   [{fam}]")
    for sid in dead:
        print(f"  {sid:48s} -- no data")
    gps = next((_get(ulog, n) for n in GPS_TOPICS if _get(ulog, n)), None)
    if gps is None:
        print("gps: none")
    else:
        s = np.asarray(gps.data.get("satellites_used", []), dtype=float)
        f = np.asarray(gps.data.get("fix_type", []), dtype=float)
        print(f"gps: {gps.name}  n={len(gps.data['timestamp'])}"
              + (f"  satellites {s.min():.0f}..{s.max():.0f}" if s.size else "")
              + (f"  fix_type {f.min():.0f}..{f.max():.0f}" if f.size else ""))
    print(f"armed spans: {len(armed_spans(ulog))}")


def _has_display():
    """True if a GUI window can plausibly be opened.  Deliberately not
    WSL-specific -- this tool is run from Windows as well as from WSL."""
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def graph(path, smooth=31.0, use_abs=False, rate_src=None, adds=(), save=None,
          show=True):
    """Build (and show/save) the figure.  Factored out of main() so the mavTerminal
    shell can call it directly, the same way it calls ulog_diag.diagnose."""
    import matplotlib
    if save and not _has_display():
        matplotlib.use("Agg")           # render to file with no window manager
    elif not _has_display() and show:
        sys.exit("No display available. Re-run with --save <file.png>.")
    import matplotlib.pyplot as plt

    topics = TEMP_TOPICS + GPS_TOPICS + [ARMED_TOPIC]
    for ref in adds:
        try:
            topics.append(parse_ref(ref)[0])
        except ValueError:
            pass
    ulog = ULog(path, message_name_filter_list=sorted(set(topics)))

    series, notes = build_series(ulog, smooth, use_abs, rate_src, list(adds))
    if not series:
        sys.exit("Nothing plottable in this log (no temperature or GPS topics).")
    fig = render(ulog, series, path, notes, use_abs, smooth)

    if save:
        fig.savefig(save, dpi=130, facecolor=C_SURFACE)
        print(f"  saved {save}")
    if show and _has_display():
        plt.show()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Plot a PX4 .ulg as toggleable time series")
    ap.add_argument("log", help="path to a .ulg file")
    ap.add_argument("--list", action="store_true",
                    help="print the channels this log carries and exit")
    ap.add_argument("--save", metavar="PNG", help="also render to a PNG file")
    ap.add_argument("--smooth", type=float, default=31.0, metavar="SEC",
                    help="rate-of-change fit window in seconds (default 31)")
    ap.add_argument("--abs", dest="use_abs", action="store_true",
                    help="plot |dT/dt| instead of the signed rate")
    ap.add_argument("--rate-src", metavar="TOPIC[i].FIELD",
                    help=f"temperature channel to differentiate (default {DEFAULT_RATE_SRC})")
    ap.add_argument("--add", action="append", default=[], metavar="TOPIC[i].FIELD",
                    help="plot an extra channel on its own axis (repeatable)")
    ap.add_argument("--no-show", action="store_true",
                    help="don't open a window (use with --save)")
    a = ap.parse_args()

    if not os.path.isfile(a.log):
        sys.exit(f"no such file: {a.log}")
    if a.list:
        list_channels(a.log)
        return
    graph(a.log, smooth=a.smooth, use_abs=a.use_abs, rate_src=a.rate_src,
          adds=a.add, save=a.save, show=not a.no_show)


if __name__ == "__main__":
    main()
