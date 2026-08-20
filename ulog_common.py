#!/usr/bin/env python3
"""ulog_common.py -- pieces shared by every .ulg plot in this toolkit.

Extracted from ulog_graph.py when a second plot (altitude estimation) arrived and
the thermal plot stopped being the only customer.  Nothing in here knows what is
being plotted; it is the vocabulary the individual plot modules are written in:

  * Series          one toggleable line, carrying its own time base
  * _get/_clean/... the pyulog access helpers
  * sliding_slope   non-uniform-timestamp derivative
  * armed_spans     the armed shading every plot draws
  * check_panel     the checkbox panel that doubles as the legend
  * add_mouse_navigation  wheel/drag navigation, and the cross-plot time link

Acronyms: ULog = PX4's binary log format, GPS = global positioning system,
EKF = extended Kalman filter, AMSL = above mean sea level.
"""
import re
from dataclasses import dataclass, field as _dc_field

import numpy as np


@dataclass
class PlotCtx:
    """Per-run options handed to every plot builder.

    One object rather than a growing argument list, because the browser, the PDF
    exporter and the CLI all have to pass the same thing through to builders they
    don't know the signatures of.  Builders read only the keys they care about;
    the thermal plot uses smooth/use_abs/rate_src/adds, the altitude plot uses
    debias/page_scroll.
    """
    smooth: float = 31.0           # thermal: rate-of-change fit window, seconds
    use_abs: bool = False          # thermal: plot |dT/dt|
    rate_src: str = None           # thermal: which temperature to differentiate
    adds: list = _dc_field(default_factory=list)   # thermal: extra channels
    debias: bool = True            # altitude: remove the constant datum offset
    page_scroll: bool = False      # True when hosted in the browser's scroll page
    notes: list = _dc_field(default_factory=list)  # builders append; host prints

    def note(self, msg):
        self.notes.append(msg)

# --- color ------------------------------------------------------------------
# One palette for the whole toolkit so a signal keeps its identity across plots
# (armed shading is the same grey in the thermal and altitude plots, and no data
# series is ever allowed to use it).
C_SATS = "#2a78d6"   # blue
C_FIX = "#4a3aa7"    # violet
C_RATE = "#1baf7a"   # aqua
C_ADD = "#e87ba4"    # magenta
C_ARMED = "#8c8c85"  # neutral -- background shading, never a data color
C_INK = "#0b0b0b"
C_MUTED = "#6f6e6a"
C_SURFACE = "#fcfcfb"
C_GRID = "#e4e3df"
C_BAD = "#c0392b"    # red -- faults and rejections only, never a normal signal

# Line style per sensor family (the secondary encoding for the thermal plot's
# warm ramp -- ~11 orange tones are not separable by hue alone).
FAMILY_STYLE = {"imu": "-", "baro": "--", "mag": ":", "other": "-."}

ARMED_TOPIC = "actuator_armed"


class Series:
    """One toggleable line: its own time base, its own axis group.

    Every series carries its OWN t vector rather than sharing one x array.  Sample
    rates in a single log differ by nearly two orders of magnitude (temperatures
    ~1 Hz, GPS ~10 Hz, distance_sensor 85 Hz), and resampling onto a common grid
    would invent transitions -- a satellite count or a validity flag that never
    actually changed.
    """

    def __init__(self, sid, label, t_min, y, group, color, ls="-", visible=False,
                 drawstyle="default", lw=2.0, alpha=1.0, zorder=None):
        self.id = sid              # canonical "topic[i].field"
        self.label = label         # short form shown in the checkbox panel
        self.t = t_min             # minutes since log start
        self.y = y
        self.group = group         # plot-defined axis group
        self.color = color
        self.ls = ls
        self.visible = visible
        self.drawstyle = drawstyle
        self.lw = lw
        self.alpha = alpha
        self.zorder = zorder
        self.line = None


# --- pyulog access ----------------------------------------------------------

def _get(ulog, topic, mid=0):
    """The dataset for topic[mid], or None if this log doesn't carry it."""
    for d in ulog.data_list:
        if d.name == topic and d.multi_id == mid:
            return d
    return None


def has_topic(ulog, topic):
    return any(d.name == topic for d in ulog.data_list)


def _time_min(ulog, dataset):
    """Timestamps as minutes since the log's start.

    43 minutes of flight on a seconds axis means reading four-digit tick labels;
    minutes is how you'd actually describe an event ("half an hour in")."""
    t0 = getattr(ulog, "start_timestamp", 0) or 0
    return (np.asarray(dataset.data["timestamp"], dtype=float) - t0) / 6e7


def _clean(t, y):
    """Drop NaN samples (several PX4 fields are published but never filled, so
    they arrive as all-NaN) and enforce monotonic time."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size > 1:                      # a logged topic can emit out-of-order
        order = np.argsort(t, kind="stable")
        t, y = t[order], y[order]
    return t, y


def field(ulog, topic, name, mid=0, scale=1.0):
    """(t_min, y) for one field, cleaned and scaled.  Empty arrays if absent.

    The empty-array return is deliberate: callers build a whole panel out of
    fields that may or may not exist in a given firmware, and forcing each one to
    guard with has_topic() first would triple the size of every plot module."""
    d = _get(ulog, topic, mid)
    if d is None or name not in d.data:
        return np.array([]), np.array([])
    t, y = _clean(_time_min(ulog, d), d.data[name])
    return t, y * scale


def parse_ref(ref):
    """Split 'topic[i].field' (or 'topic.field') into (topic, multi_id, field)."""
    m = re.match(r"^([A-Za-z0-9_]+)(?:\[(\d+)\])?\.([A-Za-z0-9_\[\]]+)$", ref.strip())
    if not m:
        raise ValueError(f"expected topic[i].field, got '{ref}'")
    return m.group(1), int(m.group(2) or 0), m.group(3)


def primary_ekf(ulog):
    """Which EKF instance the selector was using (the log carries three).

    PX4 runs one estimator per IMU and arbitrates between them; plotting
    instance 0 unconditionally can show you the innovations of a filter that was
    never steering the vehicle.  Falls back to 0 when the selector topic is
    absent (HITL logs don't carry it) -- with a single instance that IS the
    primary, so the fallback is correct rather than merely convenient."""
    d = _get(ulog, "estimator_selector_status")
    if d is None or "primary_instance" not in d.data:
        return 0, False
    v = np.asarray(d.data["primary_instance"], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0, False
    # The mode, not the last value: a momentary switch at the end of the log
    # shouldn't relabel the whole flight.
    vals, counts = np.unique(v.astype(int), return_counts=True)
    return int(vals[np.argmax(counts)]), len(vals) > 1


def duration_min(ulog):
    return (ulog.last_timestamp - ulog.start_timestamp) / 6e7


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


def gap_mask(t, y, valid, max_gap_min=None):
    """Blank out samples where `valid` is false by writing NaN.

    NaN rather than dropping the samples, because matplotlib draws a BREAK at a
    NaN but draws a straight line across a removed sample -- and a straight line
    across a rangefinder dropout is a reading that never happened.  In
    d05a88e3 the rangefinder is valid only 33% of the time, so this is the
    difference between an honest trace and a fabricated one."""
    y = np.asarray(y, dtype=float).copy()
    y[~np.asarray(valid, dtype=bool)] = np.nan
    if max_gap_min is not None and t.size > 1:
        # Also break across long sample gaps (a topic that stopped publishing).
        big = np.flatnonzero(np.diff(t) > max_gap_min)
        y[big] = np.nan
    return y


def resample_to(t_dst, t_src, y_src):
    """y_src evaluated at t_dst, with NaN outside the source's time range.

    np.interp clamps to the end values outside the range, which would silently
    invent a flat reference before the source topic started publishing.  Every
    residual in the altitude plot is a difference against a resampled series, so
    an invented reference becomes an invented error."""
    t_dst = np.asarray(t_dst, dtype=float)
    if t_src.size == 0 or t_dst.size == 0:
        return np.full(t_dst.shape, np.nan)
    out = np.interp(t_dst, t_src, y_src)
    out[(t_dst < t_src[0]) | (t_dst > t_src[-1])] = np.nan
    return out


# --- rendering helpers ------------------------------------------------------

def window_values(ax, lines, positive_only=False):
    """Y values of the VISIBLE lines that fall inside ax's current TIME window.

    Two filters, and both matter:

    Visible-only, because matplotlib's autoscale counts hidden artists, so a
    toggled-off 80 degC channel would keep the axis stretched.

    Window-only, because otherwise zooming the time axis does not change the
    picture.  The local-z plot is the case that proves it: its full-flight y
    range is set by one 89 m reset step and a 30 m origin split, so every
    zoomed-in window renders as the same flat line in the same place, and the
    zoom looks like it did nothing.  Fitting the value axis to what is actually
    on screen is what makes the gesture do something.

    A binary search rather than a boolean mask: this runs on every wheel notch,
    and the accel plot carries ~50 lines of a couple hundred thousand samples
    each.  Every series here is time-sorted (field() cleans through _clean), and
    the endpoint check falls back to a mask if some caller ever passes one that
    is not.  One sample of overhang each side keeps a line that crosses the edge
    of the window contributing, so the trace does not get cropped at the frame.
    """
    lo, hi = ax.get_xlim()
    if lo > hi:
        lo, hi = hi, lo
    out = []
    for ln in lines:
        if not ln.get_visible():
            continue
        x = np.asarray(ln.get_xdata(), dtype=float)
        v = np.asarray(ln.get_ydata(), dtype=float)
        if v.size == 0 or x.size != v.size:
            continue
        if x.size > 1 and x[0] <= x[-1]:
            i0, i1 = np.searchsorted(x, (lo, hi))
            v = v[max(int(i0) - 1, 0):min(int(i1) + 1, v.size)]
        else:
            v = v[(x >= lo) & (x <= hi)]
        v = v[np.isfinite(v)]
        if positive_only:
            v = v[v > 0]
        if v.size:
            out.append(v)
    return np.concatenate(out) if out else np.array([])


def _rescale(ax, lines, pad=0.06):
    """Fit an axis to its visible lines, within the visible time window."""
    v = window_values(ax, lines)
    if v.size == 0:
        return
    lo, hi = float(v.min()), float(v.max())
    if hi == lo:
        lo, hi = lo - 1, hi + 1
    m = (hi - lo) * pad
    ax.set_ylim(lo - m, hi + m)


def _style_axis(ax, color):
    ax.tick_params(axis="y", colors=color, labelsize=8)
    ax.yaxis.label.set_color(color)


def style_time_axis(ax, label=True):
    ax.grid(True, color=C_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", colors=C_MUTED, labelsize=8)
    if label:
        ax.set_xlabel("time since log start (minutes)", color=C_MUTED, fontsize=9)


def draw_armed(ax, spans):
    """Armed stretches, behind everything.  Returns the artists so a checkbox
    can toggle them."""
    return [ax.axvspan(a, b, color=C_ARMED, alpha=0.13, lw=0, zorder=0)
            for a, b in spans]


# --- EKF instances: one shared vocabulary -----------------------------------
# PX4 runs one EKF per IMU (EKF2_MULTI_IMU) and an EKF2Selector arbitrates which
# one publishes.  Several plots need to say "this stretch belonged to instance
# 2", and they must agree on the hue or the reader learns the mapping twice.
#
# Deliberately NOT any plot's per-SOURCE palette -- these are instances, not
# sensors, and giving instance 0 the same blue as GPS would invite reading two
# plots as if the colors meant the same thing.
INST_COLORS = ["#2a78d6",   # 0  blue
               "#d2691e",   # 1  orange
               "#1baf7a",   # 2  aqua
               "#8a7fb5"]   # 3  violet


def inst_color(i):
    return INST_COLORS[int(i) % len(INST_COLORS)]


def primary_spans(ulog):
    """[(t0_min, t1_min, instance), ...] -- which EKF instance was primary, when.

    The selector publishes at ~1 Hz AND immediately on any change
    (EKF2Selector.cpp:811), so a value holds from its own sample until the next
    one that differs.  Treating each sample as an instantaneous point instead
    would leave 1 s gaps in the shading that look like the vehicle had no
    estimator at all.
    """
    d = _get(ulog, "estimator_selector_status")
    if d is None or "primary_instance" not in d.data:
        return []
    t = _time_min(ulog, d)
    v = np.asarray(d.data["primary_instance"], dtype=float)
    m = np.isfinite(t) & np.isfinite(v)
    t, v = t[m], v[m].astype(int)
    if t.size == 0:
        return []
    spans, start, cur = [], t[0], v[0]
    for i in range(1, t.size):
        if v[i] != cur:
            spans.append((start, t[i], int(cur)))
            start, cur = t[i], v[i]
    spans.append((start, t[-1], int(cur)))
    return spans


def draw_primary_shading(ax, spans, alpha=0.13):
    """The coloured background.  Returns the artists so a checkbox can hide them.

    zorder 0 and no edge line: this is a background, and an edge on abutting
    spans draws a vertical rule at every handover that reads as a data event.
    """
    return [ax.axvspan(a, b, color=inst_color(inst), alpha=alpha, lw=0, zorder=0)
            for a, b, inst in spans]


def instance_key(fig, left, width, spans, y=0.952, pitch=0.040):
    """Name the SHADING colours, right-aligned on the title row.

    Figure text rather than a legend box so it costs no plot area.  The checkbox
    panel already carries the LINE colours; this exists for the shading, which
    has no checkbox of its own per instance.  A line of its own would sit on top
    of the first panel, where it collides with rotated event labels.
    """
    if not spans:
        return left + width
    x = left + width
    for i in sorted({i for _, _, i in spans}, reverse=True):
        fig.text(x, y, f"EKF {i}", color=inst_color(i), fontsize=8,
                 ha="right", fontweight="bold")
        x -= pitch
    fig.text(x, y, "shading:", color=C_MUTED, fontsize=8, ha="right")
    # The x it stopped at, so a second key can be chained to its left on the same
    # row rather than fighting the subtitle for space below it.
    return x - 0.055


# --- flight mode -------------------------------------------------------------
# Named the way the ground station names them, not the way the enum does:
# "Hold" is what QGC calls AUTO_LOITER and what the pilot pressed, and a plot the
# pilot has to translate is a plot they will misread.
#
# Only the codes that mean the same thing across PX4 v1.14 and v1.18 are named.
# 6, 8 and 9 were free/unused before v1.15 and mean POSITION_SLOW /
# ALTITUDE_CRUISE / (free) after it, so naming them would be wrong on half this
# project's logs -- they fall through to "mode N", which is never wrong.
NAV_STATE_NAMES = {
    0: "Manual", 1: "Altitude", 2: "Position", 3: "Mission", 4: "Hold",
    5: "Return", 10: "Acro", 12: "Descend", 13: "Termination", 14: "Offboard",
    15: "Stabilized", 17: "Takeoff", 18: "Land", 19: "Follow", 20: "Precland",
    21: "Orbit", 22: "VTOL Takeoff",
}
# One hue per MODE, because the name cannot always be shown.  Labels are dropped
# whenever two changes fall close together (d05a88e3 changes mode 53 times in 15
# minutes), and an unlabelled rule is a line you cannot identify -- which is the
# one thing this overlay exists to prevent.  Colour identifies every rule, at
# every zoom, and the key below names the colours once.
#
# Grouped by what the pilot did: assisted modes cool, auto modes warm, the
# unusual ones (Offboard, Termination) loud.  Red is reserved for Termination,
# so it keeps the meaning it has everywhere else in this toolkit.
MODE_COLORS = {
    0:  "#8d6e63",   # Manual        brown
    1:  "#26a69a",   # Altitude      teal
    2:  "#3f51b5",   # Position      indigo
    3:  "#43a047",   # Mission       green
    4:  "#fb8c00",   # Hold          orange
    5:  "#8e24aa",   # Return        purple
    10: "#6d4c41",   # Acro          dark brown
    12: "#00838f",   # Descend       dark cyan
    13: "#c0392b",   # Termination   RED
    14: "#d81b60",   # Offboard      magenta
    15: "#7cb342",   # Stabilized    light green
    17: "#00acc1",   # Takeoff       cyan
    18: "#5e35b1",   # Land          deep purple
    19: "#546e7a",   # Follow        slate
    20: "#3949ab",   # Precland      dark indigo
    21: "#f4511e",   # Orbit         deep orange
    22: "#00897b",   # VTOL Takeoff  dark teal
}
# For codes with no assigned hue (firmware-specific or future).  Neutral rather
# than a recycled colour, so an unknown mode never impersonates a known one.
C_MODE = "#6a4fa3"


def mode_color(code):
    return MODE_COLORS.get(int(code), C_MODE)


def nav_state_name(code):
    return NAV_STATE_NAMES.get(int(code), f"mode {int(code)}")


def mode_changes(ulog):
    """[(t_min, nav_state)] -- the first sample, then every transition.

    The first sample is included deliberately: without it a log that never
    changes mode draws nothing, and "no lines" would read as "no data" rather
    than "one mode the whole way".
    """
    d = _get(ulog, "vehicle_status")
    if d is None or "nav_state" not in d.data:
        return []
    t = _time_min(ulog, d)
    v = np.asarray(d.data["nav_state"], dtype=float)
    m = np.isfinite(t) & np.isfinite(v)
    t, v = t[m], v[m].astype(int)
    if t.size == 0:
        return []
    idx = np.concatenate([[0], np.flatnonzero(np.diff(v) != 0) + 1])
    return [(float(t[i]), int(v[i])) for i in idx]


def draw_mode_changes(axes, changes, text_ax=None, min_gap=0.0, label=True):
    """A rule at every mode change on every axis, coloured by the mode it starts.

    Returns (artists, codes_seen) so the caller can hang a checkbox on the
    artists and name the colours with mode_key().

    Every axis gets the RULE so a feature in any panel can be lined up against
    the mode it happened in, but only one gets the TEXT -- repeating the name on
    five panels is five times the ink for no extra information, and on the panels
    with their own annotations it collides with them.

    `min_gap` suppresses a label too close to the previous labelled one, in x
    units.  Measured need: d05a88e3 changes mode 53 times in 15 minutes, mostly
    Position -> Hold -> Offboard flicker a few seconds apart, and unfiltered
    those labels overprint into a solid bar of text.  The rule is still drawn for
    every change, in that mode's own colour -- which is why dropping the label
    costs nothing but convenience.
    """
    artists, last_label, codes = [], None, []
    text_ax = text_ax if text_ax is not None else (axes[0] if axes else None)
    for t, code in changes:
        color = mode_color(code)
        if code not in codes:
            codes.append(int(code))
        for ax in axes:
            artists.append(ax.axvline(t, color=color, lw=1.1, ls="--",
                                      alpha=0.7, zorder=1.5))
        if text_ax is None or not label:
            continue
        if last_label is not None and (t - last_label) < min_gap:
            continue
        last_label = t
        txt = text_ax.text(
            t, 0.985, " " + nav_state_name(code),
            transform=text_ax.get_xaxis_transform(), rotation=90, fontsize=6.5,
            color=color, va="top", ha="left", zorder=6,
            bbox=dict(facecolor=C_SURFACE, edgecolor="none", pad=0.8, alpha=0.7))
        # Text is NOT clipped to its axes by default, unlike a Line2D.  Zoom in
        # and every label for a change outside the window keeps drawing -- over
        # the y tick labels, over the checkbox panel, over the neighbouring
        # figure.  The rules themselves clip already, being lines.
        txt.set_clip_on(True)
        artists.append(txt)
    return artists, codes


def mode_key(fig, x_right, y, codes, fontsize=8, pitch=0.011):
    """Name the mode COLOURS once, right-aligned, as figure text.

    Returns the artists so they toggle with the rules -- a key to lines that are
    switched off is worse than no key.

    Laid out right to left so the row ends flush at `x_right` whatever it
    contains; `pitch` is per character, since the names differ in length and a
    fixed column would either overlap or leave a gap.
    """
    if not codes:
        return []
    artists, x = [], x_right
    for code in reversed(codes):
        name = nav_state_name(code)
        artists.append(fig.text(x, y, name, color=mode_color(code),
                                fontsize=fontsize, ha="right",
                                fontweight="bold"))
        x -= (len(name) + 1.6) * pitch
    artists.append(fig.text(x, y, "modes:", color=C_MUTED, fontsize=fontsize,
                            ha="right"))
    return artists


# --- boolean bands ----------------------------------------------------------

def spans_from_bool(t, ok):
    """[(t0, t1), ...] for each contiguous true run of `ok`."""
    ok = np.asarray(ok, dtype=bool)
    out, start = [], None
    for i in range(ok.size):
        if ok[i] and start is None:
            start = t[i]
        elif not ok[i] and start is not None:
            out.append((start, t[i]))
            start = None
    if start is not None and t.size:
        out.append((start, t[-1]))
    return out


def _row_lanes(row):
    """Normalise a row to (label, [(spans, color), ...], label_color).

    Two forms are accepted because two kinds of row exist.  `(label, spans,
    color)` is one bar per row -- what the altitude and local-z plots want, where
    the row IS the fact.  `(label, [(spans, color), ...], label_color)` splits a
    row into parallel lanes, which is what a per-instance condition needs: the
    row is the condition and each lane is one EKF, so a lane with nothing in it
    says "checked, this instance was clean" instead of vanishing.
    """
    label, data, color = row
    lanes_form = bool(data) and isinstance(data[0], (list, tuple)) and \
        len(data[0]) == 2 and isinstance(data[0][0], (list, tuple))
    if lanes_form:
        return label, [(list(sp), c) for sp, c in data], color
    return label, [(list(data), color)], None


def draw_band_rows(ax, rows, ylabel="", empty_msg="nothing to show",
                   min_width=0.0, track=False):
    """Render stacked boolean rows, top row first.  See _row_lanes for the forms.

    Filled bars rather than lines because every row is boolean -- a line between
    0 and 1 implies intermediate values that do not exist.

    `min_width` widens any bar narrower than it, in the axis's own x units.  A
    fault that lasts one 30 Hz sample is a real event and a 0.0005-minute bar is
    an invisible one; without a floor the panel would say "clean" about a log
    that was not.  Applied per bar, so it never merges two separate events.

    `track` draws a faint full-width rail behind every lane.  Without it an empty
    lane and an absent lane look identical -- blank -- and "this instance was
    checked and was fine" is exactly the thing a fault panel must not leave to
    inference.
    """
    if not rows:
        ax.text(0.5, 0.5, empty_msg, transform=ax.transAxes, ha="center",
                va="center", color=C_MUTED, fontsize=9)
        ax.set_yticks([])
        return
    rows = [_row_lanes(r) for r in rows]

    # The x extent the labels have to share with the bars.  Taken from the rows
    # themselves rather than from the axis, because draw_band_rows runs before
    # the shared x limits are settled.
    all_spans = [sp for _l, lanes, _c in rows for spans, _lc in lanes for sp in spans]
    t_lo = min(a for a, _b in all_spans) if all_spans else 0.0
    t_hi = max(b for _a, b in all_spans) if all_spans else 1.0
    span_x = max(t_hi - t_lo, 1e-9)

    multi = any(len(lanes) > 1 for _l, lanes, _c in rows)
    for i, (label, lanes, label_color) in enumerate(rows):
        y = len(rows) - 1 - i
        n = max(len(lanes), 1)
        # A one-lane row keeps the label centred on its bar, which is what the
        # altitude and local-z panels already read as normal.  A multi-lane row
        # CANNOT: with an odd lane count the middle lane sits exactly on the row
        # centre, so the label's background box hides it -- on the d05a88e3 log
        # that silently erased instance 1 from the driver-error row, the one row
        # where all three instances had something to say.
        row_h = 0.62 if n == 1 else 0.52
        lane_h = row_h / n
        for j, (spans, color) in enumerate(lanes):
            # Lanes run top-down in the order given, so instance 0 is the top
            # lane of every row and the panel can be read across.
            ly = y + row_h / 2 - (j + 0.5) * lane_h
            if track:
                ax.barh(ly, span_x, left=t_lo, height=lane_h * 0.88,
                        color=C_GRID, alpha=0.9, lw=0, zorder=1)
            for a, b in spans:
                ax.barh(ly, max(b - a, min_width, 1e-6), left=a,
                        height=lane_h * 0.88, color=color, alpha=0.9, lw=0,
                        zorder=3)
        # Labels go INSIDE the axes, not on the y ticks: these names run to ~22
        # characters and as tick labels they extend left into the checkbox panel
        # and get clipped by the figure edge.  Anchored in axes coordinates they
        # also survive a time-axis zoom, which data coordinates would not.
        row_spans = [sp for spans, _c in lanes for sp in spans]
        covered = sum(b - a for a, b in row_spans) / span_x
        # A dense row (armed, "fusing GPS") has no free side, so leave it on the
        # left where every other plot already puts it.  A sparse row gets the
        # emptier third: its handful of bars are the only thing on the row, and
        # the label's background box would hide them.
        if covered < 0.2 and row_spans:
            def _in(lo, hi):
                return sum(max(0.0, min(b, hi) - max(a, lo)) for a, b in row_spans)
            third = span_x / 3.0
            right = _in(t_hi - third, t_hi) < _in(t_lo, t_lo + third)
        else:
            right = False
        # Just clear of its own top lane, NOT centred in the gap: at the halfway
        # point a title is equidistant from the lanes above and below it and
        # reads as belonging to either.
        ax.text(0.996 if right else 0.004, y if n == 1 else y + row_h / 2 + 0.11,
                label, transform=ax.get_yaxis_transform(which="grid"),
                fontsize=7, color=label_color or C_MUTED, va="center",
                ha="right" if right else "left", zorder=5,
                bbox=dict(facecolor=C_SURFACE, edgecolor="none", pad=1.0,
                          alpha=0.75))
    ax.set_yticks([])
    # Headroom for the top row's label when it sits above its lanes.
    ax.set_ylim(-0.6, len(rows) - (0.15 if multi else 0.4))
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=C_MUTED)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


# Blank rows inserted between one graph's series and the next in the checkbox
# panel.  In row units, so it scales with whatever height the caller gave the
# panel.
GROUP_GAP_ROWS = 0.8


def _block_layout(n, breaks, gap, anchors):
    """Row centres, in axes fraction, for every checkbox entry.

    Without `anchors` this is matplotlib's uniform spacing plus a blank row
    between groups.  With them, each group's block is slid to sit beside the
    graph it annotates -- which is the only thing that reliably answers "which
    legend goes with which panel" on a five-panel figure.  Blocks are placed
    top-down and pushed apart on collision, so a group too tall for its panel
    displaces the ones below it rather than overlapping them.

    Returns (ys, total_rows) or None when the anchored layout does not fit, in
    which case the caller falls back to the compact one.
    """
    starts = sorted(b for b in breaks if b < n)
    blocks = []                     # (start, count, anchor or None)
    for k, st in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else n
        blocks.append((st, end - st, anchors.get(st) if anchors else None))
    if not blocks:
        return None

    total_rows = n + gap * (len(blocks) - 1) + 1.0
    pitch = 1.0 / total_rows
    if anchors is None or not any(a is not None for _s, _c, a in blocks):
        return None, total_rows

    tops, prev_bottom = [], 1.0 - 0.5 * pitch
    for _st, count, anchor in blocks:
        h = count * pitch
        want = (anchor + h / 2) if anchor is not None else prev_bottom
        top = min(want, prev_bottom)
        tops.append(top)
        prev_bottom = top - h - gap * pitch
    if prev_bottom < -0.5 * pitch:          # ran off the bottom
        return None, total_rows

    ys = [0.0] * n
    for (st, count, _a), top in zip(blocks, tops):
        for k in range(count):
            ys[st + k] = top - (k + 0.5) * pitch
    return ys, total_rows


def _respace_checkbuttons(cb, breaks, gap=GROUP_GAP_ROWS, anchors=None):
    """Re-lay-out a CheckButtons so each graph's series sit in their own block.

    matplotlib spaces rows uniformly (widgets.py: ``ys = linspace(1, 0, n+2)``),
    so a panel listing four groups reads as one undifferentiated column of twenty
    labels and you have to parse the text to find where one graph's series end
    and the next begin.

    There is no public API for row positions, so this moves the label Text
    artists and, depending on the matplotlib generation, either the two scatter
    collections (3.7+) or the Rectangle/Line2D pairs (3.6).  Hit-testing follows
    for free in both: the click handler resolves against exactly those artists.

    `breaks` is the set of entry indices that START a group.  `anchors` optionally
    maps such an index to the y (axes fraction) its block should centre on.
    Private-attribute access is guarded: a matplotlib that renames these costs
    the spacing, not the panel.
    """
    n = len(cb.labels)
    if n == 0 or not breaks:
        return
    laid = _block_layout(n, breaks, gap, anchors)
    if laid and laid[0] is not None:
        ys, total = laid
    else:
        total = laid[1] if laid else (n + 1.0)
        cum, centers = 0.5, []      # 0.5 = half-row margin above the first row
        for i in range(n):
            if i in breaks and i > 0:
                cum += gap
            centers.append(cum + 0.5)
            cum += 1.0
        total = cum + 0.5           # matching half-row margin below the last row
        ys = [1.0 - c / total for c in centers]

    for text, y in zip(cb.labels, ys):
        text.set_position((text.get_position()[0], y))

    # The glyphs have to follow the labels, and the two matplotlib generations in
    # play store them differently.  3.7+ keeps two scatter collections
    # (`_frames`, `_checks`); 3.6 keeps a Rectangle plus a pair of Line2D per row
    # (`rectangles`, `lines`).  Moving only the labels -- which is what this
    # function did before the accel plot exposed it -- leaves 3.6's boxes at
    # matplotlib's original uniform positions, so on a grouped panel the box and
    # its label drift apart and `_clicked` matches the wrong row for box clicks.
    bb = cb.ax.get_window_extent()
    pitch_px = bb.height / total
    if hasattr(cb, "_frames"):                      # matplotlib >= 3.7
        offsets = np.column_stack([np.full(n, 0.15), ys])
        cb._frames.set_offsets(offsets)
        cb._checks.set_offsets(offsets)
        # Sized from the font alone by matplotlib, so a dense panel (the accel
        # plot lists ~55 entries) overlaps its own boxes into one continuous bar.
        # Only ever shrinks: a sparse panel keeps matplotlib's proportions.
        for coll in (cb._frames, cb._checks):
            sizes = coll.get_sizes()
            want = (0.62 * pitch_px * 72.0 / cb.ax.figure.dpi) ** 2
            if len(sizes) and want < float(np.max(sizes)):
                coll.set_sizes([want])
    elif hasattr(cb, "rectangles"):                 # matplotlib 3.6
        # 3.6 builds the box as a square in AXES coordinates, which is only
        # square on a square panel.  This one is ~2.3 x 11 inches, so the
        # "checkbox" renders 2 px wide and 11 px tall -- a dash, not a box.
        # Sizing in pixels and converting back fixes both that and the density.
        # Computed once, at the figure's build size: the browser re-lays the
        # canvas out at a different aspect, so the box drifts back off square by
        # whatever that ratio is.  Approximately square beats a 2 px dash.
        side = min(9.0, 0.66 * pitch_px)
        w, h = side / bb.width, side / bb.height
        for rect, (l1, l2), y in zip(cb.rectangles, cb.lines, ys):
            x, y0 = 0.05, y - h / 2
            rect.set_bounds(x, y0, w, h)
            l1.set_data([x, x + w], [y0 + h, y0])
            l2.set_data([x, x + w], [y0, y0 + h])

    # Shrink the labels too once the rows get tighter than the type they carry.
    pitch_pt = pitch_px * 72.0 / cb.ax.figure.dpi
    if pitch_pt < 11.0:
        for text in cb.labels:
            text.set_fontsize(max(5.5, min(text.get_fontsize(), 0.72 * pitch_pt)))


def check_panel(fig, rect, series, groups, extra=(), on_change=None,
                title="series", anchors=None):
    """The checkbox panel that doubles as the legend.

    Labels are painted in their line's color, so identity is carried without a
    separate legend box eating plot area.  `groups` is [(group_key, master_label)]
    and a master appears only when its group has more than one member -- a
    "ALL x" checkbox above a single item is noise.

    `extra` is [(label, artists, initial_state)] for non-Series toggles such as
    the armed shading.
    """
    from matplotlib.widgets import CheckButtons

    entries = []          # (label, series_or_None, group_or_key)
    breaks = set()        # entry indices that begin a new block, for the spacing
    block_anchor = {}     # break index -> y (axes fraction) to centre it on
    for gname, master in groups:
        members = [s for s in series if s.group == gname]
        if not members:
            continue
        breaks.add(len(entries))
        if anchors and gname in anchors:
            block_anchor[len(entries)] = anchors[gname]
        indent = master and len(members) > 1
        if indent:
            entries.append((master, None, gname))
        for s in members:
            entries.append(("   " + s.label if indent else s.label, s, None))
    extra_art = {}
    for label, artists, state in extra:
        # The extras (armed shading, reset markers) are one block of their own:
        # they belong to no single graph, so separating them from the last
        # group's series stops them reading as more of its members.
        if not extra_art:
            breaks.add(len(entries))
        key = f"__extra_{len(extra_art)}__"
        extra_art[key] = (artists, state)
        entries.append((label, None, key))
    if not entries:
        return None

    labels = [e[0] for e in entries]
    states = []
    for lbl, s, key in entries:
        if s is not None:
            states.append(bool(s.visible))
        elif key in extra_art:
            states.append(bool(extra_art[key][1]))
        else:
            # A group master starts checked only if every member is visible.
            members = [m for m in series if m.group == key]
            states.append(bool(members) and all(m.visible for m in members))

    ax_cb = fig.add_axes(rect, facecolor=C_SURFACE)
    ax_cb.set_title(title, fontsize=9, color=C_MUTED, loc="left")
    for spine in ax_cb.spines.values():
        spine.set_visible(False)
    cb = CheckButtons(ax_cb, labels, states)
    for txt, e in zip(cb.labels, entries):
        txt.set_fontsize(8)
        txt.set_color(e[1].color if e[1] is not None else C_MUTED)

    guard = {"busy": False}

    def on_click(label):
        if guard["busy"]:
            return
        i = labels.index(label)
        state = cb.get_status()[i]
        _, s, key = entries[i]
        if s is not None:
            s.line.set_visible(state)
        elif key in extra_art:
            for art in extra_art[key][0]:
                art.set_visible(state)
        else:
            # Group master: drive the members' lines AND their check marks.  The
            # guard stops set_active's callback from re-entering this handler.
            guard["busy"] = True
            try:
                for j, (_, ms, _) in enumerate(entries):
                    if ms is not None and ms.group == key:
                        ms.line.set_visible(state)
                        if cb.get_status()[j] != state:
                            cb.set_active(j)
            finally:
                guard["busy"] = False
        if on_change:
            on_change()
        fig.canvas.draw_idle()

    cb.on_clicked(on_click)
    _respace_checkbuttons(cb, breaks, anchors=block_anchor or None)
    # A matplotlib widget whose only reference is a local gets garbage-collected
    # and silently stops responding -- keep it alive on the figure.
    fig._checkbuttons = cb
    return cb


class Nav:
    """Mouse navigation over a set of axes that share one time axis.

    Two modes, because this toolkit now has two hosts for a figure:

      standalone (page_scroll=False) -- the classic single-window behaviour:
          bare wheel zooms time, ctrl+wheel zooms values.

      browser page (page_scroll=True) -- the plots are stacked in a scroll area,
          so a bare wheel has to belong to the PAGE or you cannot get from the
          first plot to the last.  Time zoom moves to ctrl+wheel and value zoom
          to ctrl+shift+wheel.  The bare wheel is refused here and the Qt canvas
          re-raises it to the scroll area (see log_browser.PlotCanvas.wheelEvent).

    `on_xlim` is called with (lo, hi) after any change to the time axis, which is
    what lets the browser keep every plot on the same time window.
    """

    def __init__(self, fig, axes, page_scroll=False, on_xlim=None, fixed_y=(),
                 on_view=None):
        self.fig = fig
        self.axes = list(axes)
        # Called after the time window changes, to refit the value axes to what
        # is now on screen.  Each plot passes its own refresh() -- they already
        # know how to scale their own panels (percentile limits on the log
        # ratios, thresholds that must stay in view on the accel panels), and
        # re-deriving that generically here would get those wrong.
        self.on_view = on_view
        # Axes the user has value-zoomed or panned by hand.  Auto-fit leaves
        # those alone until a reset: having the axis you just set snap back on
        # the next wheel notch is worse than not auto-fitting at all.
        self.manual_y = set()
        # Axes whose y is a LAYOUT, not a measurement: the band panels put one
        # row per flag at integer y and label them in axes coordinates, so
        # zooming or panning their y scrambles the rows into nonsense.  They
        # still take part in the shared TIME axis.
        self.fixed_y = set(fixed_y)
        self.page_scroll = page_scroll
        self.on_xlim = on_xlim
        self.home = [(a, a.get_xlim(), a.get_ylim()) for a in self.axes]
        self._drag = {}
        self._echo = False        # guards the sync -> callback -> sync loop
        for name, fn in (("scroll_event", self._on_scroll),
                         ("button_press_event", self._on_press),
                         ("motion_notify_event", self._on_motion),
                         ("button_release_event", self._on_release)):
            fig.canvas.mpl_connect(name, fn)

    # -- helpers
    def _live(self, event):
        """Ignore events over the checkbox panel, and stand down while a toolbar
        mode (pan/zoom) holds the canvas widget lock -- otherwise both would act
        on the same drag and the view would move twice as fast."""
        return event.inaxes in self.axes and not self.fig.canvas.widgetlock.locked()

    @staticmethod
    def _zoomed_limits(ax, which, pixel, scale):
        """The (lo, hi) one axis WOULD have, zoomed about a pixel position.

        Separate from applying them because the time axis is shared: computing
        the new window once and assigning it to every panel is not the same as
        zooming each panel in turn.  Doing the latter compounds, since sharex
        propagates each panel's set_xlim to all the others before the next one
        is computed -- a 5-panel figure zoomed 0.87 per panel per notch, i.e.
        0.87**5 = 0.50, so one notch halved the window on the local-z plot and
        moved it 13% on the single-axis thermal plot.  Same gesture, different
        result per figure.
        """
        lo, hi = ax.get_ylim() if which == "y" else ax.get_xlim()
        to, back = ax.transData, ax.transData.inverted()
        i = 1 if which == "y" else 0
        p_lo = to.transform((0, lo))[i] if which == "y" else to.transform((lo, 0))[i]
        p_hi = to.transform((0, hi))[i] if which == "y" else to.transform((hi, 0))[i]
        n_lo = pixel - (pixel - p_lo) * scale
        n_hi = pixel + (p_hi - pixel) * scale
        if which == "y":
            a = back.transform((0, n_lo))[1]
            b = back.transform((0, n_hi))[1]
        else:
            a = back.transform((n_lo, 0))[0]
            b = back.transform((n_hi, 0))[0]
        if not (np.isfinite(a) and np.isfinite(b)) or a == b:
            return None
        return a, b

    @staticmethod
    def _zoom_about(ax, which, pixel, scale):
        """Scale one axis about a pixel position, in DISPLAY space.

        Doing the arithmetic in data coordinates is only correct on a linear
        axis.  The local-z plot's test-ratio panel is log-scaled, where
        `yc - (yc - y0) * scale` is meaningless -- it collapses the decade under
        the cursor and can produce a non-positive limit, which matplotlib then
        quietly refuses, so the axis appears not to zoom at all.  Converting
        through transData handles linear, log and symlog identically because the
        transform already knows the scale.
        """
        got = Nav._zoomed_limits(ax, which, pixel, scale)
        if got is None:
            return
        if which == "y":
            ax.set_ylim(*got)
        else:
            ax.set_xlim(*got)

    def _under(self, event):
        """The axes the pointer is actually inside -- base AND any twin.

        Twins share a rectangle exactly, so containment returns both, which is
        what you want: they are one panel to the reader and zooming half of it
        would slide the two scales apart.
        """
        return [a for a in self.axes
                if a.bbox.contains(event.x, event.y)]

    def _announce(self):
        if self.on_xlim and not self._echo:
            lo, hi = self.axes[0].get_xlim()
            self.on_xlim(lo, hi)

    def set_xlim(self, lo, hi):
        """Externally driven time window (the browser's cross-plot link)."""
        self._echo = True
        try:
            for a in self.axes:
                a.set_xlim(lo, hi)
        finally:
            self._echo = False
        self._refit()
        self.fig.canvas.draw_idle()

    def _refit(self):
        """Refit the value axes to the new window, honouring manual overrides."""
        if self.on_view is None:
            return
        keep = {a: a.get_ylim() for a in self.manual_y}
        self.on_view()
        for a, yl in keep.items():
            a.set_ylim(yl)

    def reset(self):
        self.manual_y.clear()
        for a, xl, yl in self.home:
            a.set_xlim(xl)
            a.set_ylim(yl)
        self._refit()
        self.fig.canvas.draw_idle()
        self._announce()

    # -- events
    def _on_scroll(self, event):
        if not self._live(event):
            return
        # Modifiers, from the host toolkit if it offered them.
        #
        # matplotlib fills a scroll event's `key` from keyboard state it tracks in
        # its own keyPressEvent, which only arrives when the CANVAS HAS KEYBOARD
        # FOCUS.  In the browser the focus is usually on the log tree, so
        # event.key is None however hard you hold ctrl, and ctrl+wheel silently
        # does nothing.  PlotCanvas therefore reads Qt's modifiers straight off
        # the wheel event and leaves them here; this falls back to event.key for
        # the standalone --classic window, which has no PlotCanvas.
        do_x, do_y = wheel_mods(event, self.page_scroll)
        if not (do_x or do_y):
            return
        # One notch = 15%.  event.step is +1 up / -1 down (fractional on
        # high-resolution trackpads, which this handles for free).
        scale = 1.15 ** (-event.step)

        if do_x and event.x is not None:
            # Zoom about the cursor, not the axis centre, so the sample under the
            # pointer stays put.  Computed ONCE -- on the panel the pointer is
            # over, so that is the one whose sample stays exactly put -- then
            # assigned to every axis, because the time axis is shared and one
            # panel drifting off the others is the one thing this figure must
            # never do.  See _zoomed_limits for why this is not a loop.
            ref = (self._under(event) or self.axes)[0]
            got = self._zoomed_limits(ref, "x", event.x, scale)
            if got is not None:
                for a in self.axes:
                    a.set_xlim(*got)
                self._refit()
        if do_y:
            # Value zoom applies ONLY to the panel under the pointer.  Zooming
            # every panel meant the pointer's pixel row was outside all the
            # others, so their anchor landed off-screen and they translated
            # wildly instead of zooming -- and it re-scaled the band panels,
            # whose y is a row layout.
            for a in self._under(event):
                if a not in self.fixed_y:
                    self._zoom_about(a, "y", event.y, scale)
                    self.manual_y.add(a)     # you set it; auto-fit leaves it
        self.fig.canvas.draw_idle()
        if do_x:
            self._announce()

    def _on_press(self, event):
        if event.button != 1 or not self._live(event):
            return
        if event.dblclick:                      # back to the full flight
            self._drag.clear()
            self.reset()
            return
        # Freeze the press-time transforms: the scale doesn't change during a
        # pan, so converting the pixel delta with these stays exact and avoids
        # the feedback drift you get from re-reading the transform each motion.
        self._drag["at"] = (event.x, event.y)
        self._drag["axes"] = [(a, a.get_xlim(), a.get_ylim(),
                               a.transData.inverted()) for a in self.axes]

    def _on_motion(self, event):
        if not self._drag or event.x is None:
            return
        px, py = self._drag["at"]
        moved_y = abs(event.y - py) > 1
        for a, (x0, x1), (y0, y1), inv in self._drag["axes"]:
            xa, ya = inv.transform((px, py))
            xb, yb = inv.transform((event.x, event.y))
            dx, dy = xa - xb, ya - yb
            a.set_xlim(x0 + dx, x1 + dx)
            if a not in self.fixed_y:
                a.set_ylim(y0 + dy, y1 + dy)
                if moved_y:
                    self.manual_y.add(a)
        # A pan that only moved sideways should still refit; one that moved the
        # value axis has just been positioned by hand, and _refit preserves it.
        self._refit()
        self.fig.canvas.draw_idle()
        self._announce()

    def _on_release(self, event):
        self._drag.clear()


def add_mouse_navigation(fig, axes, page_scroll=False, on_xlim=None, fixed_y=(),
                         on_view=None):
    """Wheel to zoom, drag to pan, double-click to reset -- without having to arm
    a mode on the toolbar first.  See Nav for the binding table.

    `fixed_y` lists axes whose y is a row layout rather than a scale (the band
    panels), so value zoom and vertical pan leave them alone.

    `on_view` is the plot's own refresh(), called after the time window changes
    so the value axes fit what is on screen."""
    nav = Nav(fig, axes, page_scroll=page_scroll, on_xlim=on_xlim,
              fixed_y=fixed_y, on_view=on_view)
    # Same GC hazard as the checkbuttons: mpl_connect holds only weak-ish refs
    # through the callback registry, and a Nav that dies stops responding.
    fig._nav = nav
    return nav


def nav_hint(page_scroll):
    if page_scroll:
        return ("wheel: scroll page  ·  ctrl+wheel: zoom time  ·  "
                "ctrl+shift+wheel: zoom values  ·  drag: pan  ·  double-click: reset")
    return ("wheel: zoom time  ·  ctrl+wheel: zoom values  ·  "
            "drag: pan  ·  double-click: reset")


def wheel_mods(event, page_scroll):
    """(zoom_time, zoom_values) for one wheel event, per the host's binding table.

    Split out of Nav because the flight-path figure has to decode the SAME
    gesture and it must not drift from what the time-series plots do.

    matplotlib fills a scroll event's `key` from keyboard state it tracks in its
    own keyPressEvent, which only arrives when the CANVAS HAS KEYBOARD FOCUS.  In
    the browser the focus is usually elsewhere, so event.key is None however hard
    you hold ctrl.  log_browser.PlotCanvas therefore reads Qt's modifiers off the
    wheel event and leaves them on the canvas; event.key is the fallback for the
    standalone --classic window, which has no PlotCanvas.
    """
    key = getattr(event.canvas, "_nav_mods", None)
    if key is None:
        key = event.key or ""
    ctrl = "ctrl" in key or "control" in key
    shift = "shift" in key
    if page_scroll:
        return ctrl and not shift, ctrl and shift
    return (not ctrl), (ctrl or shift)


class ViewNav:
    """Mouse navigation for SPATIAL axes -- a map and a 3D view, not a time series.

    Nav is wrong for these in two ways.  It zooms x and y independently, which on
    a ground track changes the SHAPE of the flight and is the one thing a map
    must never do; and it has no notion of a 3D axis at all.

      plan_axes  -- equal-aspect 2D maps.  Wheel scales BOTH axes by the same
                    factor about the cursor, so the aspect is preserved by
                    construction rather than by matplotlib correcting it after
                    the fact.  Drag pans, double-click resets.

      view_axes  -- mplot3d axes.  Wheel scales all three limits about their
                    centres: there is no cursor-anchored zoom that stays
                    meaningful while the view rotates, because the pixel under
                    the pointer is not a point in the data at all.  DRAG IS LEFT
                    ALONE -- that is mplot3d's own rotate, and taking it over
                    would cost the whole point of a 3D view.

    Deliberately NOT stored as fig._nav: the browser links every fig._nav onto
    one shared TIME window, and an east axis driven to a time range would empty
    the plot.
    """

    def __init__(self, fig, plan_axes=(), view_axes=(), page_scroll=False):
        self.fig = fig
        self.plan = list(plan_axes)
        self.view = list(view_axes)
        self.page_scroll = page_scroll
        self.home = [(a, a.get_xlim(), a.get_ylim()) for a in self.plan]
        self.home3 = [(a, a.get_xlim3d(), a.get_ylim3d(), a.get_zlim3d())
                      for a in self.view]
        self._drag = {}
        for name, fn in (("scroll_event", self._on_scroll),
                         ("button_press_event", self._on_press),
                         ("motion_notify_event", self._on_motion),
                         ("button_release_event", self._on_release)):
            fig.canvas.mpl_connect(name, fn)

    @staticmethod
    def _zoom_limits(lo, hi, scale):
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * scale
        return mid - half, mid + half

    def reset(self):
        for a, xl, yl in self.home:
            a.set_xlim(xl)
            a.set_ylim(yl)
        for a, xl, yl, zl in self.home3:
            a.set_xlim3d(xl)
            a.set_ylim3d(yl)
            a.set_zlim3d(zl)
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        if self.fig.canvas.widgetlock.locked():
            return
        do_x, do_y = wheel_mods(event, self.page_scroll)
        if not (do_x or do_y):
            return
        scale = 1.15 ** (-event.step)
        hit = False
        for a in self.plan:
            if event.x is None or not a.bbox.contains(event.x, event.y):
                continue
            hit = True
            # Same factor on both axes, anchored on the cursor: the point under
            # the pointer stays put and the ground track keeps its shape.
            Nav._zoom_about(a, "x", event.x, scale)
            Nav._zoom_about(a, "y", event.y, scale)
        for a in self.view:
            if event.x is None or not a.bbox.contains(event.x, event.y):
                continue
            hit = True
            a.set_xlim3d(*self._zoom_limits(*a.get_xlim3d(), scale))
            a.set_ylim3d(*self._zoom_limits(*a.get_ylim3d(), scale))
            a.set_zlim3d(*self._zoom_limits(*a.get_zlim3d(), scale))
        if hit:
            self.fig.canvas.draw_idle()

    def _on_press(self, event):
        if event.button != 1 or self.fig.canvas.widgetlock.locked():
            return
        if event.dblclick and (event.inaxes in self.plan
                               or event.inaxes in self.view):
            self._drag.clear()
            self.reset()
            return
        if event.inaxes not in self.plan:
            return              # a press in the 3D box belongs to mplot3d
        self._drag = {"at": (event.x, event.y),
                      "axes": [(event.inaxes, event.inaxes.get_xlim(),
                                event.inaxes.get_ylim(),
                                event.inaxes.transData.inverted())]}

    def _on_motion(self, event):
        if not self._drag or event.x is None:
            return
        px, py = self._drag["at"]
        for a, (x0, x1), (y0, y1), inv in self._drag["axes"]:
            xa, ya = inv.transform((px, py))
            xb, yb = inv.transform((event.x, event.y))
            dx, dy = xa - xb, ya - yb
            a.set_xlim(x0 + dx, x1 + dx)
            a.set_ylim(y0 + dy, y1 + dy)
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        self._drag.clear()


def add_view_navigation(fig, plan_axes=(), view_axes=(), page_scroll=False):
    """Wheel to zoom, drag to pan the map / rotate the 3D box, double-click to
    reset.  See ViewNav for why this is separate from add_mouse_navigation."""
    nav = ViewNav(fig, plan_axes=plan_axes, view_axes=view_axes,
                  page_scroll=page_scroll)
    # Same GC hazard as the checkbuttons -- and note the attribute name: NOT
    # _nav, which the browser would wire onto the shared time window.
    fig._view_nav = nav
    return nav


def view_nav_hint(page_scroll):
    if page_scroll:
        return ("wheel: scroll page  ·  ctrl+wheel: zoom  ·  "
                "drag in the 3D box: rotate  ·  drag the map: pan  ·  "
                "double-click: reset")
    return ("wheel: zoom  ·  drag in the 3D box: rotate  ·  "
            "drag the map: pan  ·  double-click: reset")
