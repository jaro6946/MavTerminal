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

def _rescale(ax, lines, pad=0.06):
    """Fit an axis to only its VISIBLE lines (matplotlib's autoscale counts
    hidden artists, so a toggled-off 80 degC channel would keep the axis
    stretched)."""
    vals = [ln.get_ydata() for ln in lines if ln.get_visible()]
    vals = [np.asarray(v, dtype=float) for v in vals]
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


def check_panel(fig, rect, series, groups, extra=(), on_change=None, title="series"):
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
    for gname, master in groups:
        members = [s for s in series if s.group == gname]
        if not members:
            continue
        indent = master and len(members) > 1
        if indent:
            entries.append((master, None, gname))
        for s in members:
            entries.append(("   " + s.label if indent else s.label, s, None))
    extra_art = {}
    for label, artists, state in extra:
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

    def __init__(self, fig, axes, page_scroll=False, on_xlim=None):
        self.fig = fig
        self.axes = list(axes)
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
        self.fig.canvas.draw_idle()

    def reset(self):
        for a, xl, yl in self.home:
            a.set_xlim(xl)
            a.set_ylim(yl)
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
        key = getattr(event.canvas, "_nav_mods", None)
        if key is None:
            key = event.key or ""
        ctrl = "ctrl" in key or "control" in key
        shift = "shift" in key
        if self.page_scroll:
            do_x, do_y = ctrl and not shift, ctrl and shift
        else:
            do_x, do_y = not ctrl, ctrl or shift
        if not (do_x or do_y):
            return
        # One notch = 15%.  event.step is +1 up / -1 down (fractional on
        # high-resolution trackpads, which this handles for free).
        scale = 1.15 ** (-event.step)

        if do_x and event.xdata is not None:
            # Zoom about the cursor, not the axis centre, so the sample under the
            # pointer stays put.  x data coords are identical on every twin.
            x0, x1 = self.axes[0].get_xlim()
            xc = event.xdata
            for a in self.axes:
                a.set_xlim(xc - (xc - x0) * scale, xc + (x1 - xc) * scale)
        if do_y:
            for a in self.axes:
                # The cursor's y in THIS axis's data coords -- the twins have
                # different scales, so each needs its own inverse transform.
                _, yc = a.transData.inverted().transform((event.x, event.y))
                y0, y1 = a.get_ylim()
                a.set_ylim(yc - (yc - y0) * scale, yc + (y1 - yc) * scale)
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
        for a, (x0, x1), (y0, y1), inv in self._drag["axes"]:
            xa, ya = inv.transform((px, py))
            xb, yb = inv.transform((event.x, event.y))
            dx, dy = xa - xb, ya - yb
            a.set_xlim(x0 + dx, x1 + dx)
            a.set_ylim(y0 + dy, y1 + dy)
        self.fig.canvas.draw_idle()
        self._announce()

    def _on_release(self, event):
        self._drag.clear()


def add_mouse_navigation(fig, axes, page_scroll=False, on_xlim=None):
    """Wheel to zoom, drag to pan, double-click to reset -- without having to arm
    a mode on the toolbar first.  See Nav for the binding table."""
    nav = Nav(fig, axes, page_scroll=page_scroll, on_xlim=on_xlim)
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
