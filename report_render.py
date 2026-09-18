#!/usr/bin/env python3
"""report_render.py -- turning a saved report into series, figures and numbers.

Everything here is Qt-free and side-effect-free.  It was lifted out of
GraphCard, which was the only thing that knew how to draw a report graph, and
that made two things impossible: rendering a report without a display, and
checking a report anyone (or anything) authored by hand before handing it over.

The split is deliberate.  This module decides WHAT a graph contains and builds
the figure; the Report tab decides how that figure is mounted, navigated and
resized.  Both callers therefore draw the identical picture by construction
rather than by anyone remembering to keep two code paths in step.

Acronyms: ULog = PX4's binary log format, GNSS = global navigation satellite
system.
"""
import os
import time

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from report_model import ALIGNMENTS
from ulog_cache import start_epoch
from ulog_derived import derived_field, derived_units, is_derived
from ulog_common import (C_GRID, C_INK, C_MUTED, C_SURFACE, armed_spans,
                         decimate, field, parse_ref, style_time_axis)

__all__ = ["SERIES_COLORS", "LOG_STYLES", "STAT_COLS", "DRAW_PX",
           "series_color", "log_style", "short_ref", "align_offset",
           "absolute_base", "stats_of", "fmt_stat", "gather_series",
           "auto_axis", "assign_axes", "build_figure", "fit_value_axes",
           "window_of", "plot_y", "log_date", "LANE_ON", "LANE_FRAC",
           "scatter_points", "spearman", "axis_label", "log_epoch"]

# Channel colours.  A categorical set -- these encode identity, not magnitude, so
# they are chosen to stay apart at one-pixel line width and to survive the two
# common colour-vision deficiencies, rather than to look like a ramp.
SERIES_COLORS = [
    "#2a78d6",   # blue
    "#d2691e",   # orange
    "#1baf7a",   # aqua
    "#7b2d8e",   # purple
    "#c0392b",   # red
    "#8a7fb5",   # violet
    "#4f7a28",   # olive
    "#d81b7a",   # magenta
    "#0f8f9e",   # teal
    "#8a6d3b",   # brown
    # Beyond ten the wrap used to hand log 11 the same blue as log 1, which on a
    # lanes graph -- where colour is the ONLY thing naming a log -- makes two
    # different flights look like one.  These six extend the set rather than
    # re-shading it: each is separated from all ten above in hue AND in
    # lightness, so they stay apart both in colour and printed grey.
    "#1f5f8b",   # deep blue
    "#e08a1e",   # amber
    "#3f7f3f",   # forest
    "#9b1b6b",   # plum
    "#5a5f8c",   # slate
    "#a83c1e",   # rust
]

# Log identity.  Style rather than colour, so that "which channel" and "which
# log" are read off two independent visual channels instead of competing for hue.
LOG_STYLES = ["-", "--", "-.", (0, (1, 1.4))]

DRAW_PX = 1400              # decimation budget; a little over the widest canvas

# Lanes (Graph.lanes).  A binary channel plotted across six logs is six lines
# that share two values, so they are drawn on top of one another and only the
# last one is visible.  Given a lane each -- level 1, 2, 3 ... -- every log's
# verdict is legible, and confining the stack to the bottom LANE_FRAC of the
# axis keeps it from covering the continuous channel it is there to annotate.
LANE_ON = 0.72              # height of a lane's "1" above its own baseline
LANE_FRAC = 0.34            # fraction of the value axis the lane stack may use
LANE_PAD = (0.35, 0.20)     # blank below the first baseline / above the last "1"

STAT_COLS = ["n", "min", "max", "mean", "median", "std", "first", "last"]


def series_color(i):
    return SERIES_COLORS[i % len(SERIES_COLORS)]


def log_style(i):
    return LOG_STYLES[i % len(LOG_STYLES)]


def log_epoch(ulog):
    """Sortable start time for a log: GNSS epoch, else the file's mtime, else
    +inf (undated logs sort last).  The same fallback chain as log_date, so an
    order built from this agrees with the dates printed beside it."""
    ep = start_epoch(ulog)
    if ep:
        return float(ep)
    try:
        return float(os.path.getmtime(getattr(ulog, "source_path", None)))
    except (OSError, TypeError):
        return float("inf")


def log_date(ulog):
    """"2 Sep" for a log, from its GNSS clock -- or "~2 Sep" from the file's
    modification time when the log never got a fix.

    Two logs called Test_1 and Test_2 say nothing about when either was flown,
    and a legend is where that question gets asked.  The tilde is not decoration:
    a bench log with no satellites is dated by its FILE, which is the day it was
    copied off the card, not necessarily the day it was recorded.  Same fallback
    chain the library tree uses, minus the name-stamp step (a report's logs are
    renamed by hand often enough that a stamp in the name is the least reliable
    of the three here)."""
    ep = start_epoch(ulog)
    if ep:
        return time.strftime("%-d %b", time.localtime(ep))
    path = getattr(ulog, "source_path", None)
    try:
        return "~" + time.strftime("%-d %b", time.localtime(os.path.getmtime(path)))
    except (OSError, TypeError):
        return ""


def short_ref(ref):
    """'sensor_accel[0].temperature' -> 'sensor_accel.temperature' when there is
    only one instance to speak of.  Legends are cramped enough."""
    return ref.replace("[0]", "", 1) if "[0]" in ref else ref


# --- time alignment ----------------------------------------------------------

def align_offset(ulog, mode, epoch_base=None):
    """Minutes to SUBTRACT from a log's own time base to line it up with others.

    Returns (offset, problem).  `problem` is None when the log can satisfy the
    requested alignment and a short phrase when it cannot -- the caller draws it
    anyway, at offset zero, and says so on the plot.  Silently falling back would
    put two flights on top of each other at an offset nobody chose, which is
    worse than an ugly label.
    """
    if mode == "first_arm":
        spans = armed_spans(ulog)
        if not spans:
            return 0.0, "never armed"
        return float(spans[0][0]), None
    if mode == "absolute":
        ep = start_epoch(ulog)
        if not ep:
            return 0.0, "no GNSS fix"
        if epoch_base is None:
            return 0.0, None
        return -(ep - epoch_base) / 60.0, None
    return 0.0, None


def absolute_base(ulogs):
    """The earliest wall-clock start among these logs, or None if none is fixed."""
    eps = [e for e in (start_epoch(u) for u in ulogs) if e]
    return min(eps) if eps else None


# --- statistics --------------------------------------------------------------

def stats_of(t, y, xlim=None):
    """Summary of y over the visible window, from the FULL arrays.

    Deliberately not computed from what was drawn: the drawn line is a min/max
    envelope, whose mean and standard deviation are those of the extremes rather
    than of the signal."""
    if xlim is not None and t.size:
        lo, hi = min(xlim), max(xlim)
        a, b = np.searchsorted(t, [lo, hi])
        y = y[a:b]
    # Finite samples only.  Computed channels use NaN for "not defined here" --
    # a rate before its window has filled, a heading check on the ground -- and
    # one NaN would otherwise make every statistic in the row read nan.
    y = y[np.isfinite(y)]
    if y.size == 0:
        return dict.fromkeys(STAT_COLS, None) | {"n": 0}
    return {"n": int(y.size), "min": float(y.min()), "max": float(y.max()),
            "mean": float(y.mean()), "median": float(np.median(y)),
            "std": float(y.std()), "first": float(y[0]), "last": float(y[-1])}


def fmt_stat(v):
    if v is None:
        return "—"
    if isinstance(v, int):
        return f"{v:,}"
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e5 or a < 1e-3:
        return f"{v:.3e}"
    return f"{v:,.4g}"


def window_of(series):
    """The full time span covered by these series, or None."""
    spans = [(s["t"][0], s["t"][-1]) for s in series if s["t"].size]
    if not spans:
        return None
    return min(a for a, _ in spans), max(b for _, b in spans)


# --- series ------------------------------------------------------------------

def gather_series(graph, ulogs, names=None):
    """Turn a graph's (logs x channels) selection into drawable series.

    `ulogs` is {log basename: parsed ULog}.  Returns (series, problems); a
    problem is a short human phrase, never an exception, because one absent
    channel must not cost you the other eleven.
    """
    names = list(names if names is not None else graph.logs)
    names = [n for n in names if n in ulogs]
    by_log = getattr(graph, "color_by", "channel") == "log"
    problems = []
    base = (absolute_base([ulogs[n] for n in names])
            if graph.align == "absolute" else None)
    series = []
    for li, name in enumerate(names):
        ulog = ulogs[name]
        when = log_date(ulog)
        epoch = log_epoch(ulog)
        off, why = align_offset(ulog, graph.align, base)
        if why:
            problems.append(f"{name}: {why}, drawn unaligned")
        for fi, ref in enumerate(graph.fields):
            try:
                topic, mid, fname = parse_ref(ref)
            except ValueError:
                problems.append(f"{short_ref(ref)}: not a topic[i].field name")
                continue
            t, y = field(ulog, topic, fname, mid)
            if t.size == 0 and is_derived(topic):
                # A computed channel (see ulog_derived): PX4 logs the inputs to
                # the arming check but not its verdict, and the verdict is the
                # thing worth comparing across logs.
                t, y = derived_field(ulog, topic, fname, mid)
            if t.size == 0:
                problems.append(f"{name}: {short_ref(ref)} absent")
                continue
            if graph.normalise:
                lo, hi = float(y.min()), float(y.max())
                y = (y - lo) / (hi - lo) if hi > lo else np.zeros_like(y)
            # colour/style carry one dimension each; which way round is the
            # graph's choice (see Graph.color_by).
            ci, si = (li, fi) if by_log else (fi, li)
            series.append({
                "ref": ref, "log": name, "date": when, "epoch": epoch,
                "label": f"{short_ref(ref)} · {os.path.splitext(name)[0]}",
                "t": t - off, "y": y,
                "color": series_color(ci), "ls": log_style(si),
                "unaligned": bool(why)})
    return series, problems


def auto_axis(graph, series):
    """Split channels across two scales when their magnitudes do not mix.

    Pack voltage at 15 and CPU load at 0.4 on one axis renders the second as a
    flat line on the floor.  The decades are clustered rather than thresholded so
    that a set of channels that genuinely belong together -- four temperatures,
    say -- is never split down the middle.
    """
    if graph.normalise or len(series) < 2:
        return {s["ref"]: "left" for s in series}
    decade = {}
    for s in series:
        y = np.abs(s["y"])
        y = y[y > 0]
        p95 = float(np.percentile(y, 95)) if y.size else 1.0
        decade[s["ref"]] = int(np.floor(np.log10(p95))) if p95 > 0 else 0
    if not decade:
        return {}
    vals = list(decade.values())
    # The most populated decade holds the left axis; anything more than one
    # decade away from it is unreadable beside it.
    main = max(set(vals), key=vals.count)
    return {ref: ("left" if abs(d - main) <= 1 else "right")
            for ref, d in decade.items()}


def is_binary(y):
    """True for a channel that only ever reads 0 or 1 -- a verdict, not a value.

    Tested rather than declared, because the caller asks for lanes on a GRAPH
    and the graph does not know which of its channels are flags; a temperature
    that happened to be selected for the right axis must keep its own scale."""
    y = np.asarray(y)
    if y.size == 0:
        return False
    return bool(np.all((y == 0) | (y == 1)))


def assign_lanes(graph, series):
    """Give each binary right-axis series its own level: 1, 2, 3 ...

    Stamped onto the series (`s["lane"]`, None when it is not a lane) rather
    than applied to the data, so the statistics table and the stats in the PDF
    keep reporting the real 0/1 channel while the PLOT shows the stack.
    `plot_y` is the single place the offset is applied, so the figure builder
    and the interactive re-decimation cannot disagree about where a line sits.
    """
    want = bool(getattr(graph, "lanes", False)) and not graph.normalise
    n = 0
    for s in series:
        if want and s.get("axis") == "right" and is_binary(s["y"]):
            n += 1
            s["lane"] = n
        else:
            s["lane"] = None
    return n


def plot_y(s):
    """A series' y values AS DRAWN: lane-offset when it was given a lane."""
    lane = s.get("lane")
    if not lane:
        return s["y"]
    return lane + LANE_ON * s["y"]


def lane_ylim(n):
    """(bottom, top) for the lane axis: n lanes filling its bottom LANE_FRAC."""
    lo = 1.0 - LANE_PAD[0]
    hi = n + LANE_ON + LANE_PAD[1]
    return lo, lo + (hi - lo) / LANE_FRAC


def assign_axes(graph, series):
    """Stamp each series with its final axis.  Returns the AUTOMATIC choice too,
    so a caller can show which assignments were made for the user rather than
    by them."""
    auto = auto_axis(graph, series)
    for s in series:
        s["axis"] = graph.axis.get(s["ref"], auto.get(s["ref"], "left"))
    assign_lanes(graph, series)
    return auto


# --- figure ------------------------------------------------------------------

def spearman(a, b):
    """Rank correlation, ties averaged.  None when there is nothing to correlate.

    Rank rather than Pearson because the claim being tested is "more of this
    goes with more of that", not "they are proportional" -- and because one log
    at ten times the exposure of the rest would otherwise set the answer on its
    own."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    def rank(x):
        o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
        for v in np.unique(x):
            m = x == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    a, b = rank(a[ok]), rank(b[ok])
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def scatter_points(graph, series):
    """[(log, x, y)] -- one point per log, from the first two channels.

    Each axis is the log's TIME-weighted MEAN of that channel (see _mean_over).
    For the rate channels this is the natural summary (mean of a sliding-window
    rate over the log is the log's rate); for a 0/1 verdict it is the fraction
    of the log spent at 1.  NaN-aware, so a window's undefined first seconds do
    not drag a log's value down.  Values are in the channel's own units -- any
    percentage scaling is the figure's business, not this function's.
    """
    if len(graph.fields) < 2:
        return []
    xr, yr = graph.fields[0], graph.fields[1]
    by_log = {}
    for s in series:
        by_log.setdefault(s["log"], {})[s["ref"]] = s
    out = []
    for name, d in by_log.items():
        sx, sy = d.get(xr), d.get(yr)
        if sx is None or sy is None:
            continue
        span = _common_span(sx, sy)
        if span is None:
            continue
        x, y = _mean_over(sx, span), _mean_over(sy, span)
        if np.isfinite(x) and np.isfinite(y):
            out.append((name, float(x), float(y), sx.get("date", ""),
                        sx.get("epoch", float("inf"))))
    # DATE order, oldest first: the marker numbers and the legend follow this,
    # and a legend read top to bottom is then the test campaign in sequence.
    # (Number-by-x was tried first; with twenty logs it made the legend a
    # shuffle of dates that answered nothing.)  Same-day logs keep their
    # time-of-day order because the key is the full epoch, not the date text.
    out.sort(key=lambda r: (r[4], r[0]))
    return [r[:4] for r in out]


def _finite_span(s):
    """(first, last) time at which this series has a finite value, or None."""
    t, y = s["t"], s["y"]
    ok = np.isfinite(y)
    if not t.size or not ok.any():
        return None
    i = np.nonzero(ok)[0]
    return float(t[i[0]]), float(t[i[-1]])


def _common_span(sx, sy):
    """The time both channels actually cover.  None if they never overlap.

    Averaging each channel over its OWN extent compares unlike things, and on
    the graph this was written for it does so in the worst possible direction:
    offboard_control_mode exists only while the companion is publishing, but
    the accelerometer error counter runs for the whole log -- so a log where
    the companion joined at minute 28 of 43 had its x measured over 15 minutes
    and its y over 43, diluting y by the very quantity the graph is about.
    Both are therefore reduced over the intersection.
    """
    a, b = _finite_span(sx), _finite_span(sy)
    if a is None or b is None:
        return None
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if hi > lo else None


def _mean_over(s, span):
    """TIME-weighted mean of this channel inside `span`, NaN-aware.

    Each sample holds its value until the next one (the same zero-order hold
    the plot draws), so a sample stands for as much of the log as it covers.
    A plain sample mean is wrong whenever a topic's logging rate varies, and
    the one this was written for varies a lot: `estimator_sensor_bias` logs
    far faster while the filter converges, so averaging SAMPLES of the
    accel-bias verdict said LandingTest_Accel_Bias_001 failed 78.2% of the
    time when it failed 70.8% of it, and Flight Endurance Test 1 6.1% when it
    was 0.9%.  Intervals that start on a NaN are "not defined here" and are
    dropped from both numerator and denominator.

    Falls back to the channel's whole finite mean when no SAMPLE lands inside
    the window.  That is not a fudge: a channel sampled more sparsely than the
    window still has a value throughout it, and the case that forced this is
    the honest one -- a log with no companion carries offboard_rate as two
    points, at the log's first and last instant, and a window trimmed by even a
    tenth of a second to the other channel's extent then contains neither of
    them.  Dropping those logs would have silently deleted every control from
    the graph.
    """
    lo, hi = span
    m = (s["t"] >= lo) & (s["t"] <= hi)
    t, v = s["t"][m], s["y"][m]
    if t.size:
        # The last sample holds to the end of the window.
        dt = np.diff(np.append(t, hi))
        ok = np.isfinite(v) & (dt > 0)
        if ok.any():
            return float((v[ok] * dt[ok]).sum() / dt[ok].sum())
        fin = v[np.isfinite(v)]
        if fin.size:            # samples all at one instant: nothing to weight
            return float(fin.mean())
    all_v = s["y"][np.isfinite(s["y"])]
    return float(all_v.mean()) if all_v.size else np.nan


def _percent_scale(ref):
    """(factor, label) for one scatter axis.

    A 0/1 verdict averaged over time is the FRACTION of time it read 1, and
    "0.708" on an axis labelled 0/1 reads like a verdict that is 71% true.
    Shown as a percentage and labelled as one, it says what it is.  Anything
    else is drawn in its own units."""
    try:
        topic, _mid, name = parse_ref(ref)
    except ValueError:
        return 1.0, axis_label([ref], "left")
    if derived_units(topic, name) == "0/1":
        return 100.0, f"{short_ref(ref)}  (% of time at 1)"
    return 1.0, axis_label([ref], "left")


SCATTER_MS = 13             # marker diameter, points -- room for a 2-digit number
SCATTER_GAP_PX = 2.0        # clear space kept between neighbouring markers


def _dodge_px(px, py, diam):
    """Display x positions moved just far enough that no two markers touch.

    Only x moves, and only by the minimum that clears the overlap: two markers
    whose centres are dy apart vertically need a horizontal separation of
    sqrt(d^2 - dy^2), no more.  Nothing is spread by a fixed step, so a point
    with room around it is drawn exactly where it measured, and a crowded group
    opens up by about one marker width rather than a share of the axis.

    An earlier version fanned only EXACT ties, on the grounds that a tolerance
    moves points that genuinely differ.  That is still true of a tolerance in
    DATA units; a tolerance of one marker diameter in PIXELS is different --
    it moves only what the reader could not otherwise see.

    Order is preserved (a marker never hops over its neighbour), and exact ties
    open symmetrically, in log order, about their shared value."""
    n = len(px)
    q = np.array(px, dtype=float)
    need = diam + SCATTER_GAP_PX
    order = sorted(range(n), key=lambda i: (px[i], i))
    rank = {i: k for k, i in enumerate(order)}
    for _ in range(400):
        moved = False
        for a in range(n):
            for b in range(a + 1, n):
                dy = abs(py[a] - py[b])
                if dy >= need:
                    continue
                want = float(np.sqrt(need * need - dy * dy))
                i, j = (a, b) if rank[a] < rank[b] else (b, a)
                gap = q[j] - q[i]
                if gap < want - 1e-6:
                    push = (want - gap) / 2.0
                    q[i] -= push
                    q[j] += push
                    moved = True
        if not moved:
            break
    return q


def _scatter_figure(graph, series, fig, ax):
    """Set up the one-point-per-log form: axes, limits, legend handles.

    The MARKERS are not drawn here.  Keeping them apart is a question about
    pixels, and the pixels are not settled until the legend and any problem
    note have moved the axes -- `_scatter_markers` draws them afterwards.
    Returns (handles, state) for it."""
    pts = scatter_points(graph, series)
    handles = []
    if len(graph.fields) < 2:
        return handles, None
    kx, lx = _percent_scale(graph.fields[0])
    ky, ly = _percent_scale(graph.fields[1])
    pts = [(name, x * kx, y * ky, when) for name, x, y, when in pts]
    for i, (name, x, y, when) in enumerate(pts):
        stem = os.path.splitext(name)[0]
        if len(stem) > 34:
            stem = stem[:33] + "…"
        label = f"{i + 1}  {stem}" + (f" ({when})" if when else "")
        handles.append(Line2D([], [], color=series_color(i), marker="o", ms=7,
                              ls="none", label=label))
    ax.set_xlabel(lx, fontsize=9, color=C_MUTED)
    ax.set_ylabel(ly, fontsize=9, color=C_MUTED)
    if pts:
        xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
        def pad(lo, hi, frac=0.10):
            m = (hi - lo) * frac or max(abs(hi), 1.0) * frac
            return lo - m, hi + m
        ax.set_xlim(*pad(min(xs), max(xs)))
        ax.set_ylim(*pad(min(ys), max(ys), 0.12))
    return handles, {"pts": pts}


def _scatter_markers(fig, ax, state):
    """Draw the numbered markers, dodged in pixels, plus the on-figure caption.

    A marker moved further than its own radius keeps a thin line back to a tick
    at its TRUE x, so the reader can see both that it moved and where it
    belongs.  A smaller nudge leaves the true x inside the marker, where a line
    would be hidden anyway.  The statistics page lists the unshifted numbers."""
    if not state or not state["pts"]:
        return
    pts = state["pts"]
    xs = np.array([p[1] for p in pts]); ys = np.array([p[2] for p in pts])
    diam = SCATTER_MS * fig.dpi / 72.0

    def layout():
        for _ in range(3):
            disp = ax.transData.transform(np.column_stack([xs, ys]))
            qx = _dodge_px(disp[:, 0], disp[:, 1], diam)
            box = ax.get_window_extent()
            lo, hi = qx.min() - diam, qx.max() + diam
            if lo >= box.x0 and hi <= box.x1:
                break
            # A dodge pushed a marker past the frame: widen x and dodge again
            # (the pixel scale changed, so the old answer no longer holds).
            inv = ax.transData.inverted()
            ax.set_xlim(inv.transform((min(lo, box.x0), 0))[0],
                        inv.transform((max(hi, box.x1), 0))[0])
        return qx, disp

    def caption(moved):
        bits = []
        r = spearman(xs, ys)
        if r is not None:
            bits.append(f"Spearman \u03c1 = {r:+.2f}   n = {len(pts)}")
        if moved.any():
            # Said on the figure, not only in the notes: a reader who does not
            # know the markers were moved will read the offset as data.
            # WHICH markers, with the value each really has, not a count: a
            # count tells the reader something moved but not what to discount.
            items = [f"{i + 1} ({xs[i]:.4g})" for i in np.nonzero(moved)[0]]
            rows = [", ".join(items[k:k + 8]) for k in range(0, len(items), 8)]
            bits.append("nudged sideways so none overlap -- marker (true x): "
                        + rows[0])
            bits.extend(rows[1:])
        return "\n".join(bits)

    qx, disp = layout()
    moved = np.abs(qx - disp[:, 0]) > 0.5
    text = caption(moved)
    if text:
        hits, art = _scatter_caption(fig, ax, text, qx, disp[:, 1], diam)
        if hits:
            # Every corner has data in it.  Make room rather than cover a
            # marker: raise the top of the y scale until the highest marker
            # sits below the caption, then redo the dodge on the new scale.
            box = ax.get_window_extent()
            cap = art.get_window_extent().height + 0.03 * box.height
            art.remove()
            lo, hi = ax.get_ylim()
            room = box.height - cap - diam
            if room > 0.3 * box.height:
                ax.set_ylim(lo, max(hi, lo + (ys.max() - lo) * box.height / room))
                qx, disp = layout()
                moved = np.abs(qx - disp[:, 0]) > 0.5
            ax.text(0.985, 0.97, caption(moved), transform=ax.transAxes,
                    ha="right", va="top", fontsize=9, color=C_MUTED,
                    linespacing=1.6, zorder=5)

    shifted_x = ax.transData.inverted().transform(
        np.column_stack([qx, disp[:, 1]]))[:, 0]
    far = np.abs(qx - disp[:, 0]) > diam / 2.0
    for i, (name, x, y, _when) in enumerate(pts):
        c = series_color(i)
        if far[i]:
            ax.plot([x, shifted_x[i]], [y, y], color=C_MUTED, lw=0.8, zorder=2)
            ax.plot([x], [y], marker="|", ms=7, mew=1.2, color=C_MUTED,
                    ls="none", zorder=2)
        ax.plot([shifted_x[i]], [y], marker="o", ms=SCATTER_MS, mfc=c,
                mec=C_SURFACE, mew=1.2, ls="none", zorder=3)
        ax.text(shifted_x[i], y, str(i + 1), ha="center", va="center",
                fontsize=6.5 if i >= 9 else 7.5, weight="bold",
                color="white", zorder=4)


def _scatter_caption(fig, ax, text, qx, qy, diam):
    """Put the caption in whichever corner of the frame covers fewest markers.

    A fixed corner is a guess about where the data is not; on the temperature
    graph the bottom right is clear and the top left is full, and on the
    companion graph it is the other way round.  Returns (markers covered, the
    text artist) so the caller can make room when no corner is clear."""
    corners = [(0.985, 0.03, "right", "bottom"), (0.985, 0.97, "right", "top"),
               (0.015, 0.97, "left", "top"), (0.015, 0.03, "left", "bottom")]
    best = None
    r = diam / 2.0
    for x, y, ha, va in corners:
        t = ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                    fontsize=9, color=C_MUTED, linespacing=1.6, zorder=5)
        try:
            fig.draw_without_rendering()
            bb = t.get_window_extent()
            hits = int(np.sum((qx + r > bb.x0) & (qx - r < bb.x1)
                              & (qy + r > bb.y0) & (qy - r < bb.y1)))
        except (AttributeError, ValueError, RuntimeError):
            hits = 0
        if best is None or hits < best[0]:
            if best is not None:
                best[1].remove()
            best = (hits, t)
        else:
            t.remove()
        if hits == 0:
            break
    return best


def build_figure(graph, series, problems=(), figsize=(13.0, 4.3), dpi=100,
                 auto=None):
    """The graph, exactly as both the tab and the exporter draw it.

    Returns (fig, ax, axr, lines) where `lines` pairs each Line2D with the series
    it came from -- the tab needs that to re-decimate on zoom, the exporter
    ignores it.  No navigation and no hint text: those belong to the interactive
    host, and a PDF should not advertise a mouse.
    """
    if auto is None:
        auto = assign_axes(graph, series)

    # Figure(), NOT plt.figure(): pyplot keeps a global strong reference to every
    # figure it makes, which is how this program once walked itself into the OOM
    # killer one log at a time.
    fig = Figure(figsize=figsize, dpi=dpi, facecolor=C_SURFACE)
    n_f = len({s["ref"] for s in series})
    n_l = len({s["log"] for s in series})
    # With only one channel, colour encodes nothing and a separate colour key is
    # three legend entries describing two lines, in a grey that matches neither.
    # Same the other way round with a single log.  So the split key appears only
    # when BOTH dimensions actually vary.
    n_keys = len(series) if (n_f <= 1 or n_l <= 1) else n_f + n_l
    ncol = max(2, n_keys // 2) if n_keys else 2
    # Vertical margins in INCHES, expressed as a fraction of this figure's own
    # height, rather than as a fixed fraction.  A title is a fixed number of
    # points tall and so is a row of legend text, so scaling their room with the
    # figure means a graph asked to be 1.6x taller spends most of the new space
    # on white margin -- 2.3 inches of it under the axes on a 14.4-inch page.
    # The constants are the old fractions at the 4.3-inch card that set them, so
    # a card renders exactly as it did before.
    h = float(figsize[1]) or 4.3
    fig.subplots_adjust(left=0.055, right=0.945,
                        top=1.0 - min(0.10, 0.43 / h),
                        bottom=min(0.16, 0.688 / h))
    ax = fig.add_subplot(111)
    ax.set_facecolor(C_SURFACE)
    axr = None

    if getattr(graph, "kind", "series") == "scatter":
        # A different picture entirely: no time axis, no right axis, no lines to
        # decimate or refit.  Returning lines=[] is what keeps every caller's
        # window-fitting machinery from touching it -- fit_value_axes finds
        # nothing of its own on the axis and leaves the limits set here.
        handles, state = _scatter_figure(graph, series, fig, ax)
        ax.set_title(graph.title or "untitled graph", loc="left", fontsize=11,
                     color=C_INK)
        ax.grid(True, color=C_GRID, lw=0.6)
        if handles:
            leg = ax.legend(handles=handles, loc="lower left",
                            bbox_to_anchor=(0.055, 0.012),
                            bbox_transform=fig.transFigure,
                            ncol=max(1, min(3, (len(handles) + 5) // 6)),
                            fontsize=8, frameon=False, handlelength=1.2,
                            columnspacing=1.4)
            leg.set_in_layout(False)
            _fit_legend(fig, ax)
        note = "; ".join(list(problems)[:3])
        if note:
            _problem_note(fig, note)
        # Last: both calls above move the axes, and the markers are kept apart
        # in PIXELS, so they can only be placed once the frame stops moving.
        _scatter_markers(fig, ax, state)
        return fig, ax, None, []

    lines = []
    for s in series:
        target = ax
        if s["axis"] == "right":
            if axr is None:
                axr = ax.twinx()
                axr.set_facecolor("none")
                # Read by fit_value_axes, which is where the band has to be
                # applied -- see Graph.right_frac.
                axr._band_frac = getattr(graph, "right_frac", 1.0)
            target = axr
        td, yd = decimate(s["t"], plot_y(s), DRAW_PX)
        (line,) = target.plot(td, yd, color=s["color"], ls=s["ls"], lw=1.3,
                              label=s["label"] + (" (unaligned)"
                                                  if s["unaligned"] else ""))
        lines.append((line, s))

    style_time_axis(ax, label=True)
    ax.set_xlabel(f"{ALIGNMENTS.get(graph.align, '')}  (minutes)")
    ax.set_title(graph.title or "untitled graph", loc="left", fontsize=11,
                 color=C_INK)
    ax.grid(True, color=C_GRID, lw=0.6)
    if graph.normalise:
        ax.set_ylabel("normalised 0–1")

    if series:
        _legend(ax, axr, series, graph, auto, ncol=ncol)
        _lane_axis(axr, series)
    else:
        ax.text(0.5, 0.5, "no channels selected", transform=ax.transAxes,
                ha="center", va="center", color=C_MUTED, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    if graph.xlim:
        ax.set_xlim(*graph.xlim)

    # Only now, with every label that affects the layout in place, can the room
    # under the axes be worked out -- so it is MEASURED rather than predicted.
    _fit_legend(fig, ax)

    note = "; ".join(list(problems)[:3])
    if len(problems) > 3:
        note += f"; +{len(problems) - 3} more"
    if note:
        _problem_note(fig, note)

    return fig, ax, axr, lines


def _lane_axis(axr, series):
    """Turn the right axis into a lane rack: one tick per lane, stack at the
    bottom, and a faint baseline under each lane so "off" is a line rather than
    an absence.  The scale is FIXED -- lanes are positions, not measurements, so
    there is nothing for an autoscale to fit (fit_value_axes honours this).

    With nothing on the LEFT axis the stack has nothing to annotate, so it takes
    the whole frame rather than its bottom third, and the empty left scale is
    hidden -- a graph made only of verdicts (which magnetometer, which check
    tripped) otherwise spends two thirds of its height on a blank 0-1 axis."""
    lanes = sorted({s["lane"] for s in series if s.get("lane")})
    if axr is None or not lanes:
        return
    n = max(lanes)
    if any(s.get("axis", "left") == "left" for s in series):
        lo, hi = lane_ylim(n)
    else:
        lo, hi = 1.0 - LANE_PAD[0], n + LANE_ON + LANE_PAD[1]
        # build_figure makes exactly two axes, the frame and its twin.
        for a in axr.figure.axes:
            if a is not axr:
                a.set_yticks([])
    axr.set_ylim(lo, hi)
    axr.set_yticks(list(lanes))
    axr.set_yticklabels([str(k) for k in lanes], fontsize=8, color=C_MUTED)
    axr.set_ylabel("lane, one per log  ›", fontsize=8, color=C_MUTED)
    for k in lanes:
        axr.axhline(k, color=C_GRID, lw=0.5, zorder=0)
    axr._lane_ylim = (lo, hi)


def _legend(ax, axr, series, graph, auto, ncol=2):
    """A key that matches what is on the plot.

    Two dimensions are encoded -- by default colour is the channel and line
    style is the log, and graph.color_by == "log" swaps them --
    and a combined key needs one entry per (log x channel) pair, which is
    eighteen entries for three logs and six channels.  Split into a colour block
    and a style block, it is nine.

    But that split is only worth its confusion when both dimensions actually
    vary.  Plot one channel across two logs and the colour block becomes a third
    entry for two lines, swatched in a grey that appears nowhere on the plot.  So
    in that case -- and in the mirror case of one log and several channels --
    the key reverts to one entry per LINE, drawn exactly as that line is drawn.
    """
    by_log = getattr(graph, "color_by", "channel") == "log"

    def log_label(name, width=38):
        """The log's name with its date -- "Test_1 (2 Sep)".

        Trimmed on the NAME only: dropping the date to fit would defeat the
        point, and these names share long prefixes so the tail is where they
        differ anyway."""
        when = next((x.get("date") for x in series if x["log"] == name), "")
        stem = os.path.splitext(name)[0]
        if len(stem) > width:
            stem = stem[:width - 1] + "…"
        return f"{stem} ({when})" if when else stem

    fields, logs = [], []
    for s in series:
        if s["ref"] not in [f for f, _, _ in fields]:
            fields.append((s["ref"], s["color"], s["ls"]))
        if s["log"] not in [l for l, _, _ in logs]:
            logs.append((s["log"], s["color"], s["ls"]))

    if len(fields) <= 1 or len(logs) <= 1:
        one_channel = len(fields) <= 1
        # Numbered to match the right-hand ticks when the lines ARE lanes -- the
        # same reason as the colour-by-log key below: the key is then the only
        # thing that says which lane is which.
        handles = [Line2D([], [], color=s["color"], ls=s["ls"], lw=1.8,
                          label=(f"{s['lane']} · " if s.get("lane") else "")
                                + (log_label(s["log"]) if one_channel
                                   else short_ref(s["ref"])))
                   for s in series]
    elif by_log:
        # Colour is the log, so the log block carries the swatches and the
        # channel block is the greyed one.  With lanes on, the key is also the
        # only thing that says which lane belongs to which log, so it is
        # numbered to match the right-hand ticks.
        lane_of = {s["log"]: s["lane"] for s in series if s.get("lane")}
        handles = [Line2D([], [], color=c, lw=2,
                          label=(f"{lane_of[n]} · " if n in lane_of else "")
                                + log_label(n, width=30))
                   for n, c, _ in logs]
        handles += [Line2D([], [], color=C_MUTED, lw=1.6, ls=ls,
                           label=short_ref(r) + (" ›" if graph.axis.get(
                               r, auto.get(r, "left")) == "right" else ""))
                    for r, _, ls in fields]
    else:
        handles = [Line2D([], [], color=c, lw=2,
                          label=short_ref(r) + (" ›" if graph.axis.get(
                              r, auto.get(r, "left")) == "right" else ""))
                   for r, c, _ in fields]
        # "log:" prefixed and greyed, so a style swatch is not mistaken for a
        # series that ought to be on the plot in that colour.
        handles += [Line2D([], [], color=C_MUTED, lw=1.6, ls=ls,
                           label="log: " + log_label(n, width=34))
                    for n, _, ls in logs]
    # Anchored to the FIGURE, not the axes.  Anchored to the axes, the offset has
    # to be expressed in axes fractions, which means it changes meaning every
    # time the axes are resized to make room -- the guessed offset that put this
    # key off the bottom of a 4.3-inch card while looking right on a 5-inch one.
    # Pinned to the figure, the key stays put and _fit_legend moves the AXES
    # instead, which is the thing that has room to give.
    def place(n):
        leg = ax.legend(handles=handles, loc="lower left",
                        bbox_to_anchor=(0.055, 0.012),
                        bbox_transform=ax.figure.transFigure,
                        ncol=n, fontsize=8, frameon=False, handlelength=2.6,
                        columnspacing=1.4)
        leg.set_in_layout(False)
        return leg

    leg = place(ncol)
    # The caller's ncol is derived from the NUMBER of entries and knows nothing
    # about how long the labels are or how wide the figure is, so a dozen logs
    # with dated names ran the key off the right-hand edge -- entries simply
    # absent, with nothing to show they had been dropped.  Measure it, the same
    # way _fit_legend measures the height, and take a column away until it fits.
    # Measured rather than estimated for the same reason: any character-width
    # formula is right at one font and figure size and wrong at the next.
    fig = ax.figure
    inv = fig.transFigure.inverted()
    while ncol > 1:
        try:
            fig.draw_without_rendering()
            wide = leg.get_window_extent().transformed(inv).x1 > 0.955
        except (AttributeError, ValueError, RuntimeError):
            break
        if not wide:
            break
        ncol -= 1
        leg = place(ncol)
    _label_axes(ax, axr, series, graph)


def axis_label(refs, side):
    """The name for a value scale: the channels on it, with their units.

    Every graph used to be labelled only in the special case of a single
    channel, and the right-hand scale got the placeholder "right scale ›" --
    which names the side of the frame it is on and nothing about what is
    plotted.  On a graph whose two scales are errors per minute and hertz that
    is not a label, it is a direction.

    Several channels on one scale are joined rather than dropped: they share the
    scale, so the reader needs to know that both are on it.
    """
    seen, parts = set(), []
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        try:
            topic, _mid, name = parse_ref(ref)
            unit = derived_units(topic, name)
        except ValueError:
            unit = ""
        parts.append(short_ref(ref) + (f"  ({unit})" if unit else ""))
    text = "   ·   ".join(parts)
    if len(text) > 78:                  # a label taller than the plot helps no one
        text = text[:77] + "\u2026"
    return text + ("  ›" if side == "right" else "")


def _label_axes(ax, axr, series, graph):
    """Name both value scales from what was actually drawn on them."""
    if graph.normalise:
        return                          # already labelled "normalised 0-1"
    left = [s["ref"] for s in series if s.get("axis", "left") == "left"]
    right = [s["ref"] for s in series if s.get("axis") == "right"]
    if left:
        ax.set_ylabel(axis_label(left, "left"), fontsize=9, color=C_MUTED)
    if axr is not None and right and getattr(axr, "_lane_ylim", None) is None:
        # A lane rack is labelled by _lane_axis instead: its scale is positions
        # on a stack, and naming the channel there would be a category error.
        axr.set_ylabel(axis_label(right, "right"), fontsize=9, color=C_MUTED)


def _fit_legend(fig, ax, pad=0.012):
    """Raise the axes until the legend measurably fits beneath them.

    The legend's height depends on the font, the number of rows, the label
    lengths and the figure width -- so any formula for it is a guess that holds
    at one size and fails at another, which is exactly what happened.  Render
    once, ask the legend and the x-axis furniture how much room they actually
    occupy, and give them that much.
    """
    leg = ax.get_legend()
    if leg is None:
        return
    # A figure built with Figure() carries only a FigureCanvasBase until someone
    # attaches a real one, so there is no renderer to ask yet -- and this has to
    # work BEFORE the Report tab wraps it in a PlotCanvas.  draw_without_rendering
    # lays the figure out and stashes a renderer without producing any output.
    inv = fig.transFigure.inverted()
    try:
        fig.draw_without_rendering()
        key = leg.get_window_extent().transformed(inv)
        pos = ax.get_position()
        # How far the tick labels and the x-axis title hang below the axes.
        furniture = max(0.0, pos.y0 - ax.get_tightbbox().transformed(inv).y0)
    except (AttributeError, ValueError, RuntimeError):
        return
    want = key.y1 + furniture + pad
    if want > pos.y0:
        # Capped: a legend so tall it would leave no plot is a legend to shorten,
        # not a reason to render an empty axes.
        fig.subplots_adjust(bottom=min(0.55, want))


NOTE_PT = 8                 # problem-note font size
NOTE_LEAD = 1.9             # line box as a multiple of it, with breathing room


def _problem_note(fig, note):
    """Put the "these logs are missing" line in a strip of its own at the top.

    It used to be drawn at the constant figure fraction 0.955, which cleared the
    title only while the top margin was the constant 0.90 that put the title
    below it.  Once the margins became a fixed number of INCHES -- so that a
    graph asked to be taller spends the room on the plot rather than on white
    space -- a taller figure carried the title up THROUGH the note, and the one
    line on the figure that says data is missing became the line hidden behind
    the heading.

    Measuring the title is the obvious repair and does not work: before the
    figure is drawn `Text.get_window_extent` reports a zero-height box at the
    baseline, so "just above the title" lands inside it.  So the note is not
    placed relative to the title at all.  It gets its own strip at the top of
    the figure, and the axes -- and with them the title, which hangs off the
    axes -- are pushed down by exactly that much.  No measurement, and nothing
    to collide with."""
    h = (NOTE_PT * NOTE_LEAD / 72.0) / (fig.get_figheight() or 4.3)
    box = fig.subplotpars
    fig.subplots_adjust(top=max(0.55, box.top - h))
    fig.text(0.055, 1.0 - 0.25 * h, note, fontsize=NOTE_PT, color=C_MUTED,
             va="top")

def _fit_one(axis, lines, window_values_fn=None, pad=0.06):
    """Fit ONE value axis to the data inside the current time window.

    Split out of fit_value_axes so the banded case can reuse it: a band is an
    ordinary fit followed by a stretch, and duplicating the fit to say so is how
    two copies of a calculation start disagreeing.  Returns False when the axis
    owns nothing to fit to.
    """
    mine = [ln for ln, _ in lines if ln.axes is axis]
    if not mine:
        return False
    if window_values_fn is not None:
        v = window_values_fn(axis, mine)
    else:
        lo_x, hi_x = axis.get_xlim()
        chunks = []
        for ln in mine:
            t = np.asarray(ln.get_xdata())
            y = np.asarray(ln.get_ydata())
            if t.size:
                m = (t >= min(lo_x, hi_x)) & (t <= max(lo_x, hi_x))
                chunks.append(y[m])
        v = np.concatenate(chunks) if chunks else np.array([])
    if v is None or not len(v):
        return False
    v = v[np.isfinite(v)]
    if not v.size:
        return False
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi <= lo:
        return False
    m = (hi - lo) * pad
    axis.set_ylim(lo - m, hi + m)
    return True


def fit_value_axes(axes, lines, window_values_fn=None, pad=0.06):
    """Fit each value axis to what is inside the current time window.

    Takes window_values as a parameter only so the interactive host can pass the
    binary-searching version it already uses on every wheel notch; the default
    is the same computation without that optimisation."""
    for axis in axes:
        if axis is None:
            continue
        fixed = getattr(axis, "_lane_ylim", None)
        if fixed is not None:
            # Lanes are positions on a rack, not values.  Fitting them to the
            # window would stretch the stack back over the whole frame on the
            # first zoom -- which is exactly what the lanes were keeping it off.
            axis.set_ylim(*fixed)
            continue
        if not _fit_one(axis, lines, window_values_fn, pad):
            continue
        band = getattr(axis, "_band_frac", 1.0)
        if band < 1.0:
            # Graph.right_frac: an ordinary fit, then stretch the SCALE so the
            # data occupies only that fraction of the frame, measured from the
            # bottom.  Done here rather than once in build_figure because the
            # interactive host refits on every zoom, and a band applied once
            # would spring back to full height the moment anyone looked closer.
            lo, hi = axis.get_ylim()
            if hi > lo:
                axis.set_ylim(lo, lo + (hi - lo) / band)
