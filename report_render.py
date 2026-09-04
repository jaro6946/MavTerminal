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

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from report_model import ALIGNMENTS
from ulog_cache import start_epoch
from ulog_common import (C_GRID, C_INK, C_MUTED, C_SURFACE, armed_spans,
                         decimate, field, parse_ref, style_time_axis)

__all__ = ["SERIES_COLORS", "LOG_STYLES", "STAT_COLS", "DRAW_PX",
           "series_color", "log_style", "short_ref", "align_offset",
           "absolute_base", "stats_of", "fmt_stat", "gather_series",
           "auto_axis", "assign_axes", "build_figure", "fit_value_axes",
           "window_of"]

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
]

# Log identity.  Style rather than colour, so that "which channel" and "which
# log" are read off two independent visual channels instead of competing for hue.
LOG_STYLES = ["-", "--", "-.", (0, (1, 1.4))]

DRAW_PX = 1400              # decimation budget; a little over the widest canvas

STAT_COLS = ["n", "min", "max", "mean", "median", "std", "first", "last"]


def series_color(i):
    return SERIES_COLORS[i % len(SERIES_COLORS)]


def log_style(i):
    return LOG_STYLES[i % len(LOG_STYLES)]


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
    problems = []
    base = (absolute_base([ulogs[n] for n in names])
            if graph.align == "absolute" else None)
    series = []
    for li, name in enumerate(names):
        ulog = ulogs[name]
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
            if t.size == 0:
                problems.append(f"{name}: {short_ref(ref)} absent")
                continue
            if graph.normalise:
                lo, hi = float(y.min()), float(y.max())
                y = (y - lo) / (hi - lo) if hi > lo else np.zeros_like(y)
            series.append({
                "ref": ref, "log": name,
                "label": f"{short_ref(ref)} · {os.path.splitext(name)[0]}",
                "t": t - off, "y": y,
                "color": series_color(fi), "ls": log_style(li),
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


def assign_axes(graph, series):
    """Stamp each series with its final axis.  Returns the AUTOMATIC choice too,
    so a caller can show which assignments were made for the user rather than
    by them."""
    auto = auto_axis(graph, series)
    for s in series:
        s["axis"] = graph.axis.get(s["ref"], auto.get(s["ref"], "left"))
    return auto


# --- figure ------------------------------------------------------------------

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
    fig.subplots_adjust(left=0.055, right=0.945, top=0.90, bottom=0.16)
    ax = fig.add_subplot(111)
    ax.set_facecolor(C_SURFACE)
    axr = None

    lines = []
    for s in series:
        target = ax
        if s["axis"] == "right":
            if axr is None:
                axr = ax.twinx()
                axr.set_facecolor("none")
            target = axr
        td, yd = decimate(s["t"], s["y"], DRAW_PX)
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
        fig.text(0.055, 0.955, note, fontsize=8, color=C_MUTED, va="bottom")

    return fig, ax, axr, lines


def _legend(ax, axr, series, graph, auto, ncol=2):
    """A key that matches what is on the plot.

    Two dimensions are encoded -- colour is the channel, line style is the log --
    and a combined key needs one entry per (log x channel) pair, which is
    eighteen entries for three logs and six channels.  Split into a colour block
    and a style block, it is nine.

    But that split is only worth its confusion when both dimensions actually
    vary.  Plot one channel across two logs and the colour block becomes a third
    entry for two lines, swatched in a grey that appears nowhere on the plot.  So
    in that case -- and in the mirror case of one log and several channels --
    the key reverts to one entry per LINE, drawn exactly as that line is drawn.
    """
    fields, logs = [], []
    for s in series:
        if s["ref"] not in [f for f, _ in fields]:
            fields.append((s["ref"], s["color"]))
        if s["log"] not in [l for l, _ in logs]:
            logs.append((s["log"], s["ls"]))

    if len(fields) <= 1 or len(logs) <= 1:
        one_channel = len(fields) <= 1
        handles = [Line2D([], [], color=s["color"], ls=s["ls"], lw=1.8,
                          label=(os.path.splitext(s["log"])[0][:38] if one_channel
                                 else short_ref(s["ref"])))
                   for s in series]
        # The single channel's name would otherwise be lost with its key entry.
        if one_channel and fields and not graph.normalise:
            ax.set_ylabel(short_ref(fields[0][0]), fontsize=9, color=C_MUTED)
    else:
        handles = [Line2D([], [], color=c, lw=2,
                          label=short_ref(r) + (" ›" if graph.axis.get(
                              r, auto.get(r, "left")) == "right" else ""))
                   for r, c in fields]
        # "log:" prefixed and greyed, so a style swatch is not mistaken for a
        # series that ought to be on the plot in that colour.
        handles += [Line2D([], [], color=C_MUTED, lw=1.6, ls=ls,
                           label="log: " + os.path.splitext(n)[0][:34])
                    for n, ls in logs]
    # Anchored to the FIGURE, not the axes.  Anchored to the axes, the offset has
    # to be expressed in axes fractions, which means it changes meaning every
    # time the axes are resized to make room -- the guessed offset that put this
    # key off the bottom of a 4.3-inch card while looking right on a 5-inch one.
    # Pinned to the figure, the key stays put and _fit_legend moves the AXES
    # instead, which is the thing that has room to give.
    leg = ax.legend(handles=handles, loc="lower left",
                    bbox_to_anchor=(0.055, 0.012),
                    bbox_transform=ax.figure.transFigure,
                    ncol=ncol, fontsize=8, frameon=False, handlelength=2.6,
                    columnspacing=1.4)
    leg.set_in_layout(False)
    if axr is not None:
        axr.set_ylabel("right scale ›", fontsize=8, color=C_MUTED)


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


def fit_value_axes(axes, lines, window_values_fn=None, pad=0.06):
    """Fit each value axis to what is inside the current time window.

    Takes window_values as a parameter only so the interactive host can pass the
    binary-searching version it already uses on every wheel notch; the default
    is the same computation without that optimisation."""
    for axis in axes:
        if axis is None:
            continue
        mine = [ln for ln, _ in lines if ln.axes is axis]
        if not mine:
            continue
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
            continue
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        lo, hi = float(np.min(v)), float(np.max(v))
        if hi > lo:
            m = (hi - lo) * pad
            axis.set_ylim(lo - m, hi + m)
