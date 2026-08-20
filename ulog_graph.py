#!/usr/bin/env python3
"""ulog_graph.py -- the thermal/GPS plot, and the CLI front door for the toolkit.

This file owns ONE plot: the thermal overlay this tool started life as.  It
overlays, on one time axis:

  * every temperature channel the log carries (auto-discovered), in degC
  * GPS satellite count (and fix_type), a plain count
  * the RATE of change of one temperature channel, in degC/min

Each series has a checkbox; the panel on the left doubles as the legend (labels
are colored to match their line).  Armed stretches of the flight are shaded, so
"the IMU spiked at 30 min" immediately reads as "the IMU spiked mid-flight".

Everything reusable lives in ulog_common.py; the plot registry that lets the
browser and the PDF exporter iterate over more than this one plot is in
ulog_plots.py.

Usage:
    ulog_graph.py                                 # browse: pick a log in a GUI
    ulog_graph.py <log.ulg>                       # browse, with that log loaded
    ulog_graph.py <log.ulg> --classic             # just this plot, one window
    ulog_graph.py <log.ulg> --list                # what channels are in there?
    ulog_graph.py <log.ulg> --save out.png        # headless render of this plot
    ulog_graph.py --pdf out.pdf <a.ulg> <b.ulg>   # headless multi-log report
    ulog_graph.py <log.ulg> --smooth 60 --abs     # tune the rate series
    ulog_graph.py <log.ulg> --add battery_status.voltage_v

Acronyms: FC = flight controller, IMU = inertial measurement unit,
GPS = global positioning system, ULog = PX4's binary log format.
Requires pyulog + matplotlib + numpy (all in the agc_CTOL_SE3-rotopy venv).
"""
import argparse
import os
import sys

try:
    import numpy as np
    from pyulog import ULog
except ImportError as e:  # pragma: no cover - dependency guard
    sys.exit(f"needs numpy + pyulog: {e}  (pip install pyulog)")

from ulog_common import (ARMED_TOPIC, C_ADD, C_FIX, C_INK, C_MUTED, C_RATE,
                         C_SATS, C_SURFACE, FAMILY_STYLE, PlotCtx, Series,
                         _clean, _get, _rescale, _style_axis, _time_min,
                         add_mouse_navigation, armed_spans, check_panel,
                         draw_armed, draw_mode_changes, duration_min,
                         mode_changes, mode_key, nav_hint, parse_ref,
                         sliding_slope,
                         style_time_axis)

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

THERMAL_TOPICS = TEMP_TOPICS + GPS_TOPICS + [ARMED_TOPIC,
                                             "vehicle_status"]  # mode overlay

# The cleanest channel to differentiate: vehicle_imu_status publishes the driver's
# *averaged* temperature (quantum ~1e-5 degC) whereas the raw sensor_* channels are
# quantized as coarsely as 0.125 degC, which a derivative turns into pure hash.
DEFAULT_RATE_SRC = "vehicle_imu_status[0].temperature_accel"


def _family(topic):
    if topic in ("sensor_accel", "sensor_gyro", "vehicle_imu_status"):
        return "imu"
    if topic in ("sensor_baro", "vehicle_air_data"):
        return "baro"
    if topic == "sensor_mag":
        return "mag"
    return "other"


def _short(topic, mid, field):
    """Compact channel name for the checkbox panel."""
    t = topic.replace("vehicle_", "").replace("sensor_", "")
    f = field.replace("temperature_", "").replace("temperature", "")
    f = f.replace("baro_temp_celcius", "baro").replace("_celcius", "")
    name = f"{t}[{mid}]" if mid else t
    return f"{name} {f}".strip()


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


# --- rendering --------------------------------------------------------------

def build_thermal(ulog, ctx=None, path=""):
    """The thermal/GPS figure.  Signature matches every other plot builder so the
    registry can call them uniformly."""
    import matplotlib.pyplot as plt

    ctx = ctx or PlotCtx()
    series, notes = build_series(ulog, ctx.smooth, ctx.use_abs, ctx.rate_src,
                                 list(ctx.adds))
    for n in notes:
        ctx.note(n)
    if not series:
        ctx.note("no temperature or GPS topics in this log -- nothing to plot")
        return None

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

    spans = armed_spans(ulog)
    span_art = draw_armed(ax, spans)

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

    style_time_axis(ax)
    ax.set_ylabel("temperature (degC)", fontsize=9)
    ax_cnt.set_ylabel("satellites / fix_type", fontsize=9)
    ax_rate.set_ylabel(("|dT/dt|" if ctx.use_abs else "dT/dt") + " (degC/min)",
                       fontsize=9)
    ax_add.set_ylabel("added channel", fontsize=9)
    _style_axis(ax, "#a33603")
    _style_axis(ax_cnt, C_SATS)
    _style_axis(ax_rate, C_RATE)
    _style_axis(ax_add, C_ADD)

    dur = ulog.last_timestamp - ulog.start_timestamp
    temps = [s for s in series if s.group == "temp"]
    peak = max((float(np.nanmax(s.y)) for s in temps), default=float("nan"))
    # The plot's NAME is the title now that it is one of several; the filename
    # moves to the subtitle so a standalone --save PNG still identifies itself
    # without the browser chrome around it.
    fig.text(0.20, 0.955, "Thermal / GPS", color=C_INK, fontsize=13,
             fontweight="bold", ha="left")
    who = f"{os.path.basename(path)}   |   " if path else ""
    fig.text(0.20, 0.925,
             f"{who}{dur/6e7:.1f} min   |   {len(temps)} temperature channel(s), "
             f"peak {peak:.1f} degC   |   rate window {ctx.smooth:.0f} s",
             color=C_MUTED, fontsize=9, ha="left")

    def refresh():
        _rescale(ax, [s.line for s in series if s.group == "temp"])
        _rescale(ax_cnt, [s.line for s in series if s.group == "sats"])
        _rescale(ax_rate, [s.line for s in series if s.group == "rate"])
        add_lines = [s.line for s in series if s.group == "add"]
        ax_add.set_visible(any(ln.get_visible() for ln in add_lines))
        _rescale(ax_add, add_lines)

    base_extra = [("armed (shaded)", span_art, True)] if span_art else []
    # Flight-mode overlay: a rule on every panel at each mode change, named on
    # one of them.  Toggleable, because a log that flickers between Position and
    # Hold 52 times (d05a88e3) is unreadable with it on and unanswerable with it
    # off.  min_gap keeps the LABELS legible without dropping any rule.
    mode_art, mode_codes = draw_mode_changes(
        [ax, ax_cnt, ax_rate, ax_add], mode_changes(ulog), text_ax=ax_cnt,
        min_gap=max(duration_min(ulog), 1.0) * 0.035)
    extra = list(base_extra)
    mode_art += mode_key(fig, 0.965, 0.012, mode_codes)
    if mode_art:
        extra.append(("mode changes", mode_art, True))

    # +5 leaves room for the three group masters and the armed/mode toggles,
    # which are panel rows that aren't series.
    h = min(0.86, 0.035 * (len(series) + 5) + 0.05)
    check_panel(fig, [0.015, 0.89 - h, 0.19, h], series,
                [("temp", "ALL temperatures"), ("sats", "ALL gps"),
                 ("rate", None), ("add", "ALL added")],
                extra=extra, on_change=refresh)
    refresh()
    # After refresh(), so "home" is the fitted view rather than the raw one.
    add_mouse_navigation(fig, [ax, ax_cnt, ax_rate, ax_add],
                         page_scroll=ctx.page_scroll)
    fig.text(0.20, 0.036, nav_hint(ctx.page_scroll), color=C_MUTED, fontsize=8,
             ha="left")
    return fig


# --- entry points -----------------------------------------------------------

def list_channels(path):
    """Print what the log carries, without opening a window."""
    ulog = ULog(path, message_name_filter_list=THERMAL_TOPICS)
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


def graph(path=None, smooth=31.0, use_abs=False, rate_src=None, adds=(),
          save=None, show=True, classic=False, pdf=None, extra_paths=()):
    """Build (and show/save) the plots.  Factored out of main() so the mavTerminal
    shell can call it directly, the same way it calls ulog_diag.diagnose.

    Four modes, in precedence order:
      pdf     -> headless multi-log report
      save    -> headless PNG of the thermal plot alone (the pre-browser behaviour)
      classic -> the thermal plot alone, one blocking window
      else    -> the browser: every registered plot, scrollable, with a log picker
    """
    import matplotlib

    ctx = PlotCtx(smooth=smooth, use_abs=use_abs, rate_src=rate_src,
                  adds=list(adds))

    if pdf:
        matplotlib.use("Agg")
        import ulog_report
        paths = ([path] if path else []) + list(extra_paths)
        return ulog_report.export_pdf(paths, pdf, ctx)

    if save:
        if not _has_display():
            matplotlib.use("Agg")       # render to file with no window manager
        import matplotlib.pyplot as plt
        ulog = ULog(path, message_name_filter_list=sorted(set(
            THERMAL_TOPICS + [parse_ref(r)[0] for r in adds
                              if _safe_topic(r)])))
        fig = build_thermal(ulog, ctx, path)
        for n in ctx.notes:
            print(f"  note: {n}")
        if fig is None:
            sys.exit("Nothing plottable in this log (no temperature or GPS topics).")
        fig.savefig(save, dpi=130, facecolor=C_SURFACE)
        print(f"  saved {save}")
        if show and _has_display():
            plt.show()
        return fig

    if not _has_display():
        sys.exit("No display available. Re-run with --save <file.png> or "
                 "--pdf <file.pdf>.")

    if classic:
        import matplotlib.pyplot as plt
        ulog = ULog(path, message_name_filter_list=sorted(set(
            THERMAL_TOPICS + [parse_ref(r)[0] for r in adds if _safe_topic(r)])))
        fig = build_thermal(ulog, ctx, path)
        for n in ctx.notes:
            print(f"  note: {n}")
        if fig is None:
            sys.exit("Nothing plottable in this log (no temperature or GPS topics).")
        plt.show()
        return fig

    import log_browser
    return log_browser.browse(([path] if path else []) + list(extra_paths), ctx)


def _safe_topic(ref):
    try:
        parse_ref(ref)
        return True
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser(description="Plot a PX4 .ulg as toggleable time series")
    ap.add_argument("log", nargs="*", help="path(s) to .ulg files (omit to browse)")
    ap.add_argument("--list", action="store_true",
                    help="print the channels this log carries and exit")
    ap.add_argument("--save", metavar="PNG", help="render the thermal plot to a PNG")
    ap.add_argument("--pdf", metavar="PDF",
                    help="render every plot for every given log into one PDF")
    ap.add_argument("--classic", action="store_true",
                    help="just the thermal plot in one window, no browser")
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

    for p in a.log:
        if not os.path.isfile(p):
            sys.exit(f"no such file: {p}")
    if a.list:
        if not a.log:
            sys.exit("--list needs a log file")
        list_channels(a.log[0])
        return
    if (a.save or a.classic) and not a.log:
        sys.exit("--save and --classic need a log file")
    graph(a.log[0] if a.log else None, smooth=a.smooth, use_abs=a.use_abs,
          rate_src=a.rate_src, adds=a.add, save=a.save, show=not a.no_show,
          classic=a.classic, pdf=a.pdf, extra_paths=a.log[1:])


if __name__ == "__main__":
    main()
