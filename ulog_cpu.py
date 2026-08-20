#!/usr/bin/env python3
"""ulog_cpu.py -- processor load and everything that shows the board struggling.

Four stacked panels on one time axis, answering "did the flight controller run
out of CPU, and if so what did it drop?":

  1. load        -- CPU and RAM, the two numbers PX4 measures directly
  2. falling behind -- the EKF's own time slip, and each IMU's publish rate
  3. logging     -- SD buffer pressure and the MAVLink rate throttle
  4. band        -- armed, logging, and the moments something was actually dropped

Why the last three panels exist
-------------------------------
`cpuload.load` is a 2 Hz average over the whole system, and it saturates as a
diagnostic long before the board does: on this project's logs it sits at 36% and
peaks at 64% even on flights where the estimator fell 0.3 s behind real time.  A
percentage that never reaches 100 cannot tell you whether work was being dropped.

The panels below it are the CONSEQUENCES, which are what you actually care
about, and each is a different failure:

  * `estimator_status.time_slip` -- the EKF telling you how far behind the wall
    clock it has fallen.  Reported as an absolute offset, so this plot de-biases
    it against its first sample; the SLOPE is the signal.
  * `vehicle_imu_status.*_rate_hz` -- a driver that misses its schedule publishes
    slower.  Drawn as a percentage of each channel's OWN median, because the
    three IMUs run at 797 / 803 / 746 Hz by design and a shared axis in Hz would
    make that look like a fault.
  * `logger_status.buffer_used_bytes` and `message_gaps` -- the SD writer's
    backlog, and the count of log messages it gave up on.  A gap here is data
    you will look for later and not find.
  * `telemetry_status.rate_multiplier` -- MAVLink throttles its own streams when
    it cannot keep up, so a multiplier below 1 is the radio link reporting the
    same pressure from the other side.

Acronyms: CPU = central processing unit, RAM = random-access memory,
EKF = extended Kalman filter, IMU = inertial measurement unit,
SD = secure digital (the log card), MAVLink = the telemetry protocol.
"""
import os

import numpy as np

from ulog_common import (C_ARMED, C_BAD, C_INK, C_MUTED, C_SURFACE, PlotCtx,
                         Series, _clean, _get, _rescale, _style_axis, _time_min,
                         add_mouse_navigation, armed_spans, check_panel,
                         draw_armed, draw_band_rows, draw_mode_changes,
                         duration_min, field, has_topic, inst_color,
                         mode_changes, mode_key, nav_hint, spans_from_bool,
                         style_time_axis)

CPU_TOPICS = [
    "cpuload", "estimator_status", "vehicle_imu_status", "logger_status",
    "telemetry_status", "actuator_armed",
    "vehicle_status",          # flight-mode overlay
]

C_CPU = "#c0392b"      # red-ish: the headline number
C_RAM = "#2a78d6"      # blue
C_LOGGER = "#d2691e"   # orange
C_TELEM = "#1baf7a"    # aqua
C_PUB = "#20222b"

# Where PX4 starts losing work.  Not a published constant -- NuttX has no single
# threshold -- but above ~80% the 2 Hz average is hiding peaks that are at 100%,
# and every log in this project that dropped anything was above it.
LOAD_WARN = 80.0

# --- layout, in inches (see ulog_accel for why this is not in fractions) -----
PANEL_IN = [("cpu", 2.05), ("slip", 1.85), ("log", 1.85)]
GAP_IN = 0.48
TOP_IN = 0.95
BOTTOM_IN = 1.35       # band x label + nav hint + mode key + margins
BAND_ROW_IN = 0.52
BAND_PAD_IN = 0.30
BAND_MIN_IN = 1.25
BAND_MAX_IN = 9.00
MIN_EVENT_FRAC = 0.003
PAGE_PX_PER_IN = 78


def _instances(ulog, topic):
    return sorted({d.multi_id for d in ulog.data_list if d.name == topic})


# --- panel 1: what PX4 measures directly -------------------------------------

def _series_load(ulog, ctx):
    """CPU and RAM, both as percentages, deliberately on ONE axis.

    They are the same unit and the same question -- "how much of the board is
    spoken for" -- and splitting them across a twin axis would let two unrelated
    scales make a flat RAM trace look like it was moving.
    """
    series = []
    t, load = field(ulog, "cpuload", "load", scale=100.0)
    if load.size:
        series.append(Series("cpuload.load", "CPU load", t, load, "cpu", C_CPU,
                             lw=1.6, visible=True, zorder=4))
    t, ram = field(ulog, "cpuload", "ram_usage", scale=100.0)
    if ram.size:
        series.append(Series("cpuload.ram_usage", "RAM used", t, ram, "cpu",
                             C_RAM, lw=1.4, visible=True))
    if not series:
        ctx.note("no cpuload in this log -- the board's own load measurement is "
                 "unavailable, so panels 2-4 are all there is")
    return series


# --- panel 2: the consequences ----------------------------------------------

def _series_slip(ulog, ctx):
    """EKF time slip (left) and IMU publish rate as % of nominal (right)."""
    series = []
    for i in _instances(ulog, "estimator_status"):
        d = _get(ulog, "estimator_status", i)
        if d is None or "time_slip" not in d.data:
            continue
        t, y = _clean(_time_min(ulog, d), d.data["time_slip"])
        if not y.size:
            continue
        # De-biased against the first sample: the absolute value is an arbitrary
        # offset (0.59 s on one log, 0.0 on another) and only the growth means
        # anything.  Stated in the label so the number is not silently changed.
        series.append(Series(f"estimator_status[{i}].time_slip",
                             f"EKF {i} time slip (-{y[0]:.2f}s)", t, y - y[0],
                             "slip", inst_color(i), lw=1.4, visible=True))

    for m in _instances(ulog, "vehicle_imu_status"):
        d = _get(ulog, "vehicle_imu_status", m)
        if d is None:
            continue
        for fname, kind, style in (("accel_rate_hz", "accel", "-"),
                                   ("gyro_rate_hz", "gyro", "--")):
            if fname not in d.data:
                continue
            t, y = _clean(_time_min(ulog, d), d.data[fname])
            med = float(np.median(y)) if y.size else 0.0
            if not y.size or med <= 0:
                continue
            series.append(Series(f"vehicle_imu_status[{m}].{fname}",
                                 f"IMU {m} {kind} rate ({med:.0f} Hz)",
                                 t, 100.0 * y / med, "rate", inst_color(m),
                                 ls=style, lw=1.1, alpha=0.8, visible=True))
    if not series:
        ctx.note("no estimator_status or vehicle_imu_status -- cannot show "
                 "whether anything was falling behind")
    return series


# --- panel 3: what the board was trying to get rid of ------------------------

def _series_logging(ulog, ctx):
    """SD buffer / MAVLink throttle (left, %) and throughput (right, kB/s)."""
    series = []
    d = _get(ulog, "logger_status")
    if d is not None:
        t = _time_min(ulog, d)
        used = np.asarray(d.data.get("buffer_used_bytes", []), dtype=float)
        size = np.asarray(d.data.get("buffer_size_bytes", []), dtype=float)
        if used.size and size.size and np.nanmax(size) > 0:
            tt, y = _clean(t, 100.0 * used / np.where(size > 0, size, np.nan))
            series.append(Series("logger_status.buffer_used", "SD buffer used",
                                 tt, y, "pct", C_LOGGER, lw=1.4, visible=True))
        if "write_rate_kb_s" in d.data:
            tt, y = _clean(t, d.data["write_rate_kb_s"])
            # The first sample is a startup artefact of several thousand kB/s
            # (total bytes over a near-zero interval) that would own the axis.
            if y.size > 2:
                tt, y = tt[1:], y[1:]
            series.append(Series("logger_status.write_rate_kb_s",
                                 "SD write rate", tt, y, "kb", C_LOGGER,
                                 ls="--", lw=1.1, alpha=0.85, visible=True))
    else:
        ctx.note("no logger_status in this log -- SD buffer pressure and "
                 "dropped log messages are unavailable (older firmware)")

    d = _get(ulog, "telemetry_status")
    if d is not None:
        t = _time_min(ulog, d)
        if "rate_multiplier" in d.data:
            tt, y = _clean(t, d.data["rate_multiplier"])
            series.append(Series("telemetry_status.rate_multiplier",
                                 "MAVLink rate multiplier", tt, 100.0 * y,
                                 "pct", C_TELEM, lw=1.4, visible=True))
        if "tx_rate_avg" in d.data:
            tt, y = _clean(t, np.asarray(d.data["tx_rate_avg"], dtype=float) / 1000.0)
            series.append(Series("telemetry_status.tx_rate_avg",
                                 "MAVLink tx rate", tt, y, "kb", C_TELEM,
                                 ls="--", lw=1.1, alpha=0.85, visible=True))
    return series


# --- panel 4 -----------------------------------------------------------------

def _counter_spans(t, counter, hold):
    """Spans at each increment of a cumulative counter (see ulog_accel)."""
    c = np.asarray(counter, dtype=float)
    return [(float(t[i]), float(t[i]) + hold)
            for i in np.flatnonzero(np.diff(c) > 0) + 1]


def _fault_rows(ulog, ctx, hold):
    """([(label, spans, colour)], n_clean) -- when work was actually dropped."""
    rows, n_clean = [], 0

    def add(label, spans, color, fault=True):
        nonlocal n_clean
        if not spans:
            n_clean += 1
        rows.append((label, spans, color if spans or not fault else color))

    rows.append(("armed", armed_spans(ulog), C_ARMED))

    d = _get(ulog, "logger_status")
    if d is not None:
        t = _time_min(ulog, d)
        if "is_logging" in d.data:
            v = np.asarray(d.data["is_logging"], dtype=float) > 0.5
            rows.append(("logging", spans_from_bool(t, v), C_LOGGER))
        if "message_gaps" in d.data:
            # Cumulative, so the increments are the events: each one is log data
            # the writer gave up on.
            add("SD: log messages DROPPED",
                _counter_spans(t, d.data["message_gaps"], hold), C_BAD)

    t, load = field(ulog, "cpuload", "load", scale=100.0)
    if load.size:
        add(f"CPU over {LOAD_WARN:.0f}%", spans_from_bool(t, load > LOAD_WARN),
            C_BAD)

    d = _get(ulog, "telemetry_status")
    if d is not None:
        t = _time_min(ulog, d)
        if "tx_buffer_overruns" in d.data:
            add("MAVLink: tx buffer overrun",
                _counter_spans(t, d.data["tx_buffer_overruns"], hold), C_BAD)
        if "rate_multiplier" in d.data:
            v = np.asarray(d.data["rate_multiplier"], dtype=float) < 0.999
            add("MAVLink throttled its own streams",
                spans_from_bool(t, v), C_BAD)

    # A publish rate below 90% of its own median is the driver missing its slot.
    for m in _instances(ulog, "vehicle_imu_status"):
        d = _get(ulog, "vehicle_imu_status", m)
        if d is None or "accel_rate_hz" not in d.data:
            continue
        t, y = _clean(_time_min(ulog, d), d.data["accel_rate_hz"])
        med = float(np.median(y)) if y.size else 0.0
        if med <= 0:
            continue
        add(f"IMU {m} publish rate < 90%", spans_from_bool(t, y < 0.9 * med),
            C_BAD)

    return rows, n_clean


# --- the figure --------------------------------------------------------------

def build_cpu(ulog, ctx=None, path=""):
    """The processor-load figure.  Same signature as every plot builder."""
    import matplotlib.pyplot as plt

    ctx = ctx or PlotCtx()

    if not any(has_topic(ulog, t) for t in
               ("cpuload", "logger_status", "estimator_status",
                "vehicle_imu_status")):
        ctx.note("no load, logging or scheduling topics in this log")
        return None

    dur = duration_min(ulog) or 1.0
    hold = dur * MIN_EVENT_FRAC

    series = _series_load(ulog, ctx)
    series += _series_slip(ulog, ctx)
    series += _series_logging(ulog, ctx)
    if not series:
        ctx.note("nothing load-related in this log -- nothing to plot")
        return None

    rows, n_clean = _fault_rows(ulog, ctx, hold)

    band_in = min(max(len(rows) * BAND_ROW_IN + BAND_PAD_IN, BAND_MIN_IN),
                  BAND_MAX_IN)
    fig_h = (TOP_IN + sum(h for _k, h in PANEL_IN) + GAP_IN * len(PANEL_IN)
             + band_in + BOTTOM_IN)

    fig = plt.figure(figsize=(15, fig_h), facecolor=C_SURFACE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(
            f"logGraph cpu - {os.path.basename(path)}")

    left, width = 0.260, 0.655

    def _f(inches):
        return inches / fig_h

    rects, bottom = {}, BOTTOM_IN
    rects["band"] = (_f(bottom), _f(band_in))
    bottom += band_in + GAP_IN
    for key, h in reversed(PANEL_IN):
        rects[key] = (_f(bottom), _f(h))
        bottom += h + GAP_IN

    ax_cpu, ax_slip, ax_log, ax_band = [
        fig.add_axes([left, rects[k][0], width, rects[k][1]], facecolor=C_SURFACE)
        for k in ("cpu", "slip", "log", "band")]
    for a in (ax_cpu, ax_slip, ax_log):
        a.sharex(ax_band)
    ax_rate = ax_slip.twinx()
    ax_kb = ax_log.twinx()
    for a in (ax_rate, ax_kb):
        a.set_facecolor("none")

    axis_of = {"cpu": ax_cpu, "slip": ax_slip, "rate": ax_rate,
               "pct": ax_log, "kb": ax_kb}

    armed_art = []
    for a in (ax_cpu, ax_slip, ax_log, ax_band):
        armed_art += draw_armed(a, armed_spans(ulog))

    warn_line = ax_cpu.axhline(LOAD_WARN, color=C_BAD, lw=1.1, ls="--",
                               alpha=0.8, zorder=1)
    # The LABEL is placed in data coordinates, so on a log that never approaches
    # the threshold it lands above the axes and prints over the title.  The line
    # is harmless when clipped; the text is not, so it is added only if the
    # threshold is actually on screen after the rescale below.
    warn_text = ax_cpu.text(0.012, LOAD_WARN, f"{LOAD_WARN:.0f}%",
                            transform=ax_cpu.get_yaxis_transform(), color=C_BAD,
                            fontsize=7, va="bottom", ha="left")
    ax_slip.axhline(0.0, color=C_MUTED, lw=1.0, ls=":", alpha=0.6, zorder=1)
    ax_rate.axhline(100.0, color=C_MUTED, lw=1.0, ls=":", alpha=0.6, zorder=1)

    draw_band_rows(ax_band, rows, ylabel="dropped work",
                   empty_msg="no logging or load flags in this log",
                   min_width=hold)

    for s in series:
        (line,) = axis_of[s.group].plot(
            s.t, s.y, color=s.color, ls=s.ls, lw=s.lw, label=s.label,
            drawstyle=s.drawstyle, alpha=s.alpha,
            zorder=s.zorder if s.zorder is not None else 3)
        line.set_visible(s.visible)
        s.line = line

    mode_art, mode_codes = draw_mode_changes(
        [ax_cpu, ax_slip, ax_log, ax_band], mode_changes(ulog), text_ax=ax_slip,
        min_gap=dur * 0.035)

    for a in (ax_cpu, ax_slip, ax_log):
        style_time_axis(a, label=False)
        a.tick_params(axis="x", labelbottom=False)
    style_time_axis(ax_band)

    ax_cpu.set_ylabel("CPU / RAM used (%)", fontsize=9)
    ax_slip.set_ylabel("EKF time slip (s, de-biased)", fontsize=9)
    ax_rate.set_ylabel("IMU publish rate (% of own median)", fontsize=9)
    ax_log.set_ylabel("buffer / throttle (%)", fontsize=9)
    ax_kb.set_ylabel("throughput (kB/s, dashed)", fontsize=9)
    for a in (ax_cpu, ax_slip, ax_log):
        _style_axis(a, C_INK)
    for a in (ax_rate, ax_kb):
        _style_axis(a, C_MUTED)

    fig.text(left, 1.0 - _f(0.35), "Processor load and dropped work",
             color=C_INK, fontsize=13, fontweight="bold", ha="left")
    who = f"{os.path.basename(path)}   |   " if path else ""
    t, load = field(ulog, "cpuload", "load", scale=100.0)
    peak = (f"CPU {np.median(load):.0f}% median, {load.max():.0f}% peak"
            if load.size else "no cpuload")
    fig.text(left, 1.0 - _f(0.62), f"{who}{dur:.1f} min   |   {peak}",
             color=C_MUTED, fontsize=9, ha="left")
    if n_clean:
        ctx.note(f"{n_clean} of the checked drop conditions never fired -- they "
                 f"are drawn as empty rows, not omitted")

    def refresh():
        for group, a in (("cpu", ax_cpu), ("slip", ax_slip), ("rate", ax_rate),
                         ("pct", ax_log), ("kb", ax_kb)):
            _rescale(a, [s.line for s in series if s.group == group])
        lo, hi = ax_cpu.get_ylim()
        on_screen = lo <= LOAD_WARN <= hi
        warn_line.set_visible(on_screen)
        warn_text.set_visible(on_screen)

    extra = []
    mode_art += mode_key(fig, left + width, _f(0.10), mode_codes)
    if mode_art:
        extra.append(("mode changes", mode_art, True))
    if armed_art:
        extra.append(("armed (shaded)", armed_art, True))

    cb_top = rects["cpu"][0] + rects["cpu"][1]
    cb_bot = rects["band"][0]

    def _anchor(key):
        b, ph = rects[key]
        return (b + ph / 2 - cb_bot) / (cb_top - cb_bot)

    check_panel(fig, [0.012, cb_bot, 0.155, cb_top - cb_bot], series,
                [("cpu", "ALL load"), ("slip", "ALL time slip"),
                 ("rate", "ALL publish rates"), ("pct", "ALL buffer/throttle"),
                 ("kb", "ALL throughput")],
                extra=extra, on_change=refresh,
                anchors={"cpu": _anchor("cpu"), "slip": _anchor("slip"),
                         "rate": _anchor("slip"), "pct": _anchor("log"),
                         "kb": _anchor("log")})
    refresh()
    add_mouse_navigation(fig, [ax_cpu, ax_slip, ax_rate, ax_log, ax_kb, ax_band],
                         page_scroll=ctx.page_scroll, fixed_y=[ax_band])
    fig.text(left, _f(0.32), nav_hint(ctx.page_scroll), color=C_MUTED,
             fontsize=8, ha="left")
    fig._page_height = int(round(fig_h * PAGE_PX_PER_IN))
    return fig
