#!/usr/bin/env python3
"""ulog_cpu.py -- processor load, link traffic, and everything that shows the
board struggling.

Five stacked panels on one time axis, answering "did the flight controller run
out of CPU or bandwidth, and if so what did it drop?":

  1. load        -- CPU and RAM, the two numbers PX4 measures directly
  2. falling behind -- the EKF's own time slip, and each IMU's publish rate
  3. logging     -- SD buffer pressure and the MAVLink rate throttle
  4. links       -- the companion/DDS bridge traffic and the MAVLink byte rates
  5. band        -- armed, logging, and the moments something was actually dropped

Measuring the DDS bridge (panel 4)
----------------------------------
There is no byte counter for it.  PX4's uXRCE-DDS client publishes no status
topic to uORB, so a ULog contains no direct record of what crossed the bridge --
searching for one is the first thing to get out of the way.

What a log DOES contain is every uORB topic the bridge WROTE.  When the
companion sends a setpoint, the bridge publishes it into uORB, the logger
records it, and its arrival rate is the inbound traffic level -- in messages per
second directly, and in bytes per second by multiplying by the message size,
which the log itself carries in its format definitions.

Two honesties this panel has to keep:

  * The rate is INBOUND only.  Topics PX4 streams OUT to the companion are
    published by PX4 whether a bridge is running or not, so their rate is not
    evidence of outbound traffic and is not shown as if it were.
  * A logged rate is a LOWER BOUND on the published rate.  PX4's logger
    rate-limits some topics, so a measured 20 Hz can be a 50 Hz stream that was
    only logged 20 times a second -- and when the SD writer drops messages, a
    gap in this panel is a gap in the LOG, not necessarily in the link.  The
    band's "SD: log messages DROPPED" row is directly below for exactly that
    comparison: a heartbeat gap that coincides with a drop is not evidence, and
    one that does not is.

`offboard_control_mode` is the one unambiguous witness: nothing inside PX4
publishes it, so every message is an external commander saying "I am still
here".  Its rate is the companion's heartbeat and its GAPS are what trip the
offboard-loss failsafe at COM_OF_LOSS_T -- which is read from the log's own
parameters rather than assumed, because it is 1.0 s on one of this project's
airframes and 3.0 s on the other.

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
SD = secure digital (the log card), MAVLink = the telemetry protocol,
DDS = Data Distribution Service (the ROS 2 transport), uXRCE-DDS = the PX4
client that bridges uORB to it, uORB = PX4's internal publish/subscribe bus.
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
    "vehicle_status",          # flight-mode overlay, and the Offboard spans
    # Everything the companion writes across the bridge.  Listed even when a
    # given airframe never uses one: an absent topic costs a skipped lookup, and
    # a missing one that should be there is itself the finding.
    "offboard_control_mode", "trajectory_setpoint", "vehicle_attitude_setpoint",
    "vehicle_rates_setpoint", "vehicle_thrust_setpoint", "vehicle_torque_setpoint",
    "vehicle_command", "vehicle_command_ack",
    "vehicle_visual_odometry", "vehicle_mocap_odometry",
]

C_CPU = "#c0392b"      # red-ish: the headline number
C_RAM = "#2a78d6"      # blue
C_LOGGER = "#d2691e"   # orange
C_TELEM = "#1baf7a"    # aqua
C_PUB = "#20222b"

# One hue per bridge topic.  offboard_control_mode gets the near-black reference
# treatment: it is the heartbeat every other trace is read against.
#
# The last field is WHO ELSE PUBLISHES IT, and it is the difference between a
# measurement and a fiction.  Only some of these topics are external by nature:
#
#   "external"  -- nothing inside PX4 publishes it, so every message crossed a
#                  link.  offboard_control_mode is the cleanest case there is.
#   "offboard"  -- PX4's own flight tasks and controllers publish it in normal
#                  flight, and the bridge publishes it in Offboard.  Measured on
#                  Barometer_Primary_Datum: trajectory_setpoint runs at a steady
#                  5 Hz for all 33 minutes, of which Offboard is a few.  Counting
#                  that as companion traffic would be off by an order of
#                  magnitude, so these are MASKED to the Offboard spans.
BRIDGE_TOPICS = [
    ("offboard_control_mode", "offboard heartbeat", "#20222b", 2.2, "external"),
    ("vehicle_visual_odometry", "visual odometry in", "#f4511e", 1.2, "external"),
    ("vehicle_mocap_odometry", "mocap odometry in", "#6d4c41", 1.2, "external"),
    ("vehicle_command", "commands in (external)", "#fb8c00", 1.2, "external"),
    ("trajectory_setpoint", "trajectory setpoint", "#d81b60", 1.4, "offboard"),
    ("vehicle_attitude_setpoint", "attitude setpoint", "#3f51b5", 1.3, "offboard"),
    ("vehicle_rates_setpoint", "rates setpoint", "#00838f", 1.2, "offboard"),
    ("vehicle_thrust_setpoint", "thrust setpoint", "#43a047", 1.2, "offboard"),
    ("vehicle_torque_setpoint", "torque setpoint", "#8e24aa", 1.2, "offboard"),
]

# uORB field sizes, for turning a message rate into a byte rate.  These are the
# LOGGED struct sizes; the DDS wire adds CDR padding and per-sample overhead, so
# the result is an estimate of the payload and a floor on the real bandwidth.
UORB_SIZES = {"int8_t": 1, "uint8_t": 1, "bool": 1, "char": 1,
              "int16_t": 2, "uint16_t": 2, "int32_t": 4, "uint32_t": 4,
              "float": 4, "int64_t": 8, "uint64_t": 8, "double": 8}

# PX4's OFFBOARD nav_state.  Stable across every firmware in this project.
NAV_OFFBOARD = 14

# Fallback when the log carries no COM_OF_LOSS_T (PX4's own default).
OFFBOARD_LOSS_DEFAULT_S = 1.0

# Where PX4 starts losing work.  Not a published constant -- NuttX has no single
# threshold -- but above ~80% the 2 Hz average is hiding peaks that are at 100%,
# and every log in this project that dropped anything was above it.
LOAD_WARN = 80.0

# --- layout, in inches (see ulog_accel for why this is not in fractions) -----
PANEL_IN = [("cpu", 2.05), ("slip", 1.85), ("log", 1.85), ("link", 1.95)]
GAP_IN = 0.48
TOP_IN = 0.95
BOTTOM_IN = 1.35       # band x label + nav hint + mode key + margins
BAND_ROW_IN = 0.52
BAND_PAD_IN = 0.30
BAND_MIN_IN = 1.25
BAND_MAX_IN = 10.50
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



# --- panel 4: the companion / DDS bridge -------------------------------------

def msg_bytes(ulog, topic):
    """Serialised size of one message, from the log's OWN format definition.

    Reading the format rather than hard-coding a table: these structs change
    between firmware versions (vehicle_attitude_setpoint is 56 bytes on one of
    this project's boards and 40 on the other), and a stale table would quietly
    misreport bandwidth rather than fail.
    """
    fmt = getattr(ulog, "message_formats", {}).get(topic)
    if fmt is None:
        return 0
    return sum(UORB_SIZES.get(typ, 0) * max(arr, 1)
               for typ, arr, _name in fmt.fields)


def message_rate(t_min, dur_min, bin_min):
    """(bin_centres, messages per second) by counting into fixed bins.

    Counting rather than 1/dt on purpose.  An instantaneous reciprocal is
    unreadably noisy at these rates, and -- the part that matters -- it cannot
    represent a DROPOUT: no message means no sample, so 1/dt draws a straight
    line across the gap and the outage disappears.  A bin with nothing in it is
    a zero, which is what an outage should look like.
    """
    if t_min.size == 0 or dur_min <= 0 or bin_min <= 0:
        return np.array([]), np.array([])
    n = max(int(np.ceil(dur_min / bin_min)), 1)
    edges = np.linspace(0.0, n * bin_min, n + 1)
    counts, _ = np.histogram(t_min, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, counts / (bin_min * 60.0)


def _offboard_spans(ulog):
    """[(t0, t1)] where nav_state was OFFBOARD -- when a heartbeat gap is fatal."""
    d = _get(ulog, "vehicle_status")
    if d is None or "nav_state" not in d.data:
        return []
    t = _time_min(ulog, d)
    v = np.asarray(d.data["nav_state"], dtype=float) == NAV_OFFBOARD
    return spans_from_bool(t, v)


def _offboard_loss_timeout(ulog):
    """COM_OF_LOSS_T in seconds, from the log's own parameters."""
    try:
        v = float(ulog.initial_parameters.get("COM_OF_LOSS_T",
                                              OFFBOARD_LOSS_DEFAULT_S))
    except (TypeError, ValueError):
        return OFFBOARD_LOSS_DEFAULT_S
    return v if v > 0 else OFFBOARD_LOSS_DEFAULT_S


def _bin_min(dur_min):
    """Rate bin width: one second, widened on a long log to cap the point count."""
    return max(1.0 / 60.0, dur_min / 2000.0)


def _in_spans(t, spans):
    """Boolean mask: which of `t` fall inside any of `spans`."""
    m = np.zeros(np.shape(t), dtype=bool)
    for a, b in spans:
        m |= (t >= a) & (t <= b)
    return m


def _external_commands(ulog):
    """Timestamps of vehicle_command messages that came from OFF the board.

    `from_external` is the flag PX4 itself sets when a command arrived over a
    link rather than from an internal module, which is exactly the question --
    and it avoids having to guess whether a source_system of 10 is the companion
    or the ground station.
    """
    d = _get(ulog, "vehicle_command")
    if d is None:
        return np.array([])
    t = _time_min(ulog, d)
    if "from_external" in d.data:
        t = t[np.asarray(d.data["from_external"], dtype=float) > 0.5]
    return t[np.isfinite(t)]


def _series_link(ulog, ctx, dur):
    """Panel 4: inbound bridge rates (left) and byte rates (right).

    The total is drawn as well as the parts, because "is the companion talking"
    is usually the first question and reading it off eight overlapping traces is
    not an answer.
    """
    series = []
    bin_min = _bin_min(dur)
    total_hz, total_kb, centres = None, None, None
    present, gated = [], []
    off_spans = _offboard_spans(ulog)

    for topic, label, color, lw, origin in BRIDGE_TOPICS:
        d = _get(ulog, topic)
        if d is None:
            continue
        t = (_external_commands(ulog) if topic == "vehicle_command"
             else _time_min(ulog, d))
        t = t[np.isfinite(t)]
        if t.size == 0:
            continue
        c, hz = message_rate(t, dur, bin_min)
        if c.size == 0:
            continue
        if origin == "offboard":
            if not off_spans:
                # Nothing to attribute it to.  Dropping it beats drawing PX4's
                # own controller output under a "companion -> FC" axis label.
                gated.append(topic)
                continue
            # NaN, not zero: outside Offboard this topic's rate is not "no
            # companion traffic", it is "not measurable from here", and a zero
            # would be a claim.
            hz = np.where(_in_spans(c, off_spans), hz, np.nan)
            label += " (in Offboard)"
        present.append(topic)
        nbytes = msg_bytes(ulog, topic)
        series.append(Series(f"{topic}.rate", label, c, hz, "hz", color,
                             lw=lw, visible=True,
                             zorder=5 if topic == "offboard_control_mode" else 3))
        if nbytes:
            series.append(Series(f"{topic}.bytes", f"{label} ({nbytes} B)", c,
                                 hz * nbytes / 1000.0, "linkkb", color,
                                 ls="--", lw=1.0, alpha=0.85, visible=False))
        centres = c
        safe = np.nan_to_num(hz)
        total_hz = safe if total_hz is None else total_hz + safe
        if nbytes:
            add_kb = safe * nbytes / 1000.0
            total_kb = add_kb if total_kb is None else total_kb + add_kb

    if gated:
        ctx.note(f"{len(gated)} setpoint topic(s) are published by PX4's own "
                 f"controllers as well as by the bridge and this log never "
                 f"entered Offboard, so they cannot be attributed and are not "
                 f"drawn: {', '.join(gated)}")

    if not present:
        ctx.note("none of the companion-written topics are in this log -- "
                 "either no DDS bridge was running or the logger was not "
                 "recording them")
        return series

    if total_hz is not None and len(present) > 1:
        series.append(Series("bridge.total_rate", "ALL inbound (total)", centres,
                             total_hz, "hz", C_PUB, ls=":", lw=1.6, visible=False))
    if total_kb is not None:
        series.append(Series("bridge.total_bytes", "inbound (est. kB/s)",
                             centres, total_kb, "linkkb", C_PUB, ls="--",
                             lw=1.6, visible=True, zorder=5))

    # The MAVLink link's own byte counters, on the same byte axis -- the two
    # transports are answering the same question and belong on one scale.
    for m in _instances(ulog, "telemetry_status"):
        for key, label, ls in (("tx_rate_avg", "MAVLink tx", "-"),
                               ("rx_rate_avg", "MAVLink rx", ":")):
            t, y = field(ulog, "telemetry_status", key, mid=m)
            if t.size and np.isfinite(y).any():
                series.append(Series(f"telemetry_status[{m}].{key}",
                                     f"{label} [{m}]", t, y / 1000.0, "linkkb",
                                     C_TELEM, ls=ls, lw=1.1, alpha=0.85,
                                     visible=True))

    cfg = ulog.initial_parameters.get("UXRCE_DDS_CFG")
    if cfg is not None and float(cfg) == 0:
        ctx.note("UXRCE_DDS_CFG is 0 in this log -- the DDS client was NOT "
                 "started, so any inbound traffic below arrived over MAVLink")
    return series


# --- panel 5 -----------------------------------------------------------------

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
        if "rx_buffer_overruns" in d.data:
            add("MAVLink: rx buffer overrun",
                _counter_spans(t, d.data["rx_buffer_overruns"], hold), C_BAD)
        if "rx_message_lost_count" in d.data:
            add("MAVLink: rx messages LOST",
                _counter_spans(t, d.data["rx_message_lost_count"], hold), C_BAD)
        if "rx_parse_errors" in d.data:
            add("MAVLink: rx parse errors",
                _counter_spans(t, d.data["rx_parse_errors"], hold), C_BAD)
        if "rate_multiplier" in d.data:
            v = np.asarray(d.data["rate_multiplier"], dtype=float) < 0.999
            add("MAVLink throttled its own streams",
                spans_from_bool(t, v), C_BAD)
        if "heartbeat_type_onboard_controller" in d.data:
            v = np.asarray(d.data["heartbeat_type_onboard_controller"],
                           dtype=float) > 0.5
            rows.append(("companion heartbeat (MAVLink)",
                         spans_from_bool(t, v), C_TELEM))

    # -- the bridge ----------------------------------------------------------
    # Offboard mode is what makes a heartbeat gap fatal, so the fault row is the
    # INTERSECTION of the two: a gap on the bench with nothing flying is not a
    # finding, and the same gap in Offboard is a failsafe.
    off_spans = _offboard_spans(ulog)
    if off_spans:
        rows.append(("mode: Offboard", off_spans, C_TELEM))
    d = _get(ulog, "offboard_control_mode")
    if d is not None:
        t = _time_min(ulog, d)
        t = t[np.isfinite(t)]
        timeout_min = _offboard_loss_timeout(ulog) / 60.0
        gaps = []
        if t.size > 1:
            dt = np.diff(t)
            gaps = [(float(t[i]), float(t[i + 1]))
                    for i in np.flatnonzero(dt > timeout_min)]
        add(f"offboard heartbeat gap > {_offboard_loss_timeout(ulog):g} s",
            gaps, C_MUTED, fault=False)
        fatal = []
        for g0, g1 in gaps:
            for o0, o1 in off_spans:
                lo, hi = max(g0, o0), min(g1, o1)
                if hi > lo:
                    fatal.append((lo, hi))
        add("OFFBOARD heartbeat lost WHILE IN OFFBOARD", fatal, C_BAD)
    elif has_topic(ulog, "vehicle_status"):
        ctx.note("no offboard_control_mode in this log -- no external commander "
                 "(companion or ground station) ever sent one")

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
    series += _series_link(ulog, ctx, dur)
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

    ax_cpu, ax_slip, ax_log, ax_link, ax_band = [
        fig.add_axes([left, rects[k][0], width, rects[k][1]], facecolor=C_SURFACE)
        for k in ("cpu", "slip", "log", "link", "band")]
    for a in (ax_cpu, ax_slip, ax_log, ax_link):
        a.sharex(ax_band)
    ax_rate = ax_slip.twinx()
    ax_kb = ax_log.twinx()
    ax_linkkb = ax_link.twinx()
    for a in (ax_rate, ax_kb, ax_linkkb):
        a.set_facecolor("none")

    axis_of = {"cpu": ax_cpu, "slip": ax_slip, "rate": ax_rate,
               "pct": ax_log, "kb": ax_kb, "hz": ax_link, "linkkb": ax_linkkb}

    armed_art = []
    for a in (ax_cpu, ax_slip, ax_log, ax_link, ax_band):
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
        [ax_cpu, ax_slip, ax_log, ax_link, ax_band], mode_changes(ulog),
        text_ax=ax_slip, min_gap=dur * 0.035)

    for a in (ax_cpu, ax_slip, ax_log, ax_link):
        style_time_axis(a, label=False)
        a.tick_params(axis="x", labelbottom=False)
    style_time_axis(ax_band)

    ax_cpu.set_ylabel("CPU / RAM used (%)", fontsize=9)
    ax_slip.set_ylabel("EKF time slip (s, de-biased)", fontsize=9)
    ax_rate.set_ylabel("IMU publish rate (% of own median)", fontsize=9)
    ax_log.set_ylabel("buffer / throttle (%)", fontsize=9)
    ax_kb.set_ylabel("throughput (kB/s, dashed)", fontsize=9)
    ax_link.set_ylabel("inbound rate (msg/s)\nover the link, into uORB", fontsize=9)
    ax_linkkb.set_ylabel("link (kB/s, dashed)", fontsize=9)
    for a in (ax_cpu, ax_slip, ax_log, ax_link):
        _style_axis(a, C_INK)
    for a in (ax_rate, ax_kb, ax_linkkb):
        _style_axis(a, C_MUTED)

    fig.text(left, 1.0 - _f(0.35), "Processor load, links and dropped work",
             color=C_INK, fontsize=13, fontweight="bold", ha="left")
    who = f"{os.path.basename(path)}   |   " if path else ""
    t, load = field(ulog, "cpuload", "load", scale=100.0)
    peak = (f"CPU {np.median(load):.0f}% median, {load.max():.0f}% peak"
            if load.size else "no cpuload")
    dds = ulog.initial_parameters.get("UXRCE_DDS_CFG")
    dom = ulog.initial_parameters.get("UXRCE_DDS_DOM_ID")
    if dds is None:
        bridge = "no UXRCE_DDS_CFG in this log"
    elif float(dds) == 0:
        bridge = "DDS client OFF (UXRCE_DDS_CFG=0)"
    else:
        bridge = f"DDS client on port {int(dds)}, domain {int(dom or 0)}"
    fig.text(left, 1.0 - _f(0.62),
             f"{who}{dur:.1f} min   |   {peak}   |   {bridge}",
             color=C_MUTED, fontsize=9, ha="left")
    if n_clean:
        ctx.note(f"{n_clean} of the checked drop conditions never fired -- they "
                 f"are drawn as empty rows, not omitted")

    def refresh():
        for group, a in (("cpu", ax_cpu), ("slip", ax_slip), ("rate", ax_rate),
                         ("pct", ax_log), ("kb", ax_kb), ("hz", ax_link),
                         ("linkkb", ax_linkkb)):
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
                 ("kb", "ALL throughput"), ("hz", "ALL inbound rates"),
                 ("linkkb", "ALL link byte rates")],
                extra=extra, on_change=refresh,
                anchors={"cpu": _anchor("cpu"), "slip": _anchor("slip"),
                         "rate": _anchor("slip"), "pct": _anchor("log"),
                         "kb": _anchor("log"), "hz": _anchor("link"),
                         "linkkb": _anchor("link")})
    refresh()
    add_mouse_navigation(fig, [ax_cpu, ax_slip, ax_rate, ax_log, ax_kb, ax_link,
                               ax_linkkb, ax_band],
                         page_scroll=ctx.page_scroll, fixed_y=[ax_band],
                         on_view=refresh)
    fig.text(left, _f(0.32), nav_hint(ctx.page_scroll), color=C_MUTED,
             fontsize=8, ha="left")
    fig._page_height = int(round(fig_h * PAGE_PX_PER_IN))
    return fig
