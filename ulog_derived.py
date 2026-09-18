#!/usr/bin/env python3
"""ulog_derived.py -- computed channels a report can plot.

A report Graph can only name fields PX4 actually logged, which is usually the
right constraint: it keeps a report auditable, because every line in it traces
back to a message the flight controller wrote.  A few quantities that matter are
not logged as a single field though -- they are DECISIONS PX4 makes from several
fields at once, and the decision is what you want to compare across logs.

"Is the accelerometer bias too high to arm?" is the case in point.  PX4 does not
log a boolean for it.  The answer is `|accel_bias[k]| > 0.75 * accel_bias_limit
+ 3 * sqrt(accel_bias_variance[k])`, evaluated per axis, gated on
`accel_bias_valid` -- a time-varying threshold, not a constant, so no single
logged channel is equivalent to it and `accel_bias_stable` is a different
question entirely (it tracks convergence, not magnitude).

So this module is deliberately small and deliberately closed: a fixed registry
of computed channels, each one delegating to the SAME function the interactive
plot uses where one exists, so the report and the plot can never disagree.  It
is not an expression evaluator, and it should not grow into one -- anything
that needs real arithmetic over channels belongs in a plot module where it can
be commented and shaded.

Refs look like normal ones so nothing else has to learn a new syntax:

    preflight[1].accel_bias_fail    1 while instance 1 would FAIL the
                                    "High Accelerometer Bias" arming check
                                    on any axis, 0 while it would pass
    imu[k].accel_error_rate         sensor_accel[k] driver errors per minute
    dds[0].offboard_rate            offboard_control_mode messages per second
    mag[k].failed_over              1 while the estimator is NOT being fed
                                    magnetometer #k (the voter's "MAG #k
                                    failed" decision, as a channel)
    heading[0].published_vs_gsf     |published heading - GSF heading|, degrees
    heading[0].compass_vs_gsf       |compass heading - GSF heading|, degrees,
                                    level flight only

The two heading channels are measured against the same witness -- the EKF-GSF
of whichever instance is primary, in flight, converged -- so they share one
clock and one definition of "where this means anything" (see _primary_gsf).

Acronyms: EKF = extended Kalman filter, GSF = Gaussian Sum Filter (PX4's
magnetometer-free backup yaw estimator), GNSS = satellite navigation,
ULog = PX4's binary log format.
"""
import numpy as np

from ulog_accel import preflight_bias_fail

__all__ = ["derived_field", "is_derived", "DERIVED_REFS",
           "DERIVED_UNITS", "derived_units"]

DERIVED_TOPICS = ("preflight", "imu", "dds", "mag", "heading")

_EMPTY = (np.array([]), np.array([]))


def _accel_bias_fail(ulog, inst):
    """0/1 per sample: would this instance fail the High-Accelerometer-Bias check.

    Any-axis, because the arming check fails the vehicle if ANY axis trips --
    reducing to one number per sample is the whole point of asking for the
    binary rather than the bias.
    """
    t, fail = preflight_bias_fail(ulog, inst)
    if t.size == 0:
        return _EMPTY
    return t, fail.any(axis=1).astype(float)


# How wide a window a RATE is measured over.  Rates need one: a counter that
# steps by 1 is either 0 or a division by the log interval, and neither is the
# quantity anyone means by "errors per minute".  60 s is chosen against the
# thing being measured -- the accelerometer error clusters last a few hundred
# milliseconds and recur seconds apart, so a window has to span many clusters
# to be a rate rather than a sampling of them.
RATE_WINDOW_S = 60.0
MIN_WINDOW_S = 5.0      # below this a 'rate' is an artefact, not a measurement


def _win_rate(t_s, counts, per=60.0):
    """Sliding-window rate of an EVENT SERIES, in events per `per` seconds.

    `t_s` is seconds, `counts` the number of events at each time.  Returned on
    the same time base as the input so it can be plotted and compared against
    any other channel without resampling.
    """
    if t_s.size < 2:
        return _EMPTY
    total = np.cumsum(counts, dtype=float)
    lo = np.searchsorted(t_s, t_s - RATE_WINDOW_S, side="left")
    # Events inside the window, over the window's REAL width -- which is
    # shorter than RATE_WINDOW_S at the start of the log, and pretending
    # otherwise would show a false ramp for the first minute of every log.
    span = t_s - t_s[lo]
    inside = total - np.where(lo > 0, total[lo - 1], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = inside / span * per
    # NaN, not a number, until the window has real time in it: one event
    # divided by the first millisecond of a log is 1000 Hz, and that artefact
    # is both the maximum of the series and pure arithmetic.  NaN leaves a gap
    # in the plot and drops out of nanmean, which is what it deserves.
    rate[span < MIN_WINDOW_S] = np.nan
    return t_s / 60.0, rate


def _accel_error_rate(ulog, inst):
    """Accelerometer driver errors per minute, for sensor_accel[inst].

    `sensor_accel.error_count` is a monotonic counter, and its VALUE says only
    how many errors have happened since boot -- which is a function of how long
    the log is as much as of how bad the problem is.  The rate is the
    comparable quantity.

    What it counts (identical in the ICM20602, BMI088 and ICM20948 drivers,
    PX4 src/drivers/imu/...):

        error_count = bad_register + bad_transfer + fifo_empty + fifo_overflow

    The names mislead.  PX4's SPI::_transfer() never returns an error on NuttX,
    so "bad_transfer" is a CONTENT check on the FIFO data (the InvenSense
    temperature-consistency and doubled-accel-pair checks), and on the BMI088
    accelerometer "fifo_overflow" also fires when the FIFO length in the burst
    header is impossible.  On this airframe the logs' postflight perf dumps show
    those two dominating while "DRDY missed" stays near zero -- i.e. corrupted
    reads, not late ones.  Split the sum with the perf_counter_postflight
    message when a log has one; PX4 publishes only the total here.
    """
    d = _get(ulog, "sensor_accel", inst)
    if d is None or "error_count" not in d.data:
        return _EMPTY
    t = (np.asarray(d.data["timestamp"], dtype=np.float64)
         - ulog.start_timestamp) / 1e6
    e = np.asarray(d.data["error_count"], dtype=np.float64)
    # diff, floored at 0: the counter is monotonic, but a driver restart resets
    # it and a negative step would otherwise read as a large negative rate.
    step = np.diff(e, prepend=e[0]).clip(0)
    return _win_rate(t, step)


def _offboard_rate(ulog, inst):
    """offboard_control_mode messages per second -- the companion's publish rate.

    This topic exists only while a companion is streaming setpoints into the
    flight controller, so its message rate is the closest thing the flight
    controller logs to "how hard is the companion link working".  It is the
    INBOUND half only: what the flight controller publishes back out over
    uXRCE-DDS leaves no trace in its own log.
    """
    d = _get(ulog, "offboard_control_mode", inst)
    if d is None:
        # ABSENT IS ZERO, and only for a message rate.  For an ordinary channel
        # a missing topic means "this log cannot answer the question" and has to
        # stay missing; for "how often did the companion publish", a log with no
        # offboard_control_mode in it is a measured zero -- the companion sent
        # nothing.  Returning empty here would silently drop every control log
        # from a comparison whose whole point is the controls.
        span = (ulog.last_timestamp - ulog.start_timestamp) / 6e7
        return np.array([0.0, max(span, 1e-3)]), np.array([0.0, 0.0])
    t = (np.asarray(d.data["timestamp"], dtype=np.float64)
         - ulog.start_timestamp) / 1e6
    return _win_rate(t, np.ones(t.size), per=1.0)


def _mag_failed_over(ulog, inst):
    """0/1: 1 while the estimator is NOT being fed magnetometer #inst.

    PX4's sensor voter announces "MAG #0 failed: STALE!" as TEXT and then feeds
    the estimator the next magnetometer by priority.  The text has no value to
    plot; the switch does.  `estimator_selector_status.mag_device_id` is the
    device the EKF is actually fed, so comparing it with sensor_mag[inst]'s own
    device_id turns the voter's decision into a channel.

    It says "not in use", not "failed": a priority change would flip it too.
    Check the log's messages before calling a 1 a failure -- in the grazer
    library every 0 -> 1 so far lands on a "MAG #0 failed: STALE!" line.
    """
    s = _get(ulog, "sensor_mag", inst)
    if s is None or "device_id" not in s.data:
        return _EMPTY
    ids = np.asarray(s.data["device_id"])
    ids = ids[ids != 0]
    if ids.size == 0:
        return _EMPTY
    vals, counts = np.unique(ids, return_counts=True)
    mine = vals[np.argmax(counts)]

    sel = _get(ulog, "estimator_selector_status", 0)
    if sel is not None and "mag_device_id" in sel.data:
        d = sel
        dev = np.asarray(sel.data["mag_device_id"])
    else:
        # Older logs: the voter's published output.  Right only while
        # SENS_MAG_MODE publishes the selected magnetometer alone, which is why
        # it is the fallback rather than the source.
        d = _get(ulog, "vehicle_magnetometer", 0)
        if d is None or "device_id" not in d.data:
            return _EMPTY
        dev = np.asarray(d.data["device_id"])
    t = _minutes(ulog, d)
    # 0 is "no magnetometer chosen yet" at boot -- no decision, not a failover.
    ok = dev != 0
    return t[ok], (dev[ok] != mine).astype(float)


# --- heading, against the magnetometer-free witness --------------------------
#
# The EKF-GSF is PX4's backup yaw estimator: a bank of filters that works out
# heading from GNSS velocity and acceleration alone, with no magnetometer.  When
# the magnetometer is lying it is the witness still telling the truth -- see
# ulog_heading, whose interactive plot draws the same differences with the same
# helpers imported below, so the report and the plot cannot disagree about what
# they are.  It is a WITNESS, not ground truth: two independent estimates
# agreeing within a few degrees is the statement these channels can make.

# The channels are undefined on the ground and wherever the GSF has not
# converged.  Rather than a NaN per undefined sample -- which the report's
# min/max decimation turns into dropped pixels either side of every one -- the
# undefined samples are removed and ONE NaN goes in each gap longer than this,
# so the line breaks there instead of bridging a landing with a straight stroke.
GAP_MIN = 2.0 / 60.0        # minutes


def _primary_gsf(ulog):
    """(t_min, gsf_heading_deg) where the GSF heading means something.

      * IN FLIGHT (vehicle_land_detected.landed == 0).  The GSF gets heading
        from horizontal acceleration against GNSS velocity; a vehicle sitting
        on the ground gives it nothing to work with.
      * where yaw_composite_valid says the GSF has converged.
      * from the GSF of whichever EKF instance is PRIMARY at that moment -- the
        one whose heading is being published.  The three GSFs run on three
        different IMUs and disagree with each other by several degrees, so
        comparing the published heading with another instance's GSF would
        measure the IMUs, not the heading.
    """
    sel = _get(ulog, "estimator_selector_status", 0)
    if sel is not None and "primary_instance" in sel.data:
        t_sel = _minutes(ulog, sel)
        primary = np.asarray(sel.data["primary_instance"])
    else:
        t_sel, primary = np.array([]), np.array([])
    land = _get(ulog, "vehicle_land_detected", 0)
    if land is not None and "landed" in land.data:
        t_land = _minutes(ulog, land)
        landed = np.asarray(land.data["landed"]) > 0.5
    else:
        t_land, landed = np.array([]), np.array([], dtype=bool)

    ts, ys = [], []
    for d in ulog.data_list:
        if d.name != "yaw_estimator_status" or "yaw_composite" not in d.data:
            continue
        t = _minutes(ulog, d)
        y = np.degrees(np.asarray(d.data["yaw_composite"], dtype=float))
        keep = np.isfinite(y)
        if "yaw_composite_valid" in d.data:
            keep &= np.asarray(d.data["yaw_composite_valid"]) > 0.5
        keep &= _step_at(t, t_sel, primary, 0) == d.multi_id
        if t_land.size:
            # Before the land detector's first word the vehicle is on the ground.
            keep &= ~_step_at(t, t_land, landed, True)
        ts.append(t[keep])
        ys.append(y[keep])
    if not ts:
        return _EMPTY
    t = np.concatenate(ts)
    y = np.concatenate(ys)
    order = np.argsort(t, kind="stable")
    return t[order], y[order]


def _published_vs_gsf(ulog, inst):
    """|published heading - GSF heading|, degrees: did the vehicle fly on a bad heading?

    Only [0] exists: the witness follows the primary, so there is no per-
    instance version to index.
    """
    if inst != 0:
        return _EMPTY
    # Imported here, not at the top: ulog_heading is a plot module, and the
    # report stack should not pay for it unless a heading channel is asked for.
    from ulog_heading import _attitude, circ_interp, wrap180

    t, gsf = _primary_gsf(ulog)
    tp, _roll, _pitch, yaw = _attitude(ulog)
    if t.size == 0 or tp.size == 0:
        return _EMPTY
    e = np.abs(wrap180(circ_interp(t, tp, yaw) - gsf))
    ok = np.isfinite(e)
    return _break_gaps(t[ok], e[ok])


def _compass_vs_gsf(ulog, inst):
    """|compass heading - GSF heading|, degrees: is the MAGNETOMETER pointing right?

    The compass is ulog_heading.mag_heading: the textbook tilt-compensated
    compass on `vehicle_magnetometer` -- the calibrated field of whichever
    magnetometer the voter is feeding the EKF -- plus the declination the EKF
    learned, from the same instance the heading plot takes it from.  It is
    independent of the estimator's yaw, so it says what the magnetometer is
    telling the EKF whether or not the EKF believes it.

    Level flight only (tilt below ulog_heading.TILT_LIMIT): tilt compensation
    degrades as the vehicle banks, and the heading plot's own headline number
    is measured under the same limit.  Only [0] exists, as above.
    """
    if inst != 0:
        return _EMPTY
    from ulog_heading import (TILT_LIMIT, _instances, circ_interp, declination,
                              mag_heading, wrap180)

    t, gsf = _primary_gsf(ulog)
    if t.size == 0:
        return _EMPTY
    instances = (_instances(ulog) or _instances(ulog, "estimator_status_flags")
                 or [0])
    decl_t, decl_deg, _strength = declination(ulog, instances[0])
    tm, _magnetic, true, tilt = mag_heading(ulog, decl_t, decl_deg)
    # No learned declination means no TRUE heading to compare -- a magnetic
    # heading against a true-north witness would report the declination as
    # an error, so the channel is absent rather than wrong.
    if tm.size == 0 or not np.isfinite(true).any():
        return _EMPTY
    e = np.abs(wrap180(circ_interp(t, tm, true) - gsf))
    ok = np.isfinite(e) & (np.interp(t, tm, tilt) < TILT_LIMIT)
    return _break_gaps(t[ok], e[ok])


def _break_gaps(t, e):
    """Insert one NaN in every gap longer than GAP_MIN -- see GAP_MIN."""
    if t.size < 2:
        return t, e
    gaps = np.nonzero(np.diff(t) > GAP_MIN)[0]
    if gaps.size:
        t = np.insert(t, gaps + 1, 0.5 * (t[gaps] + t[gaps + 1]))
        e = np.insert(e, gaps + 1, np.nan)
    return t, e


def _step_at(t_query, t_src, v_src, default):
    """A step signal's value (its last sample at or before) at each query time.

    `default` before the first sample, and everywhere when there is no source.
    """
    t_query = np.asarray(t_query, dtype=float)
    if t_src.size == 0:
        return np.full(t_query.shape, default)
    i = np.searchsorted(t_src, t_query, side="right") - 1
    out = np.asarray(v_src)[np.clip(i, 0, t_src.size - 1)]
    return np.where(i >= 0, out, default)


def _minutes(ulog, d):
    return (np.asarray(d.data["timestamp"], dtype=np.float64)
            - ulog.start_timestamp) / 6e7


def _get(ulog, name, inst):
    return next((d for d in ulog.data_list
                 if d.name == name and d.multi_id == inst), None)


_REGISTRY = {
    ("preflight", "accel_bias_fail"): _accel_bias_fail,
    ("imu", "accel_error_rate"): _accel_error_rate,
    ("dds", "offboard_rate"): _offboard_rate,
    ("mag", "failed_over"): _mag_failed_over,
    ("heading", "published_vs_gsf"): _published_vs_gsf,
    ("heading", "compass_vs_gsf"): _compass_vs_gsf,
}

DERIVED_REFS = tuple(f"{topic}[i].{name}" for topic, name in _REGISTRY)

# Units, for axis labels.  A logged channel carries no units anywhere in a ULog
# -- you are expected to know that sensor_accel.temperature is Celsius -- so a
# name is all a label can say for one.  A COMPUTED channel is different: this
# module chose the units, so it is the only place that can state them, and a
# rate axis with no units on it is genuinely ambiguous (per second? per
# minute?).
DERIVED_UNITS = {
    ("preflight", "accel_bias_fail"): "0/1",
    ("imu", "accel_error_rate"): "errors/min",
    ("dds", "offboard_rate"): "Hz",
    ("mag", "failed_over"): "0/1",
    ("heading", "published_vs_gsf"): "deg",
    ("heading", "compass_vs_gsf"): "deg",
}


def derived_units(topic, name):
    """The units of a computed channel, or "" for anything else."""
    return DERIVED_UNITS.get((topic, name), "")


def is_derived(topic):
    return topic in DERIVED_TOPICS


def derived_field(ulog, topic, name, mid=0):
    """(t_min, y) for a computed channel, or empty arrays if it is not one."""
    fn = _REGISTRY.get((topic, name))
    if fn is None:
        return _EMPTY
    try:
        return fn(ulog, mid)
    except Exception:
        # Same contract as ulog_common.field(): a channel that cannot be built
        # costs its own line, never the rest of the graph.
        return _EMPTY
