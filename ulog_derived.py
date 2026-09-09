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
accel plot uses, so the report and the plot can never disagree about what the
arming check says.  It is not an expression evaluator, and it should not grow
into one -- anything that needs real arithmetic over channels belongs in a plot
module where it can be commented and shaded.

Refs look like normal ones so nothing else has to learn a new syntax:

    preflight[1].accel_bias_fail    1 while instance 1 would FAIL the
                                    "High Accelerometer Bias" arming check
                                    on any axis, 0 while it would pass

Acronyms: EKF = extended Kalman filter, ULog = PX4's binary log format.
"""
import numpy as np

from ulog_accel import preflight_bias_fail

__all__ = ["derived_field", "is_derived", "DERIVED_REFS"]

DERIVED_TOPICS = ("preflight", "imu", "dds")


def _accel_bias_fail(ulog, inst):
    """0/1 per sample: would this instance fail the High-Accelerometer-Bias check.

    Any-axis, because the arming check fails the vehicle if ANY axis trips --
    reducing to one number per sample is the whole point of asking for the
    binary rather than the bias.
    """
    t, fail = preflight_bias_fail(ulog, inst)
    if t.size == 0:
        return np.array([]), np.array([])
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
        return np.array([]), np.array([])
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
    PX4 v1.15 src/drivers/imu/...):

        error_count = bad_register + bad_transfer + fifo_empty + fifo_overflow

    so it is dominated by TIMING, not by data corruption: fifo_overflow means
    the driver's scheduled read came too LATE and the sensor's FIFO filled,
    fifo_empty that it came early and found nothing.  bad_transfer is a failed
    SPI transaction and bad_register a failed register readback.  A rising
    error_count therefore reads as "this sensor is not being serviced on
    schedule", which is why it is worth plotting against anything that loads
    the flight controller.
    """
    d = _get(ulog, "sensor_accel", inst)
    if d is None or "error_count" not in d.data:
        return np.array([]), np.array([])
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


def _get(ulog, name, inst):
    return next((d for d in ulog.data_list
                 if d.name == name and d.multi_id == inst), None)


_REGISTRY = {
    ("preflight", "accel_bias_fail"): _accel_bias_fail,
    ("imu", "accel_error_rate"): _accel_error_rate,
    ("dds", "offboard_rate"): _offboard_rate,
}

DERIVED_REFS = tuple(f"{topic}[i].{name}" for topic, name in _REGISTRY)


def is_derived(topic):
    return topic in DERIVED_TOPICS


def derived_field(ulog, topic, name, mid=0):
    """(t_min, y) for a computed channel, or empty arrays if it is not one."""
    fn = _REGISTRY.get((topic, name))
    if fn is None:
        return np.array([]), np.array([])
    try:
        return fn(ulog, mid)
    except Exception:
        # Same contract as ulog_common.field(): a channel that cannot be built
        # costs its own line, never the rest of the graph.
        return np.array([]), np.array([])
