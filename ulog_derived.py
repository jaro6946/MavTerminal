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

DERIVED_TOPICS = ("preflight",)


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


_REGISTRY = {
    ("preflight", "accel_bias_fail"): _accel_bias_fail,
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
