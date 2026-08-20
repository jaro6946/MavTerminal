#!/usr/bin/env python3
"""ulog_accel.py -- accelerometer data and calibration faults, by EKF instance.

Five stacked panels on one time axis, answering "is this IMU's accelerometer
calibrated, and did the firmware ever say it wasn't?":

  1. accelerometer      -- |a| per IMU (Inertial Measurement Unit) and per axis
  2. EKF accel bias     -- what each filter thinks the sensor is off by, against
                           the exact preflight arming threshold
  3. thermal correction -- the offset TC (thermal compensation) is injecting
  4. consistency        -- inter-IMU disagreement and vibration
  5. fault band         -- every accelerometer fault flag the firmware carries

The whole figure is shaded by `estimator_selector_status.primary_instance`, so a
bias excursion is read against which filter was actually steering the vehicle.

Why the shading matters here specifically
-----------------------------------------
The "High Accelerometer Bias" preflight failure is checked on the PRIMARY
instance only -- `estimatorCheck.cpp` moves `_estimator_sensor_bias_sub` onto the
selector's choice before running the test.  So an instance can sit above the
threshold for a minute and never block arming, purely because it was not primary
at the time.  A bias plot without the shading cannot tell those two cases apart,
and they have opposite conclusions: one is "this board will refuse to arm", the
other is "this board is one selector handover away from refusing to arm".

The preflight test, reproduced exactly
--------------------------------------
From `src/modules/commander/HealthAndArmingChecks/checks/estimatorCheck.cpp`
(`checkSensorBias`), for each axis independently:

    accel_bias_valid  AND  |accel_bias[k]| > 0.75 * accel_bias_limit
                                             + 3 * sqrt(accel_bias_variance[k])

`accel_bias_limit` is `EKF2_ABL_LIM`; the 3-sigma term widens the gate on axes
the filter cannot observe well, so the threshold is a TIME SERIES, not a
constant, and panel 2 plots it as one.  `accel_bias_valid` is itself a variance
test (`EKF2.cpp:1900`: the variance must be > 0 and the vector no longer than
0.1), so a filter that has not converged is excluded rather than failed.

Acronyms: EKF = extended Kalman filter, IMU = inertial measurement unit,
TC = thermal compensation.
"""
import os

import numpy as np

from ulog_common import (C_ARMED, C_BAD, C_INK, C_MUTED, C_SURFACE, PlotCtx,
                         Series, _clean, _get, _rescale, _style_axis, _time_min,
                         add_mouse_navigation, armed_spans, check_panel,
                         draw_armed, draw_band_rows, draw_primary_shading,
                         draw_mode_changes, duration_min, has_topic,
                         inst_color, instance_key, mode_changes, mode_key,
                         nav_hint,
                         primary_ekf, primary_spans, resample_to,
                         spans_from_bool, style_time_axis)

ACCEL_TOPICS = [
    "sensor_accel", "sensor_combined", "sensors_status_imu",
    "vehicle_imu_status", "estimator_sensor_bias", "estimator_status_flags",
    "estimator_selector_status", "sensor_correction", "sensor_selection",
    "actuator_armed",

    "vehicle_status",          # flight-mode overlay
]

C_PUB = "#20222b"       # near-black -- the PUBLISHED/primary series
C_UNMAPPED = "#8c8c85"  # an IMU no EKF instance claimed

# Standard gravity.  The reference line on panel 1: at rest, |a| IS g, and the
# gap between them is the scale-factor half of the calibration.
G = 9.80665

# Axis as line style, instance as hue.  Nine bias traces cannot be separated by
# hue alone without inventing a second palette that then competes with the
# instance colours -- which are the point of the figure.
AXIS_STYLE = {0: "-", 1: "--", 2: ":"}
AXIS_NAME = {0: "x", 1: "y", 2: "z"}

# estimatorCheck.cpp:527 and :532.  Named rather than inlined because if PX4
# changes them, the reproduction in panel 2 silently stops matching the firmware.
PREFLIGHT_LIMIT_FRAC = 0.75
PREFLIGHT_SIGMA = 3.0

# --- layout, in inches ------------------------------------------------------
# The four trace panels have fixed heights; the fault band does NOT, because its
# height is content.  A log with two flags and a log with fifteen are the same
# figure with the same fraction reserved, and at fifteen the rows are 0.1 inch
# apart with 7 pt labels sitting on top of each other.  The band is therefore
# sized per row and the FIGURE grows to fit it, which also keeps every trace
# panel exactly as tall as it was regardless of how eventful the log is.
PANEL_IN = [("acc", 2.05), ("bias", 1.95), ("corr", 1.75), ("cons", 1.75)]
GAP_IN = 0.48          # between panels; wide enough that a legend block can
                       # sit beside its own graph without touching the next
TOP_IN = 0.95          # title + subtitle + instance key
BOTTOM_IN = 1.35       # band's x label + nav hint + mode key + margins
BAND_ROW_IN = 0.52     # per fault row; a 3-lane row needs 3 readable bars AND
                       # a title, so this is ~4x the label's own height
BAND_PAD_IN = 0.30
BAND_MIN_IN = 1.25     # a log with one row still needs a readable axis
BAND_MAX_IN = 9.00     # past this the band would push the traces off a screen
# Pixels per inch the browser's scroll page is asked for.  The figure's height
# now depends on the log, so a fixed PlotSpec.height would squeeze an eventful
# log's five panels into the same strip as a quiet one's -- exactly the
# compression this layout exists to remove.  Below print dpi on purpose: this is
# a page you scroll, not one you read at 1:1.
PAGE_PX_PER_IN = 78

# Minimum drawn width of a band event, as a fraction of the log duration.  An
# accel_healthy dropout in a real log lasts ONE sample; at true width that is a
# 0.0005-minute bar, i.e. invisible, and the panel would report "clean" about a
# log that is not.  Widening is a lie about duration, so the band's caption says
# events are widened to be visible.
MIN_EVENT_FRAC = 0.003


# --- device / instance mapping ----------------------------------------------

def device_to_instance(ulog):
    """{accel device_id: EKF instance} -- who consumes which physical sensor.

    `sensor_accel[2]` is NOT necessarily the sensor EKF instance 2 runs on: the
    multi_id is the order the drivers advertised in, and the EKF's binding is
    whatever `sensor_selection`/`EKF2_MULTI_IMU` handed it.  Colouring a raw
    sensor trace by its multi_id would therefore put it under the wrong
    instance's shading, which is exactly the mistake this plot exists to prevent.

    Built from `estimator_sensor_bias[i].accel_device_id`, which is the EKF's own
    statement of what it is fusing (`EKF2.cpp:1896`).
    """
    out = {}
    for d in ulog.data_list:
        if d.name != "estimator_sensor_bias" or "accel_device_id" not in d.data:
            continue
        ids = np.asarray(d.data["accel_device_id"], dtype=np.int64)
        ids = ids[ids != 0]
        if ids.size:
            # The mode: the binding is fixed for the run, but a zero-initialised
            # first sample or a mid-log rebind should not rename the instance.
            vals, counts = np.unique(ids, return_counts=True)
            out[int(vals[np.argmax(counts)])] = d.multi_id
    return out


def _accel_instances(ulog):
    """Sorted EKF instances that publish an accel bias."""
    return sorted({d.multi_id for d in ulog.data_list
                   if d.name == "estimator_sensor_bias"})


def _imu_multi_ids(ulog, topic):
    return sorted({d.multi_id for d in ulog.data_list if d.name == topic})


def _color_for_device(dev_map, device_id):
    """Instance hue if some EKF claims this sensor, neutral grey if none does."""
    inst = dev_map.get(int(device_id))
    return (inst_color(inst), inst) if inst is not None else (C_UNMAPPED, None)


# --- panel 1: the measurement -----------------------------------------------

def _series_accel(ulog, ctx, dev_map):
    """Per-IMU magnitude (on) and per-axis components (off), plus the primary.

    Magnitude first and by default, because it is the one view that is frame-
    independent: however the board is mounted, |a| at rest is g, and any IMU
    whose |a| sits somewhere else has a scale or offset error you can read
    without knowing the rotation.  The axes are there for when the magnitude
    says something is wrong and you need to know which axis.
    """
    series = []

    # sensor_combined is the CALIBRATED, rotated, primary accel -- the vector the
    # EKF actually consumed.  Drawn thin and translucent rather than thick: it is
    # logged at the full IMU rate (119k samples in an 11-minute log against 6.9k
    # for sensor_accel), so at full weight its vibration envelope would paint
    # over every raw trace underneath it.
    d = _get(ulog, "sensor_combined")
    if d is not None and "accelerometer_m_s2[0]" in d.data:
        t = _time_min(ulog, d)
        xyz = np.column_stack([d.data[f"accelerometer_m_s2[{j}]"] for j in range(3)])
        t, mag = _clean(t, np.linalg.norm(xyz, axis=1))
        series.append(Series("sensor_combined.|accel|", "primary |a| (calibrated)",
                             t, mag, "acc", C_PUB, lw=1.0, alpha=0.55,
                             visible=True, zorder=2))

    for m in _imu_multi_ids(ulog, "sensor_accel"):
        d = _get(ulog, "sensor_accel", m)
        if d is None or "x" not in d.data:
            continue
        dev = int(np.asarray(d.data["device_id"])[0])
        col, inst = _color_for_device(dev_map, dev)
        tag = f"EKF {inst}" if inst is not None else f"IMU {m}"
        t = _time_min(ulog, d)
        xyz = np.column_stack([d.data[k] for k in ("x", "y", "z")])
        tc, mag = _clean(t, np.linalg.norm(xyz, axis=1))
        series.append(Series(f"sensor_accel[{m}].|accel|", f"{tag} |a| (raw)",
                             tc, mag, "acc", col, lw=1.4, visible=True, zorder=4))
        for j, name in AXIS_NAME.items():
            ta, y = _clean(t, xyz[:, j])
            series.append(Series(f"sensor_accel[{m}].{name}", f"{tag} {name}",
                                 ta, y, "acc", col, ls=AXIS_STYLE[j], lw=1.0,
                                 alpha=0.85, visible=False))
    if not series:
        ctx.note("no sensor_accel or sensor_combined in this log -- no raw "
                 "accelerometer data to plot")
    elif not has_topic(ulog, "sensor_accel"):
        ctx.note("no sensor_accel in this log -- only the fused primary vector "
                 "is available, so the IMUs cannot be compared against each "
                 "other (raise SDLOG_PROFILE to log per-sensor data)")
    return series


def _rest_magnitudes(ulog, dev_map):
    """[(label, median |a| at rest, n)] per IMU -- the cleanest calibration read.

    At rest the true specific force is exactly g, so the at-rest median of |a| is
    a direct measurement of that sensor's residual scale/offset error with no
    model in between.  Restricted to `cs_vehicle_at_rest`, the EKF's own
    stationarity flag, rather than to "disarmed": a disarmed vehicle being
    carried is not at rest, and its |a| median would be meaningless.
    """
    ekf, _ = primary_ekf(ulog)
    fl = _get(ulog, "estimator_status_flags", ekf)
    if fl is None or "cs_vehicle_at_rest" not in fl.data:
        return []
    t_rest = _time_min(ulog, fl)
    rest = np.asarray(fl.data["cs_vehicle_at_rest"], dtype=float)
    out = []
    for m in _imu_multi_ids(ulog, "sensor_accel"):
        d = _get(ulog, "sensor_accel", m)
        if d is None or "x" not in d.data:
            continue
        dev = int(np.asarray(d.data["device_id"])[0])
        inst = dev_map.get(dev)
        t = _time_min(ulog, d)
        mag = np.linalg.norm(
            np.column_stack([d.data[k] for k in ("x", "y", "z")]), axis=1)
        # Nearest-sample hold, not interpolation: a boolean interpolated to 0.5
        # is not a state the vehicle was ever in.
        at_rest = resample_to(t, t_rest, rest) > 0.5
        sel = mag[at_rest & np.isfinite(mag)]
        if sel.size:
            tag = f"EKF {inst}" if inst is not None else f"IMU {m}"
            out.append((tag, float(np.median(sel)), int(sel.size),
                        inst_color(inst) if inst is not None else C_UNMAPPED))
    return out


# --- panel 2: the bias estimate and the arming gate -------------------------

def _preflight_threshold(d):
    """(threshold[n,3], valid[n]) -- the arming check's gate, per sample.

    Kept as one function so panel 2's dashed line and the band panel's fault row
    cannot drift apart: they are the same number, drawn twice.
    """
    n = len(d.data["timestamp"])
    lim = np.asarray(d.data.get("accel_bias_limit", np.zeros(n)), dtype=float)
    var = np.column_stack([
        np.asarray(d.data.get(f"accel_bias_variance[{j}]", np.zeros(n)),
                   dtype=float) for j in range(3)])
    thr = (PREFLIGHT_LIMIT_FRAC * lim)[:, None] + \
        PREFLIGHT_SIGMA * np.sqrt(np.maximum(var, 0.0))
    valid = np.asarray(d.data.get("accel_bias_valid", np.ones(n)), dtype=float) > 0.5
    return thr, valid


def preflight_bias_fail(ulog, inst):
    """(t_min, fail[n,3]) -- where instance `inst` would fail the arming check."""
    d = _get(ulog, "estimator_sensor_bias", inst)
    if d is None or "accel_bias[0]" not in d.data:
        return np.array([]), np.zeros((0, 3), dtype=bool)
    t = _time_min(ulog, d)
    bias = np.column_stack([d.data[f"accel_bias[{j}]"] for j in range(3)])
    thr, valid = _preflight_threshold(d)
    fail = valid[:, None] & (np.abs(bias) > thr) & np.isfinite(bias)
    return t, fail


def _series_bias(ulog, ctx, instances):
    """Signed per-axis bias plus the (time-varying) preflight threshold.

    Signed, not |bias|: the sign says which way the sensor reads heavy, which is
    what you carry into a recalibration.  The threshold is therefore drawn as a
    mirrored pair, and both halves are one checkbox group so they toggle
    together.
    """
    series = []
    for i in instances:
        d = _get(ulog, "estimator_sensor_bias", i)
        if d is None or "accel_bias[0]" not in d.data:
            continue
        col = inst_color(i)
        t = _time_min(ulog, d)
        for j, name in AXIS_NAME.items():
            tb, y = _clean(t, d.data[f"accel_bias[{j}]"])
            series.append(Series(f"estimator_sensor_bias[{i}].accel_bias[{j}]",
                                 f"EKF {i} bias {name}", tb, y, "bias", col,
                                 ls=AXIS_STYLE[j], lw=1.4, visible=True))
        thr, _v = _preflight_threshold(d)
        # One line per sign, from the worst axis: the three axis thresholds differ
        # only by their variance term (~1% apart in practice), and drawing six
        # near-coincident dashed lines per instance turns the panel to hatching.
        # The max is the conservative choice -- a crossing above it is a failure
        # on every axis, so the drawn line never over-claims.
        worst = np.nanmax(thr, axis=1)
        tt, yy = _clean(t, worst)
        # Both signs as ONE line, joined by a NaN so matplotlib breaks between
        # them.  A separate negative series would double this panel's share of
        # the checkbox column for a curve that is, by construction, the mirror of
        # one already there -- and would let the reader hide half a gate.
        tg = np.concatenate([tt, [np.nan], tt])
        yg = np.concatenate([yy, [np.nan], -yy])
        series.append(Series(f"estimator_sensor_bias[{i}].preflight_limit",
                             f"EKF {i} arming limit +/-", tg, yg, "bias", col,
                             ls="-.", lw=1.0, alpha=0.7, visible=True, zorder=2))
    if not series:
        ctx.note("no estimator_sensor_bias in this log -- the preflight "
                 "'High Accelerometer Bias' check cannot be reproduced")
    return series


# --- panel 3: what thermal compensation is injecting ------------------------

def _series_correction(ulog, ctx, dev_map):
    """sensor_correction.accel_offset_N, i.e. the TC polynomial's output.

    This panel exists because on this project's board the same mechanism
    fabricates a 3.2 kPa barometer error over a 36 degC rise; the accel channel
    of the same calibration swings ~0.7 m/s^2, which is nearly twice the
    EKF2_ABL_LIM the arming check is measured against.  If panel 2 shows a bias
    that tracks this trace, the fault is in the calibration polynomial, not in
    the sensor.

    The offset SLOT index N indexes `accel_device_ids[N]`, not the EKF instance
    and not the sensor_accel multi_id -- so it is mapped through device_id like
    everything else here.
    """
    d = _get(ulog, "sensor_correction")
    if d is None:
        ctx.note("no sensor_correction in this log -- cannot show the thermal "
                 "compensation offset (TC_A_ENABLE may be 0)")
        return []
    t = _time_min(ulog, d)
    series = []
    for slot in range(4):
        key = f"accel_offset_{slot}[0]"
        if key not in d.data:
            continue
        ids = np.asarray(d.data.get(f"accel_device_ids[{slot}]", [0]), dtype=np.int64)
        ids = ids[ids != 0]
        dev = int(ids[0]) if ids.size else 0
        col, inst = _color_for_device(dev_map, dev)
        tag = f"EKF {inst}" if inst is not None else f"slot {slot}"
        off = np.column_stack([d.data[f"accel_offset_{slot}[{j}]"] for j in range(3)])
        if not np.any(np.abs(off) > 0):
            continue                    # an unpopulated slot, not a zero offset
        to, mag = _clean(t, np.linalg.norm(off, axis=1))
        series.append(Series(f"sensor_correction.accel_offset_{slot}", f"{tag} |offset|",
                             to, mag, "corr", col, lw=1.4, visible=True))
        for j, name in AXIS_NAME.items():
            ta, y = _clean(t, off[:, j])
            series.append(Series(f"sensor_correction.accel_offset_{slot}[{j}]",
                                 f"{tag} offset {name}", ta, y, "corr", col,
                                 ls=AXIS_STYLE[j], lw=1.0, alpha=0.85,
                                 visible=False))
        tt, temp = _clean(t, d.data.get(f"accel_temperature[{slot}]", []))
        if temp.size:
            series.append(Series(f"sensor_correction.accel_temperature[{slot}]",
                                 f"{tag} temp", tt, temp, "temp", col, ls=":",
                                 lw=1.2, alpha=0.8, visible=True))
    return series


# --- panel 4: do the IMUs agree, and how hard are they shaking ---------------

def _series_consistency(ulog, ctx, dev_map):
    """Inter-IMU disagreement (left) and vibration metric (right).

    Two different questions that share an axis because they answer each other:
    `accel_inconsistency_m_s_s` is each sensor's distance from the mean of all of
    them, so a single high line is one bad sensor -- but a vibration event lifts
    every line at once and means nothing about calibration.  Without the
    vibration trace next to it, the first is easy to read into the second.
    """
    series = []
    d = _get(ulog, "sensors_status_imu")
    if d is not None:
        t = _time_min(ulog, d)
        for k in range(4):
            key = f"accel_inconsistency_m_s_s[{k}]"
            if key not in d.data:
                continue
            ids = np.asarray(d.data.get(f"accel_device_ids[{k}]", [0]), dtype=np.int64)
            ids = ids[ids != 0]
            if not ids.size:
                continue            # empty slot: all-zero inconsistency, not agreement
            col, inst = _color_for_device(dev_map, int(ids[0]))
            tag = f"EKF {inst}" if inst is not None else f"IMU slot {k}"
            tt, y = _clean(t, d.data[key])
            series.append(Series(f"sensors_status_imu.{key}", f"{tag} inconsistency",
                                 tt, y, "cons", col, lw=1.4, visible=True))
    else:
        ctx.note("no sensors_status_imu -- no inter-IMU consistency to show")

    for m in _imu_multi_ids(ulog, "vehicle_imu_status"):
        d = _get(ulog, "vehicle_imu_status", m)
        if d is None or "accel_vibration_metric" not in d.data:
            continue
        dev = int(np.asarray(d.data.get("accel_device_id", [0]))[0])
        col, inst = _color_for_device(dev_map, dev)
        tag = f"EKF {inst}" if inst is not None else f"IMU {m}"
        tt, y = _clean(_time_min(ulog, d), d.data["accel_vibration_metric"])
        series.append(Series(f"vehicle_imu_status[{m}].accel_vibration_metric",
                             f"{tag} vibration", tt, y, "vib", col, ls="--",
                             lw=1.1, alpha=0.8, visible=True))
    return series


# --- panel 5: every accelerometer fault the firmware carries -----------------

def _counter_spans(t, counter, hold):
    """Spans starting at each increment of a monotonic counter.

    Clipping and driver errors are logged as CUMULATIVE COUNTS, not as flags: the
    number goes up once and stays up forever, so plotting `counter > 0` marks the
    whole rest of the flight as faulty from the first event.  What you want is
    the instant of each increment, given a visible width."""
    c = np.asarray(counter, dtype=float)
    idx = np.flatnonzero(np.diff(c) > 0) + 1
    return [(float(t[i]), float(t[i]) + hold) for i in idx]


def _fault_rows(ulog, ctx, instances, spans, dev_map, hold):
    """([(label, lanes, label_colour)], n_clean) -- the accelerometer checklist.

    Every condition this function knows how to check gets a row WHETHER OR NOT it
    fired, and every row carries one lane per EKF instance whether or not that
    instance fired.  The earlier version emitted a row only on a hit, which made
    the panel's shape depend on the log: a reader seeing "EKF 1 UNHEALTHY" and
    "EKF 2 UNHEALTHY" could not tell whether instance 0 was healthy or simply not
    checked, and the answer lived in a flat list under the figure.  A fault panel
    that reads "nothing here" must mean it.

    Lanes are ordered by instance, so instance 0 is the top lane of every row and
    the panel reads across; their colour is the same instance hue as the shading.
    """
    rows = []
    n_clean = 0

    def lanes_for(fn):
        """[(spans, colour)] over every instance, empty lanes included."""
        return [(fn(i) or [], inst_color(i)) for i in instances]

    def add(label, lanes, fault=True):
        nonlocal n_clean
        if not any(sp for sp, _c in lanes):
            n_clean += 1
        rows.append((label, lanes, C_BAD if fault else C_MUTED))

    a = armed_spans(ulog)
    rows.append(("armed", [(a, C_ARMED)], C_MUTED))

    # -- the arming check, reproduced, and then narrowed to when it mattered ---
    fails, blocking = {}, []
    for i in instances:
        t, fail = preflight_bias_fail(ulog, i)
        if t.size == 0:
            fails[i] = []
            continue
        any_fail = fail.any(axis=1)
        fails[i] = spans_from_bool(t, any_fail)
        # Only the primary instance is tested by commander, so the same
        # excursion is either an arming refusal or a latent one depending on
        # the shading.  Intersecting here is what makes that legible.
        prim = np.zeros(t.size, dtype=bool)
        for t0, t1, who in spans:
            if who == i:
                prim |= (t >= t0) & (t < t1)
        blocking += spans_from_bool(t, any_fail & prim)
    add("PREFLIGHT FAIL: high accel bias (primary)", [(blocking, C_BAD)])
    add("bias over arming limit", lanes_for(fails.get))

    # -- the EKF's own accelerometer fault flags ------------------------------
    def _flag(key):
        def get(i):
            fl = _get(ulog, "estimator_status_flags", i)
            if fl is None or key not in fl.data:
                return []
            v = np.asarray(fl.data[key], dtype=float) > 0.5
            return spans_from_bool(_time_min(ulog, fl), v)
        return get

    for key, label in (("fs_bad_acc_bias", "EKF fault: bad accel bias"),
                       ("fs_bad_acc_clipping", "EKF fault: accel clipping"),
                       ("fs_bad_acc_vertical", "EKF fault: bad accel vertical")):
        add(label, lanes_for(_flag(key)))

    # -- the selector's view (one instance-independent flag) ------------------
    d = _get(ulog, "estimator_selector_status")
    if d is not None and "accel_fault_detected" in d.data:
        v = np.asarray(d.data["accel_fault_detected"], dtype=float) > 0.5
        add("selector: accel FAULT", [(spans_from_bool(_time_min(ulog, d), v), C_BAD)])
    else:
        ctx.note("no estimator_selector_status -- no instance shading, and the "
                 "selector's own accel fault flag is unavailable")

    # -- sensor health, by device rather than by slot index -------------------
    def _healthy(i):
        d = _get(ulog, "sensors_status_imu")
        if d is None:
            return []
        for k in range(4):
            key = f"accel_healthy[{k}]"
            if key not in d.data:
                continue
            ids = np.asarray(d.data.get(f"accel_device_ids[{k}]", [0]), dtype=np.int64)
            ids = ids[ids != 0]
            if not ids.size or dev_map.get(int(ids[0])) != i:
                continue
            bad = ~(np.asarray(d.data[key], dtype=float) > 0.5)
            return spans_from_bool(_time_min(ulog, d), bad)
        return []

    add("accel UNHEALTHY", lanes_for(_healthy))

    # -- driver counters ------------------------------------------------------
    # Named "driver:" to keep them apart from the EKF's own clipping FAULT above.
    # They were both "EKF i accel clipping" before, so one label could appear as
    # both fired and not-fired in the same figure -- they are different facts
    # from different modules and now say so.
    def _counter(field, combine3=False):
        def get(i):
            for m in _imu_multi_ids(ulog, "vehicle_imu_status"):
                d = _get(ulog, "vehicle_imu_status", m)
                if d is None:
                    continue
                dev = int(np.asarray(d.data.get("accel_device_id", [0]))[0])
                if dev_map.get(dev) != i:
                    continue
                t = _time_min(ulog, d)
                if combine3:
                    c = np.zeros(t.size)
                    for j in range(3):
                        key = f"{field}[{j}]"
                        if key in d.data:
                            c = c + np.asarray(d.data[key], dtype=float)
                elif field in d.data:
                    c = np.asarray(d.data[field], dtype=float)
                else:
                    return []
                return _counter_spans(t, c, hold)
            return []
        return get

    add("driver: clip events", lanes_for(_counter("accel_clipping", combine3=True)))
    add("driver: errors (bad transfer / FIFO)",
        lanes_for(_counter("accel_error_count")))

    # -- context rows: is the bias estimate usable, and is it being kept -------
    def _bias_flag(key, invert):
        def get(i):
            d = _get(ulog, "estimator_sensor_bias", i)
            if d is None or key not in d.data:
                return []
            v = np.asarray(d.data[key], dtype=float) > 0.5
            return spans_from_bool(_time_min(ulog, d), ~v if invert else v)
        return get

    # The NEGATIVE, because "the arming check was not run on this instance" is
    # the fault-adjacent fact; a bar covering the whole flight to say "valid"
    # carries no information.
    add("bias NOT valid (arming check skipped)",
        lanes_for(_bias_flag("accel_bias_valid", True)))
    # Not a fault -- but this is the flag that lets VehicleIMU write the learned
    # bias into the calibration parameters at disarm (VehicleIMU.cpp:866), so a
    # bad bias that is "stable" is the one that survives a reboot.
    add("bias stable -> learned into CAL_ACC",
        lanes_for(_bias_flag("accel_bias_stable", False)), fault=False)

    return rows, n_clean


def _calibration_changes(ulog):
    """[t_min] where accel_calibration_count stepped -- a recalibration event.

    PX4 bumps this whenever the accel calibration parameters change, including
    the learned-bias save at disarm.  A bias trace that resets across one of
    these did not converge; it was re-datumed underneath the filter."""
    d = _get(ulog, "sensor_combined")
    if d is None or "accel_calibration_count" not in d.data:
        return []
    t = _time_min(ulog, d)
    c = np.asarray(d.data["accel_calibration_count"], dtype=float)
    return [float(t[i]) for i in np.flatnonzero(np.diff(c) != 0) + 1]


def _rescale_clipped(ax, lines, pct=99.5, keep=None):
    """Percentile y-limits, with a range that must stay in view.

    Plain min/max is wrong for both of the panels that use this.  Panel 1's |a|
    reaches 44 m/s^2 under rotor vibration, which pushes the whole at-rest
    region -- where the calibration question actually lives -- into the bottom
    tenth of the axis.  Panel 2's bias opens with a filter-initialisation
    transient several times larger than anything that follows.

    `keep` is a (lo, hi) that the limits must contain no matter what the
    percentiles say: the g reference line and the arming-limit lines are the
    thresholds the panels are read against, and an axis that crops the threshold
    off the top is worse than one that crops data.  Returns the number of
    off-scale samples so the caller can admit to the clipping.
    """
    vals = [np.asarray(ln.get_ydata(), dtype=float)
            for ln in lines if ln.get_visible()]
    vals = [v[np.isfinite(v)] for v in vals]
    vals = [v for v in vals if v.size]
    if not vals:
        return 0
    allv = np.concatenate(vals)
    lo = float(np.percentile(allv, 100.0 - pct))
    hi = float(np.percentile(allv, pct))
    if keep is not None:
        lo, hi = min(lo, keep[0]), max(hi, keep[1])
    if hi == lo:
        lo, hi = lo - 1.0, hi + 1.0
    pad = (hi - lo) * 0.08
    ax.set_ylim(lo - pad, hi + pad)
    return int(((allv < lo) | (allv > hi)).sum())


# --- the figure -------------------------------------------------------------

def build_accel(ulog, ctx=None, path=""):
    """The accelerometer figure.  Same signature as every plot builder."""
    import matplotlib.pyplot as plt

    ctx = ctx or PlotCtx()

    if not (has_topic(ulog, "sensor_accel") or has_topic(ulog, "sensor_combined")
            or has_topic(ulog, "estimator_sensor_bias")):
        ctx.note("no accelerometer topics in this log -- nothing to plot")
        return None

    dev_map = device_to_instance(ulog)
    instances = _accel_instances(ulog)
    spans = primary_spans(ulog)
    dur = duration_min(ulog) or 1.0
    hold = dur * MIN_EVENT_FRAC

    series = _series_accel(ulog, ctx, dev_map)
    series += _series_bias(ulog, ctx, instances)
    series += _series_correction(ulog, ctx, dev_map)
    series += _series_consistency(ulog, ctx, dev_map)
    if not series:
        ctx.note("no accelerometer fields in this log -- nothing to plot")
        return None

    unmapped = [s for s in series if s.color == C_UNMAPPED]
    if unmapped:
        ctx.note(f"{len(unmapped)} series come from an IMU no EKF instance "
                 f"claims (drawn grey) -- check EKF2_MULTI_IMU")

    # --- figure -------------------------------------------------------------
    # The fault rows are needed before the figure exists, because they decide how
    # tall it is.  _fault_rows touches no axes, so this is only a reordering.
    rows, n_clean = _fault_rows(ulog, ctx, instances, spans, dev_map, hold)

    band_in = min(max(len(rows) * BAND_ROW_IN + BAND_PAD_IN, BAND_MIN_IN),
                  BAND_MAX_IN)
    fig_h = (TOP_IN + sum(h for _k, h in PANEL_IN) + GAP_IN * len(PANEL_IN)
             + band_in + BOTTOM_IN)
    if len(rows) * BAND_ROW_IN + BAND_PAD_IN > BAND_MAX_IN:
        ctx.note(f"{len(rows)} fault rows do not fit the band at full spacing -- "
                 f"it is capped at {BAND_MAX_IN:g} in and the rows are tighter "
                 f"than the rest of the figure")

    fig = plt.figure(figsize=(15, fig_h), facecolor=C_SURFACE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(
            f"logGraph accelerometer - {os.path.basename(path)}")

    # Panel 1 carries the raw measurement and panel 2 the argument, so those two
    # get the room; the correction and consistency panels are supporting, and the
    # band is sized above.  The gap between the checkbox panel's right edge and
    # `left` has to hold both the tick labels and the axis label.
    #
    # Stacked bottom-up in inches and converted at the end: with a figure whose
    # height depends on the content, hand-written fractions would have to be
    # re-derived every time a panel changed, and silently overlap when they were
    # not.
    left, width = 0.260, 0.655

    def _f(inches):
        return inches / fig_h

    rects, bottom = {}, BOTTOM_IN
    rects["band"] = (_f(bottom), _f(band_in))
    bottom += band_in + GAP_IN
    for key, h in reversed(PANEL_IN):
        rects[key] = (_f(bottom), _f(h))
        bottom += h + GAP_IN

    ax_acc, ax_bias, ax_corr, ax_cons, ax_band = [
        fig.add_axes([left, rects[k][0], width, rects[k][1]], facecolor=C_SURFACE)
        for k in ("acc", "bias", "corr", "cons", "band")]
    for a in (ax_acc, ax_bias, ax_corr, ax_cons):
        a.sharex(ax_band)
    ax_temp = ax_corr.twinx()
    ax_vib = ax_cons.twinx()
    for a in (ax_temp, ax_vib):
        a.set_facecolor("none")

    axis_of = {"acc": ax_acc, "bias": ax_bias, "corr": ax_corr,
               "temp": ax_temp, "cons": ax_cons, "vib": ax_vib}

    shade_art = []
    for a in (ax_acc, ax_bias, ax_corr, ax_cons, ax_band):
        shade_art += draw_primary_shading(a, spans)

    # Armed shading is available but OFF by default: it would overlay the
    # instance colours it exists alongside.  The band panel carries it always.
    armed_art = []
    for a in (ax_acc, ax_bias, ax_corr, ax_cons):
        armed_art += draw_armed(a, armed_spans(ulog))
    for art in armed_art:
        art.set_visible(False)

    ax_acc.axhline(G, color=C_MUTED, lw=1.0, ls="--", alpha=0.8, zorder=1)
    ax_acc.text(0.012, G, f"g = {G:.3f}", transform=ax_acc.get_yaxis_transform(),
                color=C_MUTED, fontsize=7, va="bottom", ha="left")
    ax_bias.axhline(0.0, color=C_MUTED, lw=1.0, ls=":", alpha=0.6, zorder=1)
    ax_corr.axhline(0.0, color=C_MUTED, lw=1.0, ls=":", alpha=0.6, zorder=1)

    draw_band_rows(ax_band, rows, ylabel="accel faults",
                   empty_msg="no accelerometer flags in this log",
                   min_width=hold, track=True)

    for s in series:
        (line,) = axis_of[s.group].plot(
            s.t, s.y, color=s.color, ls=s.ls, lw=s.lw, label=s.label,
            drawstyle=s.drawstyle, alpha=s.alpha,
            zorder=s.zorder if s.zorder is not None else 3)
        line.set_visible(s.visible)
        s.line = line

    # Recalibration markers on the bias panel: a bias step across one of these is
    # the datum moving, not the estimate converging.
    cal_art = []
    for t_c in _calibration_changes(ulog):
        cal_art.append(ax_bias.axvline(t_c, color=C_BAD, lw=1.0, ls=":",
                                       alpha=0.75, zorder=2))
        cal_art.append(ax_bias.text(
            t_c, 0.985, " accel cal changed", transform=ax_bias.get_xaxis_transform(),
            rotation=90, fontsize=6, color=C_BAD, va="top", ha="left", zorder=6))

    # --- axis furniture -----------------------------------------------------
    for a in (ax_acc, ax_bias, ax_corr, ax_cons):
        style_time_axis(a, label=False)
        a.tick_params(axis="x", labelbottom=False)
    style_time_axis(ax_band)

    ax_acc.set_ylabel("specific force (m/s^2)", fontsize=9)
    ax_bias.set_ylabel("EKF accel bias (m/s^2)", fontsize=9)
    ax_corr.set_ylabel("thermal offset (m/s^2)", fontsize=9)
    ax_temp.set_ylabel("accel temp (degC, dotted)", fontsize=9)
    ax_cons.set_ylabel("inter-IMU inconsistency (m/s^2)", fontsize=9)
    ax_vib.set_ylabel("vibration metric (dashed)", fontsize=9)
    for a in (ax_acc, ax_bias, ax_corr, ax_cons):
        _style_axis(a, C_INK)
    for a in (ax_temp, ax_vib):
        _style_axis(a, C_MUTED)

    y_title = 1.0 - _f(0.35)
    fig.text(left, y_title, "Accelerometer and calibration faults by EKF instance",
             color=C_INK, fontsize=13, fontweight="bold", ha="left")
    who = f"{os.path.basename(path)}   |   " if path else ""
    if spans:
        used = sorted({i for _, _, i in spans})
        shade_note = (f"shaded by primary instance ({', '.join(map(str, used))}); "
                      f"{max(len(spans) - 1, 0)} handover(s)")
    else:
        shade_note = "no selector topic -- single EKF, no shading"
    fig.text(left, 1.0 - _f(0.62), f"{who}{dur:.1f} min   |   {shade_note}",
             color=C_MUTED, fontsize=9, ha="left")
    key_x = instance_key(fig, left, width, spans, y=y_title - _f(0.02))

    # The at-rest calibration read, on the panel it belongs to.  Text rather than
    # a note, because it is a per-IMU number the reader wants next to the traces
    # it summarises.
    # zorder above the traces and on an opaque-ish box: at full rotor vibration
    # panel 1's data reaches the top corner, and a note about clipping that is
    # itself hidden by the clipped data is a joke at the reader's expense.
    _note_box = dict(facecolor=C_SURFACE, edgecolor="none", pad=1.5, alpha=0.85)
    off_acc = ax_acc.text(0.995, 0.96, "", transform=ax_acc.transAxes,
                          ha="right", va="top", fontsize=7, color=C_MUTED,
                          zorder=8, bbox=_note_box)
    off_bias = ax_bias.text(0.995, 0.96, "", transform=ax_bias.transAxes,
                            ha="right", va="top", fontsize=7, color=C_MUTED,
                            zorder=8, bbox=dict(_note_box))

    rest = _rest_magnitudes(ulog, dev_map)
    if rest:
        parts = "   ".join(f"{tag} {mag:.3f}" for tag, mag, _n, _c in rest)
        ax_acc.text(0.995, 0.03, f"median |a| at rest:  {parts}   (g = {G:.3f})",
                    transform=ax_acc.transAxes, ha="right", va="bottom",
                    fontsize=7, color=C_MUTED, zorder=8,
                    bbox=dict(facecolor=C_SURFACE, edgecolor="none", alpha=0.85,
                              pad=1.5))
        worst = max(abs(m - G) for _t, m, _n, _c in rest)
        if worst > 0.05:
            ctx.note(f"at rest the IMUs disagree with g by up to {worst:.3f} m/s^2 "
                     f"({100 * worst / G:.1f}%) -- "
                     + ", ".join(f"{tag} {mag:.3f}" for tag, mag, _n, _c in rest))

    # A band panel with three bars on it does not say what it looked for and did
    # not find, and "no red" has to be readable as a RESULT, not as an omission.
    if n_clean:
        ctx.note(f"{n_clean} of the {len(rows) - 1} accelerometer conditions "
                 f"never fired on any instance -- they are drawn as empty rows, "
                 f"not omitted")
    if hold:
        ctx.note(f"band events are drawn at least {hold * 60:.1f} s wide so "
                 f"single-sample faults stay visible -- read their width as "
                 f"'at least this short', not as a duration")

    # The gates each clipped panel must never crop off: g on panel 1, and the
    # widest arming limit on panel 2.
    lim_keep = 0.0
    for i in instances:
        d = _get(ulog, "estimator_sensor_bias", i)
        if d is not None and "accel_bias_limit" in d.data:
            thr, _v = _preflight_threshold(d)
            if np.isfinite(thr).any():
                lim_keep = max(lim_keep, float(np.nanmax(thr)))

    # An empty panel with a grid on it reads as "these values were all zero",
    # which is a claim about the data.  Say instead that the log does not carry
    # them -- HITL logs have no sensor_correction at all, and a single-IMU build
    # has no inter-IMU anything.
    for ax_e, ax_tw, groups, msg in (
            (ax_corr, ax_temp, ("corr", "temp"),
             "no sensor_correction in this log -- thermal compensation is off "
             "or not logged"),
            (ax_cons, ax_vib, ("cons", "vib"),
             "no sensors_status_imu or vehicle_imu_status in this log")):
        if not any(s.group in groups for s in series):
            ax_e.text(0.5, 0.5, msg, transform=ax_e.transAxes, ha="center",
                      va="center", color=C_MUTED, fontsize=9)
            ax_e.set_yticks([])
            ax_tw.set_yticks([])

    # Flight-mode overlay: a rule on every panel at each mode change, named on
    # ax_bias.  Toggleable, because a log that flickers between Position and
    # Hold 52 times (d05a88e3) is unreadable with it on and unanswerable with it
    # off.  min_gap keeps the LABELS legible without dropping any rule.
    mode_art, mode_codes = draw_mode_changes(
        [ax_acc, ax_bias, ax_corr, ax_cons, ax_band], mode_changes(ulog),
        text_ax=ax_bias, min_gap=max(duration_min(ulog), 1.0) * 0.035)

    def refresh():
        n_acc = _rescale_clipped(ax_acc, [s.line for s in series if s.group == "acc"],
                                 keep=(G, G))
        n_bias = _rescale_clipped(ax_bias, [s.line for s in series if s.group == "bias"],
                                  keep=(-lim_keep, lim_keep) if lim_keep else None)
        for tag, n, note in (("acc", n_acc, off_acc), ("bias", n_bias, off_bias)):
            note.set_text(f"{n} sample(s) off-scale (ctrl+wheel to zoom out)"
                          if n else "")
        for group, a in (("corr", ax_corr), ("temp", ax_temp),
                         ("cons", ax_cons), ("vib", ax_vib)):
            _rescale(a, [s.line for s in series if s.group == group])

    extra = []
    # The key rides on the same toggle as the rules it names: a key to lines
    # that are switched off is worse than no key at all.
    mode_art += mode_key(fig, left + width, _f(0.10), mode_codes)
    if mode_art:
        extra.append(("mode changes", mode_art, True))
    if cal_art:
        extra.append(("accel cal changes", cal_art, True))
    if shade_art:
        extra.append(("instance shading", shade_art, True))
    if armed_art:
        extra.append(("armed (shaded)", armed_art, False))

    # The checkbox column spans exactly the plot stack, so a group's block can be
    # anchored beside the panel it belongs to.  Answering "which legend goes with
    # which graph" by POSITION costs nothing and never has to be read.
    cb_top = rects["acc"][0] + rects["acc"][1]
    cb_bot = rects["band"][0]
    h = cb_top - cb_bot

    def _anchor(key):
        """Panel centre, as a fraction of the checkbox axes."""
        b, ph = rects[key]
        return (b + ph / 2 - cb_bot) / h

    check_panel(fig, [0.012, cb_bot, 0.155, h], series,
                [("acc", "ALL accelerometer"), ("bias", "ALL bias"),
                 ("corr", "ALL thermal offset"), ("temp", "ALL temperature"),
                 ("cons", "ALL inconsistency"), ("vib", "ALL vibration")],
                extra=extra, on_change=refresh,
                anchors={"acc": _anchor("acc"), "bias": _anchor("bias"),
                         "corr": _anchor("corr"), "temp": _anchor("corr"),
                         "cons": _anchor("cons"), "vib": _anchor("cons")})
    refresh()
    add_mouse_navigation(fig, [ax_acc, ax_bias, ax_corr, ax_temp, ax_cons,
                               ax_vib, ax_band], page_scroll=ctx.page_scroll)
    fig.text(left, _f(0.32), nav_hint(ctx.page_scroll), color=C_MUTED,
             fontsize=8, ha="left")
    fig._page_height = int(round(fig_h * PAGE_PX_PER_IN))
    return fig
