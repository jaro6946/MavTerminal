#!/usr/bin/env python3
"""ulog_heading.py -- where the heading estimate came from, and who disagrees.

Yaw is the one attitude axis gravity cannot fix.  Roll and pitch are observable
from the accelerometer the moment the vehicle sits still; yaw is not observable
from any inertial measurement at all, so the EKF (Extended Kalman Filter) has to
be TOLD which way is north -- by the magnetometer, by GNSS (Global Navigation
Satellite System) course, by a dual-antenna GPS heading, or by its own
velocity-vs-acceleration reasoning.  Every heading bug is therefore a question of
which source it believed and whether that source was right.

So this plot puts every independent witness on one axis and then, underneath,
their disagreement with the published estimate:

  1. heading        -- the fused estimate, each EKF instance, the EKF-GSF, the
                       magnetometer worked out from scratch, and GNSS course
  2. disagreement   -- each of those minus the published heading, wrapped
  3. innovations    -- what the EKF thought of the mag data it was fed
  4. mag bias, declination and field strength
  5. band           -- yaw alignment, which source was fused, and every mag fault

Reading it
----------
A CONSTANT offset between the fused heading and the magnetometer is a
declination or a frame problem -- the estimate tracks every turn correctly and
sits a fixed number of degrees off.  Panel 4 carries the declination the EKF
learned, so a constant offset that equals it is the declination not being
applied, and one that does not is a rotation error.

A heading that is FLAT while the vehicle turns is a stuck estimate: yaw fusion
has stopped, and the band panel says which source stopped.  The EKF-GSF trace is
the check that matters here, because it is derived from velocity and
acceleration alone -- no magnetometer -- so when the mag is lying, GSF is the
witness that is still telling the truth.

Innovations that are large but test ratios below 1.0 mean the EKF is accepting
data it disagrees with; ratios above 1.0 mean it is rejecting them, and a yaw
that then drifts is dead reckoning on gyro bias.

The magnetometer heading here is computed independently of the EKF -- the
standard tilt-compensated compass, using only roll and pitch from the attitude
(which are gravity-referenced and trustworthy) and the raw field vector.  That
is deliberate: it is a witness the estimator cannot influence, so when it and
the estimator disagree, the disagreement is real.

Acronyms: EKF = extended Kalman filter, GSF = Gaussian Sum Filter (PX4's
magnetometer-free backup yaw estimator), NED = North-East-Down, COG = course
over ground, GNSS/GPS = satellite navigation, decl = magnetic declination
(the angle from true north to magnetic north).
"""
import os

import numpy as np

from ulog_common import (C_ARMED, C_BAD, C_GRID, C_INK, C_MUTED, C_SURFACE,
                         INST_COLORS, PlotCtx, Series, _get, _rescale,
                         _style_axis, _time_min, add_mouse_navigation,
                         armed_spans, check_panel, draw_armed, draw_band_rows,
                         draw_mode_changes, draw_primary_shading, duration_min,
                         field, has_topic, inst_color, instance_key,
                         mode_changes, mode_key, nav_hint, primary_spans,
                         spans_from_bool, style_time_axis, window_values)

HEADING_TOPICS = [
    "vehicle_attitude",              # the published attitude -> published yaw
    "estimator_attitude",            # per instance
    "vehicle_local_position",        # heading, heading_good_for_control, resets
    "yaw_estimator_status",          # the EKF-GSF: yaw WITHOUT the magnetometer
    "vehicle_magnetometer",          # the fused field vector, for our own compass
    "sensor_mag",                    # per-device field, for the strength check
    "estimator_innovations",
    "estimator_innovation_test_ratios",
    "estimator_status",              # mag_test_ratio, pre-flight heading check
    "estimator_status_flags",        # which yaw source was fused, and mag faults
    "estimator_sensor_bias",         # learned mag bias
    "estimator_states",              # earth field -> learned declination
    "vehicle_gps_position",          # course over ground, dual-antenna heading
    "vehicle_attitude_setpoint",     # what was COMMANDED
    "estimator_selector_status",
    "actuator_armed",
    "vehicle_status",
]

# --- colour ------------------------------------------------------------------
# Instance hues are shared with the local-z and accelerometer plots, so a trace
# here and the shading under it mean the same filter they do there.
C_PUB = "#20222b"        # near-black -- the PUBLISHED estimate, the reference
C_MAG = "#c2185b"        # magenta -- the independent compass
C_MAG_RAW = "#f06292"    # lighter magenta -- the same, without declination
C_COG = "#00838f"        # dark cyan -- GNSS course over ground
C_GPSYAW = "#1565c0"     # blue -- dual-antenna GPS heading
C_SP = "#8d6e63"         # brown -- the commanded yaw
C_DECL = "#6a4fa3"       # violet -- declination

# PX4's EKF2 state vector, 24 states.  16..18 are the earth magnetic field in
# NED (Gauss) and 19..21 the body-frame magnetometer bias.  The declination the
# filter is actually using is the direction of that earth field in the
# horizontal plane -- not the EKF2_MAG_DECL parameter, which is only its prior.
S_MAG_EARTH_N, S_MAG_EARTH_E, S_MAG_EARTH_D = 16, 17, 18

# GNSS course is the direction of TRAVEL, which equals heading only when the
# vehicle is going where it points.  Below this it is the direction of whatever
# noise the receiver last saw, and plotting it as a heading is an invention.
COG_MIN_SPEED = 2.0     # m/s

# Past this bank angle the tilt compensation in mag_heading loses accuracy, so
# the headline magnetometer-vs-estimate number is measured below it only.
TILT_LIMIT = 20.0       # deg

# A test ratio at or above 1.0 is a rejected measurement (EKF2 innovation gate).
REJECT_RATIO = 1.0

GAUSS_TO_MGAUSS = 1e3

PANEL_IN = [("head", 2.30), ("diff", 1.95), ("innov", 1.75), ("bias", 1.75)]
GAP_IN = 0.48
TOP_IN = 0.95
BOTTOM_IN = 1.35
BAND_ROW_IN = 0.52
BAND_PAD_IN = 0.30
BAND_MIN_IN = 1.25
# 22 rows at full spacing is 11.7 in.  The accelerometer plot caps at 9 because
# its row count varies with the log; this checklist is FIXED, so a cap that
# always bites would just be a permanent note about a permanent condition.
BAND_MAX_IN = 12.00
PAGE_PX_PER_IN = 78


# --- angle helpers -----------------------------------------------------------

def wrap180(deg):
    """Fold degrees into (-180, 180]."""
    return (np.asarray(deg, dtype=float) + 180.0) % 360.0 - 180.0


def circ_interp(t_dst, t_src, deg):
    """Interpolate a WRAPPED angle onto another time base.

    np.interp on degrees is wrong at the wrap: going 179 -> -179 is a 2 degree
    turn, and linear interpolation sweeps the long way round through zero
    instead.  Measured cost of getting this wrong on Barometer_Primary_Datum:
    the EKF-vs-GSF disagreement reported a 101 degree maximum that is really
    25 degrees -- every sample near the wrap was fiction.

    Interpolating the unit vector and taking the angle back is exact, because
    the vector has no discontinuity to sweep across.
    """
    t_dst = np.asarray(t_dst, dtype=float)
    if t_src.size == 0 or t_dst.size == 0:
        return np.full(t_dst.shape, np.nan)
    r = np.radians(np.asarray(deg, dtype=float))
    c = np.interp(t_dst, t_src, np.cos(r))
    s = np.interp(t_dst, t_src, np.sin(r))
    out = np.degrees(np.arctan2(s, c))
    out[(t_dst < t_src[0]) | (t_dst > t_src[-1])] = np.nan
    return out


def break_wraps(t, deg, jump=180.0):
    """NaN out the sample after a wrap so the line does not draw a vertical bar.

    A heading crossing +-180 is continuous in the world and discontinuous on the
    axis, and matplotlib joins the two ends with a stripe through every other
    trace on the panel.  Returned as a copy: the caller still needs the
    unbroken series for arithmetic.
    """
    y = np.asarray(deg, dtype=float).copy()
    if y.size > 1:
        y[1:][np.abs(np.diff(y)) > jump] = np.nan
    return y


def heading_center(deg):
    """The circular mean of a heading series, used as the WRAP BRANCH.

    Folding headings into a fixed (-180, 180] is only a good choice for a
    vehicle pointing near north.  One sitting near south -- which is most of
    Barometer_Primary_Datum -- crosses the branch cut constantly, and the panel
    fills with traces jumping between +170 and -170 that are the same heading.

    Wrapping about the series' own mean direction instead puts the cut where the
    vehicle never points.  Every heading trace on the panel uses the SAME centre,
    from the published estimate, so they stay directly comparable; the axis then
    reads in degrees that may run outside +-180, which is the price and is worth
    it.  A vehicle that genuinely turns through 360 has no good branch, and those
    wraps are broken by break_wraps as before.
    """
    d = np.asarray(deg, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0
    r = np.radians(d)
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def about(deg, center):
    """Fold degrees into (center - 180, center + 180]."""
    return center + wrap180(np.asarray(deg, dtype=float) - center)


def _euler(q0, q1, q2, q3):
    """(roll, pitch, yaw) in radians from a PX4 quaternion (w, x, y, z)."""
    roll = np.arctan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1 ** 2 + q2 ** 2))
    pitch = np.arcsin(np.clip(2.0 * (q0 * q2 - q3 * q1), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 ** 2 + q3 ** 2))
    return roll, pitch, yaw


def _attitude(ulog, topic="vehicle_attitude", mid=0):
    """(t_min, roll, pitch, yaw_deg) from a quaternion topic; empties if absent."""
    d = _get(ulog, topic, mid)
    if d is None or "q[0]" not in d.data:
        return (np.array([]),) * 4
    q = [np.asarray(d.data[f"q[{i}]"], dtype=float) for i in range(4)]
    t = _time_min(ulog, d)
    roll, pitch, yaw = _euler(*q)
    ok = np.isfinite(t) & np.isfinite(yaw)
    return t[ok], roll[ok], pitch[ok], np.degrees(yaw[ok])


# --- the independent compass -------------------------------------------------

def mag_heading(ulog, decl_t=None, decl_deg=None):
    """(t_min, magnetic_heading_deg, true_heading_deg, tilt_deg) -- our own compass.

    The textbook tilt-compensated compass.  Roll and pitch come from the
    published attitude, which is safe to trust even when yaw is not: they are
    observable from gravity, and an EKF that has lost its heading still has them.
    The field vector is `vehicle_magnetometer`, the calibrated and rotated field
    the estimator itself is fed.

        X_h = mx cos(p) + my sin(r) sin(p) + mz cos(r) sin(p)
        Y_h = my cos(r) - mz sin(r)
        heading_magnetic = atan2(-Y_h, X_h)

    X_h and Y_h are the field's north and east components after the tilt is
    rotated out, so their ratio is the direction of magnetic north relative to
    the nose.  The minus sign puts the result in NED, where heading grows
    clockwise from north.

    `true = magnetic + declination`, because declination is measured from true
    north TO magnetic north.  Both are returned: the magnetic one is what the
    sensor actually sees and the true one is what the EKF should be reporting,
    and having both is what separates "declination is not being applied" from
    "the field is being rotated wrongly".

    `tilt` comes back so the caller can say how level the vehicle was: this
    compensation degrades as the vehicle tips, and a disagreement measured at
    45 degrees of bank is not the same evidence as one measured on the ground.
    """
    d = _get(ulog, "vehicle_magnetometer")
    if d is None or "magnetometer_ga[0]" not in d.data:
        return (np.array([]),) * 4
    tm = _time_min(ulog, d)
    mx, my, mz = (np.asarray(d.data[f"magnetometer_ga[{i}]"], dtype=float)
                  for i in range(3))
    ta, roll, pitch, yaw = _attitude(ulog)
    if ta.size == 0:
        return (np.array([]),) * 4
    r = np.interp(tm, ta, roll)
    p = np.interp(tm, ta, pitch)
    x_h = mx * np.cos(p) + my * np.sin(r) * np.sin(p) + mz * np.cos(r) * np.sin(p)
    y_h = my * np.cos(r) - mz * np.sin(r)
    magnetic = np.degrees(np.arctan2(-y_h, x_h))
    if decl_t is not None and decl_t.size:
        true = wrap180(magnetic + circ_interp(tm, decl_t, decl_deg))
    else:
        true = np.full_like(magnetic, np.nan)
    tilt = np.degrees(np.hypot(r, p))
    ok = np.isfinite(tm) & np.isfinite(magnetic)
    return tm[ok], magnetic[ok], true[ok], tilt[ok]


def declination(ulog, mid=0):
    """(t_min, declination_deg, field_strength_gauss) the EKF is actually using.

    Derived from the earth-field states, not from EKF2_MAG_DECL: the parameter
    is the prior the filter starts with, and with EKF2_MAG_DECL_A / mag_dec
    fusion enabled the filter learns its own.  When a heading sits a constant
    few degrees off, the number that explains it is this one.
    """
    d = _get(ulog, "estimator_states", mid)
    if d is None or f"states[{S_MAG_EARTH_N}]" not in d.data:
        return np.array([]), np.array([]), np.array([])
    t = _time_min(ulog, d)
    n = np.asarray(d.data[f"states[{S_MAG_EARTH_N}]"], dtype=float)
    e = np.asarray(d.data[f"states[{S_MAG_EARTH_E}]"], dtype=float)
    z = np.asarray(d.data[f"states[{S_MAG_EARTH_D}]"], dtype=float)
    dec = np.degrees(np.arctan2(e, n))
    strength = np.sqrt(n ** 2 + e ** 2 + z ** 2)
    ok = np.isfinite(t) & np.isfinite(dec) & (np.abs(n) + np.abs(e) > 1e-9)
    return t[ok], dec[ok], strength[ok]


def _gps_cog(ulog):
    """(t_min, course_deg) with the stationary samples removed -- see COG_MIN_SPEED."""
    d = _get(ulog, "vehicle_gps_position")
    if d is None or "cog_rad" not in d.data:
        return np.array([]), np.array([])
    t = _time_min(ulog, d)
    cog = np.degrees(np.asarray(d.data["cog_rad"], dtype=float))
    spd = np.asarray(d.data.get("vel_m_s", np.full_like(cog, np.inf)), dtype=float)
    ok = np.isfinite(t) & np.isfinite(cog) & (spd >= COG_MIN_SPEED)
    return t[ok], wrap180(cog[ok])


def _gps_yaw(ulog):
    """(t_min, heading_deg) from a dual-antenna receiver; empty when not fitted."""
    d = _get(ulog, "vehicle_gps_position")
    if d is None or "heading" not in d.data:
        return np.array([]), np.array([])
    t = _time_min(ulog, d)
    h = np.degrees(np.asarray(d.data["heading"], dtype=float))
    ok = np.isfinite(t) & np.isfinite(h)
    return t[ok], wrap180(h[ok])


def _instances(ulog, topic="estimator_attitude"):
    return sorted({d.multi_id for d in ulog.data_list if d.name == topic})


def _reset_events(ulog):
    """[(t_min, delta_deg, old, new)] from heading_reset_counter.

    A heading reset is the filter throwing its yaw away and re-initialising it,
    usually from the GSF.  It is the signature of an estimator that KNOWS its
    heading was wrong -- which is a different finding from one that is wrong and
    content.
    """
    d = _get(ulog, "vehicle_local_position")
    if d is None or "heading_reset_counter" not in d.data:
        return []
    t = _time_min(ulog, d)
    c = np.asarray(d.data["heading_reset_counter"], dtype=float)
    dh = np.degrees(np.asarray(d.data.get("delta_heading", np.zeros_like(c)),
                               dtype=float))
    return [(float(t[i]), float(dh[i]), int(c[i - 1]), int(c[i]))
            for i in np.where(np.diff(c) != 0)[0] + 1]


# --- series ------------------------------------------------------------------

def _heading_series(ulog, ctx, instances, compass, center):
    """Panels 1 and 2: every witness, and every witness minus the published one."""
    series = []

    t_pub, _r, _p, yaw_pub = _attitude(ulog)
    if t_pub.size == 0:
        return series, (np.array([]), np.array([]))

    def add(group, sid, label, t, y, color, **kw):
        """Plot a HEADING: folded onto the shared branch, then broken at wraps."""
        if t.size == 0 or not np.isfinite(y).any():
            return
        y = about(y, center)
        series.append(Series(sid, label, t, break_wraps(t, y), group, color, **kw))

    def add_diff(sid, label, t, y, color, **kw):
        """The witness minus the published estimate, on the witness's own clock.

        Resampling the PUBLISHED heading onto the witness rather than the other
        way round: the published estimate is the smooth, fast, always-present
        one, so it is the safe thing to interpolate, and the witness keeps its
        own sample times and its own gaps.
        """
        if t.size == 0 or not np.isfinite(y).any():
            return
        d = wrap180(y - circ_interp(t, t_pub, yaw_pub))
        series.append(Series(sid + ".diff", label, t, break_wraps(t, d, 180.0),
                             "diff", color, **kw))

    add("head", "vehicle_attitude.yaw", "published (primary)", t_pub, yaw_pub,
        C_PUB, lw=2.4, visible=True, zorder=5)

    # Per-instance yaw.  Where the instances disagree with each other, the
    # selector's choice IS the published heading, and a handover is a step.
    for i in instances:
        t, _r, _p, y = _attitude(ulog, "estimator_attitude", i)
        add("head", f"estimator_attitude[{i}].yaw", f"EKF {i} yaw", t, y,
            inst_color(i), lw=1.3, visible=len(instances) > 1)
        add_diff(f"estimator_attitude[{i}].yaw", f"EKF {i} - published", t, y,
                 inst_color(i), lw=1.2, visible=len(instances) > 1)

    # The EKF-GSF: yaw from velocity and acceleration only, no magnetometer.
    # This is the trace that stays honest when the mag is the problem.
    for i in _instances(ulog, "yaw_estimator_status"):
        d = _get(ulog, "yaw_estimator_status", i)
        if d is None or "yaw_composite" not in d.data:
            continue
        t = _time_min(ulog, d)
        y = np.degrees(np.asarray(d.data["yaw_composite"], dtype=float))
        valid = np.asarray(d.data.get("yaw_composite_valid",
                                      np.ones_like(y)), dtype=float) > 0.5
        y = np.where(valid, y, np.nan)      # an invalid composite is not a heading
        add("head", f"yaw_estimator_status[{i}].yaw_composite", f"GSF {i} (no mag)",
            t, wrap180(y), inst_color(i), ls="--", lw=1.4, visible=True)
        add_diff(f"yaw_estimator_status[{i}].yaw_composite", f"GSF {i} - published",
                 t, wrap180(y), inst_color(i), ls="--", lw=1.4, visible=True)

    # Our own compass, computed once by the caller.
    tm, magnetic, true, tilt = compass
    add("head", "mag.heading_true", "magnetometer (+decl)", tm, true, C_MAG,
        lw=1.5, visible=True)
    add("head", "mag.heading_magnetic", "magnetometer (magnetic)", tm, magnetic,
        C_MAG_RAW, lw=1.1, visible=False)
    add_diff("mag.heading_true", "magnetometer - published", tm, true, C_MAG,
             lw=1.4, visible=True)
    add_diff("mag.heading_magnetic", "mag (magnetic) - published", tm, magnetic,
             C_MAG_RAW, lw=1.1, visible=False)
    if tm.size:
        frac = float(np.mean(tilt > TILT_LIMIT)) * 100.0
        if frac > 5.0:
            ctx.note(f"the vehicle was banked past {TILT_LIMIT:g} deg for {frac:.0f}% of the "
                     f"log -- tilt compensation degrades there, so read the "
                     f"magnetometer disagreement from the level stretches")

    t, cog = _gps_cog(ulog)
    add("head", "vehicle_gps_position.cog_rad", f"GNSS course (>{COG_MIN_SPEED:g} m/s)",
        t, cog, C_COG, lw=1.1, visible=False)
    add_diff("vehicle_gps_position.cog_rad", "GNSS course - published", t, cog,
             C_COG, lw=1.1, visible=False)

    t, gy = _gps_yaw(ulog)
    add("head", "vehicle_gps_position.heading", "GPS yaw (dual antenna)", t, gy,
        C_GPSYAW, lw=1.4, visible=True)
    add_diff("vehicle_gps_position.heading", "GPS yaw - published", t, gy,
             C_GPSYAW, lw=1.4, visible=True)

    t, sp = field(ulog, "vehicle_attitude_setpoint", "yaw_body")
    add("head", "vehicle_attitude_setpoint.yaw_body", "commanded yaw",
        t, wrap180(np.degrees(sp)), C_SP, lw=1.0, visible=False)

    return series, (t_pub, yaw_pub)


def _innovation_series(ulog, instances):
    """Panel 3: the EKF's own opinion of the heading data it was fed."""
    series = []
    for i in instances:
        t, v = field(ulog, "estimator_innovations", "heading", mid=i)
        if t.size and np.isfinite(v).any():
            series.append(Series(f"estimator_innovations[{i}].heading",
                                 f"EKF {i} heading innovation", t, np.degrees(v),
                                 "innov", inst_color(i), lw=1.3, visible=True))
        for k in range(3):
            t, v = field(ulog, "estimator_innovations", f"mag_field[{k}]", mid=i)
            if t.size and np.isfinite(v).any():
                series.append(Series(
                    f"estimator_innovations[{i}].mag_field[{k}]",
                    f"EKF {i} mag {'xyz'[k]} innov", t, v * GAUSS_TO_MGAUSS,
                    "innov_mag", inst_color(i), ls=[":", "--", "-."][k],
                    lw=1.0, visible=False))
        t, v = field(ulog, "estimator_innovation_test_ratios", "heading", mid=i)
        if t.size and np.isfinite(v).any():
            series.append(Series(f"estimator_innovation_test_ratios[{i}].heading",
                                 f"EKF {i} heading ratio", t, v, "ratio",
                                 inst_color(i), ls="--", lw=1.2, visible=True))
        t, v = field(ulog, "estimator_status", "mag_test_ratio", mid=i)
        if t.size and np.isfinite(v).any():
            series.append(Series(f"estimator_status[{i}].mag_test_ratio",
                                 f"EKF {i} mag test ratio", t, v, "ratio",
                                 inst_color(i), ls=":", lw=1.2, visible=False))
    return series


def _bias_series(ulog, instances):
    """Panel 4: learned mag bias, learned declination, and field strength."""
    series = []
    for i in instances:
        for k in range(3):
            t, v = field(ulog, "estimator_sensor_bias", f"mag_bias[{k}]", mid=i)
            if t.size and np.isfinite(v).any() and np.any(v != 0):
                series.append(Series(
                    f"estimator_sensor_bias[{i}].mag_bias[{k}]",
                    f"EKF {i} mag bias {'xyz'[k]}", t, v * GAUSS_TO_MGAUSS,
                    "bias", inst_color(i), ls=["-", "--", ":"][k], lw=1.2,
                    visible=True))
        t, dec, strength = declination(ulog, i)
        if t.size:
            series.append(Series(f"estimator_states[{i}].declination",
                                 f"EKF {i} declination", t, dec, "decl",
                                 inst_color(i) if len(instances) > 1 else C_DECL,
                                 lw=1.6, visible=True))
            series.append(Series(f"estimator_states[{i}].field_strength",
                                 f"EKF {i} earth field", t,
                                 strength * GAUSS_TO_MGAUSS, "bias",
                                 inst_color(i), ls="-.", lw=1.0, visible=False))
    return series


# --- the fault band ----------------------------------------------------------

def _band_rows(ulog, ctx, instances):
    """([(label, lanes, colour)], n_clean) -- the heading checklist.

    Same contract as the accelerometer plot's band: every condition gets a row
    whether or not it fired, and every row gets one lane per instance whether or
    not that instance fired.  An empty row means "checked, clean" -- which is a
    different statement from "not checked", and the whole value of the panel is
    that you can tell them apart.
    """
    rows, n_clean = [], 0

    def lanes_for(fn):
        return [(fn(i) or [], inst_color(i)) for i in instances]

    def add(label, lanes, fault=True):
        nonlocal n_clean
        if not any(sp for sp, _c in lanes):
            n_clean += 1
        rows.append((label, lanes, C_BAD if fault else C_MUTED))

    rows.append(("armed", [(armed_spans(ulog), C_ARMED)], C_MUTED))

    def _flag(key, invert=False):
        def get(i):
            d = _get(ulog, "estimator_status_flags", i)
            if d is None or key not in d.data:
                return []
            v = np.asarray(d.data[key], dtype=float) > 0.5
            return spans_from_bool(_time_min(ulog, d), ~v if invert else v)
        return get

    # -- is yaw observable at all, and from what ------------------------------
    add("yaw NOT aligned", lanes_for(_flag("cs_yaw_align", invert=True)))
    for key, label in (("cs_mag_hdg", "fusing: mag heading (1-axis)"),
                       ("cs_mag_3d", "fusing: mag 3-axis"),
                       ("cs_mag_dec", "fusing: declination"),
                       ("cs_gps_yaw", "fusing: GPS yaw"),
                       ("cs_ev_yaw", "fusing: external vision yaw"),
                       ("cs_mag_aligned_in_flight", "mag aligned in flight")):
        add(label, lanes_for(_flag(key)), fault=False)

    # -- and what went wrong with it -----------------------------------------
    for key, label in (("reject_yaw", "REJECTING yaw measurements"),
                       ("cs_mag_fault", "mag FAULT"),
                       ("cs_mag_field_disturbed", "mag field disturbed"),
                       ("cs_gps_yaw_fault", "GPS yaw fault"),
                       ("cs_synthetic_mag_z", "synthetic mag z (z not usable)"),
                       ("fs_bad_mag_x", "EKF fault: bad mag x"),
                       ("fs_bad_mag_y", "EKF fault: bad mag y"),
                       ("fs_bad_mag_z", "EKF fault: bad mag z"),
                       ("fs_bad_mag_decl", "EKF fault: bad declination")):
        add(label, lanes_for(_flag(key)))

    # -- the arming checks commander actually runs ---------------------------
    def _pre(key):
        def get(i):
            d = _get(ulog, "estimator_status", i)
            if d is None or key not in d.data:
                return []
            v = np.asarray(d.data[key], dtype=float) > 0.5
            return spans_from_bool(_time_min(ulog, d), v)
        return get

    add("PREFLIGHT FAIL: heading innovation", lanes_for(_pre("pre_flt_fail_innov_heading")))
    add("PREFLIGHT FAIL: mag field disturbed",
        lanes_for(_pre("pre_flt_fail_mag_field_disturbed")))

    # -- gate the estimate was rejected by ------------------------------------
    def _ratio_over(i):
        t, v = field(ulog, "estimator_innovation_test_ratios", "heading", mid=i)
        if t.size == 0:
            return []
        return spans_from_bool(t, np.asarray(v, dtype=float) >= REJECT_RATIO)

    add(f"heading test ratio >= {REJECT_RATIO:g}", lanes_for(_ratio_over))

    def _bias_invalid(i):
        d = _get(ulog, "estimator_sensor_bias", i)
        if d is None or "mag_bias_valid" not in d.data:
            return []
        v = np.asarray(d.data["mag_bias_valid"], dtype=float) > 0.5
        return spans_from_bool(_time_min(ulog, d), ~v)

    add("mag bias not valid", lanes_for(_bias_invalid))

    # -- and one flag from outside the estimator ------------------------------
    d = _get(ulog, "vehicle_local_position")
    if d is not None and "heading_good_for_control" in d.data:
        v = np.asarray(d.data["heading_good_for_control"], dtype=float) > 0.5
        add("heading NOT good for control",
            [(spans_from_bool(_time_min(ulog, d), ~v), C_BAD)])

    return rows, n_clean


# --- rescaling ---------------------------------------------------------------

def _rescale_ratio(ax, lines, pct=99.5):
    """Percentile limits for the log-scaled test-ratio axis, straddling 1.0.

    Same reasoning as the altitude and local-z plots: min..max on a quantity
    that spikes to 1e3 at a reset leaves the 0.01..1 range every normal sample
    lives in as a hairline, and a log axis silently refuses a limit <= 0.
    """
    v = window_values(ax, lines, positive_only=True)
    if v.size == 0:
        return 0, 0.0
    hi = max(float(np.percentile(v, pct)) * 1.4, REJECT_RATIO * 2.0)
    lo = min(max(float(np.percentile(v, 100.0 - pct)) / 1.4, 1e-4),
             REJECT_RATIO / 10.0)
    ax.set_ylim(lo, hi)
    return int((v > hi).sum()), float(v.max())


def _rescale_heading(ax, lines, center=0.0):
    """Fit the heading panel, but never to less than a readable span.

    A vehicle that never turns gives min..max of a fraction of a degree, and the
    panel then renders sensor noise as if it were a manoeuvre.  Anything under
    10 degrees is padded out to 10.  The clamp is one full turn about the branch
    centre -- outside that there is nothing to show, because every heading has
    been folded into it.
    """
    v = window_values(ax, lines)
    if v.size == 0:
        return
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 10.0:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 5.0, mid + 5.0
    m = (hi - lo) * 0.06
    ax.set_ylim(max(lo - m, center - 190.0), min(hi + m, center + 190.0))


# --- the figure --------------------------------------------------------------

def build_heading(ulog, ctx=None, path=""):
    """The heading figure.  Same signature as every plot builder."""
    import matplotlib.pyplot as plt

    ctx = ctx or PlotCtx()

    if not has_topic(ulog, "vehicle_attitude"):
        ctx.note("no vehicle_attitude in this log -- no heading to plot")
        return None

    instances = _instances(ulog) or _instances(ulog, "estimator_status_flags") or [0]
    spans = primary_spans(ulog)

    # The declination the EKF learned, from whichever instance was primary
    # longest -- it is what our own compass is corrected by, so it has to be
    # settled before the series are built.
    decl_t, decl_deg, _ = declination(ulog, instances[0])
    if decl_t.size == 0:
        ctx.note("no estimator_states in this log -- the magnetometer heading "
                 "is shown WITHOUT a declination correction, so a constant "
                 "offset from the estimate is expected")

    # The compass is computed ONCE: the series builder plots it and the summary
    # line under panel 2 measures it, and it is ~30 k samples of trigonometry.
    compass = mag_heading(ulog, decl_t, decl_deg)

    # The wrap branch, from the published estimate, shared by every heading
    # trace on panel 1 so they stay comparable.
    t_pub0, _r0, _p0, yaw_pub0 = _attitude(ulog)
    center = heading_center(yaw_pub0)

    series, (t_pub, yaw_pub) = _heading_series(ulog, ctx, instances,
                                               compass, center)
    if not series:
        ctx.note("vehicle_attitude carries no usable quaternion -- nothing to plot")
        return None
    series += _innovation_series(ulog, instances)
    series += _bias_series(ulog, instances)

    rows, n_clean = _band_rows(ulog, ctx, instances)

    # --- figure, sized from the band ----------------------------------------
    band_in = min(max(len(rows) * BAND_ROW_IN + BAND_PAD_IN, BAND_MIN_IN),
                  BAND_MAX_IN)
    fig_h = (TOP_IN + sum(h for _k, h in PANEL_IN) + GAP_IN * len(PANEL_IN)
             + band_in + BOTTOM_IN)
    if len(rows) * BAND_ROW_IN + BAND_PAD_IN > BAND_MAX_IN:
        ctx.note(f"{len(rows)} band rows do not fit at full spacing -- the band "
                 f"is capped at {BAND_MAX_IN:g} in and its rows are tighter than "
                 f"the rest of the figure")

    fig = plt.figure(figsize=(15, fig_h), facecolor=C_SURFACE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(
            f"logGraph heading - {os.path.basename(path)}")

    left, width = 0.260, 0.655

    def _f(inches):
        return inches / fig_h

    rects, bottom = {}, BOTTOM_IN
    rects["band"] = (_f(bottom), _f(band_in))
    bottom += band_in + GAP_IN
    for key, h in reversed(PANEL_IN):
        rects[key] = (_f(bottom), _f(h))
        bottom += h + GAP_IN

    ax_head, ax_diff, ax_innov, ax_bias, ax_band = [
        fig.add_axes([left, rects[k][0], width, rects[k][1]], facecolor=C_SURFACE)
        for k in ("head", "diff", "innov", "bias", "band")]
    for a in (ax_head, ax_diff, ax_innov, ax_bias):
        a.sharex(ax_band)
    ax_ratio = ax_innov.twinx()
    ax_decl = ax_bias.twinx()
    for a in (ax_ratio, ax_decl):
        a.set_facecolor("none")

    axis_of = {"head": ax_head, "diff": ax_diff, "innov": ax_innov,
               "innov_mag": ax_innov, "ratio": ax_ratio, "bias": ax_bias,
               "decl": ax_decl}

    shade_art = []
    for a in (ax_head, ax_diff, ax_innov, ax_bias, ax_band):
        shade_art += draw_primary_shading(a, spans)
    armed_art = []
    for a in (ax_head, ax_diff, ax_innov, ax_bias):
        armed_art += draw_armed(a, armed_spans(ulog))
    for art in armed_art:
        art.set_visible(False)

    ax_ratio.set_yscale("log")
    ax_ratio.axhline(REJECT_RATIO, color=C_BAD, lw=1.1, ls="--", alpha=0.8, zorder=1)
    ax_ratio.text(0.012, REJECT_RATIO, "rejected >= 1.0",
                  transform=ax_ratio.get_yaxis_transform(), color=C_BAD,
                  fontsize=7, va="bottom", ha="left")
    for a in (ax_diff, ax_innov):
        a.axhline(0.0, color=C_MUTED, lw=1, ls=":", alpha=0.6, zorder=1)

    draw_band_rows(ax_band, rows, ylabel="yaw source / faults",
                   empty_msg="no estimator status flags in this log")

    for s in series:
        (line,) = axis_of[s.group].plot(
            s.t, s.y, color=s.color, ls=s.ls, lw=s.lw, label=s.label,
            drawstyle=s.drawstyle, alpha=s.alpha,
            zorder=s.zorder if s.zorder is not None else 3)
        line.set_visible(s.visible)
        s.line = line

    # Heading resets, on the panel they moved.
    reset_art = []
    for t_r, dh, c0, c1 in _reset_events(ulog):
        reset_art.append(ax_head.axvline(t_r, color=C_BAD, lw=1.0, ls=":",
                                         alpha=0.75, zorder=2))
        if abs(dh) < 1.0:
            continue          # sub-degree resets are the filter settling
        lbl = ax_head.text(t_r, 0.985, f" reset {c0}->{c1}  {dh:+.0f} deg",
                           transform=ax_head.get_xaxis_transform(), rotation=90,
                           fontsize=6, color=C_BAD, va="top", ha="left", zorder=6)
        lbl.set_clip_on(True)
        reset_art.append(lbl)

    # --- axis furniture -----------------------------------------------------
    for a in (ax_head, ax_diff, ax_innov, ax_bias):
        style_time_axis(a, label=False)
        a.tick_params(axis="x", labelbottom=False)
    style_time_axis(ax_band)

    ax_head.set_ylabel("heading (deg)\nNED, clockwise from north", fontsize=9)
    if abs(center) > 45.0:
        # Say so, or a reader sees -250 on the axis and doubts the plot rather
        # than the branch.
        ax_head.text(0.995, 0.04, f"wrapped about {center:+.0f} deg to keep the "
                     f"branch cut off the trace", transform=ax_head.transAxes,
                     ha="right", va="bottom", fontsize=7, color=C_MUTED)
    ax_diff.set_ylabel("minus published (deg)", fontsize=9)
    ax_innov.set_ylabel("heading innov (deg)\nmag innov (mGauss)", fontsize=9)
    ax_ratio.set_ylabel("test ratio (dashed)", fontsize=9)
    ax_bias.set_ylabel("mag bias (mGauss)", fontsize=9)
    ax_decl.set_ylabel("declination (deg)", fontsize=9)
    for a, c in ((ax_head, C_INK), (ax_diff, C_INK), (ax_innov, C_INK),
                 (ax_bias, C_INK), (ax_ratio, C_MUTED), (ax_decl, C_MUTED)):
        _style_axis(a, c)

    fig.text(left, 1.0 - _f(0.30), "Heading estimation", color=C_INK,
             fontsize=13, fontweight="bold", ha="left")
    who = f"{os.path.basename(path)}   |   " if path else ""
    if decl_t.size:
        decl_note = f"EKF declination {decl_deg[-1]:+.2f} deg at the end"
    else:
        decl_note = "no learned declination in this log"
    fig.text(left, 1.0 - _f(0.60),
             f"{who}{duration_min(ulog):.1f} min   |   {decl_note}   |   "
             f"{len(rows) - n_clean} of {len(rows)} band rows fired",
             color=C_MUTED, fontsize=9, ha="left")
    instance_key(fig, left, width, spans, y=1.0 - _f(0.30))

    off_note = ax_ratio.text(0.995, 0.96, "", transform=ax_ratio.transAxes,
                             ha="right", va="top", fontsize=7, color=C_BAD)

    # The headline number: how far the independent compass sits from the
    # estimate, measured only where the vehicle was level enough for the
    # tilt compensation to mean anything.
    tm, magnetic, true, tilt = compass
    if tm.size and np.isfinite(true).any():
        d = wrap180(true - circ_interp(tm, t_pub, yaw_pub))
        lvl = np.isfinite(d) & (tilt < TILT_LIMIT)
        if lvl.sum() > 10:
            med = float(np.median(d[lvl]))
            iqr = float(np.subtract(*np.percentile(d[lvl], [75, 25])))
            ax_diff.text(0.995, 0.04,
                         f"magnetometer vs estimate (level only): "
                         f"median {med:+.1f} deg, IQR {iqr:.1f} deg",
                         transform=ax_diff.transAxes, ha="right", va="bottom",
                         fontsize=7,
                         color=C_BAD if abs(med) > 5.0 else C_MUTED)

    mode_art, mode_codes = draw_mode_changes(
        [ax_head, ax_diff, ax_innov, ax_bias, ax_band], mode_changes(ulog),
        text_ax=ax_diff, min_gap=max(duration_min(ulog), 1.0) * 0.035)

    def refresh():
        _rescale_heading(ax_head, [s.line for s in series if s.group == "head"],
                         center)
        for group, a in (("diff", ax_diff), ("innov", ax_innov),
                         ("bias", ax_bias), ("decl", ax_decl)):
            lines = [s.line for s in series
                     if s.group == group or (group == "innov"
                                             and s.group == "innov_mag")]
            _rescale(a, lines)
        n, mx = _rescale_ratio(ax_ratio,
                               [s.line for s in series if s.group == "ratio"])
        off_note.set_text(f"{n} test ratio(s) off-scale (max {mx:.0f})" if n else "")

    extra = []
    mode_art += mode_key(fig, left + width, _f(0.10), mode_codes)
    if mode_art:
        extra.append(("mode changes", mode_art, True))
    if reset_art:
        extra.append(("heading resets (marked)", reset_art, True))
    if shade_art:
        extra.append(("instance shading", shade_art, True))
    if armed_art:
        extra.append(("armed (shaded)", armed_art, False))

    cb_top = rects["head"][0] + rects["head"][1]
    cb_bot = rects["band"][0]
    h = cb_top - cb_bot

    def _anchor(key):
        b, ph = rects[key]
        return (b + ph / 2 - cb_bot) / h

    check_panel(fig, [0.012, cb_bot, 0.155, h], series,
                [("head", "ALL headings"), ("diff", "ALL differences"),
                 ("innov", "ALL heading innovations"),
                 ("innov_mag", "ALL mag innovations"),
                 ("ratio", "ALL test ratios"), ("bias", "ALL mag bias"),
                 ("decl", "ALL declination")],
                extra=extra, on_change=refresh,
                anchors={"head": _anchor("head"), "diff": _anchor("diff"),
                         "innov": _anchor("innov"), "innov_mag": _anchor("innov"),
                         "ratio": _anchor("innov"), "bias": _anchor("bias"),
                         "decl": _anchor("bias")})
    refresh()
    add_mouse_navigation(fig, [ax_head, ax_diff, ax_innov, ax_ratio, ax_bias,
                               ax_decl, ax_band], page_scroll=ctx.page_scroll,
                         fixed_y=[ax_band], on_view=refresh)
    fig.text(left, _f(0.32), nav_hint(ctx.page_scroll), color=C_MUTED,
             fontsize=8, ha="left")
    fig._page_height = int(round(fig_h * PAGE_PX_PER_IN))
    return fig
