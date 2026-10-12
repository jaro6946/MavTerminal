#!/usr/bin/env python3
"""ulog_alt.py -- the altitude estimation plot.

Eight stacked panels on one time axis, answering "why does the vehicle think it
is at that height?":

  1. AMSL overlay   -- every height source as logged, plus the EKF's fused answer
  2. residual       -- each source MINUS the fused answer, which is where drift lives
  3. innovations    -- what the EKF itself thought of each measurement
  4. source band    -- which source it was actually fusing, and when each was valid
  5-7. GPS quality  -- EPV/EPH/speed accuracy, satellites/PDOP and on-ground drift,
                       each against the EKF2_REQ_* limit that gates GPS fusion
  8. gate band      -- per-gate pass/fail, marked enforced or ignored by
                       EKF2_GPS_CHECK; the reason panel 4's "fusing GPS" drops out

Why four panels and not one overlay
-----------------------------------
The sources do not share a datum.  Measured on a real flight (d05a88e3):

    fused (vehicle_global_position.alt)   84.2 .. 93.5 m
    GPS   (vehicle_gps_position.alt)      median 10.2 m BELOW fused
    baro  (vehicle_air_data.baro_alt_meter)  median 84.0 m BELOW fused

so on a single AMSL axis the vehicle's actual 9 m of flying is a squiggle inside a
90 m span, and a 2 m barometer drift -- the thing you opened the plot to see -- is
about one line width.  Panel 2 subtracts the common motion so the y axis can be
metres instead of tens of metres.

Acronyms: AMSL = above mean sea level, HAGL = height above ground level,
EKF = extended Kalman filter, GNSS/GPS = global navigation satellite system,
QNH = the pressure setting that makes an altimeter read elevation.
"""
import numpy as np

from ulog_common import (C_ARMED, C_BAD, C_INK, C_MUTED, C_SURFACE, PlotCtx,
                         Series, _clean, _get, _rescale, _style_axis,
                         _time_min, add_mouse_navigation, armed_spans,
                         check_panel, draw_armed, draw_band_rows,
                         duration_min, field, draw_mode_changes, gap_mask,
                         has_topic, mode_changes, mode_key, nav_hint,
                         primary_ekf, resample_to, spans_from_bool,
                         style_time_axis, window_values)

ALT_TOPICS = [
    "vehicle_global_position", "vehicle_local_position", "vehicle_gps_position",
    "vehicle_air_data", "distance_sensor", "home_position",
    "estimator_status_flags", "estimator_innovations",
    "estimator_innovation_test_ratios", "estimator_gnss_hgt_bias",
    "estimator_rng_hgt_bias", "estimator_selector_status", "actuator_armed",

    "vehicle_status",          # flight-mode overlay
    "estimator_gps_status",    # GPS gate verdicts + drift rates (panels 7-8)
]

# --- color ------------------------------------------------------------------
# One hue per PHYSICAL SOURCE, held constant across all four panels: the baro
# line in panel 1, the baro residual in panel 2, the baro innovation in panel 3
# and the baro band in panel 4 are all the same color.  That is the whole reason
# the panels can be read together.  Fused is near-black because it is the
# reference every other panel is measured against, not a peer of the others.
C_FUSED = "#20222b"   # near-black
C_GPS = "#2a78d6"     # blue    -- same blue the thermal plot uses for GPS
C_BARO = "#d2691e"    # orange
C_RNG = "#1baf7a"     # aqua
C_HOME = "#8a7fb5"    # muted violet
C_BIAS = "#b0348c"    # magenta -- PX4's OWN estimate of an offset, not a measurement

# Where GPS altitude lives, and what it is in, depends on the firmware.  The lab's
# older build logs vehicle_gps_position.alt as int32 MILLIMETRES; PX4 v1.18 (on the
# HITL board) renamed it to altitude_msl_m and made it float METRES.  Both are in
# the wild in this project's logs, and reading the new field with the old scale
# would put a 30 m hover at 0.03 m -- silently, since nothing errors.  Tried in
# order, first hit wins.
GPS_ALT_FIELDS = [("alt", 1e-3), ("altitude_msl_m", 1.0)]
# A 3D fix or better.  Below this, vehicle_gps_position.alt in a real log ranges
# to +2617 m and -17 m -- garbage that would set the AMSL axis single-handedly.
MIN_FIX = 3

# --- GPS fusion gates (panels 5-8) ------------------------------------------
# One hue per DIRECTION, held across the three GPS value panels: blue is
# vertical (the one this plot is about), violet horizontal, goldenrod speed.
# Neutral slate / purple for the satellite-geometry pair, which are neither.
C_VERT = C_GPS
C_HORZ = "#4a3aa7"
C_SPD = "#b8860b"
C_NSAT = "#56677a"
C_PDOP = "#7b2d8e"

# Figure layout, inches (see build_altitude).
TOP_IN = 0.95          # title + subtitle
GAP_IN = 0.34          # between panels: each keeps its own time tick labels
SECTION_GAP_IN = 0.62  # above the GPS panels, which carry a section heading
BOTTOM_IN = 0.95       # x label + nav hint + mode key
GATE_ROW_IN = 0.24     # per gate row; 12 rows of 7 pt labels
PAGE_PX_PER_IN = 78    # the browser's scroll page, same as ulog_accel

# The EKF2_GPS_CHECK bitmask, from the flown firmware's
# src/modules/ekf2/EKF/gps_checks.cpp (MASK_GPS_*).  Every check_fail_* flag in
# estimator_gps_status is COMPUTED whether or not its bit is set -- the mask only
# decides whether a failure blocks fusion.  So a flag firing is not by itself a
# problem, and the rows below say which ones were actually enforced.  The fix
# check has no bit: fix_type < 3 always blocks.
#
# How a failure turns into "not fusing GPS" (v1.14 flown firmware): any
# enforced failure stamps _last_gps_fail_us, and checks_passed (_gps_checks_passed)
# is "no enforced failure for the last 10 s".
#   GPS HEIGHT (gnss_height_control.cpp): needs checks_passed to START *and* to
#     CONTINUE, so one enforced failure stops it on that sample and it cannot
#     restart for 10 s.  The "checks passed" row is therefore, in effect, the
#     "GPS height allowed" row -- compare it with "fusing GPS" in panel 4.
#   GPS horizontal pos/vel (gps_control.cpp): starts after 5 s clean, stops after
#     5 s failing -- EXCEPT that while GPS is the only horizontal aiding source it
#     keeps fusing regardless, since dropping it would leave nothing.
#
#   (estimator_gps_status field, label, mask bit or None, EKF2_REQ_* param, unit)
GATE_SHORT = {0: "NSATS", 1: "PDOP", 2: "EPH", 3: "EPV", 4: "SACC",
              5: "HDRIFT", 6: "VDRIFT", 7: "HSPD", 8: "VSPD"}
GPS_GATES = [
    ("check_fail_gps_fix", "fix < 3D", None, None, ""),
    ("check_fail_min_sat_count", "sats <", 0, "EKF2_REQ_NSATS", ""),
    ("check_fail_max_pdop", "PDOP >", 1, "EKF2_REQ_PDOP", ""),
    ("check_fail_max_horz_err", "EPH >", 2, "EKF2_REQ_EPH", " m"),
    ("check_fail_max_vert_err", "EPV >", 3, "EKF2_REQ_EPV", " m"),
    ("check_fail_max_spd_err", "speed acc >", 4, "EKF2_REQ_SACC", " m/s"),
    ("check_fail_max_horz_drift", "horiz drift (gnd) >", 5, "EKF2_REQ_HDRIFT", " m/s"),
    ("check_fail_max_vert_drift", "vert drift (gnd) >", 6, "EKF2_REQ_VDRIFT", " m/s"),
    ("check_fail_max_horz_spd_err", "horiz speed (gnd) >", 7, "EKF2_REQ_HDRIFT", " m/s"),
    ("check_fail_max_vert_spd_err", "vert vel mismatch >", 8, "EKF2_REQ_VDRIFT", " m/s"),
]

def _fused_amsl(ulog, ctx):
    """The EKF's altitude, AMSL, with a fallback for logs missing the global topic.

    vehicle_global_position.alt is the direct answer.  Where it is absent, the
    same quantity is ref_alt - z from vehicle_local_position: ref_alt is the AMSL
    altitude of the local origin and z is DOWN-positive from it.  These agree
    exactly (checked on d05a88e3: both span 84.18..93.51 m), so the fallback is a
    genuine substitute rather than an approximation.
    """
    t, y = field(ulog, "vehicle_global_position", "alt")
    if y.size:
        return t, y, "vehicle_global_position.alt"

    d = _get(ulog, "vehicle_local_position")
    if d is None or "ref_alt" not in d.data or "z" not in d.data:
        ctx.note("no fused altitude in this log (no vehicle_global_position, "
                 "no vehicle_local_position.ref_alt/z)")
        return np.array([]), np.array([]), None
    t, y = _clean(_time_min(ulog, d),
                  np.asarray(d.data["ref_alt"], dtype=float)
                  - np.asarray(d.data["z"], dtype=float))
    ctx.note("no vehicle_global_position -- fused altitude reconstructed as "
             "vehicle_local_position.ref_alt - z")
    return t, y, "ref_alt - z"


def _gps_amsl(ulog, ctx):
    """GPS altitude in metres, blanked wherever the fix was not 3D.

    Gating rather than clipping: an unfixed receiver still reports a number, and
    on d05a88e3 those numbers reach 2617 m.  Blanking leaves a visible hole,
    which is the honest rendering of "we did not know"."""
    d = _get(ulog, "vehicle_gps_position") or _get(ulog, "sensor_gps")
    if d is None:
        ctx.note("no GPS topic in this log")
        return np.array([]), np.array([])
    fname, scale = next(((f, s) for f, s in GPS_ALT_FIELDS if f in d.data),
                        (None, None))
    if fname is None:
        ctx.note(f"no GPS altitude field in {d.name} "
                 f"(looked for {', '.join(f for f, _ in GPS_ALT_FIELDS)})")
        return np.array([]), np.array([])
    t = _time_min(ulog, d)
    y = np.asarray(d.data[fname], dtype=float) * scale
    if "fix_type" in d.data:
        fix = np.asarray(d.data["fix_type"], dtype=float)
        bad = int((fix < MIN_FIX).sum())
        if bad:
            ctx.note(f"GPS altitude blanked for {bad} sample(s) "
                     f"({100.0*bad/max(fix.size,1):.1f}%) with fix_type < {MIN_FIX}")
        y = gap_mask(t, y, fix >= MIN_FIX)
    m = np.isfinite(t)
    return t[m], y[m]


def _rng_hagl(ulog, ctx):
    """Rangefinder height above ground, gated on the EKF's own validity flag.

    Prefers vehicle_local_position.dist_bottom over the raw distance_sensor
    topic, because dist_bottom is already rotated into the vertical and comes
    with dist_bottom_valid.  On d05a88e3 that flag is true only 33.2% of the
    time, and 63.4% of raw distance_sensor samples carry signal_quality 0 with
    current_distance pinned at the 0.05 m minimum -- so an ungated trace is
    mostly the sensor saying "nothing here", drawn as if it were a measurement.
    """
    d = _get(ulog, "vehicle_local_position")
    if d is not None and "dist_bottom" in d.data:
        t = _time_min(ulog, d)
        y = np.asarray(d.data["dist_bottom"], dtype=float)
        if "dist_bottom_valid" in d.data:
            v = np.asarray(d.data["dist_bottom_valid"], dtype=float) > 0.5
            frac = 100.0 * v.mean() if v.size else 0.0
            ctx.note(f"rangefinder valid {frac:.1f}% of samples "
                     f"(dist_bottom_valid); invalid stretches drawn as gaps")
            y = gap_mask(t, y, v)
        m = np.isfinite(t)
        return t[m], y[m]

    # No dist_bottom: fall back to the raw sensor, gated on signal_quality.
    d = _get(ulog, "distance_sensor")
    if d is None or "current_distance" not in d.data:
        ctx.note("no rangefinder in this log (no distance_sensor, "
                 "no vehicle_local_position.dist_bottom)")
        return np.array([]), np.array([])
    t = _time_min(ulog, d)
    y = np.asarray(d.data["current_distance"], dtype=float)
    if "signal_quality" in d.data:
        q = np.asarray(d.data["signal_quality"], dtype=float)
        y = gap_mask(t, y, q > 0)
        ctx.note("rangefinder taken from raw distance_sensor, gated on "
                 "signal_quality > 0")
    m = np.isfinite(t)
    return t[m], y[m]


def _visible_values(ax, lines, positive_only=False):
    """Visible lines' values inside the visible time window -- see
    ulog_common.window_values for why the window filter is not optional."""
    return window_values(ax, lines, positive_only=positive_only)


def _rescale_robust(ax, lines, pct=99.5, min_span=1.0):
    """Fit a linear axis to the BULK of the data, not to its extremes.

    The plain _rescale fits min..max, which is right for a temperature curve and
    wrong here.  Real numbers from d05a88e3: the vertical innovations sit inside
    +-1.4 m at the 99th percentile but spike to 135 m a handful of times, so
    min..max gives a 190 m axis on which every normal sample is a flat line at
    zero -- the panel shows only that an outlier happened, and nothing about the
    behaviour around it.

    Outliers are not hidden: they still draw, running off the top of the panel,
    and the caller annotates how many and how large.  Zooming out (ctrl+shift+
    wheel) brings them back into view.
    """
    v = _visible_values(ax, lines)
    if v.size == 0:
        return 0, 0.0
    lo = float(np.percentile(v, 100.0 - pct))
    hi = float(np.percentile(v, pct))
    if hi - lo < min_span:                  # don't zoom into pure quantisation noise
        mid = 0.5 * (hi + lo)
        lo, hi = mid - min_span / 2, mid + min_span / 2
    m = (hi - lo) * 0.08
    ax.set_ylim(lo - m, hi + m)
    off = int(((v < lo - m) | (v > hi + m)).sum())
    return off, float(np.max(np.abs(v)))


def _rescale_log(ax, lines, pct=99.5):
    """_rescale's counterpart for the log-scaled test-ratio axis.

    Two problems the shared _rescale cannot handle.  It would set a limit at or
    below zero, which a log axis silently refuses -- leaving the previous range
    in place and quietly mismatching the data.  And the ratios span from below
    1e-10 up to inf, so even a correct min..max is a twelve-decade axis on which
    the 1.0 threshold is a hairline at the top.

    So: percentile bounds, floored at 1e-3 (below that a measurement is agreeing
    so well the exact value carries nothing) and always straddling 1.0, which is
    the accept/reject border the whole panel is read against.
    """
    v = _visible_values(ax, lines, positive_only=True)
    if v.size == 0:
        return 0, 0.0
    hi = float(np.percentile(v, pct))
    lo = float(np.percentile(v, 100.0 - pct))
    lo = max(lo / 1.4, 1e-3)
    hi = max(hi * 1.4, 2.0)
    ax.set_ylim(min(lo, 0.5), hi)
    off = int((v > hi).sum())
    return off, float(np.max(v))


def _blank_unfused(y):
    """NaN out samples that are not measurements.

    Exact 0.0 means PX4 did not run that fusion step this cycle and left the
    field at its initialiser -- drawn literally, those are long flat runs at zero
    that read as a perfectly agreeing sensor.  inf appears in the test ratios
    when the innovation variance is zero, i.e. the same "no measurement" case
    arriving by a different route.  Both have to go or the panel lies.
    """
    y = np.asarray(y, dtype=float).copy()
    y[(y == 0.0) | ~np.isfinite(y)] = np.nan
    return y


def _residual(t_src, y_src, t_fused, y_fused, debias):
    """sensor - fused, resampled onto the SENSOR's timestamps.

    The fused series is the one moved, never the sensor: the sensor's own sample
    times are the ground truth about when it spoke, and resampling it would
    smear a dropout into a slope.

    With `debias`, the median offset is removed and returned so the caller can
    state it in the label.  On real data the offsets are -84 m (baro) and -10 m
    (GPS); leaving them in means the axis has to span 90 m and the drift you care
    about is invisible.  The offset is reported rather than discarded, because
    "baro reads 84 m low" is itself a finding.
    """
    ref = resample_to(t_src, t_fused, y_fused)
    r = y_src - ref
    if not debias:
        return r, 0.0
    finite = r[np.isfinite(r)]
    off = float(np.median(finite)) if finite.size else 0.0
    return r - off, off


# --- GPS fusion gates -------------------------------------------------------

def _param_steps(ulog, name):
    """A parameter as a step series over the whole log: initial value, then
    every in-flight change.  Empty if this firmware has no such parameter.

    A series rather than a scalar because a gate limit drawn as a constant would
    lie on any log where someone changed it mid-session -- and the limit line is
    the thing every trace on these panels is read against."""
    if name not in ulog.initial_parameters:
        return np.array([]), np.array([])
    t0 = getattr(ulog, "start_timestamp", 0) or 0
    t, y = [0.0], [float(ulog.initial_parameters[name])]
    for ts, n, v in ulog.changed_parameters:
        if n == name:
            t.append(max((ts - t0) / 6e7, 0.0))
            y.append(float(v))
    t.append(max(duration_min(ulog), t[-1]))
    y.append(y[-1])
    return np.asarray(t), np.asarray(y)


def _gps_mask(ulog, ctx):
    """EKF2_GPS_CHECK as logged at boot, plus a note if it changed in flight."""
    mask = int(ulog.initial_parameters.get("EKF2_GPS_CHECK", 0))
    if any(n == "EKF2_GPS_CHECK" for _t, n, _v in ulog.changed_parameters):
        ctx.note("EKF2_GPS_CHECK changed during this log; 'enforced' labels use "
                 "the boot value")
    return mask


def _enforced(mask, bit):
    return bit is None or bool(mask & (1 << bit))


def _limit_series(ulog, sid, param, unit, group, color, enforced):
    """The gate limit as a toggleable dashed step line in its metric's colour.

    A gate whose EKF2_GPS_CHECK bit is off is still drawn -- the receiver can
    cross it and the flag still fires -- but faded and labelled, because a trace
    over a line that blocks nothing is not a problem."""
    t, y = _param_steps(ulog, param)
    if not y.size:
        return None
    short = param.replace("EKF2_REQ_", "REQ_")
    tag = "" if enforced else "  (gate OFF)"
    return Series(sid, f"{short} {y[0]:g}{unit}{tag}", t, y, group, color,
                  ls="--", visible=True, lw=1.3 if enforced else 1.0,
                  alpha=0.95 if enforced else 0.45, drawstyle="steps-post",
                  zorder=4)


def _gps_quality_series(ulog, ekf, mask, ctx):
    """Panels 5-7: each gated quantity with its EKF2_REQ_* limit beside it.

    Read from vehicle_gps_position because that is exactly the message ekf2
    feeds to its checks (EKF2.cpp: eph/epv straight through, sacc =
    s_variance_m_s, nsats = satellites_used, pdop = sqrt(hdop^2 + vdop^2)).
    The drift rates exist only inside the EKF, so those come from
    estimator_gps_status of the instance this figure follows.
    """
    out = []
    g = _get(ulog, "vehicle_gps_position") or _get(ulog, "sensor_gps")
    if g is None:
        ctx.note("no GPS topic -- GPS gate panels empty")
        return out
    t = _time_min(ulog, g)
    ok = (np.asarray(g.data["fix_type"], dtype=float) >= MIN_FIX
          if "fix_type" in g.data else np.ones(t.size, bool))

    def gfield(name, scale=1.0):
        if name not in g.data:
            return None
        # Blanked without a 3D fix: the fix gate is failing anyway, and the
        # numbers are receiver placeholders -- hdop/vdop 99.99 gives PDOP 141,
        # which would set the axis for the whole flight.
        return gap_mask(t, np.asarray(g.data[name], dtype=float) * scale, ok)

    # Panel 5: reported accuracy.  Log axis: on the real flights these sit at
    # 0.1-0.6 while the limits are 0.5-5 and the bad-sky spikes reach 36 m, and
    # a linear axis can show the limit or the normal data, never both.
    for key, fname, label, color, ls, lw, param, unit, bit in (
            ("epv", "epv", "EPV (vert acc)", C_VERT, "-", 1.6, "EKF2_REQ_EPV", " m", 3),
            ("eph", "eph", "EPH (horiz acc)", C_HORZ, "-", 1.2, "EKF2_REQ_EPH", " m", 2),
            ("sacc", "s_variance_m_s", "speed acc", C_SPD, "-.", 1.1,
             "EKF2_REQ_SACC", " m/s", 4)):
        y = gfield(fname)
        if y is None:
            continue
        out.append(Series(f"gps_{key}", label, t, y, "gpsacc", color, ls=ls,
                          visible=True, lw=lw))
        lim = _limit_series(ulog, f"lim_{key}", param, unit, "gpsacc", color,
                            _enforced(mask, bit))
        if lim:
            out.append(lim)

    # Panel 6: satellite geometry.
    y = gfield("satellites_used")
    if y is not None:
        out.append(Series("gps_nsats", "satellites used", t, y, "gpsnsat",
                          C_NSAT, visible=True, lw=1.3, drawstyle="steps-post"))
        lim = _limit_series(ulog, "lim_nsats", "EKF2_REQ_NSATS", "", "gpsnsat",
                            C_NSAT, _enforced(mask, 0))
        if lim:
            out.append(lim)
    if "hdop" in g.data and "vdop" in g.data:
        pdop = gap_mask(t, np.hypot(np.asarray(g.data["hdop"], dtype=float),
                                    np.asarray(g.data["vdop"], dtype=float)), ok)
        out.append(Series("gps_pdop", "PDOP (hdop+vdop)", t, pdop, "gpspdop",
                          C_PDOP, ls="-", visible=True, lw=1.1, alpha=0.85))
        lim = _limit_series(ulog, "lim_pdop", "EKF2_REQ_PDOP", "", "gpspdop",
                            C_PDOP, _enforced(mask, 1))
        if lim:
            out.append(lim)

    # Panel 7: drift, which PX4 only evaluates while landed AND at rest
    # (gps_checks.cpp: in air the flags are forced false).  These are the gates
    # that hold a vehicle on the pad with "GPS not good enough" while the
    # accuracy numbers above look fine.
    s = _get(ulog, "estimator_gps_status", ekf)
    if s is None:
        ctx.note("no estimator_gps_status -- drift panel and gate band empty")
        return out
    ts = _time_min(ulog, s)
    for key, fname, label, color, ls in (
            ("vdrift", "position_drift_rate_vertical_m_s", "vert drift rate", C_VERT, "-"),
            ("hdrift", "position_drift_rate_horizontal_m_s", "horiz drift rate", C_HORZ, "-"),
            ("hspd", "filtered_horizontal_speed_m_s", "horiz speed (filt)", C_SPD, "-.")):
        if fname in s.data:
            out.append(Series(f"gps_{key}", label, ts,
                              np.asarray(s.data[fname], dtype=float), "gpsdrift",
                              color, ls=ls, visible=True, lw=1.2))
    for key, param, color, bits in (("vdrift", "EKF2_REQ_VDRIFT", C_VERT, (6, 8)),
                                    ("hdrift", "EKF2_REQ_HDRIFT", C_HORZ, (5, 7))):
        lim = _limit_series(ulog, f"lim_{key}", param, " m/s", "gpsdrift", color,
                            any(_enforced(mask, b) for b in bits))
        if lim:
            out.append(lim)
    return out


def _gate_rows(ulog, ekf, mask, ctx):
    """Panel 8: the pass/fail verdict per gate, as a fixed checklist.

    Every gate gets a row whether or not it fired, with a grey rail behind it,
    so an empty row reads "checked, clean" rather than "not shown".  Bar colour
    is the consequence: red = enforced, so this blocked GPS; grey = the flag
    fired but EKF2_GPS_CHECK has that bit off, so it blocked nothing.  The top
    row ORs the enforced ones -- the single answer to "was GPS being refused".
    """
    used = _gps_used_rows(ulog, ekf)
    s = _get(ulog, "estimator_gps_status", ekf)
    if s is None:
        return used
    t = _time_min(ulog, s)
    params = ulog.initial_parameters
    rows, blocking = [], np.zeros(t.size, bool)
    for fname, label, bit, param, unit in GPS_GATES:
        if fname not in s.data:
            continue
        v = np.asarray(s.data[fname], dtype=float) > 0.5
        enf = _enforced(mask, bit)
        if enf:
            blocking |= v
        lim = f" {float(params[param]):g}{unit}" if param in params else ""
        # Short on purpose: the labels sit at the left edge, over the boot
        # period -- which is exactly where gate failures cluster.  The section
        # heading spells out the mask.
        state = "" if enf else "  (off)"
        rows.append((f"{label}{lim}{state}", spans_from_bool(t, v),
                     C_BAD if enf else C_MUTED))
    head = [("ANY ENFORCED GATE FAILING", spans_from_bool(t, blocking), C_BAD)]
    if "checks_passed" in s.data:
        v = np.asarray(s.data["checks_passed"], dtype=float) > 0.5
        head.append(("checks passed (10 s clean)", spans_from_bool(t, v), C_GPS))
    return used + head + rows


def _gps_used_rows(ulog, ekf):
    """Top of panel 8: each raw GPS dropout, and whether the EKF was actually
    fusing GPS (height / horizontal) -- so cause and consequence share a time
    axis.  Dropouts come from vehicle_gps_position at its full ~10 Hz:
    estimator_gps_status is ~1 Hz and misses most single-sample dropouts.
    """
    rows = []
    g = _get(ulog, "vehicle_gps_position")
    if g is not None and "fix_type" in g.data:
        fix = np.asarray(g.data["fix_type"]) >= 3
        if fix.any():
            # Only after the first fix: the boot-time search is not a dropout.
            drop = ~fix & (np.arange(fix.size) > np.flatnonzero(fix)[0])
            n = int(np.sum(np.diff(drop.astype(int)) == 1))
            rows.append((f"GPS dropout (fix<3)  n={n}",
                         spans_from_bool(_time_min(ulog, g), drop), C_BAD))
    f = _get(ulog, "estimator_status_flags", ekf)
    if f is not None:
        tf = _time_min(ulog, f)
        for field, label in (("cs_gps_hgt", "EKF FUSING GPS HEIGHT"),
                             ("cs_gps", "EKF FUSING GPS HORIZONTAL")):
            if field in f.data:
                v = np.asarray(f.data[field], dtype=float) > 0.5
                rows.append((label, spans_from_bool(tf, v), C_GPS))
    return rows


def _rescale_with_limits(ax, lines, log=False, pct=99.5):
    """Fit the bulk of the data AND every visible limit line.

    The limits are the reference these panels exist to show, so they are never
    allowed off-screen; the data gets percentile bounds for the same reason as
    _rescale_robust (one 36 m EPV spike would otherwise flatten the rest).
    """
    data = [l for l in lines if not l.get_gid()]
    lims = [l for l in lines if l.get_gid() == "limit"]
    v = window_values(ax, data, positive_only=log)
    w = window_values(ax, lims, positive_only=log)
    if v.size == 0 and w.size == 0:
        return
    pts = []
    if v.size:
        pts += [float(np.percentile(v, 100.0 - pct)), float(np.percentile(v, pct))]
    if w.size:
        pts += [float(w.min()), float(w.max())]
    lo, hi = min(pts), max(pts)
    if log:
        ax.set_ylim(max(lo / 1.6, 1e-3), max(hi * 1.6, lo * 4))
    else:
        if hi == lo:
            lo, hi = lo - 1, hi + 1
        m = (hi - lo) * 0.10
        # Counts and rates are read from zero: "6 sats against a limit of 6"
        # and "16 against 6" must not look the same height.
        ax.set_ylim(0.0 if lo >= 0 else lo - m, hi + m)


def build_altitude(ulog, ctx=None, path=""):
    """The altitude estimation figure.  Same signature as every plot builder."""
    import matplotlib.pyplot as plt

    ctx = ctx or PlotCtx()
    ekf, switched = primary_ekf(ulog)
    if switched:
        ctx.note(f"the estimator selector switched instances during this log; "
                 f"showing the most-used one, EKF {ekf}")

    t_f, y_f, fused_src = _fused_amsl(ulog, ctx)
    t_g, y_g = _gps_amsl(ulog, ctx)
    t_b, y_b = field(ulog, "vehicle_air_data", "baro_alt_meter")
    t_r, y_r = _rng_hagl(ulog, ctx)
    t_h, y_h = field(ulog, "home_position", "alt")

    # The rangefinder measures HAGL, not altitude.  Putting it on an AMSL axis
    # needs a ground altitude, and the only one available is ref_alt -- the local
    # origin, i.e. the elevation where the vehicle booted.  That is a FLAT GROUND
    # assumption and it is labelled as one.  Note it does not contaminate panel 2:
    # (ref_alt + dist_bottom) - (ref_alt - z) = dist_bottom + z, so ref_alt
    # cancels out of the residual entirely.
    t_ra, y_ra = field(ulog, "vehicle_local_position", "ref_alt")
    if t_r.size and t_ra.size:
        y_r_amsl = y_r + resample_to(t_r, t_ra, y_ra)
    else:
        y_r_amsl = np.array([])

    series = []

    # --- panel 1: AMSL overlay ---------------------------------------------
    if y_f.size:
        # Short label: the checkbox panel is ~3 inches wide and doubles as the
        # legend, so a full topic path here pushes into the plot area.
        short = fused_src.replace("vehicle_", "").replace(".alt", "")
        series.append(Series("fused", f"fused ({short})", t_f, y_f, "amsl",
                             C_FUSED, visible=True, lw=2.2, zorder=5))
    if y_g.size:
        series.append(Series("gps", "GPS alt", t_g, y_g, "amsl", C_GPS,
                             visible=True, lw=1.4))
    if y_b.size:
        series.append(Series("baro", "baro alt", t_b, y_b, "amsl", C_BARO,
                             ls="--", visible=True, lw=1.4))
    if y_r_amsl.size:
        # OFF by default.  Where the flat-ground assumption fails it fails big:
        # on d05a88e3 the local origin sits ~50 m above the ground actually being
        # overflown, so this trace lands at 120-140 m against a 84-93 m flight and
        # single-handedly sets the axis range, squashing the sources you came to
        # compare.  The residual version in panel 2 has ref_alt cancelled out and
        # is the trustworthy one, so nothing is lost by making this opt-in.
        series.append(Series("rng", "rangefinder +ref_alt (flat gnd)", t_r,
                             y_r_amsl, "amsl", C_RNG, ls=":", visible=False,
                             lw=1.6))
        ctx.note("rangefinder-on-AMSL is off by default: it assumes flat ground "
                 "at ref_alt, which can be far from the real ground; use the "
                 "residual panel instead")
    if y_h.size:
        series.append(Series("home", "home alt", t_h, y_h, "amsl", C_HOME,
                             ls="-.", lw=1.2, drawstyle="steps-post"))

    # --- panel 2: residual vs fused ----------------------------------------
    offsets = {}
    if y_f.size:
        for key, label, t, y, color, ls in (
                ("res_gps", "GPS", t_g, y_g, C_GPS, "-"),
                ("res_baro", "baro", t_b, y_b, C_BARO, "--"),
        ):
            if not y.size:
                continue
            r, off = _residual(t, y, t_f, y_f, ctx.debias)
            offsets[key] = off
            tag = f" ({off:+.1f})" if ctx.debias and abs(off) > 0.05 else ""
            series.append(Series(key, f"{label} - fused{tag}", t, r, "resid",
                                 color, ls=ls, visible=True, lw=1.4))
        if y_r.size:
            # ref_alt cancels here (see above), so this residual is
            # dist_bottom - (fused - ref_alt) with no flat-ground assumption.
            r, off = _residual(t_r, y_r_amsl, t_f, y_f, ctx.debias) \
                if y_r_amsl.size else (np.array([]), 0.0)
            if r.size:
                offsets["res_rng"] = off
                tag = f" ({off:+.1f})" if ctx.debias and abs(off) > 0.05 else ""
                series.append(Series("res_rng", f"rng - fused{tag}", t_r,
                                     r, "resid", C_RNG, ls=":", visible=True,
                                     lw=1.6))

    # PX4's OWN estimate of these offsets, for cross-check.  Where the learned
    # bias and the measured residual disagree, the EKF is not tracking an offset
    # that is really there -- which is a different and worse problem than drift.
    for topic, label, ls in (("estimator_gnss_hgt_bias", "GNSS hgt bias (PX4)", "-"),
                             ("estimator_rng_hgt_bias", "rng hgt bias (PX4)", ":")):
        t, y = field(ulog, topic, "bias", mid=ekf)
        if y.size:
            series.append(Series(topic, label, t, y, "resid", C_BIAS, ls=ls,
                                 lw=1.2, alpha=0.9))
        else:
            ctx.note(f"{topic} absent -- PX4's learned offset not available")

    # --- panel 3: innovations and test ratios -------------------------------
    # An exact 0.0 in these topics does not mean "the measurement agreed
    # perfectly" -- it means PX4 did not run that fusion step this cycle and left
    # the field at its initialiser.  Drawn as-is they are long flat runs at zero
    # that look like a well-behaved sensor, which is the opposite of the truth.
    # Blanking them turns "fused and agreed" and "never fused" back into two
    # visually different things.
    innov_specs = (("baro_vpos", "baro", C_BARO, "--", True),
                   ("gps_vpos", "GPS", C_GPS, "-", True),
                   # The rangefinder innovation reaches 130 m on d05a88e3 while
                   # baro/GPS sit under a metre; on by default it would flatten
                   # the two series you normally read.  Opt-in, like its AMSL twin.
                   ("rng_vpos", "rangefinder", C_RNG, ":", False))
    for fname, label, color, ls, vis in innov_specs:
        t, y = field(ulog, "estimator_innovations", fname, mid=ekf)
        if y.size and np.any(y != 0):
            series.append(Series(f"innov_{fname}", f"{label} innov", t,
                                 _blank_unfused(y), "innov", color, ls=ls,
                                 visible=vis, lw=1.3))
    for fname, label, color, ls, vis in innov_specs:
        t, y = field(ulog, "estimator_innovation_test_ratios", fname, mid=ekf)
        if y.size and np.any(y != 0):
            series.append(Series(f"ratio_{fname}", f"{label} ratio", t,
                                 _blank_unfused(y), "ratio", color, ls=ls,
                                 visible=True, lw=1.1, alpha=0.75))

    # --- panels 5-8: GPS fusion gates ---------------------------------------
    gps_mask = _gps_mask(ulog, ctx)
    series += _gps_quality_series(ulog, ekf, gps_mask, ctx)
    gate_rows = _gate_rows(ulog, ekf, gps_mask, ctx)

    if not series:
        ctx.note("no altitude topics in this log -- nothing to plot")
        return None

    # --- figure -------------------------------------------------------------
    # Laid out in INCHES, bottom-up, and divided by the figure height at the
    # end (the ulog_accel convention): the gate band's height follows its row
    # count, and hand-written figure fractions would squash every other panel
    # whenever it grew.  The first four panels keep the inches they had when
    # this was a fixed 10 in figure.
    band_in = max(1.5, GATE_ROW_IN * len(gate_rows) + 0.25)
    stack = [("amsl", 1.85), ("res", 1.85), ("inn", 1.70), ("band", 1.50),
             ("gacc", 1.80), ("gsat", 1.45), ("gdrift", 1.45),
             ("gate", band_in)]
    gap_above = {"gacc": SECTION_GAP_IN}       # room for the GPS section heading
    fig_h = TOP_IN + sum(h for _k, h in stack) + BOTTOM_IN + sum(
        gap_above.get(k, GAP_IN) for k, _h in stack[1:])
    fig = plt.figure(figsize=(15, fig_h), facecolor=C_SURFACE)
    if fig.canvas.manager is not None:
        import os
        fig.canvas.manager.set_window_title(
            f"logGraph altitude - {os.path.basename(path)}")

    def _f(inches):
        return inches / fig_h

    # The gap between the checkbox panel's right edge (0.167) and `left` has to
    # hold BOTH the tick labels and the axis label.  The thermal plot gets away
    # with a 0.02 gap only because it stacks its y-axes on the right; here every
    # panel labels on the left, so the margin is nearly five times wider.
    left, width = 0.260, 0.655
    rects, bottom = {}, BOTTOM_IN
    for key, h in reversed(stack):
        rects[key] = (_f(bottom), _f(h))
        bottom += h + gap_above.get(key, GAP_IN)
    ax = {k: fig.add_axes([left, rects[k][0], width, rects[k][1]],
                          facecolor=C_SURFACE) for k, _h in stack}
    ax_amsl, ax_res, ax_inn, ax_band = ax["amsl"], ax["res"], ax["inn"], ax["band"]
    ax_gacc, ax_gsat, ax_gdrift, ax_gate = (ax["gacc"], ax["gsat"],
                                            ax["gdrift"], ax["gate"])
    for a in ax.values():
        if a is not ax_gate:
            a.sharex(ax_gate)
    ax_ratio = ax_inn.twinx()
    ax_ratio.set_facecolor("none")
    ax_pdop = ax_gsat.twinx()
    ax_pdop.set_facecolor("none")
    ax_gacc.set_yscale("log")

    axis_of = {"amsl": ax_amsl, "resid": ax_res, "innov": ax_inn,
               "ratio": ax_ratio, "gpsacc": ax_gacc, "gpsnsat": ax_gsat,
               "gpspdop": ax_pdop, "gpsdrift": ax_gdrift}

    spans = armed_spans(ulog)
    span_art = []
    for a in ax.values():
        span_art += draw_armed(a, spans)

    # Zero lines: on the residual and innovation panels, zero is "the sensor and
    # the filter agree", which is the only value either panel is really asking about.
    for a in (ax_res, ax_inn):
        a.axhline(0.0, color=C_MUTED, lw=1, ls=":", alpha=0.6, zorder=1)
    # The test-ratio rejection threshold.  PX4 rejects a measurement when its
    # normalised innovation exceeds 1.0, so this line is the accept/reject border.
    #
    # LOG scale, because the ratios span 0.001 to 2500 on a real log.  Linear, the
    # 1.0 threshold sits indistinguishably on the zero line and the whole panel
    # becomes "one series spikes, the rest are flat" -- which loses exactly the
    # information the panel exists for, namely how close to rejection the healthy
    # sources ran.  On a log axis 1.0 is a real midline with decades either side.
    ax_ratio.set_yscale("log")
    ax_ratio.axhline(1.0, color=C_BAD, lw=1.1, ls="--", alpha=0.8, zorder=1)
    ax_ratio.text(0.012, 1.0, "reject > 1.0",
                  transform=ax_ratio.get_yaxis_transform(),
                  color=C_BAD, fontsize=7, va="bottom", ha="left")

    for s in series:
        (line,) = axis_of[s.group].plot(
            s.t, s.y, color=s.color, ls=s.ls, lw=s.lw, label=s.label,
            drawstyle=s.drawstyle, alpha=s.alpha,
            zorder=s.zorder if s.zorder is not None else 3)
        line.set_visible(s.visible)
        if s.id.startswith("lim_"):
            line.set_gid("limit")      # _rescale_with_limits keeps these on-screen
        s.line = line

    _draw_band(ax_band, ulog, ekf, ctx)
    # min_width: a gate that fails for one 5 Hz GPS sample is a real event,
    # and at full-log zoom it would otherwise be narrower than a pixel.
    draw_band_rows(ax_gate, gate_rows, ylabel="GPS gates",
                   empty_msg="no estimator_gps_status in this log",
                   min_width=max(duration_min(ulog), 1.0) * 0.002, track=True)

    # --- axis furniture -----------------------------------------------------
    for a in ax.values():
        style_time_axis(a, label=a is ax_gate)

    ax_amsl.set_ylabel("altitude AMSL (m)", fontsize=9)
    ax_res.set_ylabel("residual vs fused (m)", fontsize=9)
    ax_inn.set_ylabel("innovation (m)", fontsize=9)
    ax_ratio.set_ylabel("test ratio", fontsize=9)
    ax_gacc.set_ylabel("GPS accuracy\n(m, m/s; log)", fontsize=9)
    ax_gsat.set_ylabel("satellites used", fontsize=9)
    ax_pdop.set_ylabel("PDOP", fontsize=9)
    ax_gdrift.set_ylabel("drift on ground\n(m/s)", fontsize=9)
    for a in (ax_amsl, ax_res, ax_inn, ax_gacc, ax_gsat, ax_gdrift):
        _style_axis(a, C_INK)
    _style_axis(ax_ratio, C_MUTED)
    _style_axis(ax_pdop, C_PDOP)

    # A section heading over the GPS panels: from here down the question changes
    # from "what height does it think" to "was it ALLOWED to use GPS".
    on = [n for b, n in GATE_SHORT.items() if _enforced(gps_mask, b)]
    off = [n for b, n in GATE_SHORT.items() if not _enforced(gps_mask, b)]
    head = (f"GPS fusion gates  —  EKF2_GPS_CHECK = {gps_mask}:  enforced "
            f"fix, {', '.join(on) or '-'}   |   OFF {', '.join(off) or '-'}")
    acc_top = rects["gacc"][0] + rects["gacc"][1]
    fig.text(left, acc_top + _f(0.12),
             head, color=C_INK, fontsize=10,
             fontweight="bold", ha="left")

    fig.text(left, 1.0 - _f(0.45), "Altitude estimation", color=C_INK,
             fontsize=13, fontweight="bold", ha="left")
    import os as _os
    who = f"{_os.path.basename(path)}   |   " if path else ""
    debias_note = ("residuals de-biased (constant offset removed, stated in the "
                   "label)" if ctx.debias else "residuals raw")
    fig.text(left, 1.0 - _f(0.75),
             f"{who}{duration_min(ulog):.1f} min   |   EKF instance {ekf}   |   "
             f"{debias_note}",
             color=C_MUTED, fontsize=9, ha="left")

    # Off-scale counters, redrawn on every toggle.  Kept as figure text rather
    # than axis text so they never get wiped by an axis rescale.
    off_note = ax_inn.text(0.995, 0.96, "", transform=ax_inn.transAxes,
                           ha="right", va="top", fontsize=7, color=C_BAD)

    def refresh():
        for group, a in (("amsl", ax_amsl), ("resid", ax_res)):
            _rescale(a, [s.line for s in series if s.group == group])
        n_i, max_i = _rescale_robust(
            ax_inn, [s.line for s in series if s.group == "innov"], min_span=2.0)
        n_r, max_r = _rescale_log(
            ax_ratio, [s.line for s in series if s.group == "ratio"])
        bits = []
        if n_i:
            bits.append(f"{n_i} innovation sample(s) off-scale (max |{max_i:.0f}| m)")
        if n_r:
            bits.append(f"{n_r} test ratio(s) off-scale (max {max_r:.0f})")
        off_note.set_text("   ".join(bits))
        _rescale_with_limits(ax_gacc, [s.line for s in series
                                       if s.group == "gpsacc"], log=True)
        for group, a in (("gpsnsat", ax_gsat), ("gpspdop", ax_pdop),
                         ("gpsdrift", ax_gdrift)):
            _rescale_with_limits(a, [s.line for s in series if s.group == group])

    base_extra = [("armed (shaded)", span_art, True)] if span_art else []
    # Flight-mode overlay: a rule on every panel at each mode change, named on
    # one of them.  Toggleable, because a log that flickers between Position and
    # Hold 52 times (d05a88e3) is unreadable with it on and unanswerable with it
    # off.  min_gap keeps the LABELS legible without dropping any rule.
    mode_art, mode_codes = draw_mode_changes(
        list(ax.values()) + [ax_ratio, ax_pdop], mode_changes(ulog),
        text_ax=ax_res, min_gap=max(duration_min(ulog), 1.0) * 0.035)
    extra = list(base_extra)
    mode_art += mode_key(fig, left + width, _f(0.18), mode_codes)
    if mode_art:
        extra.append(("mode changes", mode_art, True))

    # The checkbox panel spans the whole stack and each group's block is
    # anchored beside its own panel -- with eight panels, "which legend goes
    # with which graph" has to be answered by position, not by reading.
    cb_top = rects["amsl"][0] + rects["amsl"][1]
    cb_bot = rects["gate"][0]
    cb_h = cb_top - cb_bot

    def _anchor(key):
        b, ph = rects[key]
        return (b + ph / 2 - cb_bot) / cb_h

    check_panel(fig, [0.012, cb_bot, 0.155, cb_h], series,
                [("amsl", "ALL sources"), ("resid", "ALL residuals"),
                 ("innov", "ALL innovations"), ("ratio", "ALL test ratios"),
                 ("gpsacc", "ALL GPS accuracy"), ("gpsnsat", "ALL satellites"),
                 ("gpspdop", "ALL PDOP"), ("gpsdrift", "ALL drift")],
                extra=extra, on_change=refresh,
                anchors={"amsl": _anchor("amsl"), "resid": _anchor("res"),
                         "innov": _anchor("inn"), "ratio": _anchor("inn"),
                         "gpsacc": _anchor("gacc"), "gpsnsat": _anchor("gsat"),
                         "gpspdop": _anchor("gsat"),
                         "gpsdrift": _anchor("gdrift")})
    refresh()
    add_mouse_navigation(fig, list(ax.values()) + [ax_ratio, ax_pdop],
                         page_scroll=ctx.page_scroll,
                         fixed_y=[ax_band, ax_gate], on_view=refresh)
    fig.text(left, _f(0.40), nav_hint(ctx.page_scroll), color=C_MUTED,
             fontsize=8, ha="left")
    fig._page_height = int(round(fig_h * PAGE_PX_PER_IN))
    return fig


# --- panel 4 ----------------------------------------------------------------

def _draw_band(ax, ulog, ekf, ctx):
    """Which height source the EKF was fusing, and when each input was usable.

    This is the panel that turns "baro drifted" into "the EKF stopped fusing baro
    at 12 min", which is a different sentence with a different fix.  Drawn as
    filled bars rather than lines because every row is boolean -- a line would
    imply intermediate values that do not exist.
    """
    rows = []          # (label, spans, color)

    fl = _get(ulog, "estimator_status_flags", ekf)
    if fl is not None:
        t = _time_min(ulog, fl)
        for fname, label, color in (("cs_baro_hgt", "fusing baro", C_BARO),
                                    ("cs_gps_hgt", "fusing GPS", C_GPS),
                                    ("cs_rng_hgt", "fusing rangefinder", C_RNG),
                                    ("cs_ev_hgt", "fusing vision", C_HOME)):
            if fname not in fl.data:
                continue
            v = np.asarray(fl.data[fname], dtype=float) > 0.5
            if v.any():
                rows.append((label, spans_from_bool(t, v), color))
        # Faults and rejections share the one red row set: they are all "this
        # went wrong", and separating them by hue would spend the palette's most
        # attention-grabbing color on distinctions you read from the label.
        for fname, label in (("cs_rng_fault", "rng FAULT"),
                             ("cs_rng_stuck", "rng STUCK"),
                             ("reject_ver_pos", "reject vert pos"),
                             ("reject_hagl", "reject HAGL")):
            if fname not in fl.data:
                continue
            v = np.asarray(fl.data[fname], dtype=float) > 0.5
            if v.any():
                rows.append((label, spans_from_bool(t, v), C_BAD))
    else:
        ctx.note("no estimator_status_flags -- cannot show which height source "
                 "was being fused")

    # Input validity, below the fusion rows: "the EKF was not using the
    # rangefinder" and "the rangefinder had nothing to say" look identical on the
    # traces above and are completely different problems.
    d = _get(ulog, "vehicle_local_position")
    if d is not None and "dist_bottom_valid" in d.data:
        t = _time_min(ulog, d)
        v = np.asarray(d.data["dist_bottom_valid"], dtype=float) > 0.5
        rows.append(("dist_bottom valid", spans_from_bool(t, v), C_RNG))
    ds = _get(ulog, "distance_sensor")
    if ds is not None and "signal_quality" in ds.data:
        t = _time_min(ulog, ds)
        v = np.asarray(ds.data["signal_quality"], dtype=float) > 0
        rows.append(("rng signal quality > 0", spans_from_bool(t, v), C_RNG))
    g = _get(ulog, "vehicle_gps_position") or _get(ulog, "sensor_gps")
    if g is not None and "fix_type" in g.data:
        t = _time_min(ulog, g)
        v = np.asarray(g.data["fix_type"], dtype=float) >= MIN_FIX
        rows.append((f"GPS fix >= {MIN_FIX}", spans_from_bool(t, v), C_GPS))

    draw_band_rows(ax, rows, ylabel="source / validity",
                   empty_msg="no fusion-source or validity flags in this log")
