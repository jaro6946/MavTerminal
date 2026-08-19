#!/usr/bin/env python3
"""ulog_alt.py -- the altitude estimation plot.

Four stacked panels on one time axis, answering "why does the vehicle think it is
at that height?":

  1. AMSL overlay   -- every height source as logged, plus the EKF's fused answer
  2. residual       -- each source MINUS the fused answer, which is where drift lives
  3. innovations    -- what the EKF itself thought of each measurement
  4. source band    -- which source it was actually fusing, and when each was valid

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
                         Series, _clean, _get, _rescale, _style_axis, _time_min,
                         add_mouse_navigation, armed_spans, check_panel,
                         draw_armed, draw_band_rows, duration_min, field,
                         gap_mask, has_topic, nav_hint, primary_ekf,
                         resample_to, spans_from_bool, style_time_axis)

ALT_TOPICS = [
    "vehicle_global_position", "vehicle_local_position", "vehicle_gps_position",
    "vehicle_air_data", "distance_sensor", "home_position",
    "estimator_status_flags", "estimator_innovations",
    "estimator_innovation_test_ratios", "estimator_gnss_hgt_bias",
    "estimator_rng_hgt_bias", "estimator_selector_status", "actuator_armed",
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


def _visible_values(lines, positive_only=False):
    out = []
    for ln in lines:
        if not ln.get_visible():
            continue
        v = np.asarray(ln.get_ydata(), dtype=float)
        v = v[np.isfinite(v)]
        if positive_only:
            v = v[v > 0]
        if v.size:
            out.append(v)
    return np.concatenate(out) if out else np.array([])


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
    v = _visible_values(lines)
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
    v = _visible_values(lines, positive_only=True)
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

    if not series:
        ctx.note("no altitude topics in this log -- nothing to plot")
        return None

    # --- figure -------------------------------------------------------------
    fig = plt.figure(figsize=(15, 10), facecolor=C_SURFACE)
    if fig.canvas.manager is not None:
        import os
        fig.canvas.manager.set_window_title(
            f"logGraph altitude - {os.path.basename(path)}")

    # Heights: the overlay and the residual carry the argument, so they get the
    # room; innovations are a supporting check; the band is categorical and needs
    # only enough height to distinguish its rows.
    # The gap between the checkbox panel's right edge (0.167) and `left` has to
    # hold BOTH the tick labels and the axis label.  The thermal plot gets away
    # with a 0.02 gap only because it stacks its y-axes on the right; here every
    # panel labels on the left, so the margin is nearly five times wider.
    left, width = 0.260, 0.655
    rects = [(0.735, 0.185), (0.520, 0.185), (0.320, 0.170), (0.130, 0.150)]
    ax_amsl, ax_res, ax_inn, ax_band = [
        fig.add_axes([left, b, width, h], facecolor=C_SURFACE) for b, h in rects]
    for a in (ax_amsl, ax_res, ax_inn):
        a.sharex(ax_band)
    ax_ratio = ax_inn.twinx()
    ax_ratio.set_facecolor("none")

    axis_of = {"amsl": ax_amsl, "resid": ax_res, "innov": ax_inn,
               "ratio": ax_ratio}

    spans = armed_spans(ulog)
    span_art = []
    for a in (ax_amsl, ax_res, ax_inn, ax_band):
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
        s.line = line

    _draw_band(ax_band, ulog, ekf, ctx)

    # --- axis furniture -----------------------------------------------------
    for a in (ax_amsl, ax_res, ax_inn):
        style_time_axis(a, label=False)
        a.tick_params(axis="x", labelbottom=False)
    style_time_axis(ax_band)

    ax_amsl.set_ylabel("altitude AMSL (m)", fontsize=9)
    ax_res.set_ylabel("residual vs fused (m)", fontsize=9)
    ax_inn.set_ylabel("innovation (m)", fontsize=9)
    ax_ratio.set_ylabel("test ratio", fontsize=9)
    _style_axis(ax_amsl, C_INK)
    _style_axis(ax_res, C_INK)
    _style_axis(ax_inn, C_INK)
    _style_axis(ax_ratio, C_MUTED)

    fig.text(left, 0.955, "Altitude estimation", color=C_INK, fontsize=13,
             fontweight="bold", ha="left")
    import os as _os
    who = f"{_os.path.basename(path)}   |   " if path else ""
    debias_note = ("residuals de-biased (constant offset removed, stated in the "
                   "label)" if ctx.debias else "residuals raw")
    fig.text(left, 0.925,
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

    h = min(0.86, 0.035 * (len(series) + 5) + 0.05)
    check_panel(fig, [0.012, 0.89 - h, 0.155, h], series,
                [("amsl", "ALL sources"), ("resid", "ALL residuals"),
                 ("innov", "ALL innovations"), ("ratio", "ALL test ratios")],
                extra=[("armed (shaded)", span_art, True)] if span_art else (),
                on_change=refresh)
    refresh()
    add_mouse_navigation(fig, [ax_amsl, ax_res, ax_inn, ax_ratio, ax_band],
                         page_scroll=ctx.page_scroll)
    fig.text(left, 0.03, nav_hint(ctx.page_scroll), color=C_MUTED, fontsize=8,
             ha="left")
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
