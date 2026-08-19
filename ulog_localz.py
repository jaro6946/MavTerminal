#!/usr/bin/env python3
"""ulog_localz.py -- local position z, shaded by which EKF instance was primary.

Four stacked panels on one time axis, answering "whose z am I looking at, and did
the frame move under me?":

  1. height above local origin -- the published z, plus each instance's own z
  2. ref_alt                   -- where each instance put its local origin
  3. selector scores           -- why the primary changed
  4. band                      -- armed spans and per-instance health

The whole figure is shaded by `estimator_selector_status.primary_instance`, so
every panel is read against the same coloured background: which filter was
actually steering the vehicle at that moment.

Why this plot exists
--------------------
`vehicle_local_position` publishes ONE instance -- whichever the selector had
chosen -- and PX4 runs one EKF (Extended Kalman Filter) per IMU (Inertial
Measurement Unit), typically three.  Each instance anchors its OWN local origin,
so a handover republishes z against a different datum.  Measured on
SquareWaypointMission_1.ulg: the instances' origins sat a permanent 30.25 m apart
(`ref_alt` 10.28 / 10.35 / 40.53), and `vehicle_local_position.z` stepped by up to
89 m at a handover while the vehicle did not move at all.

Read from `vehicle_local_position` alone, a step like that is ambiguous between
"the filter reset its height" and "a different filter started publishing".  Those
have different causes and different fixes, and only the per-instance topic
separates them -- hence panel 1 overlays `estimator_local_position[i].z` for every
instance, not just the published one.

Sign convention: PX4's z is NED (North-East-Down), so z is DOWN-positive and a
climbing vehicle has a falling z.  Panel 1 plots -z, labelled "height above local
origin", because an altitude that goes down when you go up is a reliable way to
misread a plot.  It is NOT height above ground and NOT height above takeoff: it is
measured from that instance's origin, which the ref_alt panel below shows moving.

Acronyms: EKF = extended Kalman filter, IMU = inertial measurement unit,
NED = North-East-Down, AMSL = above mean sea level.
"""
import os

import numpy as np

from ulog_common import (C_ARMED, C_GRID, C_INK, C_MUTED, C_SURFACE, PlotCtx,
                         Series, _clean, _get, _rescale, _style_axis, _time_min,
                         add_mouse_navigation, armed_spans, check_panel,
                         draw_armed, duration_min, field, has_topic, nav_hint,
                         style_time_axis)

LOCALZ_TOPICS = [
    "vehicle_local_position", "estimator_local_position",
    "estimator_selector_status", "actuator_armed",
]

# --- color ------------------------------------------------------------------
# One hue per INSTANCE, held constant across all four panels and reused for the
# background shading.  That is the whole mechanic of this plot: the shading under
# a stretch of trace is the same colour as the line belonging to the filter that
# produced it, so "who was flying" needs no legend lookup.
#
# Deliberately NOT the altitude plot's per-SOURCE palette -- these are instances,
# not sensors, and giving instance 0 the same blue as GPS would invite reading the
# two plots as if the colours meant the same thing.
INST_COLORS = ["#2a78d6",   # 0  blue
               "#d2691e",   # 1  orange
               "#1baf7a",   # 2  aqua
               "#8a7fb5"]   # 3  violet
C_PUB = "#20222b"           # near-black -- the PUBLISHED series, everyone's reference
C_BAD = "#c0392b"           # red -- faults and rejections only

# PX4 rejects a measurement when its normalised innovation exceeds 1.0, and the
# selector calls an instance "warning" at the same threshold (EKF2Selector.cpp:301).
WARN_RATIO = 1.0

# Height resets below this are marked with a rule but not labelled -- see the
# annotation loop in build_local_z for why.
LABEL_RESET_M = 1.0


def _inst_color(i):
    return INST_COLORS[i % len(INST_COLORS)]


def _available_instances(ulog):
    """Which estimator_local_position multi-ids this log actually carries.

    Returned sorted.  Empty is a normal outcome, not an error: HITL logs pulled
    off the flight controller carry only the fused `vehicle_local_position`."""
    return sorted({d.multi_id for d in ulog.data_list
                   if d.name == "estimator_local_position"})


def primary_spans(ulog):
    """[(t0_min, t1_min, instance), ...] -- who was primary, when.

    The selector publishes at ~1 Hz AND immediately on any change
    (EKF2Selector.cpp:811), so a value holds from its own sample until the next
    one that differs.  Treating each sample as an instantaneous point instead
    would leave 1 s gaps in the shading that look like the vehicle had no
    estimator at all.
    """
    d = _get(ulog, "estimator_selector_status")
    if d is None or "primary_instance" not in d.data:
        return []
    t = _time_min(ulog, d)
    v = np.asarray(d.data["primary_instance"], dtype=float)
    m = np.isfinite(t) & np.isfinite(v)
    t, v = t[m], v[m].astype(int)
    if t.size == 0:
        return []
    spans, start, cur = [], t[0], v[0]
    for i in range(1, t.size):
        if v[i] != cur:
            spans.append((start, t[i], int(cur)))
            start, cur = t[i], v[i]
    spans.append((start, t[-1], int(cur)))
    return spans


def draw_primary_shading(ax, spans, alpha=0.13):
    """The coloured background.  Returns the artists so a checkbox can hide them.

    zorder 0 and no edge line: this is a background, and an edge on abutting
    spans draws a vertical rule at every handover that reads as a data event.
    """
    return [ax.axvspan(a, b, color=_inst_color(inst), alpha=alpha, lw=0, zorder=0)
            for a, b, inst in spans]


def _reset_events(ulog):
    """[(t_min, delta_z, old_counter, new_counter)] from the PUBLISHED position.

    `z_reset_counter` increments whenever the estimator resets its height state,
    and `delta_z` carries the jump so a consumer can compensate.  Marking these
    is what lets you tell a reset apart from a handover: a handover moves the
    shading, a reset bumps the counter, and the ugly ones do both at once.
    """
    d = _get(ulog, "vehicle_local_position")
    if d is None or "z_reset_counter" not in d.data:
        return []
    t = _time_min(ulog, d)
    c = np.asarray(d.data["z_reset_counter"], dtype=float)
    dz = np.asarray(d.data.get("delta_z", np.zeros_like(c)), dtype=float)
    out = []
    for i in np.where(np.diff(c) != 0)[0] + 1:
        out.append((float(t[i]), float(dz[i]), int(c[i - 1]), int(c[i])))
    return out


def _series_for_instances(ulog, ctx, instances):
    """Panel 1 and 2 series: published first, then one pair per instance."""
    series = []

    # The published series is the one every other tool shows, so it is drawn
    # thickest and in near-black -- the reference the instance lines are compared
    # against, not a peer of them.
    t, z = field(ulog, "vehicle_local_position", "z")
    if z.size:
        series.append(Series("vehicle_local_position.z", "published (primary)",
                             t, -z, "z", C_PUB, lw=2.4, visible=True, zorder=5))
    t, ra = field(ulog, "vehicle_local_position", "ref_alt")
    if ra.size:
        series.append(Series("vehicle_local_position.ref_alt", "published ref_alt",
                             t, ra, "ref", C_PUB, lw=2.4, visible=True,
                             drawstyle="steps-post", zorder=5))

    for i in instances:
        d = _get(ulog, "estimator_local_position", i)
        if d is None:
            continue
        col = _inst_color(i)
        if "z" in d.data:
            t, z = _clean(_time_min(ulog, d), d.data["z"])
            series.append(Series(f"estimator_local_position[{i}].z",
                                 f"EKF {i} z", t, -z, "z", col, lw=1.4,
                                 visible=True, alpha=0.9))
        if "ref_alt" in d.data:
            t, ra = _clean(_time_min(ulog, d), d.data["ref_alt"])
            series.append(Series(f"estimator_local_position[{i}].ref_alt",
                                 f"EKF {i} ref_alt", t, ra, "ref", col, lw=1.4,
                                 visible=True, drawstyle="steps-post", alpha=0.9))

    if not instances:
        ctx.note("no estimator_local_position in this log -- showing only the "
                 "published vehicle_local_position, so a step in z cannot be "
                 "attributed to a reset vs. a handover")
    return series


def _series_for_selector(ulog, ctx, instances):
    """Panel 3: the scores the selector actually decides on.

    combined_test_ratio is max(0.5*(vel+pos), hgt) per instance
    (EKF2Selector.cpp:294) -- note the max, which lets HEIGHT alone drive a
    handover.  relative_test_ratio is the accumulated advantage over the current
    primary; reaching -EKF2_SEL_ERR_RED is what makes an instance a candidate.
    """
    d = _get(ulog, "estimator_selector_status")
    if d is None:
        return []
    t = _time_min(ulog, d)
    series = []
    for i in instances or range(4):
        col = _inst_color(i)
        key = f"combined_test_ratio[{i}]"
        if key in d.data:
            tt, y = _clean(t, d.data[key])
            if y.size and np.nanmax(y) > 0:
                series.append(Series(key, f"EKF {i} combined ratio", tt, y,
                                     "sel", col, lw=1.4, visible=True))
        key = f"relative_test_ratio[{i}]"
        if key in d.data:
            tt, y = _clean(t, d.data[key])
            if y.size:
                series.append(Series(key, f"EKF {i} relative", tt, y,
                                     "rel", col, lw=1.2, ls="--", alpha=0.85))
    return series


def _draw_band(ax, ulog, ctx, instances, spans):
    """Armed spans and per-instance health, as filled bars.

    Boolean rows get bars rather than lines for the same reason as the altitude
    plot's band: a line between 0 and 1 implies intermediate values that do not
    exist.  Armed lives HERE rather than as a second full-height shading, because
    two overlapping translucent fills turn the instance colours to mud -- and the
    instance shading is the whole point of this figure.
    """
    rows = []          # (label, spans, color)

    a = armed_spans(ulog)
    if a:
        rows.append(("armed", a, C_ARMED))

    d = _get(ulog, "estimator_selector_status")
    if d is not None:
        t = _time_min(ulog, d)
        for i in instances or range(4):
            key = f"healthy[{i}]"
            if key not in d.data:
                continue
            v = np.asarray(d.data[key], dtype=float) > 0.5
            if v.any():
                rows.append((f"EKF {i} healthy", _spans_from_bool(t, v),
                             _inst_color(i)))
        # Faults share the red row: they are all "this went wrong", and spending
        # the palette's loudest colour on telling them apart buys nothing you
        # cannot read off the label.
        for key, label in (("accel_fault_detected", "accel FAULT"),
                           ("gyro_fault_detected", "gyro FAULT")):
            if key not in d.data:
                continue
            v = np.asarray(d.data[key], dtype=float) > 0.5
            if v.any():
                rows.append((label, _spans_from_bool(t, v), C_BAD))
    else:
        ctx.note("no estimator_selector_status -- no instance shading is "
                 "possible (single-EKF or HITL log)")

    if not rows:
        ax.text(0.5, 0.5, "no selector or armed flags in this log",
                transform=ax.transAxes, ha="center", va="center",
                color=C_MUTED, fontsize=9)
        ax.set_yticks([])
        return

    for i, (label, sp, color) in enumerate(rows):
        y = len(rows) - 1 - i
        for a0, b0 in sp:
            ax.barh(y, max(b0 - a0, 1e-6), left=a0, height=0.62, color=color,
                    alpha=0.85, lw=0, zorder=3)
        # Labels inside the axes, in axes coordinates: as y-tick labels these
        # names extend left into the checkbox panel and get clipped, and in data
        # coordinates they would slide away on a time zoom.
        ax.text(0.004, y, label, transform=ax.get_yaxis_transform(which="grid"),
                fontsize=7, color=C_MUTED, va="center", ha="left", zorder=5,
                bbox=dict(facecolor=C_SURFACE, edgecolor="none", pad=1.0,
                          alpha=0.75))
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_ylabel("armed / health", fontsize=9, color=C_MUTED)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def _spans_from_bool(t, ok):
    """[(t0, t1), ...] for each contiguous true run of `ok`."""
    ok = np.asarray(ok, dtype=bool)
    out, start = [], None
    for i in range(ok.size):
        if ok[i] and start is None:
            start = t[i]
        elif not ok[i] and start is not None:
            out.append((start, t[i]))
            start = None
    if start is not None and t.size:
        out.append((start, t[-1]))
    return out


def _rescale_log(ax, lines, pct=99.5):
    """Percentile-limited log rescale for the test-ratio axis.

    The ratios span three decades and spike past 25 at a height reset, so min/max
    limits squash the 0.01-1.0 range everything interesting lives in.  Returns
    (n_offscale, max_value) so the caller can say what was cut."""
    vals = [np.asarray(ln.get_ydata(), dtype=float)
            for ln in lines if ln.get_visible()]
    vals = [v[np.isfinite(v) & (v > 0)] for v in vals]
    vals = [v for v in vals if v.size]
    if not vals:
        return 0, 0.0
    allv = np.concatenate(vals)
    hi = float(np.percentile(allv, pct))
    lo = float(np.percentile(allv, 100.0 - pct))
    hi = max(hi, WARN_RATIO * 2.0)
    lo = min(max(lo, 1e-4), WARN_RATIO / 10.0)
    ax.set_ylim(lo, hi)
    n = int((allv > hi).sum())
    return n, float(allv.max())


def build_local_z(ulog, ctx=None, path=""):
    """The local-z figure.  Same signature as every plot builder."""
    import matplotlib.pyplot as plt

    ctx = ctx or PlotCtx()

    if not has_topic(ulog, "vehicle_local_position"):
        ctx.note("no vehicle_local_position in this log -- nothing to plot")
        return None

    instances = _available_instances(ulog)
    spans = primary_spans(ulog)

    series = _series_for_instances(ulog, ctx, instances)
    series += _series_for_selector(ulog, ctx, instances)
    if not series:
        ctx.note("no local position fields in this log -- nothing to plot")
        return None

    # --- figure -------------------------------------------------------------
    fig = plt.figure(figsize=(15, 10), facecolor=C_SURFACE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(
            f"logGraph local z - {os.path.basename(path)}")

    # Panel 1 carries the argument and gets the room; ref_alt is a step function
    # that needs only enough height to read its levels; the selector scores are a
    # supporting explanation; the band is categorical.
    # The gap between the checkbox panel's right edge (0.167) and `left` has to
    # hold both the tick labels and the axis label, as in the altitude plot.
    left, width = 0.260, 0.655
    rects = [(0.700, 0.220), (0.520, 0.150), (0.320, 0.170), (0.130, 0.150)]
    ax_z, ax_ref, ax_sel, ax_band = [
        fig.add_axes([left, b, width, h], facecolor=C_SURFACE) for b, h in rects]
    for a in (ax_z, ax_ref, ax_sel):
        a.sharex(ax_band)
    ax_rel = ax_sel.twinx()
    ax_rel.set_facecolor("none")

    axis_of = {"z": ax_z, "ref": ax_ref, "sel": ax_sel, "rel": ax_rel}

    # The shading, on every panel -- that is what makes them one figure.
    shade_art = []
    for a in (ax_z, ax_ref, ax_sel, ax_band):
        shade_art += draw_primary_shading(a, spans)

    # Armed shading is available but OFF by default: it would overlay the
    # instance colours it exists alongside.  The band panel carries it always.
    armed_art = []
    for a in (ax_z, ax_ref, ax_sel):
        armed_art += draw_armed(a, armed_spans(ulog))
    for art in armed_art:
        art.set_visible(False)

    ax_sel.set_yscale("log")
    ax_sel.axhline(WARN_RATIO, color=C_BAD, lw=1.1, ls="--", alpha=0.8, zorder=1)
    ax_sel.text(0.012, WARN_RATIO, "warning >= 1.0",
                transform=ax_sel.get_yaxis_transform(),
                color=C_BAD, fontsize=7, va="bottom", ha="left")
    ax_rel.axhline(0.0, color=C_MUTED, lw=1, ls=":", alpha=0.6, zorder=1)

    _draw_band(ax_band, ulog, ctx, instances, spans)

    for s in series:
        (line,) = axis_of[s.group].plot(
            s.t, s.y, color=s.color, ls=s.ls, lw=s.lw, label=s.label,
            drawstyle=s.drawstyle, alpha=s.alpha,
            zorder=s.zorder if s.zorder is not None else 3)
        line.set_visible(s.visible)
        s.line = line

    # Height-reset markers on panel 1.  A handover moves the shading; a reset
    # bumps z_reset_counter.  Seeing both marks on the same instant is the
    # signature of the nasty case -- the frame changed AND the state jumped.
    reset_art, n_small = [], 0
    for t_r, dz, c0, c1 in _reset_events(ulog):
        reset_art.append(ax_z.axvline(t_r, color=C_BAD, lw=1.0, ls=":",
                                      alpha=0.75, zorder=2))
        # Every reset gets a rule, but only the ones big enough to move the trace
        # get a label.  A real log opens with a burst of sub-metre resets while
        # the filters initialise, and labelling those overprints the panel's top
        # edge into an unreadable stripe -- burying the +-30 m ones that matter.
        if abs(dz) < LABEL_RESET_M:
            n_small += 1
            continue
        reset_art.append(ax_z.text(
            t_r, 0.985, f" reset {c0}->{c1}  dz={dz:+.1f}",
            transform=ax_z.get_xaxis_transform(), rotation=90, fontsize=6,
            color=C_BAD, va="top", ha="left", zorder=6))
    if n_small:
        ctx.note(f"{n_small} height reset(s) smaller than {LABEL_RESET_M:g} m are "
                 f"marked but not labelled")

    # --- axis furniture -----------------------------------------------------
    for a in (ax_z, ax_ref, ax_sel):
        style_time_axis(a, label=False)
        a.tick_params(axis="x", labelbottom=False)
    style_time_axis(ax_band)

    ax_z.set_ylabel("height above local origin (m)\n= -z, NED down-positive",
                    fontsize=9)
    ax_ref.set_ylabel("ref_alt (m AMSL)", fontsize=9)
    ax_sel.set_ylabel("combined test ratio", fontsize=9)
    ax_rel.set_ylabel("relative (dashed)", fontsize=9)
    _style_axis(ax_z, C_INK)
    _style_axis(ax_ref, C_INK)
    _style_axis(ax_sel, C_INK)
    _style_axis(ax_rel, C_MUTED)

    fig.text(left, 0.955, "Local position z by EKF instance", color=C_INK,
             fontsize=13, fontweight="bold", ha="left")
    who = f"{os.path.basename(path)}   |   " if path else ""
    if spans:
        n_sw = max(len(spans) - 1, 0)
        used = sorted({i for _, _, i in spans})
        shade_note = (f"shaded by primary instance ({', '.join(str(i) for i in used)}"
                      f"); {n_sw} handover(s)")
    else:
        shade_note = "no selector topic -- single EKF, no shading"
    fig.text(left, 0.925,
             f"{who}{duration_min(ulog):.1f} min   |   {shade_note}",
             color=C_MUTED, fontsize=9, ha="left")

    # Per-instance colour key, drawn as figure text rather than a legend box so it
    # costs no plot area.  The checkbox panel already carries the LINE colours;
    # this exists to name the SHADING, which has no checkbox of its own per
    # instance.  Right-aligned on the title row: a line of its own sits on top of
    # panel 1, where it collides with the rotated reset labels.
    x = left + width
    for i in (sorted({i for _, _, i in spans}, reverse=True) if spans else []):
        fig.text(x, 0.952, f"EKF {i}", color=_inst_color(i), fontsize=8,
                 ha="right", fontweight="bold")
        x -= 0.040
    if spans:
        fig.text(x, 0.952, "shading:", color=C_MUTED, fontsize=8, ha="right")

    off_note = ax_sel.text(0.995, 0.96, "", transform=ax_sel.transAxes,
                           ha="right", va="top", fontsize=7, color=C_BAD)

    # An empty panel with a grid on it reads as "the scores were all zero", which
    # is a claim about the data.  Say instead that the log does not carry them --
    # HITL logs pulled off the board have no selector topic at all.
    if not any(s.group in ("sel", "rel") for s in series):
        ax_sel.text(0.5, 0.5, "no estimator_selector_status in this log",
                    transform=ax_sel.transAxes, ha="center", va="center",
                    color=C_MUTED, fontsize=9)
        ax_sel.set_yticks([])
        ax_rel.set_yticks([])

    def refresh():
        for group, a in (("z", ax_z), ("ref", ax_ref), ("rel", ax_rel)):
            _rescale(a, [s.line for s in series if s.group == group])
        n, mx = _rescale_log(ax_sel, [s.line for s in series if s.group == "sel"])
        off_note.set_text(f"{n} test ratio(s) off-scale (max {mx:.0f})" if n else "")

    extra = []
    if reset_art:
        extra.append(("height resets (marked)", reset_art, True))
    if shade_art:
        extra.append(("instance shading", shade_art, True))
    if armed_art:
        extra.append(("armed (shaded)", armed_art, False))

    h = min(0.86, 0.035 * (len(series) + 5) + 0.05)
    check_panel(fig, [0.012, 0.89 - h, 0.155, h], series,
                [("z", "ALL z"), ("ref", "ALL ref_alt"),
                 ("sel", "ALL combined ratios"), ("rel", "ALL relative")],
                extra=extra, on_change=refresh)
    refresh()
    add_mouse_navigation(fig, [ax_z, ax_ref, ax_sel, ax_rel, ax_band],
                         page_scroll=ctx.page_scroll)
    fig.text(left, 0.03, nav_hint(ctx.page_scroll), color=C_MUTED, fontsize=8,
             ha="left")
    return fig
