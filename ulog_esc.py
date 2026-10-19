#!/usr/bin/env python3
"""ulog_esc.py -- measured rotor speed from the ESCs (Electronic Speed Controllers).

Three stacked panels on one time axis:

  1. RPM      -- each motor's measured speed (esc_status.esc[i].esc_rpm)
  2. split    -- each motor's RPM as % above/below the four-motor mean, armed only.
                 A steady split is TRIM: a yaw-torque imbalance (one diagonal pair
                 working harder) or a CG (centre of gravity) offset (front or rear
                 pair working harder).  Control activity is the wiggle on top.
  3. errors   -- telemetry decode failures per second per motor

Where the numbers come from
---------------------------
Bidirectional DShot (PWM_MAIN_TIM0 = -8/-7/-6): each ESC answers every DShot
frame with its eRPM on the same signal wire.  PX4 divides by
DSHOT_MOT_POLn / 2 (pole pairs) to get mechanical RPM, so a wrong pole count
scales the whole panel -- the count used is printed in the header.  It is logged
at 10 Hz by the default profile and 200 Hz with SDLOG_PROFILE bit 4 (high rate).

esc_errorcount is cumulative BDShot failures PLUS serial-telemetry timeouts, so
panel 3 is their time derivative.  With DSHOT_TEL_CFG set but no telemetry wire
(grazer17, 2026-10-16) the serial timeouts alone give ~3/s at rest.

DSHOT_TEL_CFG != 0 also has two silent effects in v1.18's DShot driver
(consume_esc_data): esc_online_flags requires BOTH sources online, so it stays 0
with no wire even while BDShot RPM is valid; and BDShot EDT (Extended DShot
Telemetry) voltage/current/temperature are DISCARDED in favour of the serial
source.  So all-zero V/I/T with DSHOT_TEL_CFG set says nothing about the ESC.
esc_online_flags is therefore not used here.
"""
import os
import warnings

import numpy as np

from ulog_common import (C_GRID, C_INK, C_MUTED, C_SURFACE, INST_COLORS,
                         PlotCtx, Series, _get, _rescale, _style_axis,
                         _time_min, add_mouse_navigation, armed_spans,
                         check_panel, draw_armed, draw_mode_changes,
                         duration_min, mode_changes, mode_key, nav_hint,
                         style_time_axis)
from ulog_commands import _header, _no_data, _plot_series

ESC_TOPICS = [
    "esc_status",
    "actuator_armed",
    "vehicle_status",
]

# actuator_function 101..112 = Motor 1..12 (PX4 OutputFunction).
_MOTOR_FN0 = 101


def _escs(ulog):
    """[(motor_number, t_min, rpm, errcount, edt)] for each ESC slot in use."""
    d = _get(ulog, "esc_status")
    if d is None:
        return []
    t = _time_min(ulog, d)
    order = np.argsort(t, kind="stable")
    t = t[order]
    out = []
    for i in range(12):
        k = f"esc[{i}].esc_rpm"
        if k not in d.data:
            break
        fn = np.asarray(d.data.get(f"esc[{i}].actuator_function", [0]), int)
        rpm = np.asarray(d.data[k], float)[order]
        # Unused slots: no function assigned and never a reading.
        if not (fn > 0).any() and not (rpm != 0).any():
            continue
        fn_used = fn[fn > 0]
        motor = int(np.bincount(fn_used).argmax()) - _MOTOR_FN0 + 1 if fn_used.size else i + 1
        err = np.asarray(d.data[f"esc[{i}].esc_errorcount"], float)[order]
        edt = any(np.any(np.asarray(d.data.get(f"esc[{i}].{f}", [0]), float) != 0)
                  for f in ("esc_voltage", "esc_current", "esc_temperature"))
        out.append((motor, t, rpm, err, edt))
    return out


def _rolling(t, y, win_min):
    """Centered time-window mean ignoring NaN; NaN where the input is NaN."""
    ok = np.isfinite(y)
    c = np.r_[0.0, np.cumsum(np.where(ok, y, 0.0))]
    n = np.r_[0, np.cumsum(ok)]
    lo = np.searchsorted(t, t - win_min / 2)
    hi = np.searchsorted(t, t + win_min / 2, side="right")
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (c[hi] - c[lo]) / cnt
    out[~ok] = np.nan
    return out


SMOOTH_S = 2.0      # 10 Hz logging is too noisy to read a trim by eye


def _armed_mask(ulog, t):
    spans = armed_spans(ulog)
    if not spans:
        return np.ones(t.size, bool)
    m = np.zeros(t.size, bool)
    for a, b in spans:
        m |= (t >= a) & (t <= b)
    return m


def build_esc(ulog, ctx=None, path=""):
    """Measured ESC RPM, per-motor split and telemetry errors.  Same signature
    as every builder."""
    import matplotlib.pyplot as plt

    ctx = ctx or PlotCtx()
    escs = _escs(ulog)
    if not escs:
        ctx.note("no esc_status in this log -- ESC telemetry needs bidirectional "
                 "DShot (PWM_MAIN_TIMx = -8/-7/-6) or a serial telemetry wire")
        return None

    params = getattr(ulog, "initial_parameters", {}) or {}
    t = escs[0][1]
    armed = _armed_mask(ulog, t)
    spinning = armed & np.all([e[2] > 0 for e in escs], axis=0)

    series = []
    rpm_all = np.vstack([e[2] for e in escs])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(np.where(rpm_all > 0, rpm_all, np.nan), axis=0)
    for (motor, tt, rpm, err, _), col in zip(escs, INST_COLORS * 3):
        series.append(Series(f"esc_status.esc[{motor - 1}].esc_rpm", f"motor {motor} RPM",
                             tt, rpm, "rpm", col, lw=1.0, alpha=0.85, visible=True))
        split = 100.0 * (rpm / mean - 1.0)
        split[~spinning] = np.nan
        series.append(Series(f"esc[{motor - 1}] split", f"motor {motor} {SMOOTH_S:g} s mean",
                             tt, _rolling(tt, split, SMOOTH_S / 60), "split", col,
                             lw=1.4, visible=True))
        series.append(Series(f"esc[{motor - 1}] split raw", f"motor {motor} raw",
                             tt, split, "split", col, lw=0.7, alpha=0.35, visible=False))
        dt = np.diff(tt) * 60.0
        de = np.diff(err)
        rate = np.full(tt.size, np.nan)
        ok = (dt > 0) & (de >= 0)       # a negative step is a counter reset
        rate[1:][ok] = de[ok] / dt[ok]
        series.append(Series(f"esc[{motor - 1}] errors/s", f"motor {motor} errors/s",
                             tt, _rolling(tt, rate, SMOOTH_S / 60), "err", col,
                             lw=1.0, alpha=0.9, visible=True))
    m_mean = mean.copy()
    m_mean[~armed] = np.nan
    series.append(Series("esc mean", "mean RPM", t, m_mean, "rpm", "#20222b",
                         lw=1.6, visible=False, zorder=5))

    fig = plt.figure(figsize=(15, 11), facecolor=C_SURFACE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(f"logGraph ESC - {os.path.basename(path)}")
    left, width = 0.260, 0.655
    ax_r = fig.add_axes([left, 0.62, width, 0.27], facecolor=C_SURFACE)
    ax_s = fig.add_axes([left, 0.36, width, 0.20], facecolor=C_SURFACE)
    ax_e = fig.add_axes([left, 0.11, width, 0.19], facecolor=C_SURFACE)
    for a in (ax_s, ax_e):
        a.sharex(ax_r)
    axes = [ax_r, ax_s, ax_e]

    spans = armed_spans(ulog)
    armed_art = sum((draw_armed(a, spans) for a in axes), [])
    _plot_series(series, {"rpm": ax_r, "split": ax_s, "err": ax_e})
    ax_s.axhline(0, color=C_MUTED, lw=0.8, zorder=2)
    if not spinning.any():
        _no_data(ax_s, "motors never spinning while armed -- no split to show")

    pol = params.get("DSHOT_MOT_POL1")
    has_edt = any(e[4] for e in escs)
    style_time_axis(ax_r, label=False)
    style_time_axis(ax_s, label=False)
    style_time_axis(ax_e)
    ax_r.set_ylabel("rotor speed (rev/min)", fontsize=9)
    ax_s.set_ylabel("RPM vs 4-motor mean (%)", fontsize=9)
    ax_e.set_ylabel("telemetry errors / s", fontsize=9)
    if has_edt:
        edt_note = ""
    elif params.get("DSHOT_TEL_CFG"):
        edt_note = "; no voltage/current/temp: DSHOT_TEL_CFG is set, so EDT is discarded"
    else:
        edt_note = "; no voltage/current/temp: ESC sends no EDT"
    ax_r.set_title("esc_status.esc_rpm -- MEASURED by each ESC (bidirectional "
                   "DShot)" + edt_note,
                   fontsize=8, color=C_MUTED, loc="left")
    ax_s.set_title("split while armed and spinning -- a steady offset is trim "
                   "(diagonal pair = yaw imbalance, front/rear pair = CG)",
                   fontsize=8, color=C_MUTED, loc="left")
    ax_e.set_title(f"failed telemetry replies per second (d esc_errorcount / dt, {SMOOTH_S:g} s mean)",
                   fontsize=8, color=C_MUTED, loc="left")
    for a in axes:
        _style_axis(a, C_INK)

    if spinning.any():
        parts = []
        for motor, _, rpm, _, _ in escs:
            parts.append(f"M{motor} {np.mean(rpm[spinning]):.0f}")
        detail = "armed mean RPM  " + "  ".join(parts)
        if pol is not None:
            detail += f"   |   pole count {pol:g} (DSHOT_MOT_POL1)"
    else:
        detail = "motors never spun while armed"
    _header(fig, left, "ESC telemetry", ulog, path, detail)

    mode_art, mode_codes = draw_mode_changes(
        axes, mode_changes(ulog), text_ax=ax_r,
        min_gap=max(duration_min(ulog), 1.0) * 0.035)
    mode_art += mode_key(fig, left + width, 0.015, mode_codes)

    def refresh():
        for a, g in ((ax_r, "rpm"), (ax_s, "split"), (ax_e, "err")):
            _rescale(a, [s.line for s in series if s.group == g])

    extra = []
    if mode_art:
        extra.append(("mode changes", mode_art, True))
    if armed_art:
        extra.append(("armed (shaded)", armed_art, True))
    h = min(0.82, 0.026 * (len(series) + len(extra) + 4) + 0.05)
    check_panel(fig, [0.012, 0.89 - h, 0.155, h], series,
                [("rpm", "ALL RPM"), ("split", "ALL split"), ("err", "ALL errors")],
                extra=extra, on_change=refresh)
    refresh()
    add_mouse_navigation(fig, axes, page_scroll=ctx.page_scroll, on_view=refresh)
    fig.text(left, 0.045, nav_hint(ctx.page_scroll), color=C_MUTED,
             fontsize=8, ha="left")
    return fig


if __name__ == "__main__":
    import argparse
    from pyulog import ULog
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log")
    ap.add_argument("--save", help="write a PNG instead of showing")
    a = ap.parse_args()
    if a.save:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ctx = PlotCtx()
    f = build_esc(ULog(a.log, ESC_TOPICS), ctx, a.log)
    for n in getattr(ctx, "notes", []):
        print("note:", n)
    if f is None:
        raise SystemExit(1)
    if a.save:
        f.savefig(a.save, dpi=90)
    else:
        plt.show()
