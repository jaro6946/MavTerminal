#!/usr/bin/env python3
"""ulog_faults.py -- every event and fault a log recorded, and when each happened.

Two halves:

  collect_faults(ulog)      -> [Fault]   the catalogue: only faults that actually
                                         OCCURRED in this log, never a checklist
  build_fault_detail(...)   -> Figure    one fault's occurrences over time
  build_faults(ulog,ctx,p)  -> Figure    the registry entry: every fault as one
                                         row (the PDF page, and the browser's
                                         "all faults" view)

The browser wraps this in a dropdown (log_browser.FaultPanel) so a fault is
picked from the list and its timeline drawn; this module holds no Qt so the PDF
exporter and the CLI can use it too.

Where a "fault" comes from -- five places, because PX4 reports trouble in five
different ways and none of them is complete on its own:

  message   STATUSTEXT-style log lines at WARNING or worse ("Preflight Fail: Gyro
            1 inconsistent", "RTT too high for timesync: 21 ms").  The most
            readable source, and the only one that names arming-check failures.
  failsafe  failsafe_flags booleans (gcs_connection_lost, local_position_invalid
            ...) and vehicle_status.failsafe itself.  These are STATES with a
            duration, not events.
  ekf       estimator_status_flags fs_* (filter faults), reject_* (innovation
            rejected a measurement) and cs_inertial_dead_reckoning, per EKF
            instance.
  driver    sensor_* error_count increments, per sensor instance -- the counter
            is cumulative, so each step up is one occurrence.
  gps       sensor_gps spoofing / jamming state at "indicated" or worse.

Plus one source that is not a fault but explains many of them:

  event     sensor calibrations: each run from "[cal] calibration started" to
            done / failed / cancelled, with the CAL_* parameters it changed.  A
            calibration tips the vehicle on every side, so it trips the failure
            detector (fd_critical_failure, "Attitude failure (roll)") and mag
            and attitude checks; without the run drawn beside them those read as
            real faults (MTL111, 6:08-9:41).
            EKF resets: every step of the yaw / position / height / velocity
            reset counters of the PUBLISHED estimate, sized by its delta, and
            every change of the primary EKF instance.  A reset is a jump in the
            estimate that no motion caused -- the explanation for a heading or
            position step in any other plot.
            Parameter changes made while logging, other than those a calibration
            run wrote: name, old -> new.  A change mid-log silently changes what
            the rest of the log means.

Occurrence semantics: an EVENT (message, counter step) is one instant; a STATE
(flag) occurs once per rising edge and lasts until it clears -- so "occurrences"
is always "how many times did it start", whichever kind it is.

Acronyms: EKF = extended Kalman filter, GCS = ground control station,
GPS = global positioning system, PDF = portable document format.
"""
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field as _dc_field
from typing import List, Optional

import numpy as np

from ulog_common import (C_ARMED, C_BAD, C_GRID, C_INK, C_MUTED, C_SURFACE,
                         PlotCtx, _get, _time_min, add_mouse_navigation,
                         armed_spans, draw_armed, draw_band_rows,
                         draw_mode_changes, duration_min, inst_color,
                         mode_changes, mode_key, nav_hint, style_time_axis)

__all__ = ["Fault", "FAULT_TOPICS", "collect_faults", "build_faults",
           "build_fault_detail", "fmt_clock"]

FAULT_TOPICS = [
    "vehicle_local_position", "estimator_selector_status",
    "failsafe_flags", "vehicle_status", "actuator_armed",
    "estimator_status_flags",
    "sensor_accel", "sensor_gyro", "sensor_mag", "sensor_baro", "sensor_gps",
]

# Severity colours.  Red stays reserved for faults (C_BAD); a WARNING-level
# message is amber so a screen full of "no GCS connection" chatter does not look
# as alarming as one real ERROR.
C_WARN = "#d9822b"
C_EVENT = "#2a78d6"     # blue: something that HAPPENED, not something wrong
LEVEL_COLOR = {"EMERGENCY": C_BAD, "ALERT": C_BAD, "CRITICAL": C_BAD,
               "ERROR": C_BAD, "WARNING": C_WARN}
SOURCE_ORDER = ["event", "message", "failsafe", "ekf", "driver", "gps"]
SOURCE_NAME = {"event": "event", "message": "log message", "failsafe": "failsafe flag",
               "ekf": "EKF flag", "driver": "driver errors", "gps": "GPS"}

# Numbers in a message are VALUES ("RTT too high: 18 ms", "Compasses 104°
# inconsistent") and are masked out of the fault's name -- except a lone digit
# straight after a word, which is an instance index ("Gyro 1 inconsistent",
# "EKF changed 0 -> 1") and names a different sensor, so stays in the name.
# Digits glued to a word are part of an identifier ("[ekf2]", "TC_GN_TREF0").
_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
_INDEX = re.compile(r"(?<=[A-Za-z] )\d(?![\d.])")


def _template(text):
    """(name with values masked as 'N', [values]) for one message."""
    out, vals, pos = [], [], 0
    for m in _NUM.finditer(text):
        # match(text, pos) lets the lookbehind see the word before the digit.
        idx = _INDEX.match(text, m.start())
        if idx and idx.end() == m.end():
            continue
        out.append(text[pos:m.start()] + "N")
        vals.append(float(m.group(0)))
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), vals


@dataclass
class Fault:
    key: str                 # stable id: survives a re-open of the same log
    source: str              # one of SOURCE_ORDER
    label: str               # what the dropdown and the row show
    color: str
    # One entry per lane: (lane name, [(t0, t1) spans], onset times).  A message
    # has one lane and no spans; an EKF flag has one lane per instance.
    lanes: list = _dc_field(default_factory=list)
    # Optional number carried by each occurrence (the ms in "RTT too high", the
    # size of an error-count step), aligned with `value_t`.
    value_t: Optional[np.ndarray] = None
    values: Optional[np.ndarray] = None
    value_name: str = ""
    texts: List[str] = _dc_field(default_factory=list)   # distinct wordings

    @property
    def onsets(self):
        ts = [t for _n, _sp, on in self.lanes for t in on]
        return np.sort(np.asarray(ts, dtype=float))

    @property
    def count(self):
        return int(sum(len(on) for _n, _sp, on in self.lanes))

    @property
    def active_min(self):
        return float(sum(b - a for _n, sp, _on in self.lanes for a, b in sp))

    def summary(self):
        on = self.onsets
        if on.size == 0:
            return f"{self.label}: no occurrences"
        s = (f"{self.count} occurrence{'s' if self.count != 1 else ''}, "
             f"first {fmt_clock(on[0])}, last {fmt_clock(on[-1])}")
        if self.active_min > 0:
            s += f", active {fmt_clock(self.active_min)} in total"
        return s


def fmt_clock(t_min):
    """Minutes -> 'm:ss.s'.  The axes are in minutes; a fault time read off one
    wants seconds, and '5:22.4' is quicker to match to a video than 5.373."""
    s = max(float(t_min), 0.0) * 60.0
    return f"{int(s // 60)}:{s % 60:04.1f}"


# --- collection ---------------------------------------------------------------

def _state_spans(t, on, t_end):
    """(spans, onsets) for a boolean state sampled at `t`.

    Unlike ulog_common.spans_from_bool, a state still true at the last sample is
    closed at the END OF THE LOG, not at that sample: these topics publish on
    change, so the last sample can be minutes before the log stops and the
    fault did not clear just because nothing new was published."""
    on = np.asarray(on, dtype=bool)
    spans, onsets, start = [], [], None
    for i in range(on.size):
        if on[i] and start is None:
            start = t[i]
            onsets.append(float(t[i]))
        elif not on[i] and start is not None:
            spans.append((float(start), float(t[i])))
            start = None
    if start is not None:
        spans.append((float(start), float(max(t_end, start))))
    return spans, onsets


def _instances(ulog, topic):
    return sorted({d.multi_id for d in ulog.data_list if d.name == topic})


def _messages(ulog):
    """Log lines at WARNING or worse, grouped into faults by _template: one
    fault per message with its values masked, the varying value kept per
    occurrence so it can be plotted."""
    t0 = getattr(ulog, "start_timestamp", 0) or 0
    rows = []
    for m in getattr(ulog, "logged_messages", []) or []:
        level = m.log_level_str()
        if level not in LEVEL_COLOR:
            continue
        # PX4 pads some lines with a literal tab; DejaVu Sans Mono has no glyph
        # for it and it would also split one message into two groups.
        text = " ".join(m.message.split())
        rows.append(((m.timestamp - t0) / 6e7, level, text))
    groups = OrderedDict()
    for t, level, text in rows:
        name, vals = _template(text)
        groups.setdefault(name, []).append((t, level, text, vals))

    faults = []
    order = list(LEVEL_COLOR)
    for name, items in groups.items():
        ts = np.array([o[0] for o in items])
        worst = min((o[1] for o in items), key=order.index)
        f = Fault(key=f"msg:{name}", source="message", label=name,
                  color=LEVEL_COLOR[worst], lanes=[("", [], list(ts))],
                  texts=list(OrderedDict((o[2], None) for o in items)))
        # Plot the first value that actually VARIES -- a constant one carries
        # no information over time.
        nums = [o[3] for o in items]
        width = min(len(n) for n in nums)
        pos = next((k for k in range(width) if len({n[k] for n in nums}) > 1),
                   None)
        if pos is not None:
            f.value_t = ts
            f.values = np.array([n[pos] for n in nums])
            f.value_name = "value in message"
        faults.append(f)
    return faults


def _flag_fault(ulog, topic, name, label, source, t_end, pred=None,
                lane_fmt=None):
    """A boolean field -> Fault, one lane per instance, or None if never set."""
    lanes = []
    for mid in _instances(ulog, topic):
        d = _get(ulog, topic, mid)
        if d is None or name not in d.data:
            continue
        v = np.asarray(d.data[name])
        on = pred(v) if pred else v.astype(float) > 0.5
        if not np.any(on):
            continue
        spans, onsets = _state_spans(_time_min(ulog, d), on, t_end)
        lanes.append((lane_fmt.format(mid) if lane_fmt else "", spans, onsets))
    if not lanes:
        return None
    return Fault(key=f"{source}:{topic}.{name}", source=source, label=label,
                 color=C_BAD, lanes=lanes)


def _failsafe(ulog, t_end):
    out = []
    d = _get(ulog, "failsafe_flags")
    if d is not None:
        for name, v in d.data.items():
            # mode_req_* are bitmasks of which modes NEED a thing, not faults.
            if name == "timestamp" or name.startswith("mode_req_"):
                continue
            f = _flag_fault(ulog, "failsafe_flags", name, name, "failsafe", t_end)
            if f:
                out.append(f)
    for topic, name, label in (
            ("vehicle_status", "failsafe", "FAILSAFE ACTIVE (vehicle_status)"),
            ("vehicle_status", "failure_detector_status",
             "failure detector tripped"),
            ("actuator_armed", "lockdown", "actuator lockdown"),
            ("actuator_armed", "manual_lockdown", "kill switch (manual lockdown)"),
            ("actuator_armed", "force_failsafe", "force failsafe")):
        f = _flag_fault(ulog, topic, name, label, "failsafe", t_end,
                        pred=lambda v: np.asarray(v).astype(np.int64) != 0)
        if f:
            out.append(f)
    return out


def _ekf(ulog, t_end):
    out = []
    mids = _instances(ulog, "estimator_status_flags")
    if not mids:
        return out
    d0 = _get(ulog, "estimator_status_flags", mids[0])
    names = [n for n in d0.data if n.startswith(("fs_", "reject_", "cs_bad"))
             or n == "cs_inertial_dead_reckoning"]
    for name in names:
        f = _flag_fault(ulog, "estimator_status_flags", name, f"EKF {name}",
                        "ekf", t_end, lane_fmt="EKF {}")
        if f:
            out.append(f)
    return out


def _driver(ulog):
    """sensor_*.error_count is cumulative: each step up is one occurrence."""
    out = []
    for topic in ("sensor_accel", "sensor_gyro", "sensor_mag", "sensor_baro"):
        lanes, vt, vv = [], [], []
        for mid in _instances(ulog, topic):
            d = _get(ulog, topic, mid)
            if d is None or "error_count" not in d.data:
                continue
            t = _time_min(ulog, d)
            e = np.asarray(d.data["error_count"], dtype=float)
            step = np.diff(e)
            # A DROP is a counter reset (driver restart), not negative errors.
            idx = np.flatnonzero(step > 0) + 1
            if idx.size == 0:
                continue
            lanes.append((f"{topic.split('_')[1]} {mid}", [], list(t[idx])))
            vt.extend(t[idx])
            vv.extend(step[idx - 1])
        if lanes:
            order = np.argsort(vt)
            out.append(Fault(key=f"driver:{topic}", source="driver",
                             label=f"{topic} driver error_count increments",
                             color=C_BAD, lanes=lanes,
                             value_t=np.asarray(vt)[order],
                             values=np.asarray(vv)[order],
                             value_name="errors per step"))
    return out


def _gps(ulog, t_end):
    out = []
    # 0 unknown, 1 none/ok, 2 indicated/warning, 3 multiple/critical.
    for name, label in (("spoofing_state", "GPS spoofing indicated"),
                        ("jamming_state", "GPS jamming indicated")):
        f = _flag_fault(ulog, "sensor_gps", name, label, "gps", t_end,
                        pred=lambda v: np.asarray(v).astype(int) >= 2,
                        lane_fmt="gps {}")
        if f:
            out.append(f)
    return out


# PX4 commander/calibration_messages.h.  The "2" after "started:" is a protocol
# version, not a count.
_CAL_START = re.compile(r"\[cal\] calibration started: \d+ (\w+)")
_CAL_END = re.compile(r"\[cal\] calibration (done|failed|cancelled)")
# Which parameters each calibration writes.  "level" writes the board offsets.
_CAL_PARAMS = {"accel": ("CAL_ACC",), "gyro": ("CAL_GYRO",), "mag": ("CAL_MAG",),
               "level": ("SENS_BOARD_X_OFF", "SENS_BOARD_Y_OFF"),
               "baro": ("CAL_BARO",), "airspeed": ("SENS_DPRES_OFF",)}


def _calibration(ulog, t_end):
    """One event row per calibration TYPE: a bar per run, red if it failed.

    The value carried per run is how many parameters it changed -- 0 on a
    "done" run means nothing was saved, worth knowing before trusting it."""
    t0 = getattr(ulog, "start_timestamp", 0) or 0
    runs, open_ = [], None
    for m in getattr(ulog, "logged_messages", []) or []:
        t = (m.timestamp - t0) / 6e7
        ms = _CAL_START.search(m.message)
        if ms:
            if open_:
                runs.append(open_ + (t, "interrupted"))
            open_ = (ms.group(1), t)
            continue
        me = _CAL_END.search(m.message)
        if me and open_:
            runs.append(open_ + (t, me.group(1)))
            open_ = None
    if open_:
        runs.append(open_ + (t_end, "unfinished"))
    if not runs:
        return []
    changed = [((ts - t0) / 6e7, name) for ts, name, _v in
               (getattr(ulog, "changed_parameters", []) or []) if ts > t0]
    out = []
    for kind in OrderedDict((r[0], None) for r in runs):
        mine = [r for r in runs if r[0] == kind]
        prefixes = _CAL_PARAMS.get(kind, ("CAL_",))
        spans, onsets, vals, texts = [], [], [], []
        for _k, a, b, how in mine:
            # Parameters land at or just after the "done" line.
            n = sum(1 for t, name in changed
                    if a <= t <= b + 0.05 and name.startswith(prefixes))
            spans.append((a, b))
            onsets.append(a)
            vals.append(n)
            texts.append(f"{fmt_clock(a)}-{fmt_clock(b)} {how}, {n} params changed")
        bad = any(r[3] != "done" for r in mine)
        outcome = ", ".join(sorted({r[3] for r in mine}))
        out.append(Fault(key=f"event:cal:{kind}", source="event",
                         label=f"{kind} calibration ({outcome}, "
                               f"{sum(vals)} params changed)",
                         color=C_BAD if bad else C_EVENT,
                         lanes=[("", spans, onsets)],
                         value_t=np.asarray(onsets), values=np.asarray(vals, float),
                         value_name="parameters changed", texts=texts))
    return out


# (counter field, label, value name, delta -> number)
_RESETS = (
    ("heading_reset_counter", "EKF yaw reset", "yaw step (deg)",
     lambda d, i: np.degrees(d["delta_heading"][i])),
    ("xy_reset_counter", "EKF horizontal position reset", "position step (m)",
     lambda d, i: np.hypot(d["delta_xy[0]"][i], d["delta_xy[1]"][i])),
    ("z_reset_counter", "EKF height reset", "height step (m, +down)",
     lambda d, i: d["delta_z"][i]),
    ("vxy_reset_counter", "EKF horizontal velocity reset", "velocity step (m/s)",
     lambda d, i: np.hypot(d["delta_vxy[0]"][i], d["delta_vxy[1]"][i])),
    ("vz_reset_counter", "EKF vertical velocity reset", "velocity step (m/s, +down)",
     lambda d, i: d["delta_vz"][i]),
)


def _ekf_resets(ulog):
    """Steps of the published estimate's reset counters, plus primary-instance
    switches.  Only steps WITHIN the log count: the first sample's counter
    already includes resets from before logging started (the boot alignment),
    which have no time here."""
    out = []
    d = _get(ulog, "vehicle_local_position")
    if d is not None:
        t = _time_min(ulog, d)
        for field, label, vname, delta in _RESETS:
            if field not in d.data:
                continue
            c = np.asarray(d.data[field], dtype=np.int64)
            # uint8 counter: a wrap 255 -> 0 is still one step.
            idx = np.flatnonzero(np.diff(c) % 256 != 0) + 1
            if idx.size == 0:
                continue
            vals = np.array([float(delta(d.data, i)) for i in idx])
            unit = vname[vname.index("(") + 1:].split(",")[0].rstrip(")")
            big = vals[np.argmax(np.abs(vals))]
            out.append(Fault(key=f"event:reset:{field}", source="event",
                             label=f"{label} (largest {big:+.3g} {unit})",
                             color=C_EVENT,
                             lanes=[("", [], list(t[idx]))],
                             value_t=t[idx], values=vals, value_name=vname,
                             texts=[f"{fmt_clock(t[i])} {v:+.3g} {unit}"
                                    for i, v in zip(idx, vals)]))
    d = _get(ulog, "estimator_selector_status")
    if d is not None and "primary_instance" in d.data:
        t = _time_min(ulog, d)
        p = np.asarray(d.data["primary_instance"], dtype=np.int64)
        idx = np.flatnonzero(np.diff(p) != 0) + 1
        if idx.size:
            out.append(Fault(
                key="event:ekf_primary", source="event",
                label="EKF primary instance switched", color=C_EVENT,
                lanes=[("", [], list(t[idx]))], value_t=t[idx],
                values=p[idx].astype(float), value_name="new primary instance",
                texts=[f"{fmt_clock(t[i])} EKF {p[i - 1]} -> {p[i]}" for i in idx]))
    return out


def _fmt_param(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def _param_changes(ulog, cal_faults):
    """Parameter changes made while logging, minus the ones a calibration run
    wrote.  One row per parameter; the wording list carries old -> new."""
    t0 = getattr(ulog, "start_timestamp", 0) or 0
    cal = []
    for f in cal_faults:
        kind = f.key.rsplit(":", 1)[-1]
        prefixes = _CAL_PARAMS.get(kind, ("CAL_",))
        for _lane, spans, _on in f.lanes:
            cal += [(a, b + 0.05, prefixes) for a, b in spans]
    current = dict(getattr(ulog, "initial_parameters", {}) or {})
    rows = OrderedDict()
    for ts, name, val in getattr(ulog, "changed_parameters", []) or []:
        old = current.get(name)
        current[name] = val
        if ts <= t0 or old == val:
            continue
        t = (ts - t0) / 6e7
        if any(a <= t <= b and name.startswith(px) for a, b, px in cal):
            continue
        rows.setdefault(name, []).append((t, old, val))
    out = []
    for name, ch in rows.items():
        ts = np.array([c[0] for c in ch])
        first, last = ch[0][1], ch[-1][2]
        label = (f"param {name}: {_fmt_param(first)} -> {_fmt_param(last)}"
                 if first is not None else f"param {name} set to {_fmt_param(last)}")
        out.append(Fault(key=f"event:param:{name}", source="event", label=label,
                         color=C_EVENT, lanes=[("", [], list(ts))],
                         value_t=ts, values=np.array([float(c[2]) for c in ch]),
                         value_name=name,
                         texts=[f"{fmt_clock(t)} {_fmt_param(o)} -> {_fmt_param(v)}"
                                for t, o, v in ch]))
    return out


def collect_faults(ulog):
    """Every fault that occurred in this log, grouped by source, busiest first."""
    t_end = duration_min(ulog)
    cal = _calibration(ulog, t_end)
    faults = (cal + _ekf_resets(ulog) + _param_changes(ulog, cal)
              + _messages(ulog) + _failsafe(ulog, t_end) + _ekf(ulog, t_end)
              + _driver(ulog) + _gps(ulog, t_end))
    faults = [f for f in faults if f.count]
    # Events keep their collection order (calibrations, resets, parameters):
    # grouped by kind reads better than busiest-first there.
    seq = {id(f): i for i, f in enumerate(faults)}
    faults.sort(key=lambda f: (SOURCE_ORDER.index(f.source),
                               seq[id(f)] if f.source == "event" else -f.count,
                               f.label))
    return faults


# --- drawing ------------------------------------------------------------------

PAGE_PX_PER_IN = 78
MIN_EVENT_FRAC = 0.003          # an instant event is drawn this wide (of the log)


def _decorate_time(fig, axes, ulog, text_ax):
    """Armed shading and mode-change rules -- the context every fault is read
    against ("was it armed?", "what mode was it in?")."""
    spans = armed_spans(ulog)
    for ax in axes:
        draw_armed(ax, spans)
    changes = mode_changes(ulog)
    codes = []
    if changes:
        _art, codes = draw_mode_changes(axes, changes, text_ax=text_ax,
                                        min_gap=duration_min(ulog) * 0.04)
    return codes


def build_faults(ulog, ctx=None, path="", faults=None):
    """Overview: one row per fault.  The registry entry (PDF, CLI, browser)."""
    import matplotlib.pyplot as plt

    ctx = ctx or PlotCtx()
    faults = collect_faults(ulog) if faults is None else faults
    dur = duration_min(ulog) or 1.0
    hold = dur * MIN_EVENT_FRAC

    rows = []
    for f in faults:
        data = []
        for lane, spans, onsets in f.lanes:
            data.append((list(spans) + [(t, t) for t in onsets if not spans],
                         f.color if len(f.lanes) == 1 else inst_color(
                             int(re.sub(r"\D", "", lane) or 0))))
        rows.append((f"{f.label}  ({f.count}x)", data if len(data) > 1
                     else data[0][0], f.color if len(data) > 1 else data[0][1]))
    n = max(len(rows), 1)
    band_in = min(max(n * 0.34 + 0.4, 1.4), 30.0)
    fig_h = 1.0 + band_in + 1.0
    fig = plt.figure(figsize=(15, fig_h), facecolor=C_SURFACE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(
            f"logGraph faults - {os.path.basename(path)}")
    left, right = 0.04, 0.985
    f_ = lambda inches: inches / fig_h
    ax = fig.add_axes([left, f_(0.85), right - left, f_(band_in)])
    ax.set_facecolor(C_SURFACE)
    draw_band_rows(ax, rows, empty_msg="no faults recorded in this log",
                   min_width=hold, track=True)
    # Calibration runs shaded down the whole chart: the faults a calibration
    # causes (failure detector, attitude/heading checks) line up under it.
    for f in faults:
        if f.key.startswith("event:cal:"):
            for _lane, spans, _on in f.lanes:
                for s0, s1 in spans:
                    ax.axvspan(s0, s1, color=C_EVENT, alpha=0.07, lw=0, zorder=0)
    ax.set_xlim(0, dur)
    style_time_axis(ax)
    codes = _decorate_time(fig, [ax], ulog, ax)
    fig.text(left, 1.0 - f_(0.35), "Events and faults over time", fontsize=13,
             fontweight="bold", color=C_INK)
    by_src = OrderedDict()
    for f in faults:
        by_src[f.source] = by_src.get(f.source, 0) + 1
    fig.text(left, 1.0 - f_(0.62),
             f"{len(faults)} distinct events/faults: " + ", ".join(
                 f"{k} {SOURCE_NAME[s]}" for s, k in by_src.items())
             if faults else "no events, warnings, failsafes, EKF faults or driver errors",
             fontsize=9, color=C_MUTED)
    if codes:
        mode_key(fig, right, 1.0 - f_(0.35), codes)
    add_mouse_navigation(fig, [ax], page_scroll=ctx.page_scroll, fixed_y=[ax])
    fig.text(left, f_(0.2), nav_hint(ctx.page_scroll), color=C_MUTED, fontsize=8)
    fig._page_height = int(round(fig_h * PAGE_PX_PER_IN))
    return fig


def build_fault_detail(ulog, fault, ctx=None, path=""):
    """One fault's occurrences over time.

    Top: when it happened -- a tick per event, a bar per active stretch, one
    lane per instance.  Middle: cumulative count, whose SLOPE is the rate (a
    burst reads as a cliff, a steady nag as a ramp).  Bottom, only if the fault
    carries a number: that number per occurrence."""
    import matplotlib.pyplot as plt

    ctx = ctx or PlotCtx()
    dur = duration_min(ulog) or 1.0
    has_val = fault.values is not None and len(fault.values)
    n_lanes = max(len(fault.lanes), 1)
    lanes_in = max(0.9, 0.42 * n_lanes + 0.35)
    panels = [("when", lanes_in), ("count", 1.9)] + ([("value", 1.9)] if has_val else [])
    gap, top, bottom = 0.45, 1.25, 0.95
    fig_h = top + sum(h for _k, h in panels) + gap * (len(panels) - 1) + bottom
    fig = plt.figure(figsize=(15, fig_h), facecolor=C_SURFACE)
    left, right = 0.06, 0.985
    f_ = lambda inches: inches / fig_h

    axes, y = {}, fig_h - top
    for key, h in panels:
        y -= h
        ax = fig.add_axes([left, f_(y), right - left, f_(h)],
                          sharex=axes.get("when"))
        ax.set_facecolor(C_SURFACE)
        axes[key] = ax
        y -= gap

    # --- when -----------------------------------------------------------------
    ax = axes["when"]
    for i, (lane, spans, onsets) in enumerate(fault.lanes):
        yy = n_lanes - 1 - i
        c = fault.color if n_lanes == 1 else inst_color(
            int(re.sub(r"\D", "", lane) or 0))
        ax.barh(yy, dur, left=0, height=0.7, color=C_GRID, alpha=0.6, lw=0,
                zorder=1)
        # One collection, not one barh per span -- see draw_band_rows.
        if spans:
            coll = ax.broken_barh([(a, max(b - a, dur * MIN_EVENT_FRAC))
                                   for a, b in spans], (yy - 0.35, 0.7),
                                  facecolors=c, alpha=0.85, lw=0, zorder=3)
            coll.sticky_edges.x.append(min(a for a, _b in spans))
        # A tick at every onset even on a state fault: a flag that flickers
        # on/off at 10 Hz is one solid bar otherwise, and the ticks show it.
        ax.vlines(onsets, yy - 0.38, yy + 0.38, color=c if not spans else C_INK,
                  lw=1.2, zorder=4)
        if lane:
            ax.text(0.004, yy, f"{lane}  ({len(onsets)}x)", fontsize=8,
                    color=C_INK, va="center", transform=ax.get_yaxis_transform(),
                    zorder=5, bbox=dict(facecolor=C_SURFACE, edgecolor="none",
                                        pad=1.0, alpha=0.8))
    ax.set_yticks([])
    ax.set_ylim(-0.6, n_lanes - 0.4)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_ylabel("occurrences", fontsize=9, color=C_MUTED)

    # --- cumulative count -----------------------------------------------------
    ax = axes["count"]
    on = fault.onsets
    ax.step(np.concatenate(([0.0], on, [dur])),
            np.concatenate(([0], np.arange(1, on.size + 1), [on.size])),
            where="post", color=fault.color, lw=1.6, zorder=3)
    ax.set_ylim(0, max(on.size, 1) * 1.08)
    ax.set_ylabel("cumulative count", fontsize=9, color=C_MUTED)

    if has_val:
        ax = axes["value"]
        ax.plot(fault.value_t, fault.values, "o", ms=4, color=fault.color,
                zorder=3)
        ax.set_ylabel(fault.value_name, fontsize=9, color=C_MUTED)

    all_axes = list(axes.values())
    for k, a in axes.items():
        style_time_axis(a, label=(a is all_axes[-1]))
    all_axes[0].set_xlim(0, dur)
    codes = _decorate_time(fig, all_axes, ulog, axes["when"])

    fig.text(left, 1.0 - f_(0.38), fault.label, fontsize=13, fontweight="bold",
             color=fault.color if fault.color != C_WARN else C_INK)
    fig.text(left, 1.0 - f_(0.66),
             f"{SOURCE_NAME[fault.source]}  ·  {fault.summary()}",
             fontsize=9, color=C_MUTED)
    if len(fault.texts) > 1:
        shown = fault.texts[:4]
        more = f"  (+{len(fault.texts) - 4} more)" if len(fault.texts) > 4 else ""
        fig.text(left, 1.0 - f_(0.92), "wordings: " + "  |  ".join(shown) + more,
                 fontsize=8, color=C_MUTED)
    if codes:
        mode_key(fig, right, 1.0 - f_(0.38), codes)
    add_mouse_navigation(fig, all_axes, page_scroll=ctx.page_scroll,
                         fixed_y=[axes["when"]])
    fig.text(left, f_(0.2), nav_hint(ctx.page_scroll), color=C_MUTED, fontsize=8)
    fig._page_height = int(round(fig_h * PAGE_PX_PER_IN))
    return fig
