#!/usr/bin/env python3
"""ulog_link.py -- link bandwidth: how much each serial link can carry, and what
is using it.

Four stacked panels on one time axis:

  1. DDS out  -- flight controller -> companion, as % of the link's capacity,
                 stacked by topic with the protocol overhead on top
  2. DDS in   -- companion -> flight controller (the SE3 setpoints), same form
  3. MAVLink  -- every MAVLink instance's tx and rx as % of its own budget
  4. band     -- connection, reconnects, dropped samples, overruns, throttling

Where the numbers come from
---------------------------
DDS: `dds_link_status`, a 1 Hz uORB topic added in the grazer custom firmware
(jacob/v1.18-custom).  Upstream PX4 publishes nothing about the uXRCE-DDS link,
so on any other firmware panels 1-2 are empty and say so.  The topic carries:

  * data_rate -- the capacity, baudrate / 10 (8N1: ten bits on the wire per byte)
  * tx_rate / rx_rate -- bytes on the UART, MEASURED at the transport: every
    XRCE frame including session, ping, time-sync and acknowledgement traffic,
    plus the serial framing (flag, addresses, length, CRC, byte stuffing)
  * topic_tx_rate[i] / topic_rx_rate[i] -- serialized payload per topic

The stack is the per-topic payloads, and the top layer is "protocol overhead" =
wire total - sum of payloads.  So the top of the stack IS the measured wire
rate, and every layer under it is a measurement, not an estimate.

Topic names: the arrays are indexed in dds_topics.yaml order and the message
carries only a hash of the name list (topic_list_hash).  The names are looked up
in dds_topic_maps.json beside this file; an unknown hash is resolved by hashing
the dds_topics.yaml files in the known PX4 trees (DDS_YAML_CANDIDATES, plus
$MAVTERMINAL_DDS_YAML), and the match is saved.  Register a tree by hand with
    python3 ulog_link.py --register path/to/dds_topics.yaml

MAVLink: `telemetry_status`, which every PX4 logs at 1 Hz per instance.  Two
limits apply and they are usually NOT the same:

  * the UART capacity, SER_<port>_BAUD / 10 -- what the wire could carry
  * data_rate, the budget MAVLink schedules its streams against (MAV_x_RATE, or
    half the capacity when that is 0).  When the streams want more than the
    budget, MAVLink scales them all down and reports rate_multiplier < 1.

The panel is drawn as % of the UART capacity, with each instance's budget as a
dashed line, so a link pinned at a budget far below 100% reads as what it is: a
parameter limit, not a full link.  telemetry_status does not say which port an
instance is on, so it is matched to MAV_x_CONFIG by its data_rate; an instance
that cannot be matched (USB, typically) is drawn as % of its own budget and
labelled so.  For the Doodle radio the UART is TELEM1 to the companion, which
forwards over Ethernet -- the radio's own RF rate is not in the log.

There is no per-message breakdown in the log; live, `mavlink status streams`
in the nsh shell gives one.

Acronyms: DDS = Data Distribution Service (the ROS 2 transport), uXRCE-DDS = the
PX4 client that bridges uORB to it, XRCE = eXtremely Resource Constrained
Environments (the wire protocol), uORB = PX4's internal publish/subscribe bus,
MAVLink = the telemetry protocol, 8N1 = 8 data bits, no parity, 1 stop bit,
CRC = cyclic redundancy check, SE3 = the companion's geometric controller.
"""
import glob
import json
import os
import sys

import numpy as np

from ulog_common import (C_ARMED, C_BAD, C_INK, C_MUTED, C_SURFACE, PlotCtx, Series, _get, _style_axis, _time_min,
                         add_mouse_navigation, armed_spans, check_panel,
                         draw_armed, draw_band_rows, draw_mode_changes,
                         duration_min, has_topic, inst_color, mode_changes,
                         mode_key, nav_hint, spans_from_bool, style_time_axis,
                         window_values)

LINK_TOPICS = ["dds_link_status", "telemetry_status", "actuator_armed",
               "vehicle_status"]

HERE = os.path.dirname(os.path.abspath(__file__))
MAPS_FILE = os.path.join(HERE, "dds_topic_maps.json")

# PX4 trees whose dds_topics.yaml may have built the firmware that wrote a log.
DDS_YAML_CANDIDATES = [
    "~/PX4-Autopilot-v1.18-custom/src/modules/uxrce_dds_client/dds_topics.yaml",
    "~/mavericks/src/px4/src/modules/uxrce_dds_client/dds_topics.yaml",
    "~/PX4-Autopilot/src/modules/uxrce_dds_client/dds_topics.yaml",
]

# Layers drawn individually in a stack; the rest are summed into "other topics".
# Eight distinct hues is about what stays tellable apart in a filled stack.
STACK_TOP_N = 8
STACK_COLORS = ["#2a78d6", "#d81b60", "#43a047", "#fb8c00", "#8e24aa",
                "#00838f", "#6d4c41", "#3f51b5"]
C_OTHER = "#9e9d97"
C_OVERHEAD = "#cfcdc6"
C_WIRE = C_INK
C_LINK = "#1baf7a"     # aqua, as the MAVLink traces in ulog_cpu
C_CAP = C_BAD

# "Busy" threshold for the band: above this a 1 s average is hiding bursts that
# fill the UART, and the output stream starts dropping samples.
BUSY_PCT = 80.0

# --- layout, in inches (see ulog_accel for why this is not in fractions) -----
PANEL_IN = {"out": 2.6, "in": 2.1, "mav": 1.9}
GAP_IN = 0.48
TOP_IN = 0.95
BOTTOM_IN = 1.35
BAND_ROW_IN = 0.40
BAND_PAD_IN = 0.30
BAND_MIN_IN = 1.25
BAND_MAX_IN = 6.0
MIN_EVENT_FRAC = 0.003
PAGE_PX_PER_IN = 78


# --- topic names ---------------------------------------------------------------

def fnv1a_32(text):
    """Same hash the firmware's dds_topics.h.em computes at build time."""
    h = 0x811C9DC5
    for b in text.encode("utf-8"):
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def topic_lists(yaml_path, namespace=""):
    """(tx names, rx names) in the order generate_dds_topics.py emits them.

    Mirrors process_message_type / process_message_instance: an optional
    namespace prefix, and an explicit `instance` appended to the topic name."""
    import yaml
    with open(yaml_path) as f:
        m = yaml.safe_load(f) or {}

    def name(entry, with_instance):
        t = entry["topic"]
        t = f"/{namespace}{t}" if namespace else t
        if with_instance and "instance" in entry:
            t = f"{t}{entry['instance']}"
        return t

    tx = [name(p, True) for p in (m.get("publications") or [])]
    rx = ([name(s, False) for s in (m.get("subscriptions") or [])]
          + [name(s, False) for s in (m.get("subscriptions_multi") or [])])
    return tx, rx


def list_hash(tx, rx):
    return fnv1a_32("\n".join(tx) + "\n#\n" + "\n".join(rx))


def _load_maps():
    try:
        with open(MAPS_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_maps(maps):
    try:
        tmp = MAPS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(maps, f, indent=1, sort_keys=True)
        os.replace(tmp, MAPS_FILE)
    except OSError:
        pass          # a read-only checkout still plots, it just re-searches


def register(yaml_path, maps=None):
    """Hash one dds_topics.yaml and store its name lists.  Returns the key."""
    tx, rx = topic_lists(yaml_path)
    key = f"{list_hash(tx, rx):08x}"
    maps = _load_maps() if maps is None else maps
    maps[key] = {"tx": tx, "rx": rx, "source": os.path.abspath(yaml_path)}
    _save_maps(maps)
    return key


def topic_names(topic_hash, n_tx, n_rx):
    """(tx names, rx names, found) for a log's topic_list_hash.

    Falls back to positional names ("tx 3") so an unknown firmware still plots
    with the right numbers, just without the labels."""
    key = f"{int(topic_hash) & 0xFFFFFFFF:08x}"
    maps = _load_maps()
    if key not in maps:
        paths = [os.path.expanduser(p) for p in DDS_YAML_CANDIDATES]
        env = os.environ.get("MAVTERMINAL_DDS_YAML")
        if env:
            paths = glob.glob(os.path.expanduser(env)) + paths
        for p in paths:
            if not os.path.isfile(p):
                continue
            try:
                tx, rx = topic_lists(p)
            except Exception:
                continue
            if f"{list_hash(tx, rx):08x}" == key:
                maps[key] = {"tx": tx, "rx": rx, "source": p}
                _save_maps(maps)
                break
    entry = maps.get(key)
    if entry and len(entry["tx"]) == n_tx and len(entry["rx"]) == n_rx:
        return entry["tx"], entry["rx"], True
    return ([f"tx {i}" for i in range(n_tx)], [f"rx {i}" for i in range(n_rx)],
            False)


def short_name(topic):
    """'/fmu/out/vehicle_odometry' -> 'vehicle_odometry'."""
    return topic.rstrip("/").split("/")[-1]


# --- DDS panels -----------------------------------------------------------------

def _arr(d, name):
    return np.asarray(d.data.get(name, []), dtype=float)


def _dds_direction(ulog, d, t, direction, names, cap):
    """Series for one direction plus the stack layers drawn under them.

    Returns (series, layers) where layers is [(label, color, y_pct)] bottom-up.
    Everything is % of capacity so headroom reads directly off the axis."""
    pre = "topic_tx" if direction == "tx" else "topic_rx"
    wire = _arr(d, f"{direction}_rate")
    n = len(names)
    per = []
    for i in range(n):
        y = _arr(d, f"{pre}_rate[{i}]")
        hz = _arr(d, f"{pre}_sample_rate[{i}]")
        if y.size and np.nanmax(y) > 0:
            per.append((names[i], y, hz))
    per.sort(key=lambda p: -float(np.nanmean(p[1])))

    pct = 100.0 / cap
    group = "out" if direction == "tx" else "in"
    series, layers = [], []
    payload = np.zeros_like(wire)
    for k, (name, y, hz) in enumerate(per):
        payload = payload + y
        mean_hz = float(np.nanmedian(hz)) if hz.size else 0.0
        label = f"{short_name(name)}  ({mean_hz:.0f} Hz)"
        if k < STACK_TOP_N:
            color = STACK_COLORS[k]
            layers.append((label, color, y * pct))
            series.append(Series(f"dds_link_status.{pre}_rate.{short_name(name)}",
                                 label, t, y * pct, group, color, lw=1.2,
                                 visible=False))
    if len(per) > STACK_TOP_N:
        rest = sum(p[1] for p in per[STACK_TOP_N:])
        label = f"other topics ({len(per) - STACK_TOP_N})"
        layers.append((label, C_OTHER, rest * pct))
        series.append(Series(f"dds_link_status.{pre}_rate.other", label, t,
                             rest * pct, group, C_OTHER, lw=1.2, visible=False))
    overhead = np.clip(wire - payload, 0.0, None)
    layers.append(("protocol overhead", C_OVERHEAD, overhead * pct))
    series.append(Series(f"dds_link_status.{direction}_overhead",
                         "protocol overhead (framing, session, acks)", t,
                         overhead * pct, group, "#8c8a83", lw=1.0, visible=False))
    series.append(Series(f"dds_link_status.{direction}_rate",
                         f"wire total {'out' if direction == 'tx' else 'in'} "
                         f"(measured)", t, wire * pct, group, C_WIRE, lw=1.5,
                         visible=True, zorder=6))
    return series, layers


def _dds_series(ulog, ctx):
    d = _get(ulog, "dds_link_status")
    if d is None:
        ctx.note("no dds_link_status in this log -- the firmware does not report "
                 "DDS link usage (needs the grazer custom v1.18 build); the "
                 "MAVLink panel is still drawn")
        return None
    t = _time_min(ulog, d)
    cap = float(np.nanmedian(_arr(d, "data_rate"))) if "data_rate" in d.data else 0.0
    n_tx = int(np.nanmax(_arr(d, "num_tx_topics"))) if "num_tx_topics" in d.data else 0
    n_rx = int(np.nanmax(_arr(d, "num_rx_topics"))) if "num_rx_topics" in d.data else 0
    h = int(_arr(d, "topic_list_hash")[-1]) if "topic_list_hash" in d.data else 0
    tx_names, rx_names, found = topic_names(h, n_tx, n_rx)
    if not found:
        ctx.note(f"DDS topic list {h & 0xFFFFFFFF:08x} is not in "
                 f"dds_topic_maps.json and no known dds_topics.yaml matches it -- "
                 f"topics are shown by index; register the firmware's yaml with "
                 f"`python3 ulog_link.py --register <dds_topics.yaml>`")
    if cap <= 0:
        # UDP (SITL) has no fixed capacity; percentages of it mean nothing, so
        # fall back to a nominal 921600-baud link and say so.
        cap = 92160.0
        ctx.note("dds_link_status has no link capacity (UDP transport) -- "
                 "percentages are of a nominal 921600-baud serial link")
        nominal = True
    else:
        nominal = False
    tx_s, tx_layers = _dds_direction(ulog, d, t, "tx", tx_names, cap)
    rx_s, rx_layers = _dds_direction(ulog, d, t, "rx", rx_names, cap)
    return dict(d=d, t=t, cap=cap, nominal=nominal, tx=tx_s, rx=rx_s,
                tx_layers=tx_layers, rx_layers=rx_layers)


# --- MAVLink panel -----------------------------------------------------------------

LINK_TYPES = {0: "generic", 1: "Ubiquiti", 2: "wire", 3: "USB", 4: "Iridium"}

# MAV_x_CONFIG / UXRCE_DDS_CFG serial port codes -> the baud-rate parameter
PORT_BAUD = {101: ("TELEM1", "SER_TEL1_BAUD"), 102: ("TELEM2", "SER_TEL2_BAUD"),
             103: ("TELEM3", "SER_TEL3_BAUD"), 104: ("TELEM4", "SER_TEL4_BAUD"),
             201: ("GPS1", "SER_GPS1_BAUD"), 202: ("GPS2", "SER_GPS2_BAUD"),
             203: ("GPS3", "SER_GPS3_BAUD"), 300: ("RC", "SER_RC_BAUD")}


def _mav_ports(ulog):
    """[(x, port name, capacity B/s, budget B/s)] for every enabled MAV_x."""
    p = ulog.initial_parameters
    out = []
    for x in range(8):
        cfg = int(p.get(f"MAV_{x}_CONFIG", 0) or 0)
        if cfg not in PORT_BAUD:
            continue
        name, baud_param = PORT_BAUD[cfg]
        baud = float(p.get(baud_param, 0) or 0)
        if baud <= 0:
            continue
        cap = baud / 10.0
        rate = float(p.get(f"MAV_{x}_RATE", 0) or 0)
        out.append((x, name, cap, rate if rate > 0 else cap / 2.0))
    return out


def _instances(ulog, topic):
    return sorted({x.multi_id for x in ulog.data_list if x.name == topic})


def _mav_series(ulog, ctx):
    """tx/rx of each MAVLink instance as % of its UART (or, unmatched, budget)."""
    series = []
    ports = _mav_ports(ulog)
    for m in _instances(ulog, "telemetry_status"):
        d = _get(ulog, "telemetry_status", m)
        if d is None or "data_rate" not in d.data:
            continue
        t = _time_min(ulog, d)
        budget = _arr(d, "data_rate")
        ok = budget > 0
        if not ok.any():
            continue
        kind = LINK_TYPES.get(int(_arr(d, "type")[-1]), "?") if "type" in d.data else "?"
        b = float(np.nanmedian(budget[ok]))
        color = inst_color(m)
        match = [pt for pt in ports if abs(pt[3] - b) < 0.5]
        if len(match) == 1:
            x, port, cap, _budget = match[0]
            where = f"MAV_{x} on {port}"
            denom = np.full_like(budget, cap)
            series.append(Series(f"telemetry_status[{m}].data_rate",
                                 f"MAVLink {m} budget ({b / 1000:.1f} kB/s of "
                                 f"{cap / 1000:.0f})", t, 100.0 * budget / cap,
                                 "mav", color, ls="--", lw=1.0, alpha=0.8,
                                 visible=True))
        else:
            where = f"{kind}, % of budget {b / 1000:.1f} kB/s -- port unknown"
            denom = np.where(ok, budget, np.nan)
            ctx.note(f"MAVLink instance {m} (data_rate {b:.0f} B/s) matches no "
                     f"single MAV_x_CONFIG port, so it is drawn as % of its own "
                     f"budget, not of a UART")
        for key, word, ls in (("tx_rate_avg", "tx", "-"), ("rx_rate_avg", "rx", ":")):
            y = _arr(d, key)
            if not y.size:
                continue
            series.append(Series(f"telemetry_status[{m}].{key}",
                                 f"MAVLink {m} {word}  ({where})", t,
                                 100.0 * y / denom, "mav", color, ls=ls, lw=1.4,
                                 visible=True))
        if "rate_multiplier" in d.data:
            rm = _arr(d, "rate_multiplier")
            if rm.size and np.nanmin(rm) < 0.999:
                ctx.note(f"MAVLink instance {m} throttled its streams to "
                         f"{100 * np.nanmedian(rm):.0f}% (median) of their "
                         f"configured rates: they want more than the "
                         f"{b:.0f} B/s budget")
    if not series:
        ctx.note("no telemetry_status with a data rate in this log -- no MAVLink "
                 "usage to show")
    return series


# --- band ------------------------------------------------------------------------

def _counter_spans(t, counter, hold):
    c = np.asarray(counter, dtype=float)
    return [(float(t[i]), float(t[i]) + hold)
            for i in np.flatnonzero(np.diff(c) > 0) + 1]


def _band_rows(ulog, dds, hold):
    rows = [("armed", armed_spans(ulog), C_ARMED)]
    if dds is not None:
        d, t = dds["d"], dds["t"]
        if "connected" in d.data:
            rows.append(("DDS connected", spans_from_bool(t, _arr(d, "connected") > 0.5),
                         C_LINK))
            rows.append(("DDS NOT connected",
                         spans_from_bool(t, _arr(d, "connected") < 0.5), C_BAD))
        if "session_count" in d.data:
            rows.append(("DDS session restarted",
                         _counter_spans(t, d.data["session_count"], hold), C_BAD))
        if "tx_samples_dropped" in d.data:
            rows.append(("DDS out: samples DROPPED (stream full)",
                         _counter_spans(t, d.data["tx_samples_dropped"], hold), C_BAD))
        if "tx_send_errors" in d.data:
            rows.append(("DDS out: UART write failed",
                         _counter_spans(t, d.data["tx_send_errors"], hold), C_BAD))
        busy = 100.0 * _arr(d, "tx_rate") / dds["cap"]
        rows.append((f"DDS out > {BUSY_PCT:.0f}% of link",
                     spans_from_bool(t, busy > BUSY_PCT), C_BAD))
        busy = 100.0 * _arr(d, "rx_rate") / dds["cap"]
        rows.append((f"DDS in > {BUSY_PCT:.0f}% of link",
                     spans_from_bool(t, busy > BUSY_PCT), C_BAD))
    for m in _instances(ulog, "telemetry_status"):
        d = _get(ulog, "telemetry_status", m)
        if d is None:
            continue
        t = _time_min(ulog, d)
        for key, label in (("tx_buffer_overruns", "tx buffer overrun"),
                           ("rx_buffer_overruns", "rx buffer overrun"),
                           ("rx_message_lost_count", "rx messages LOST")):
            if key in d.data:
                spans = _counter_spans(t, d.data[key], hold)
                if spans:
                    rows.append((f"MAVLink {m}: {label}", spans, C_BAD))
        if "rate_multiplier" in d.data:
            spans = spans_from_bool(t, _arr(d, "rate_multiplier") < 0.999)
            if spans:
                rows.append((f"MAVLink {m}: throttled its streams", spans, C_BAD))
    return rows


# --- the figure -----------------------------------------------------------------

def _stack(ax, t, layers):
    """Filled stack, bottom-up.  Returns the artists (toggled as one group)."""
    art, base = [], np.zeros_like(t, dtype=float)
    for _label, color, y in layers:
        y = np.nan_to_num(np.asarray(y, dtype=float))
        top = base + y
        art.append(ax.fill_between(t, base, top, color=color, alpha=0.55,
                                   lw=0, step="post", zorder=2))
        base = top
    return art


def build_link(ulog, ctx=None, path=""):
    """The link-bandwidth figure.  Same signature as every plot builder."""
    import matplotlib.pyplot as plt

    ctx = ctx or PlotCtx()
    if not (has_topic(ulog, "dds_link_status") or has_topic(ulog, "telemetry_status")):
        ctx.note("no dds_link_status or telemetry_status in this log")
        return None

    dur = duration_min(ulog) or 1.0
    hold = dur * MIN_EVENT_FRAC
    dds = _dds_series(ulog, ctx)
    mav = _mav_series(ulog, ctx)
    series = (dds["tx"] + dds["rx"] if dds else []) + mav
    if not series:
        return None
    rows = _band_rows(ulog, dds, hold)
    # Without dds_link_status the two DDS panels would be empty boxes.
    keys = (["out", "in"] if dds else []) + (["mav"] if mav else [])
    panel_in = [(k, PANEL_IN[k]) for k in keys]

    band_in = min(max(len(rows) * BAND_ROW_IN + BAND_PAD_IN, BAND_MIN_IN), BAND_MAX_IN)
    fig_h = (TOP_IN + sum(h for _k, h in panel_in) + GAP_IN * len(panel_in)
             + band_in + BOTTOM_IN)
    fig = plt.figure(figsize=(15, fig_h), facecolor=C_SURFACE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(f"logGraph link - {os.path.basename(path)}")

    left, width = 0.260, 0.655

    def _f(inches):
        return inches / fig_h

    rects, bottom = {}, BOTTOM_IN
    rects["band"] = (_f(bottom), _f(band_in))
    bottom += band_in + GAP_IN
    for key, h in reversed(panel_in):
        rects[key] = (_f(bottom), _f(h))
        bottom += h + GAP_IN

    axis_of = {k: fig.add_axes([left, rects[k][0], width, rects[k][1]],
                               facecolor=C_SURFACE) for k in keys}
    ax_band = fig.add_axes([left, rects["band"][0], width, rects["band"][1]],
                           facecolor=C_SURFACE)
    panels = [axis_of[k] for k in keys]
    for a in panels:
        a.sharex(ax_band)

    armed_art = []
    for a in panels + [ax_band]:
        armed_art += draw_armed(a, armed_spans(ulog))

    # 100% = the link is full.  Drawn as a line, not a Series: the axes are
    # fitted to the data, so on a quiet link it is off screen and the usage is
    # readable; the right-hand axis and the subtitle still state the capacity.
    for a in panels:
        a.axhline(100.0, color=C_CAP, lw=1.1, ls="--", alpha=0.8, zorder=1)

    stack_art = []
    if dds:
        stack_art += _stack(axis_of["out"], dds["t"], dds["tx_layers"])
        stack_art += _stack(axis_of["in"], dds["t"], dds["rx_layers"])
        cap = dds["cap"]
        for k in ("out", "in"):
            sec = axis_of[k].secondary_yaxis("right", functions=(
                lambda p, c=cap: p * c / 100.0 / 1000.0,
                lambda kb, c=cap: kb * 1000.0 * 100.0 / c))
            sec.set_ylabel("kB/s", fontsize=9)
            _style_axis(sec, C_MUTED)

    draw_band_rows(ax_band, rows, ylabel="link events",
                   empty_msg="no link status in this log", min_width=hold)

    for s in series:
        (line,) = axis_of[s.group].plot(
            s.t, s.y, color=s.color, ls=s.ls, lw=s.lw, label=s.label,
            drawstyle="steps-post" if s.group != "mav" else "default",
            alpha=s.alpha, zorder=s.zorder if s.zorder is not None else 3)
        line.set_visible(s.visible)
        s.line = line

    mode_art, mode_codes = draw_mode_changes(
        panels + [ax_band], mode_changes(ulog), text_ax=panels[0],
        min_gap=dur * 0.035)

    for a in panels:
        style_time_axis(a, label=False)
        _style_axis(a, C_INK)
    style_time_axis(ax_band)
    ylabels = {"out": "DDS out: FC -> companion\n(% of UART)",
               "in": "DDS in: companion -> FC\n(% of UART)",
               "mav": "MAVLink\n(% of UART, dashed = budget)"}
    for k in keys:
        axis_of[k].set_ylabel(ylabels[k], fontsize=9)

    fig.text(left, 1.0 - _f(0.35), "Link bandwidth: capacity and what uses it",
             color=C_INK, fontsize=13, fontweight="bold", ha="left")
    who = f"{os.path.basename(path)}   |   " if path else ""
    if dds:
        d = dds["d"]
        tx = _arr(d, "tx_rate")
        rx = _arr(d, "rx_rate")
        cap = dds["cap"]
        baud = int(np.nanmedian(_arr(d, "baudrate"))) if "baudrate" in d.data else 0
        link = (f"DDS link {cap / 1000:.1f} kB/s"
                + (" (nominal, UDP)" if dds["nominal"] else f" ({baud} baud 8N1)")
                + f"   |   out {np.nanmedian(tx) / cap * 100:.0f}% median, "
                  f"{np.nanmax(tx) / cap * 100:.0f}% peak   |   in "
                  f"{np.nanmedian(rx) / cap * 100:.0f}% median, "
                  f"{np.nanmax(rx) / cap * 100:.0f}% peak")
    else:
        link = "no DDS link status"
    fig.text(left, 1.0 - _f(0.62), f"{who}{dur:.1f} min   |   {link}",
             color=C_MUTED, fontsize=9, ha="left")

    def _fit(ax, group):
        """Floor at 0: these are rates, and the stack's base must stay on screen."""
        lines = [s.line for s in series if s.group == group]
        v = window_values(ax, lines)
        hi = float(v.max()) if v.size else 1.0
        ax.set_ylim(0.0, max(hi, 1.0) * 1.08)

    def refresh():
        for k in keys:
            _fit(axis_of[k], k)

    extra = []
    if stack_art:
        extra.append(("stacked by topic (shaded)", stack_art, True))
    mode_art += mode_key(fig, left + width, _f(0.10), mode_codes)
    if mode_art:
        extra.append(("mode changes", mode_art, True))
    if armed_art:
        extra.append(("armed (shaded)", armed_art, True))

    cb_top = rects[keys[0]][0] + rects[keys[0]][1]
    cb_bot = rects["band"][0]

    def _anchor(key):
        b, ph = rects[key]
        return (b + ph / 2 - cb_bot) / (cb_top - cb_bot)

    check_panel(fig, [0.012, cb_bot, 0.200, cb_top - cb_bot], series,
                [("out", "ALL DDS out"), ("in", "ALL DDS in"),
                 ("mav", "ALL MAVLink")],
                extra=extra, on_change=refresh,
                anchors={k: _anchor(k) for k in keys})
    refresh()
    add_mouse_navigation(fig, panels + [ax_band],
                         page_scroll=ctx.page_scroll, fixed_y=[ax_band],
                         on_view=refresh)
    fig.text(left, _f(0.32), nav_hint(ctx.page_scroll), color=C_MUTED,
             fontsize=8, ha="left")
    fig._page_height = int(round(fig_h * PAGE_PX_PER_IN))
    return fig


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ulog", nargs="?", help="log to plot")
    ap.add_argument("--register", metavar="YAML",
                    help="add a dds_topics.yaml's topic names to dds_topic_maps.json")
    ap.add_argument("--save", metavar="PNG", help="write the figure instead of showing it")
    a = ap.parse_args(argv)
    if a.register:
        print(f"registered {register(a.register)} from {a.register}")
        if not a.ulog:
            return 0
    if not a.ulog:
        ap.error("give a log, or --register")
    import matplotlib
    if a.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pyulog import ULog
    ctx = PlotCtx()
    fig = build_link(ULog(a.ulog, LINK_TOPICS), ctx, a.ulog)
    for n in ctx.notes:
        print(f"note: {n}")
    if fig is None:
        return 1
    if a.save:
        fig.savefig(a.save, dpi=90, facecolor=fig.get_facecolor())
        print(f"saved {a.save}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
