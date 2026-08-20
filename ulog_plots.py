#!/usr/bin/env python3
"""ulog_plots.py -- the registry of plots the browser and the PDF exporter draw.

Adding a plot is one entry here plus one module exposing
``build(ulog, ctx, path) -> Figure | None``.  Nothing else needs to change: the
browser stacks whatever is in PLOTS, the PDF exporter pages through it, and the
topic filter list is derived from it.

The `topics` field exists so ONE ULog parse can serve every plot.  Measured on the
137 MB d05a88e3 log, filtering does NOT save wall-clock time (1.9 s either way --
pyulog walks the whole file regardless); what it saves is memory, by not
materialising ~100 topics of arrays nobody plots.  The real win here is opening
the file once instead of once per plot, which is linear in the number of plots and
is the thing that would actually get slow as the registry grows.
"""
from dataclasses import dataclass
from typing import Callable, List

import ulog_accel
import ulog_alt
import ulog_cpu
import ulog_graph
import ulog_localz
from ulog_common import PlotCtx  # re-exported: callers build one and pass it on

__all__ = ["PlotSpec", "PLOTS", "PlotCtx", "all_topics", "by_key"]


@dataclass(frozen=True)
class PlotSpec:
    key: str                # stable id, used in CLI flags and cached state
    title: str              # sidebar entry and PDF section heading
    topics: List[str]       # contributed to the union filter list
    build: Callable         # (ulog, ctx, path) -> Figure or None
    blurb: str = ""         # one line, shown in the PDF table of contents
    # Pixel height this plot wants in the browser's scrolling page.  Not one
    # constant for all plots: the thermal plot is a single axis, the altitude
    # plot stacks four, and giving them the same strip squashes the latter into
    # unreadability.
    height: int = 520


PLOTS = [
    PlotSpec(
        key="thermal",
        title="Thermal / GPS",
        topics=ulog_graph.THERMAL_TOPICS,
        build=ulog_graph.build_thermal,
        blurb="every temperature channel, satellite count, and dT/dt on one time axis",
        height=520,
    ),
    PlotSpec(
        key="altitude",
        title="Altitude estimation",
        topics=ulog_alt.ALT_TOPICS,
        build=ulog_alt.build_altitude,
        blurb="GPS / barometer / rangefinder against the EKF's fused altitude, "
              "with residuals, innovations and the fusion source",
        height=780,
    ),
    PlotSpec(
        key="localz",
        title="Local z / EKF instance",
        topics=ulog_localz.LOCALZ_TOPICS,
        build=ulog_localz.build_local_z,
        blurb="local position z and each instance's own z and ref_alt, shaded by "
              "which EKF instance the selector had primary",
        height=780,
    ),
    PlotSpec(
        key="accel",
        title="Accelerometer / calibration",
        topics=ulog_accel.ACCEL_TOPICS,
        build=ulog_accel.build_accel,
        blurb="per-IMU accelerometer, the EKF's bias estimate against the exact "
              "preflight arming threshold, the thermal-compensation offset, and "
              "every accel fault flag -- shaded by primary EKF instance",
        # Fallback only: build_accel sizes its own figure from the fault-row
        # count and sets fig._page_height to match.
        height=1000,
    ),
    PlotSpec(
        key="cpu",
        title="Processor load",
        topics=ulog_cpu.CPU_TOPICS,
        build=ulog_cpu.build_cpu,
        blurb="CPU and RAM, plus the things that show the board struggling: EKF "
              "time slip, IMU publish rates, SD buffer pressure and the MAVLink "
              "rate throttle",
        height=900,
    ),
]


def all_topics(ctx=None):
    """The union filter list: every plot's topics, plus any --add channels.

    Sorted and de-duplicated because pyulog does a linear scan per message name
    and a duplicate is wasted work on every one of the log's millions of records.
    """
    topics = set()
    for spec in PLOTS:
        topics.update(spec.topics)
    for ref in (ctx.adds if ctx else []):
        try:
            from ulog_common import parse_ref
            topics.add(parse_ref(ref)[0])
        except ValueError:
            pass          # bad --add refs are reported by the thermal builder
    return sorted(topics)


def by_key(key):
    for spec in PLOTS:
        if spec.key == key:
            return spec
    return None
