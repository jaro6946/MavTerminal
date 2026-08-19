#!/usr/bin/env python3
"""ulog_report.py -- render one or more .ulg logs into a single PDF report.

Layout, in order:

    page 1     table of contents, with page numbers
    per log:   a text summary page (or several, if the diagnosis is long)
               then every registered plot, one page each

Why a contents page and not PDF bookmarks: matplotlib's PdfPages has no outline
API at all.  Adding real bookmarks would mean a second library (pypdf) to
post-process the file, which is not worth a dependency for a report you scroll
through -- so the contents page carries the page numbers instead.

Everything runs under the Agg backend, so this works over SSH with no display and
is the toolkit's headless regression test: if the PDF renders for a log, every
plot builder survived that log's topic set.
"""
import contextlib
import io
import os
import time

import numpy as np
from pyulog import ULog

import ulog_plots
from ulog_common import (C_INK, C_MUTED, C_SURFACE, PlotCtx, _get, _time_min,
                         armed_spans, duration_min)

# One page size for the whole document.  The plot builders lay themselves out in
# figure fractions, so they re-flow into this cleanly; mixed page sizes in one PDF
# are legal but read as a mistake.
PAGE_W, PAGE_H = 14.0, 9.0

# Parameters worth having in front of you when reading an altitude plot: they say
# which height source the EKF was configured to trust, and how high the
# rangefinder was allowed to be used.
KEY_PARAMS = ["SYS_HITL", "SYS_AUTOSTART", "EKF2_HGT_REF", "EKF2_BARO_CTRL",
              "EKF2_GPS_CTRL", "EKF2_RNG_CTRL", "EKF2_RNG_A_HMAX",
              "EKF2_GND_EFF_DZ", "SENS_IMU_AUTOCAL", "SDLOG_PROFILE"]

MONO = {"family": "monospace", "fontsize": 8}
LINES_PER_PAGE = 78


def _fmt_dur(minutes):
    return f"{int(minutes)}m {int((minutes % 1) * 60):02d}s"


def _height_source_usage(ulog):
    """Percentage of samples the EKF spent fusing each height source.

    Reported per LOG rather than per armed span because the flags are also
    informative on the ground -- "it never once fused GPS height" is the same
    finding whether or not the props were spinning.
    """
    d = _get(ulog, "estimator_status_flags")
    if d is None:
        return []
    out = []
    for fname, label in (("cs_baro_hgt", "baro"), ("cs_gps_hgt", "GPS"),
                         ("cs_rng_hgt", "rangefinder"), ("cs_ev_hgt", "vision")):
        if fname not in d.data:
            continue
        v = np.asarray(d.data[fname], dtype=float) > 0.5
        if v.size:
            out.append((label, 100.0 * v.mean()))
    return out


def summarize(path, ulog):
    """The text block for a log's summary page(s).

    Reuses ulog_diag.diagnose rather than reimplementing it -- that function
    already decodes the real failsafe cause from failsafe_flags at the moment of
    onset, which is subtle enough that a second implementation would be a second
    thing to get wrong.  It prints, so its stdout is captured.
    """
    lines = []
    st = os.stat(path)
    lines.append(f"{os.path.basename(path)}")
    lines.append(f"  path      {path}")
    lines.append(f"  size      {st.st_size / 1e6:.1f} MB")
    lines.append(f"  modified  {time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}")
    lines.append(f"  duration  {_fmt_dur(duration_min(ulog))}")

    spans = armed_spans(ulog)
    armed_min = sum(b - a for a, b in spans)
    lines.append(f"  armed     {len(spans)} span(s), {_fmt_dur(armed_min)} total")

    usage = _height_source_usage(ulog)
    if usage:
        lines.append("  height source fused: "
                     + ", ".join(f"{k} {v:.0f}%" for k, v in usage))

    # Peak temperature, the headline number from the thermal plot.
    peaks = []
    for d in ulog.data_list:
        for fname in d.data:
            f = fname.lower()
            if "temp" not in f or f.endswith(("_source", "_valid", "_count", "_id")):
                continue
            v = np.asarray(d.data[fname], dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                peaks.append((float(v.max()), f"{d.name}[{d.multi_id}].{fname}"))
    if peaks:
        hi, who = max(peaks)
        lines.append(f"  peak temp {hi:.1f} degC  ({who})")

    params = {k: ulog.initial_parameters[k] for k in KEY_PARAMS
              if k in getattr(ulog, "initial_parameters", {})}
    if params:
        lines.append("  params    "
                     + "  ".join(f"{k}={v}" for k, v in params.items()))

    lines.append("")
    lines.append("-" * 100)
    lines.append("ulog_diag output")
    lines.append("-" * 100)
    try:
        from ulog_diag import diagnose
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            diagnose(path)
        # expandtabs: PX4 STATUSTEXT lines carry literal tabs, and matplotlib's
        # font has no glyph for one -- it warns and draws a missing-glyph box,
        # which silently mangles the column alignment the mono font is here for.
        lines.extend(buf.getvalue().rstrip().expandtabs(4).splitlines())
    except Exception as e:
        # A failing diagnosis must not lose the plots; the report is still useful
        # without it, and the reason is more helpful in the PDF than in a traceback.
        lines.append(f"  (ulog_diag failed: {type(e).__name__}: {e})")
    return lines


def _text_pages(lines, title):
    """Yield figures holding `lines`, paginated.  Monospace, because the
    ulog_diag output is column-aligned and a proportional font destroys it."""
    import matplotlib.pyplot as plt

    for start in range(0, max(len(lines), 1), LINES_PER_PAGE):
        chunk = lines[start:start + LINES_PER_PAGE]
        fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=C_SURFACE)
        fig.text(0.04, 0.965, title, color=C_INK, fontsize=13,
                 fontweight="bold", ha="left", va="top")
        if start:
            fig.text(0.96, 0.965, f"(continued)", color=C_MUTED, fontsize=9,
                     ha="right", va="top")
        fig.text(0.04, 0.925, "\n".join(chunk), color=C_INK, ha="left",
                 va="top", linespacing=1.35, **MONO)
        yield fig


def _toc_page(entries, n_pages):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=C_SURFACE)
    fig.text(0.04, 0.94, "ULog report", color=C_INK, fontsize=20,
             fontweight="bold", ha="left", va="top")
    fig.text(0.04, 0.885,
             f"{len({e[2] for e in entries})} log(s), {n_pages} pages   |   "
             f"generated {time.strftime('%Y-%m-%d %H:%M')}",
             color=C_MUTED, fontsize=10, ha="left", va="top")

    y = 0.82
    last_log = None
    for label, page, logpath, blurb in entries:
        if logpath != last_log:
            y -= 0.018
            last_log = logpath
        indent = 0.04 if blurb is None else 0.075
        weight = "bold" if blurb is None else "normal"
        color = C_INK if blurb is None else C_MUTED
        fig.text(indent, y, label, color=color, fontsize=10.5 if blurb is None else 9.5,
                 fontweight=weight, ha="left", va="top")
        fig.text(0.93, y, str(page), color=color, fontsize=9.5, ha="right", va="top")
        y -= 0.026
        if blurb:
            fig.text(indent + 0.012, y, blurb, color=C_MUTED, fontsize=8,
                     ha="left", va="top", style="italic")
            y -= 0.024
        if y < 0.05:
            break
    return fig


def export_pdf(paths, out_path, ctx=None, log=print):
    """Render every registered plot for every path into one PDF.

    Memory note: the figures are built BEFORE the contents page is written,
    because the contents page needs page numbers and a plot only contributes a
    page if its builder found something to draw in that particular log.  So peak
    memory scales with the number of selected logs.  That is the right trade for
    the intended use (a handful of logs compared side by side); selecting all 42
    HITL logs at once would be a different tool.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    ctx = ctx or PlotCtx()
    paths = [p for p in paths if p]
    if not paths:
        raise ValueError("no logs given to export")

    topics = ulog_plots.all_topics(ctx)
    entries = []          # (label, page_number, logpath, blurb_or_None)
    pages = []            # figures, in order, after the contents page
    page_no = 2           # page 1 is the contents

    for path in paths:
        log(f"  reading {os.path.basename(path)} ...")
        try:
            ulog = ULog(path, message_name_filter_list=topics)
        except Exception as e:
            log(f"  !! {os.path.basename(path)}: {type(e).__name__}: {e}")
            continue

        entries.append((os.path.basename(path), page_no, path, None))
        text = summarize(path, ulog)
        for fig in _text_pages(text, f"{os.path.basename(path)} - summary"):
            pages.append(fig)
            page_no += 1

        for spec in ulog_plots.PLOTS:
            # Each plot gets its OWN ctx so one plot's notes don't leak into the
            # next plot's caption, but the user's options carry across.
            sub = PlotCtx(smooth=ctx.smooth, use_abs=ctx.use_abs,
                          rate_src=ctx.rate_src, adds=list(ctx.adds),
                          debias=ctx.debias, page_scroll=False)
            try:
                fig = spec.build(ulog, sub, path)
            except Exception as e:
                # One broken plot must not cost you the rest of the report.
                log(f"  !! {spec.title} failed on {os.path.basename(path)}: "
                    f"{type(e).__name__}: {e}")
                continue
            for n in sub.notes:
                log(f"     note: {n}")
            if fig is None:
                log(f"     {spec.title}: nothing plottable in this log, skipped")
                continue
            _stamp(fig, os.path.basename(path), sub.notes)
            pages.append(fig)
            entries.append((spec.title, page_no, path, spec.blurb))
            page_no += 1

    if not pages:
        raise RuntimeError("nothing could be rendered from the given log(s)")

    with PdfPages(out_path) as pdf:
        toc = _toc_page(entries, page_no - 1)
        pdf.savefig(toc, facecolor=C_SURFACE)
        plt.close(toc)
        for fig in pages:
            fig.set_size_inches(PAGE_W, PAGE_H)
            pdf.savefig(fig, facecolor=C_SURFACE)
            plt.close(fig)          # free as we go, not at the end
        d = pdf.infodict()
        d["Title"] = f"ULog report - {', '.join(os.path.basename(p) for p in paths)}"
        d["Creator"] = "mavTerminal logGraph"

    log(f"  wrote {out_path}  ({page_no - 1} pages, {len(paths)} log(s))")
    return out_path


def _stamp(fig, name, notes):
    """Running header on every plot page, plus whatever the builder had to say.

    In a multi-log report the plots look alike, so a page without the filename on
    it is a page you cannot attribute.  The notes matter even more on paper than
    on screen: "rangefinder valid 33% of samples" is the difference between
    reading a gappy trace as a broken sensor and as a correctly-gated one, and on
    screen it is printed to the terminal, which a PDF reader does not have.
    """
    fig.text(0.995, 0.995, name, color=C_MUTED, fontsize=8, ha="right", va="top")
    if notes:
        shown = notes[:4]
        tail = f"   (+{len(notes) - 4} more)" if len(notes) > 4 else ""
        fig.text(0.02, 0.008, "notes: " + " · ".join(shown) + tail,
                 color=C_MUTED, fontsize=6.5, ha="left", va="bottom", wrap=True)
