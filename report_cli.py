#!/usr/bin/env python3
"""report_cli.py -- author, check and render logGraph reports without the GUI.

The Report tab is for building a report by hand.  This is for the other way
round: writing one from a description ("compare IMU temperature across the three
heat-sink soaks"), confirming it actually resolves against the logs, and looking
at the result -- none of which needs a display, and all of which was impossible
while the drawing lived inside a Qt widget.

    report_cli.py channels <log> [--grep temp] [--all]
    report_cli.py list
    report_cli.py validate <report.json>
    report_cli.py render <report.json> -o out.pdf

`validate` matters more than it looks.  A field reference that does not exist in
a log is not an error anywhere -- the graph simply draws one fewer line, and the
person who finds out is whoever opens the tab.  So anything that authors a report
without looking at it should validate it first.

Acronyms: CLI = command-line interface, ULog = PX4's binary log format,
PDF = portable document format.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")                   # before anything imports pyplot

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from report_model import (ALIGNMENTS, Report, list_reports, reports_dir)
from report_render import (STAT_COLS, assign_axes, build_figure, fit_value_axes,
                           fmt_stat, gather_series, short_ref, stats_of,
                           window_of)
from ulog_cache import parse_ulog
from ulog_common import C_INK, VARY, field_inventory, parse_ref

PAGE = (14.0, 9.0)          # matches ulog_report.py, so the two read as one tool


def _default_root():
    return os.path.expanduser("~/jacobAtGar/Log Analysis")


def _find_log(name, root=None):
    """A log by path, or by basename anywhere under the library root."""
    if os.path.exists(name):
        return name
    root = root or _default_root()
    for base, _dirs, files in os.walk(root):
        for f in files:
            if f == name or os.path.splitext(f)[0] == name:
                return os.path.join(base, f)
    return None


# --- channels ----------------------------------------------------------------

def cmd_channels(args):
    path = _find_log(args.log, args.root)
    if path is None:
        print(f"no such log: {args.log}", file=sys.stderr)
        return 2
    ulog = parse_ulog(path, None)
    inv = field_inventory(ulog)
    needle = (args.grep or "").lower()
    rows = [r for r in inv
            if (args.all or r[4] == VARY) and needle in r[0].lower()]
    print(f"# {os.path.basename(path)}: {len(inv)} fields, "
          f"{sum(1 for r in inv if r[4] == VARY)} varying, showing {len(rows)}")
    for ref, topic, mid, name, kind in rows:
        d = next((d for d in ulog.data_list
                  if d.name == topic and d.multi_id == mid), None)
        n = len(d.data["timestamp"]) if d is not None else 0
        flag = "" if kind == VARY else f"  ({kind})"
        print(f"{ref:58s} {n:8,d} samples{flag}")
    return 0


# --- list --------------------------------------------------------------------

def cmd_list(args):
    found = list_reports(args.root)
    if not found:
        print(f"no reports in {reports_dir(args.root)}")
        return 0
    for path, title in found:
        try:
            r = Report.load(path)
            detail = f"{len(r.logs)} log(s), {len(r.graphs)} graph(s)"
        except (OSError, ValueError) as e:
            detail = f"unreadable: {e}"
        print(f"{title:44s} {detail:26s} {path}")
    return 0


# --- validate ----------------------------------------------------------------

def _load_logs(report, root, log=print):
    """{basename: ULog} for every log the report resolves.  Reports what it can't."""
    known = []
    base = root or _default_root()
    for d, _dirs, files in os.walk(base):
        known += [os.path.join(d, f) for f in files if f.endswith(".ulg")]
    resolved = report.resolved(known)
    ulogs, missing = {}, []
    for ref in report.logs:
        p = resolved.get(ref.name)
        if p is None:
            missing.append(ref.name)
            continue
        ulogs[ref.name] = parse_ulog(p, None)
        if os.path.basename(p) != ref.name:
            log(f"  note: {ref.name} resolved to {os.path.basename(p)} (renamed)")
    return ulogs, missing


def cmd_validate(args):
    report = Report.load(args.report)
    print(f"report: {report.title or '(untitled)'}   {args.report}")
    ulogs, missing = _load_logs(report, args.root)
    for name in missing:
        print(f"  !! MISSING LOG: {name}")
    print(f"  {len(ulogs)} of {len(report.logs)} log(s) resolved")

    bad = len(missing)
    for g in report.graphs:
        series, problems = gather_series(g, ulogs)
        drew = {s["ref"] for s in series}
        print(f"\n  graph {g.id}: {g.title or '(untitled)'}")
        print(f"    align={g.align} ({ALIGNMENTS.get(g.align, '?')})  "
              f"logs={len(g.logs)}  channels={len(g.fields)}  series={len(series)}")
        for ref in g.fields:
            try:
                parse_ref(ref)
            except ValueError:
                print(f"    !! MALFORMED: {ref}")
                bad += 1
                continue
            n = sum(1 for s in series if s["ref"] == ref)
            mark = "ok " if ref in drew else "!! "
            if ref not in drew:
                bad += 1
            print(f"    {mark}{short_ref(ref):46s} in {n}/{len(g.logs)} log(s)")
        for p in problems:
            print(f"    - {p}")
    print(f"\n{'OK' if not bad else str(bad) + ' PROBLEM(S)'}")
    return 1 if bad else 0


# --- render ------------------------------------------------------------------

PAGE_LINES = 70             # what fits under the heading at 8.5 pt on this page


def _text_page(pdf, title, lines):
    """A monospace page of text, SPILLING onto continuation pages.

    It used to render `lines[:70]` and stop, which is the worst way to run out
    of room: the tail of a graph's note -- reliably the conclusions, since they
    are written last -- vanished from the PDF with nothing to show it had, and
    the only way to find out was to count the lines yourself.  Returns the
    number of pages written so the run can report a true page count."""
    chunks = [lines[i:i + PAGE_LINES]
              for i in range(0, len(lines), PAGE_LINES)] or [[]]
    for n, chunk in enumerate(chunks):
        fig = Figure(figsize=PAGE)
        fig.text(0.06, 0.94, title if n == 0 else f"{title}  (continued)",
                 fontsize=15, weight="bold", color=C_INK)
        fig.text(0.06, 0.90, "\n".join(chunk), fontsize=8.5, va="top",
                 family="monospace", color=C_INK)
        pdf.savefig(fig)
        fig.clear()
    return len(chunks)


def cmd_render(args):
    report = Report.load(args.report)
    ulogs, missing = _load_logs(report, args.root)
    out = args.out or os.path.splitext(args.report)[0] + ".pdf"

    with PdfPages(out) as pdf:
        head = [f"logs ({len(report.logs)}):"]
        head += [f"    {r.name}" + ("   !! MISSING" if r.name in missing else "")
                 for r in report.logs]
        if report.notes.strip():
            head += ["", "notes:"] + ["    " + l for l in
                                      report.notes.strip().splitlines()]
        head += ["", f"graphs ({len(report.graphs)}):"]
        head += [f"    {i+1}. {g.title or '(untitled)'}"
                 for i, g in enumerate(report.graphs)]
        pages = _text_page(pdf, report.title or "Untitled report", head)

        for g in report.graphs:
            series, problems = gather_series(g, ulogs)
            auto = assign_axes(g, series)
            # Graph.height scales the PAGE, not the axes inside a fixed page:
            # a taller axis with the same margins is what "give this graph more
            # room" means, and PdfPages is happy to hold pages of mixed size.
            page = (PAGE[0], PAGE[1] * getattr(g, "height", 1.0))
            fig, ax, axr, lines = build_figure(g, series, problems,
                                               figsize=page, auto=auto)
            if not g.xlim:
                span = window_of(series)
                if span:
                    ax.set_xlim(*span)
            fit_value_axes([ax, axr], lines)
            pdf.savefig(fig)
            fig.clear()
            pages += 1

            xlim = g.xlim or window_of(series)
            rows = [f"{'log':<26} {'channel':<34} " +
                    " ".join(f"{c:>12}" for c in STAT_COLS),
                    "-" * (26 + 34 + 13 * len(STAT_COLS))]
            for s in series:
                st = stats_of(s["t"], s["y"], xlim)
                rows.append(f"{os.path.splitext(s['log'])[0][:25]:<26} "
                            f"{short_ref(s['ref'])[:33]:<34} " +
                            " ".join(f"{fmt_stat(st[c]):>12}" for c in STAT_COLS))
            if g.notes.strip():
                rows += ["", "notes:"] + ["    " + l for l in
                                          g.notes.strip().splitlines()]
            if problems:
                rows += ["", "problems:"] + ["    " + p for p in problems]
            pages += _text_page(
                pdf, f"{g.title or '(untitled)'} — statistics"
                     + ("" if not xlim else
                        f"  ({xlim[0]:.2f} – {xlim[1]:.2f} min)"), rows)
    print(f"wrote {out}  ({len(report.graphs)} graph(s), {pages} pages)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="report_cli", description=__doc__.split("\n")[0])
    p.add_argument("--root", default=None,
                   help="log library root (default ~/jacobAtGar/Log Analysis)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("channels", help="list a log's plottable channels")
    c.add_argument("log")
    c.add_argument("--grep", help="substring filter on topic[i].field")
    c.add_argument("--all", action="store_true",
                   help="include constant/empty channels (hidden by default)")
    c.set_defaults(fn=cmd_channels)

    c = sub.add_parser("list", help="list saved reports")
    c.set_defaults(fn=cmd_list)

    c = sub.add_parser("validate", help="check a report resolves against the logs")
    c.add_argument("report")
    c.set_defaults(fn=cmd_validate)

    c = sub.add_parser("render", help="render a report to PDF")
    c.add_argument("report")
    c.add_argument("-o", "--out")
    c.set_defaults(fn=cmd_render)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
