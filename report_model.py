#!/usr/bin/env python3
"""report_model.py -- what a saved comparison report IS, on disk and in memory.

A report is a small JSON document: which logs it is about, a free-text note, and
an ordered list of graphs, each naming its own subset of those logs, the channels
to draw, how to line the logs up in time, and its own note.  No Qt in here, so
the whole format can be exercised without a display.

It is stored beside the logs (``<Log Analysis>/reports/``) rather than in the
browser's config, because that is where the thing it describes lives: copy the
log folder to another machine and the reports come with it, the same bargain the
per-log ``_notes.txt`` sidecars already make.

Acronyms: JSON = JavaScript object notation, ULog = PX4's binary log format.
"""
import json
import os
import re
import time

__all__ = ["Report", "Graph", "LogRef", "reports_dir", "list_reports",
           "slugify", "ALIGNMENTS", "DEFAULT_ALIGN"]

SCHEMA = 1
REPORT_EXT = ".json"

# How several logs are laid over one another on the time axis.  There is no
# universally right answer -- bench soak tests want the file's own start, flight
# comparisons want the moment the vehicle armed -- so it is per graph.
ALIGNMENTS = {
    "log_start": "since log start",
    "first_arm": "since first arm",
    "absolute": "absolute clock",
}
DEFAULT_ALIGN = "log_start"

DEFAULT_ROOT = os.path.expanduser("~/jacobAtGar/Log Analysis")


def reports_dir(root=None, create=False):
    """Where reports live: ``<Log Analysis>/reports``.

    Falls back to the browser's own config directory when the log folder will not
    take a write -- a mounted card or a read-only share -- so a report is never
    lost merely because of where the logs happen to sit.  Same fallback the
    per-log notes sidecars use."""
    primary = os.path.join(root or DEFAULT_ROOT, "reports")
    if not create:
        return primary
    try:
        os.makedirs(primary, exist_ok=True)
        probe = os.path.join(primary, ".write_probe")
        with open(probe, "w") as f:
            f.write("")
        os.remove(probe)
        return primary
    except OSError:
        fallback = os.path.join(
            os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
            "mavterminal", "reports")
        os.makedirs(fallback, exist_ok=True)
        return fallback


def slugify(title, default="untitled"):
    """A filename that still resembles the title someone typed."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", (title or "").strip()).strip("_")
    return (slug[:60] or default).lower()


def list_reports(root=None):
    """Every saved report: ``[(path, title)]``, by title.

    Reads each file to get its title, because the filename is fixed at creation
    and the title is not -- see Report.save."""
    out = []
    for d in {reports_dir(root), reports_dir("")}:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if not name.endswith(REPORT_EXT):
                continue
            path = os.path.join(d, name)
            try:
                with open(path) as f:
                    title = (json.load(f).get("title") or "").strip()
            except (OSError, ValueError):
                continue
            out.append((path, title or os.path.splitext(name)[0]))
    out.sort(key=lambda r: r[1].lower())
    return out


class LogRef:
    """A log a report points at, recorded three ways so it survives a rename.

    logGraph can rename logs (all 36 HITL logs are called FC_log.ulg, so it has
    to), and a report saved before a rename would otherwise dangle.  Path first,
    then basename among the logs the library currently knows about, then size --
    each one weaker than the last, none of them silently wrong: a reference that
    resolves to nothing stays in the report and is shown as missing rather than
    being dropped on load.
    """

    def __init__(self, path, name=None, size=0):
        self.path = path
        self.name = name or os.path.basename(path)
        self.size = int(size or 0)

    @classmethod
    def of(cls, path):
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        return cls(path, os.path.basename(path), size)

    def to_dict(self):
        return {"path": self.path, "name": self.name, "size": self.size}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("path", ""), d.get("name"), d.get("size", 0))

    def resolve(self, known=()):
        """The best current path for this log, or None.

        `known` is the set of log paths the library can see right now."""
        if self.path and os.path.exists(self.path):
            return self.path
        for p in known:
            if os.path.basename(p) == self.name:
                return p
        if self.size:
            for p in known:
                try:
                    if os.path.getsize(p) == self.size:
                        return p
                except OSError:
                    continue
        return None


class Graph:
    """One panel: some channels, from some logs, on one time axis."""

    def __init__(self, gid, title="", logs=None, fields=None,
                 align=DEFAULT_ALIGN, axis=None, normalise=False, xlim=None,
                 notes=""):
        self.id = gid
        self.title = title
        # Basenames, not paths: the graph's log subset has to survive the same
        # renames LogRef.resolve handles for the report as a whole.
        self.logs = list(logs or [])
        self.fields = list(fields or [])        # canonical "topic[i].field"
        self.align = align if align in ALIGNMENTS else DEFAULT_ALIGN
        self.axis = dict(axis or {})            # field -> "left" | "right"
        self.normalise = bool(normalise)
        self.xlim = tuple(xlim) if xlim else None
        self.notes = notes

    def to_dict(self):
        return {"id": self.id, "title": self.title, "logs": self.logs,
                "fields": self.fields, "align": self.align, "axis": self.axis,
                "normalise": self.normalise,
                "xlim": list(self.xlim) if self.xlim else None,
                "notes": self.notes}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("id") or f"g{int(time.time() * 1000) % 10 ** 9}",
                   d.get("title", ""), d.get("logs"), d.get("fields"),
                   d.get("align", DEFAULT_ALIGN), d.get("axis"),
                   d.get("normalise", False), d.get("xlim"), d.get("notes", ""))


class Report:
    """The whole document.  `path` is None until it has been saved once."""

    def __init__(self, title="", notes="", logs=None, graphs=None, path=None):
        self.title = title
        self.notes = notes
        self.logs = list(logs or [])            # [LogRef]
        self.graphs = list(graphs or [])        # [Graph]
        self.path = path
        self._next_id = 1

    # -- identity
    def new_graph_id(self):
        used = {g.id for g in self.graphs}
        while f"g{self._next_id}" in used:
            self._next_id += 1
        gid = f"g{self._next_id}"
        self._next_id += 1
        return gid

    def add_graph(self, **kw):
        g = Graph(self.new_graph_id(), **kw)
        self.graphs.append(g)
        return g

    def graph(self, gid):
        return next((g for g in self.graphs if g.id == gid), None)

    def remove_graph(self, gid):
        self.graphs = [g for g in self.graphs if g.id != gid]

    def log_names(self):
        return [ref.name for ref in self.logs]

    def resolved(self, known=()):
        """{basename: path or None} for every log in the report."""
        return {ref.name: ref.resolve(known) for ref in self.logs}

    # -- persistence
    def to_dict(self):
        return {"schema": SCHEMA, "title": self.title, "notes": self.notes,
                "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                "logs": [r.to_dict() for r in self.logs],
                "graphs": [g.to_dict() for g in self.graphs]}

    @classmethod
    def from_dict(cls, d, path=None):
        got = d.get("schema", SCHEMA)
        if got > SCHEMA:
            raise ValueError(f"report schema {got} is newer than this build "
                             f"understands (max {SCHEMA})")
        return cls(d.get("title", ""), d.get("notes", ""),
                   [LogRef.from_dict(x) for x in d.get("logs", [])],
                   [Graph.from_dict(x) for x in d.get("graphs", [])],
                   path=path)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f), path=path)

    def save(self, path=None, root=None):
        """Write to `path`, or to this report's own path, or to a new file.

        A new file's NAME is derived from the title once and then never changes,
        even when the title does.  Renaming the file under the user every time
        they edit the heading would break any link to it and buys nothing: the
        picker lists titles, not filenames.
        """
        if path is None:
            path = self.path
        if path is None:
            d = reports_dir(root, create=True)
            base = slugify(self.title)
            path = os.path.join(d, base + REPORT_EXT)
            n = 2
            while os.path.exists(path):
                path = os.path.join(d, f"{base}_{n}{REPORT_EXT}")
                n += 1
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write-then-replace: a crash mid-write leaves the previous report
        # intact rather than a half-truncated file that will not parse.
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, indent=1)
        os.replace(tmp, path)
        self.path = path
        return path
