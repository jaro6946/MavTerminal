#!/usr/bin/env python3
"""log_browser.py -- pick a .ulg from a library, scroll through its plots.

The GUI half of logGraph.  Left pane is a library of the logs on this machine;
right pane is every registered plot for the selected log, stacked in one
scrollable page with their time axes linked, in the spirit of review.px4.io.

Also does the two things you cannot do from a command line without knowing the
path already: RENAME a log (all 36 HITL logs are called FC_log.ulg and are told
apart only by their run folder), and export one or several logs to a PDF.

PyQt5 rather than Tkinter because matplotlib's interactive backend in this venv
is already qtagg -- a Tk shell would mean forcing a backend switch and running two
event loops' worth of dependencies for the same result.

Acronyms: ULog = PX4's binary log format, HITL = hardware in the loop,
GUI = graphical user interface, PDF = portable document format.
"""
import json
import os
import re
import sys
import time

import matplotlib
matplotlib.use("QtAgg")            # before any pyplot import, to match the shell

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from pyulog import ULog

import ulog_plots
from ulog_common import C_MUTED, C_SURFACE, PlotCtx, duration_min

# Where logs are looked for, in order.  Each entry is (label, path, is_hitl_tree).
# MAV_LOG_DIR is where `log pull` drops downloads, so a log you just pulled shows
# up here without being told about it.
def _default_roots():
    home = os.path.expanduser("~")
    roots = [("Log Analysis", os.path.join(home, "jacobAtGar", "Log Analysis"), False)]
    mav = os.environ.get("MAV_LOG_DIR")
    if mav:
        roots.append(("MAV_LOG_DIR", mav, False))
    data_out = os.environ.get("ROTORPY_DATA_OUT") or os.path.join(
        home, "jacobAtGar", "agc_CTOL_SE3-rotopy", "rotorpy", "data_out")
    roots.append(("HITL / run folders", data_out, True))
    roots.append(("current directory", os.getcwd(), False))
    return roots


STATE_PATH = os.path.join(
    os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
    "mavterminal", "log_browser.json")

PLOT_HEIGHT = 470          # px per stacked plot; ~2 fit on a 1080p screen


# --- state ------------------------------------------------------------------

def _load_state():
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_state(state):
    try:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=1)
    except OSError:
        pass                    # a browser that cannot cache still works


def _natural_key(name):
    """Digit runs compared as numbers, so log_9 sorts before log_10.

    Same rule as pull_log.natural_key; duplicated rather than imported because
    pull_log pulls in pymavlink, and the browser has no business requiring a
    MAVLink stack to list files on disk."""
    return [int(p) if p.isdigit() else p.lower()
            for p in re.split(r"(\d+)", name)]


def _fmt_size(n):
    return f"{n/1e6:.0f} MB" if n >= 1e6 else f"{n/1e3:.0f} kB"


# --- the scrollable plot page -----------------------------------------------

class PlotCanvas(FigureCanvasQTAgg):
    """A matplotlib canvas that gives the bare mouse wheel back to the page.

    Without this the canvas eats every wheel event to zoom, and once the pointer
    is over a plot -- which is most of the window -- the scroll area is stuck.
    Ctrl+wheel still reaches matplotlib, which is where Nav has moved zooming to.
    """

    def __init__(self, figure):
        super().__init__(figure)
        # Without this the canvas never takes keyboard focus, which is also why
        # matplotlib's own modifier tracking cannot be relied on here (see below).
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self._nav_mods = None

    def wheelEvent(self, event):
        mods = event.modifiers()
        if not (mods & QtCore.Qt.ControlModifier):
            event.ignore()          # bubbles up to the QScrollArea
            return
        # Hand Nav the modifiers explicitly.  matplotlib would otherwise report
        # key=None unless this canvas happened to hold keyboard focus, so
        # ctrl+wheel would do nothing while the log tree was focused -- i.e.
        # almost always.
        self._nav_mods = "ctrl+shift" if mods & QtCore.Qt.ShiftModifier else "ctrl"
        try:
            super().wheelEvent(event)
        finally:
            self._nav_mods = None


class PlotPage(QtWidgets.QScrollArea):
    """The stack of plots for one log, with their time axes linked."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._inner = QtWidgets.QWidget()
        self._inner.setStyleSheet(f"background: {C_SURFACE};")
        self._box = QtWidgets.QVBoxLayout(self._inner)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(2)
        self.setWidget(self._inner)
        self._navs = []
        self._syncing = False
        self._anchors = {}          # plot key -> widget, for the sidebar jumps

    def clear(self):
        while self._box.count():
            item = self._box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._navs = []
        self._anchors = {}

    def add(self, key, fig, height=PLOT_HEIGHT):
        canvas = PlotCanvas(fig)
        # setFixedHeight, not setMinimumHeight: FigureCanvasQTAgg derives its
        # sizeHint from the figure's inches * dpi, so a 15x10 figure asks for
        # 1500x1000 px and gets it -- the page then scrolls sideways, which it
        # must never do.  Pinning the height and dropping the minimum width lets
        # the canvas track the viewport instead, and matplotlib re-lays the
        # figure out on resize (every layout here is in figure fractions, so it
        # re-flows rather than clipping).
        canvas.setFixedHeight(height)
        canvas.setMinimumWidth(320)
        canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                             QtWidgets.QSizePolicy.Fixed)
        self._box.addWidget(canvas)
        canvas.show()           # widgets added after the parent is shown stay hidden
        self._anchors[key] = canvas
        nav = getattr(fig, "_nav", None)
        if nav is not None:
            nav.on_xlim = self._broadcast
            self._navs.append(nav)
        canvas.draw_idle()

    def finish(self):
        self._box.addStretch(1)

    def jump_to(self, key):
        w = self._anchors.get(key)
        if w is not None:
            self.ensureWidgetVisible(w, 0, 0)

    def _broadcast(self, lo, hi):
        """One plot's time window becomes every plot's time window.

        The guard is not optional: set_xlim on a sibling fires that sibling's own
        callback, which would come straight back here and recurse until the stack
        gives out."""
        if self._syncing:
            return
        self._syncing = True
        try:
            for nav in self._navs:
                cur = nav.axes[0].get_xlim()
                if abs(cur[0] - lo) > 1e-9 or abs(cur[1] - hi) > 1e-9:
                    nav.set_xlim(lo, hi)
        finally:
            self._syncing = False


# --- when did this flight happen --------------------------------------------
# The file's mtime answers "when was this file last written", which for a log
# pulled off an SD card is the DOWNLOAD time, not the flight.  Measured on
# SquareWaypointMission_1.ulg: mtime 2026-08-19 13:00, actual flight
# 2026-08-17 14:30 -- two days out.  Three sources, best first.

GPS_TOPICS = ["vehicle_gps_position", "sensor_gps"]

# A wall-clock stamp inside a file or folder name.  Covers both conventions in
# this project's libraries: QGC downloads (`log_24_2026-7-24-13-48-16.ulg`) and
# rotorpy run folders (`HITL_PX4_waypoint_mission_1_2026-07-20_12-32-28/`).
# Both write LOCAL time, so it is read back as local.
_NAME_STAMP = re.compile(
    r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})[-_ T]+(\d{1,2})[-_:](\d{2})(?:[-_:](\d{2}))?")


def _stamp_from_name(path):
    """Epoch seconds from the file name, or failing that its folder's name.

    The folder matters as much as the file: every HITL run writes a `FC_log.ulg`
    and the run folder is the only thing that dates it."""
    for part in (os.path.basename(path), os.path.basename(os.path.dirname(path))):
        m = _NAME_STAMP.search(part)
        if not m:
            continue
        y, mo, d, h, mi = (int(m.group(i)) for i in range(1, 6))
        sec = int(m.group(6) or 0)
        try:
            return time.mktime((y, mo, d, h, mi, sec, 0, 0, -1))
        except (ValueError, OverflowError):
            continue
    return None


def _start_from_parsed(ulog):
    """Epoch seconds of the log's FIRST sample, from GNSS UTC.  None if no fix.

    `time_utc_usec` is absolute (microseconds since the Unix epoch) while
    `timestamp` is microseconds since boot, so one sample carrying both pins the
    whole log to wall clock:

        start_epoch = utc[i] - (timestamp[i] - ulog.start_timestamp)

    Samples before the first fix carry 0, hence the 1e15 floor (~year 2001) --
    without it the answer is 1970 and looks like a bug in this function rather
    than an absent fix.
    """
    import numpy as np
    for d in ulog.data_list:
        if "time_utc_usec" not in d.data:
            continue
        utc = np.asarray(d.data["time_utc_usec"], dtype=np.float64)
        ts = np.asarray(d.data["timestamp"], dtype=np.float64)
        ok = utc > 1e15
        if not ok.any():
            continue
        i = int(np.argmax(ok))
        return float(utc[i] - (ts[i] - ulog.start_timestamp)) / 1e6
    return None


def log_start_from_gps(path):
    """_start_from_parsed, for a file we have not already read."""
    return _start_from_parsed(ULog(path, message_name_filter_list=GPS_TOPICS))


class DateScanner(QtCore.QObject):
    """Fills in GNSS-derived log dates in the background, one log at a time.

    A GPS-only parse is 0.7-1.8 s on this project's logs, because pyulog walks
    the whole file whatever you filter -- so 40 logs is around a minute.  Doing
    that at startup would mean an empty window for a minute to populate one
    column, so rows open with the name/mtime fallback and are corrected here as
    the answers arrive.

    Yields to the foreground parse: the user waiting on a plot they asked for
    outranks a column filling itself in.
    """
    found = QtCore.pyqtSignal(str, float, str)
    finished = QtCore.pyqtSignal()

    def __init__(self, paths, busy):
        super().__init__()
        self.paths = list(paths)
        self._busy = busy           # callable -> True while a plot parse runs
        self._stop = False

    def stop(self):
        self._stop = True

    @QtCore.pyqtSlot()
    def run(self):
        for path in self.paths:
            while self._busy() and not self._stop:
                time.sleep(0.2)
            if self._stop:
                break
            try:
                started = log_start_from_gps(path)
            except Exception:
                started = None      # unreadable or truncated: leave the fallback
            if started:
                self.found.emit(path, started, "gps")
            elif not self._stop:
                # Record the miss too, so the next session does not re-parse a
                # log that simply has no GNSS in it (every HITL log).
                self.found.emit(path, 0.0, "none")
        self.finished.emit()


# --- background parse -------------------------------------------------------

class ParseWorker(QtCore.QObject):
    """Parses a ULog off the GUI thread.

    Only the PARSE moves off-thread.  The figures are built back on the main
    thread, because their canvases are Qt widgets and Qt does not allow widget
    construction anywhere else -- doing it in the worker looks fine until it
    crashes at random.
    """
    done = QtCore.pyqtSignal(object, str, float)
    failed = QtCore.pyqtSignal(str, str)

    def __init__(self, path, topics):
        super().__init__()
        self.path, self.topics = path, topics

    @QtCore.pyqtSlot()
    def run(self):
        try:
            t0 = time.time()
            ulog = ULog(self.path, message_name_filter_list=self.topics)
            self.done.emit(ulog, self.path, time.time() - t0)
        except Exception as e:
            self.failed.emit(self.path, f"{type(e).__name__}: {e}")


# --- main window ------------------------------------------------------------

class Browser(QtWidgets.QMainWindow):

    def __init__(self, paths=(), ctx=None):
        super().__init__()
        self.ctx = ctx or PlotCtx()
        self.ctx.page_scroll = True     # bare wheel belongs to the page here
        self.state = _load_state()
        self.state.setdefault("folders", [])
        self.state.setdefault("durations", {})
        self._thread = None
        self._worker = None
        self._current = None
        self._proc = None
        self._scan_thread = None
        self._scanner = None

        self.setWindowTitle("logGraph - ULog browser")
        self.resize(1600, 950)
        self._build_ui()
        self._populate(extra=list(paths))
        if paths:
            self._select_path(paths[0])

    # -- construction
    def _build_ui(self):
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(split)

        # left: the library
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)

        row = QtWidgets.QHBoxLayout()
        for label, slot in (("Add folder…", self._add_folder),
                            ("Open file…", self._open_file),
                            ("Refresh", lambda: self._populate())):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(slot)
            row.addWidget(b)
        lv.addLayout(row)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["log", "duration", "size", "date", "time"])
        # The name column absorbs the slack and the rest size to their contents;
        # fixed widths truncated the "duration"/"size" headers at the default
        # pane width, and a horizontal scrollbar to read a 6-character column is
        # not a trade worth making.
        header = self.tree.header()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for col in (1, 2, 3, 4):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        self.tree.setTextElideMode(QtCore.Qt.ElideMiddle)
        self.tree.setAlternatingRowColors(True)
        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.tree.itemSelectionChanged.connect(self._selection_changed)
        self.tree.itemDoubleClicked.connect(lambda *_: self._load_selected())
        lv.addWidget(self.tree, 1)

        # A selection change starts a 400 ms timer rather than loading at once.
        # Arrow-keying down a list of 137 MB logs would otherwise kick off a
        # parse per keystroke, and the last one you actually want lands behind a
        # queue of ones you do not.
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(400)
        self._debounce.timeout.connect(self._load_selected)

        row2 = QtWidgets.QHBoxLayout()
        self.btn_rename = QtWidgets.QPushButton("Rename…")
        self.btn_rename.clicked.connect(self._rename_selected)
        self.btn_pdf = QtWidgets.QPushButton("Export PDF…")
        self.btn_pdf.clicked.connect(self._export_pdf)
        row2.addWidget(self.btn_rename)
        row2.addWidget(self.btn_pdf)
        lv.addLayout(row2)

        self.check_hint = QtWidgets.QLabel("tick logs to include them in a PDF")
        self.check_hint.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        lv.addWidget(self.check_hint)

        # right: header + plot page + console
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)

        head = QtWidgets.QWidget()
        hh = QtWidgets.QHBoxLayout(head)
        hh.setContentsMargins(10, 6, 10, 6)
        self.title = QtWidgets.QLabel("no log loaded")
        self.title.setStyleSheet("font-size: 14px; font-weight: 600;")
        hh.addWidget(self.title)
        hh.addStretch(1)
        self.jump = QtWidgets.QComboBox()
        self.jump.addItem("jump to plot…")
        self.jump.activated.connect(self._jump)
        hh.addWidget(self.jump)
        self.busy = QtWidgets.QProgressBar()
        self.busy.setRange(0, 0)            # indeterminate
        self.busy.setFixedWidth(140)
        self.busy.hide()
        hh.addWidget(self.busy)
        rv.addWidget(head)

        self.page = PlotPage()
        rv.addWidget(self.page, 1)

        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(120)
        self.console.setStyleSheet("font-family: monospace; font-size: 11px;")
        rv.addWidget(self.console)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([470, 1130])

    # -- library
    def _log(self, msg):
        self.console.appendPlainText(msg)
        self.console.verticalScrollBar().setValue(
            self.console.verticalScrollBar().maximum())

    def _cached_duration(self, path, st):
        """Minutes, if we have parsed this exact file before.

        Keyed on (size, mtime) so an edited or replaced file re-measures itself.
        Duration is not cheap -- pyulog walks the whole file to find the last
        timestamp -- and doing that for 40+ logs at startup would mean a minute
        of staring at an empty window."""
        rec = self._cached_record(path, st)
        return rec.get("minutes") if rec else None

    def _remember_duration(self, path, minutes):
        self._update_record(path, minutes=minutes)

    def _update_record(self, path, **fields):
        """Merge fields into this file's cache record, re-stamping size/mtime.

        Merge rather than replace: the duration and the log date are learned at
        different times by different code paths, and a plain assignment from
        either one silently drops what the other found."""
        st = os.stat(path)
        key = os.path.abspath(path)
        rec = dict(self.state["durations"].get(key) or {})
        if rec.get("size") != st.st_size or rec.get("mtime") != int(st.st_mtime):
            rec = {}                # a replaced file: everything cached is stale
        rec.update(fields)
        rec["size"], rec["mtime"] = st.st_size, int(st.st_mtime)
        self.state["durations"][key] = rec
        _save_state(self.state)

    def _cached_record(self, path, st):
        rec = self.state["durations"].get(os.path.abspath(path))
        if rec and rec.get("size") == st.st_size and rec.get("mtime") == int(st.st_mtime):
            return rec
        return None

    def _log_date(self, path, st):
        """(epoch seconds, source) for the row's date/time columns.

        `started` is cached as 0.0 to mean "parsed, and this log has no GNSS
        time" -- distinct from a missing key, which means "not looked at yet".
        Without that distinction every HITL log is re-parsed on every startup.
        """
        rec = self._cached_record(path, st) or {}
        started = rec.get("started")
        if started:
            return started, rec.get("date_src", "gps")
        stamp = _stamp_from_name(path)
        if stamp:
            return stamp, "name"
        return st.st_mtime, "mtime"

    def _populate(self, extra=()):
        checked = self._checked_paths()
        current = self._selected_path()
        self.tree.clear()
        seen = set()

        roots = _default_roots() + [(os.path.basename(p.rstrip("/")) or p, p, False)
                                    for p in self.state["folders"]]
        for label, root, is_tree in roots:
            if not os.path.isdir(root):
                continue
            files = self._scan(root, is_tree)
            files = [f for f in files if f[1] not in seen]
            if not files:
                continue
            seen.update(f[1] for f in files)
            node = QtWidgets.QTreeWidgetItem([f"{label}  ({len(files)})"])
            node.setFirstColumnSpanned(True)
            f = node.font(0)
            f.setBold(True)
            node.setFont(0, f)
            node.setData(0, QtCore.Qt.UserRole, None)
            self.tree.addTopLevelItem(node)
            for name, path in files:
                self._add_row(node, name, path, checked)
            node.setExpanded(not is_tree)

        loose = [p for p in extra if os.path.isfile(p) and os.path.abspath(p) not in
                 {os.path.abspath(s) for s in seen}]
        if loose:
            node = QtWidgets.QTreeWidgetItem([f"opened  ({len(loose)})"])
            node.setFirstColumnSpanned(True)
            self.tree.addTopLevelItem(node)
            for p in loose:
                self._add_row(node, os.path.basename(p), p, checked)
            node.setExpanded(True)

        if current:
            self._select_path(current)
        self._start_date_scan()

    # -- log dates, filled in behind the library
    def _start_date_scan(self):
        """(Re)start the background GNSS-date scan over rows still guessing."""
        self._stop_date_scan()
        todo = []
        for it in self._iter_items():
            path = it.data(0, QtCore.Qt.UserRole)
            try:
                st = os.stat(path)
            except OSError:
                continue
            rec = self._cached_record(path, st) or {}
            if "started" not in rec:        # 0.0 counts as answered: no GNSS
                todo.append(path)
        if not todo:
            return
        self._scan_thread = QtCore.QThread(self)
        self._scanner = DateScanner(todo, lambda: self._thread is not None)
        self._scanner.moveToThread(self._scan_thread)
        self._scan_thread.started.connect(self._scanner.run)
        self._scanner.found.connect(self._on_log_date)
        self._scanner.finished.connect(self._stop_date_scan)
        self._scan_thread.start()

    def _stop_date_scan(self):
        if self._scanner is not None:
            self._scanner.stop()
        if self._scan_thread is not None:
            self._scan_thread.quit()
            self._scan_thread.wait()
        self._scan_thread = None
        self._scanner = None

    @QtCore.pyqtSlot(str, float, str)
    def _on_log_date(self, path, started, source):
        try:
            self._update_record(path, started=started, date_src=source)
        except OSError:
            return                  # the file went away mid-scan
        if not started:
            return                  # no GNSS in it: the row keeps its fallback
        for it in self._iter_items():
            if it.data(0, QtCore.Qt.UserRole) == path:
                self._set_date_cells(it, started, source)

    def _scan(self, root, is_tree):
        """(display name, path) for the .ulg files under `root`.

        `is_tree` means a directory of run folders: every log inside is called
        FC_log.ulg, so the RUN FOLDER is the identifying name and showing the
        filename would give 36 identical rows."""
        out = []
        if is_tree:
            try:
                runs = sorted(os.scandir(root), key=lambda e: e.name, reverse=True)
            except OSError:
                return out
            for entry in runs:
                if not entry.is_dir():
                    continue
                for f in sorted(os.listdir(entry.path)):
                    if f.endswith(".ulg"):
                        out.append((f"{entry.name}/{f}", os.path.join(entry.path, f)))
        else:
            try:
                names = os.listdir(root)
            except OSError:
                return out
            # ":Zone.Identifier" is the alternate-data-stream file Windows writes
            # beside anything downloaded from the internet, and WSL exposes it as
            # a real file.  It is not a log.
            names = [n for n in names if n.endswith(".ulg")]
            for n in sorted(names, key=_natural_key):
                out.append((n, os.path.join(root, n)))
        return out

    def _add_row(self, parent, name, path, checked):
        try:
            st = os.stat(path)
        except OSError:
            return
        mins = self._cached_duration(path, st)
        item = QtWidgets.QTreeWidgetItem([
            name,
            f"{mins:.1f} min" if mins is not None else "—",
            _fmt_size(st.st_size),
            "", "",
        ])
        self._set_date_cells(item, *self._log_date(path, st))
        item.setData(0, QtCore.Qt.UserRole, path)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(0, QtCore.Qt.Checked
                           if os.path.abspath(path) in checked
                           else QtCore.Qt.Unchecked)
        item.setToolTip(0, path)
        parent.addChild(item)

    # Both columns are one fact, so they are written by one function -- a date
    # from GNSS next to a time from the file system would be a sentence nobody
    # wrote and nobody could check.
    DATE_SOURCE_NOTE = {
        "gps": "flight time, from the log's own GNSS clock",
        "name": "inferred from the file or run-folder name -- the log carries "
                "no GNSS time yet",
        "mtime": "file modified time -- NOT the flight; a log pulled off an SD "
                 "card is dated when it was downloaded",
    }

    def _set_date_cells(self, item, epoch, source):
        lt = time.localtime(epoch)
        item.setText(3, time.strftime("%Y-%m-%d", lt))
        item.setText(4, time.strftime("%H:%M:%S", lt))
        # Anything but GNSS is an inference, and greying it is the difference
        # between "this flight was on the 17th" and "this file was touched on
        # the 19th" -- which is exactly the confusion this column replaced.
        colour = None if source == "gps" else QtGui.QColor(C_MUTED)
        note = self.DATE_SOURCE_NOTE.get(source, "")
        for col in (3, 4):
            if colour is not None:
                item.setForeground(col, colour)
            else:
                item.setData(col, QtCore.Qt.ForegroundRole, None)
            item.setToolTip(col, note)

    def _iter_items(self):
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            for j in range(top.childCount()):
                yield top.child(j)

    def _checked_paths(self):
        return {os.path.abspath(it.data(0, QtCore.Qt.UserRole))
                for it in self._iter_items()
                if it.checkState(0) == QtCore.Qt.Checked
                and it.data(0, QtCore.Qt.UserRole)}

    def _selected_path(self):
        items = self.tree.selectedItems()
        return items[0].data(0, QtCore.Qt.UserRole) if items else None

    def _select_path(self, path):
        target = os.path.abspath(path)
        for it in self._iter_items():
            p = it.data(0, QtCore.Qt.UserRole)
            if p and os.path.abspath(p) == target:
                self.tree.setCurrentItem(it)
                return True
        return False

    # -- loading
    def _selection_changed(self):
        if self._selected_path():
            self._debounce.start()

    def _load_selected(self):
        self._debounce.stop()
        path = self._selected_path()
        if not path or path == self._current:
            return
        if self._thread is not None:
            return                  # a parse is already running; ignore
        self._current = path
        self.title.setText(f"{os.path.basename(path)}   —   reading…")
        self.busy.show()
        self._log(f"reading {path}")

        self._thread = QtCore.QThread(self)
        self._worker = ParseWorker(path, ulog_plots.all_topics(self.ctx))
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_parsed)
        self._worker.failed.connect(self._on_parse_failed)
        self._thread.start()

    def _teardown_thread(self):
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None
        self.busy.hide()

    @QtCore.pyqtSlot(object, str, float)
    def _on_parsed(self, ulog, path, secs):
        self._teardown_thread()
        self._log(f"  parsed in {secs:.1f}s")
        mins = duration_min(ulog)
        self._remember_duration(path, mins)
        # The date comes free here: this parse already asked for the GPS topics
        # (the altitude plot needs them), so re-reading the file in the scanner
        # for a log the user just opened would be pure waste.
        started, src = _start_from_parsed(ulog), "gps"
        if started is None:
            started, src = 0.0, "none"
        self._update_record(path, started=started, date_src=src)
        for it in self._iter_items():
            if it.data(0, QtCore.Qt.UserRole) == path:
                it.setText(1, f"{mins:.1f} min")
                if started:
                    self._set_date_cells(it, started, src)

        self.page.clear()
        self.jump.clear()
        self.jump.addItem("jump to plot…")
        self.title.setText(f"{os.path.basename(path)}   —   {mins:.1f} min")

        for spec in ulog_plots.PLOTS:
            sub = PlotCtx(smooth=self.ctx.smooth, use_abs=self.ctx.use_abs,
                          rate_src=self.ctx.rate_src, adds=list(self.ctx.adds),
                          debias=self.ctx.debias, page_scroll=True)
            try:
                fig = spec.build(ulog, sub, path)
            except Exception as e:
                # A plot that cannot render this log must not take the others
                # (and the whole window) down with it.
                self._log(f"  !! {spec.title}: {type(e).__name__}: {e}")
                continue
            for n in sub.notes:
                self._log(f"  note [{spec.key}]: {n}")
            if fig is None:
                self._log(f"  {spec.title}: nothing plottable in this log")
                continue
            self.page.add(spec.key, fig, spec.height)
            self.jump.addItem(spec.title, spec.key)
        self.page.finish()

    @QtCore.pyqtSlot(str, str)
    def _on_parse_failed(self, path, msg):
        self._teardown_thread()
        self._current = None
        self.title.setText(f"{os.path.basename(path)}   —   failed")
        self._log(f"  !! {msg}")
        QtWidgets.QMessageBox.warning(self, "Could not read log",
                                      f"{os.path.basename(path)}\n\n{msg}")

    def _jump(self, index):
        key = self.jump.itemData(index)
        if key:
            self.page.jump_to(key)

    # -- library actions
    def _add_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Add a folder of logs")
        if d:
            if d not in self.state["folders"]:
                self.state["folders"].append(d)
                _save_state(self.state)
            self._populate()

    def _open_file(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open a .ulg log", "", "PX4 logs (*.ulg);;All files (*)")
        if f:
            self._populate(extra=[f])
            self._select_path(f)

    def _context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None or not item.data(0, QtCore.Qt.UserRole):
            return
        menu = QtWidgets.QMenu(self)
        menu.addAction("Plot", self._load_selected)
        menu.addAction("Rename…", self._rename_selected)
        menu.addAction("Copy path", lambda: QtWidgets.QApplication.clipboard()
                       .setText(item.data(0, QtCore.Qt.UserRole)))
        menu.exec_(self.tree.viewport().mapToGlobal(pos))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_F2 and self.tree.hasFocus():
            self._rename_selected()
            return
        super().keyPressEvent(event)

    def _rename_selected(self):
        path = self._selected_path()
        if not path:
            return
        folder, old = os.path.split(path)
        new, ok = QtWidgets.QInputDialog.getText(
            self, "Rename log", f"New name (in {folder}):",
            QtWidgets.QLineEdit.Normal, old)
        if not ok:
            return
        new = new.strip()
        if not new or new == old:
            return
        if os.sep in new or (os.altsep and os.altsep in new):
            QtWidgets.QMessageBox.warning(
                self, "Rename", "A name cannot contain a path separator.\n"
                "Renaming only ever moves a log within its own folder.")
            return
        if not new.endswith(".ulg"):
            new += ".ulg"
        target = os.path.join(folder, new)
        if os.path.exists(target):
            QtWidgets.QMessageBox.warning(
                self, "Rename", f"'{new}' already exists in that folder.\n"
                "Renaming onto an existing log would destroy it, so nothing "
                "was changed.")
            return

        # Sidecars travel with the log or they become orphans.  The HITL case is
        # the one that matters: FC_log.ulg always sits beside FC_log_diag.txt,
        # and renaming only the .ulg leaves a diagnosis nothing points at.
        moves = [(path, target)]
        old_stem = old[:-4]
        new_stem = new[:-4]
        for suffix in ("_diag.txt", ".ulg:Zone.Identifier"):
            src = os.path.join(folder, old_stem + suffix)
            if os.path.exists(src):
                moves.append((src, os.path.join(folder, new_stem + suffix)))
        try:
            for src, dst in moves:
                os.rename(src, dst)
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Rename failed", str(e))
            self._populate()
            return

        rec = self.state["durations"].pop(os.path.abspath(path), None)
        if rec:
            self.state["durations"][os.path.abspath(target)] = rec
        _save_state(self.state)
        for src, dst in moves:
            self._log(f"renamed {os.path.basename(src)} -> {os.path.basename(dst)}")
        if self._current == path:
            self._current = target
        self._populate()
        self._select_path(target)

    # -- pdf
    def _export_pdf(self):
        paths = sorted(self._checked_paths())
        if not paths:
            sel = self._selected_path()
            if not sel:
                QtWidgets.QMessageBox.information(
                    self, "Export PDF",
                    "Tick one or more logs in the list (or select one) first.")
                return
            paths = [sel]

        default = os.path.splitext(os.path.basename(paths[0]))[0]
        default += "_report.pdf" if len(paths) == 1 else f"_and_{len(paths)-1}_more.pdf"
        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save PDF report", os.path.join(os.getcwd(), default),
            "PDF (*.pdf)")
        if not out:
            return
        if self._proc is not None:
            QtWidgets.QMessageBox.information(self, "Export PDF",
                                              "An export is already running.")
            return

        # A SUBPROCESS, not a thread.  export_pdf builds matplotlib figures, and
        # this process' backend is QtAgg -- building figures off the GUI thread
        # would be creating Qt objects outside it.  The child forces Agg, keeps
        # this window responsive, and is the exact code path the CLI already
        # exercises, so there is only one PDF pipeline to keep working.
        argv = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "ulog_graph.py"), "--pdf", out] + paths
        self._log(f"exporting {len(paths)} log(s) -> {out}")
        self.busy.show()
        self.btn_pdf.setEnabled(False)
        self._proc = QtCore.QProcess(self)
        self._proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._pdf_output)
        self._proc.finished.connect(lambda code, _s: self._pdf_done(code, out))
        self._proc.start(argv[0], argv[1:])

    def _pdf_output(self):
        text = bytes(self._proc.readAllStandardOutput()).decode(errors="replace")
        for line in text.splitlines():
            if line.strip():
                self._log(line.rstrip())

    def _pdf_done(self, code, out):
        self._proc = None
        self.busy.hide()
        self.btn_pdf.setEnabled(True)
        if code == 0 and os.path.exists(out):
            self._log(f"PDF written: {out}")
            QtWidgets.QMessageBox.information(
                self, "Export PDF", f"Wrote\n{out}")
        else:
            self._log(f"!! PDF export failed (exit {code})")
            QtWidgets.QMessageBox.warning(
                self, "Export PDF",
                f"Export failed (exit code {code}).\nSee the console pane for "
                f"the reason.")

    def closeEvent(self, event):
        self._stop_date_scan()
        self._teardown_thread()
        if self._proc is not None:
            self._proc.kill()
        _save_state(self.state)
        super().closeEvent(event)


def browse(paths=(), ctx=None):
    """Open the browser.  Blocks until the window is closed."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = Browser(paths=[p for p in paths if p], ctx=ctx)
    win.show()
    app.exec_()
    return win


if __name__ == "__main__":
    browse(sys.argv[1:])
