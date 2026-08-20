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
import contextlib
import io
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
from ulog_common import C_BAD, C_MUTED, C_SURFACE, PlotCtx, duration_min

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

# Bumped whenever the library scan learns a new fact.  Cached rows below this
# version are re-scanned once, rather than sitting blank in a column that did not
# exist when they were cached.
SCAN_VERSION = 2

# (The library sidebar became a dropdown; its width constants went with it.)


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
        # ACCEPT, or the zoom happens AND the page scrolls out from under it.
        # Qt calls ignore() on a wheel event before delivering it and walks up
        # the parent chain until someone accepts; matplotlib's
        # FigureCanvasQT.wheelEvent handles the event but never accepts it, so
        # without this the QScrollArea gets it next and scrolls the page away
        # from the plot you were zooming.  Unconditional on ctrl: this gesture
        # belongs to the canvas whether or not the notch resolved to a step.
        event.accept()


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
        self._last_nav = None       # the plot whose time window moved last
        # Whether zooming one plot re-ranges the others.  OFF by default: the
        # pointer is over ONE plot, and having the other five jump under it is
        # not what the gesture looks like it should do -- you lose the view you
        # had scrolled to on every other plot to zoom the one in front of you.
        # The toolbar checkbox turns it back on for a cross-plot comparison.
        self.link_time = False

    def clear(self):
        while self._box.count():
            item = self._box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._navs = []
        self._anchors = {}
        self._last_nav = None

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
            # Bound per-nav so the page knows WHICH plot moved, not just that
            # something did -- that is what set_link_time adopts the window of.
            nav.on_xlim = lambda lo, hi, n=nav: self._nav_changed(n, lo, hi)
            self._navs.append(nav)
        canvas.draw_idle()

    def finish(self):
        self._box.addStretch(1)

    def jump_to(self, key):
        w = self._anchors.get(key)
        if w is not None:
            self.ensureWidgetVisible(w, 0, 0)

    def _nav_changed(self, nav, lo, hi):
        self._last_nav = nav
        self._broadcast(lo, hi)

    def set_link_time(self, on):
        """Turn cross-plot time linking on or off.

        Switching it ON adopts the window of the plot you were last working in,
        so the plots agree immediately -- rather than staying disagreed until the
        next wheel notch, and rather than snapping to whichever plot happens to
        be first on the page."""
        self.link_time = bool(on)
        if not self.link_time or not self._navs:
            return
        src = self._last_nav if self._last_nav in self._navs else self._navs[0]
        lo, hi = src.axes[0].get_xlim()
        self._broadcast(lo, hi)

    def _broadcast(self, lo, hi):
        """One plot's time window becomes every plot's time window.

        Only when link_time is on -- see the flag in __init__.

        The guard is not optional: set_xlim on a sibling fires that sibling's own
        callback, which would come straight back here and recurse until the stack
        gives out."""
        if self._syncing or not self.link_time:
            return
        self._syncing = True
        try:
            for nav in self._navs:
                cur = nav.axes[0].get_xlim()
                if abs(cur[0] - lo) > 1e-9 or abs(cur[1] - hi) > 1e-9:
                    nav.set_xlim(lo, hi)
        finally:
            self._syncing = False


# --- parameters ---------------------------------------------------------------

def params_of(path):
    """{name: value} as the log was booted with, plus the firmware it booted.

    `parse_header_only` reads the definition section and stops: 16 ms for a
    160 MB log against ~2 s for a full parse, because the parameters all live in
    the header.  That is what makes comparing two arbitrary logs a click rather
    than a wait.

    `initial_parameters` is the boot-time set.  In-flight changes live in
    `changed_parameters` and are deliberately NOT merged: "what was this aircraft
    configured with" is the question being asked, and folding a mid-flight tweak
    into it would answer a different one silently.
    """
    ulog = ULog(path, parse_header_only=True)
    return dict(ulog.initial_parameters), dict(ulog.msg_info_dict)


def fmt_param(v):
    """PX4 shows ints as ints and floats to 6 significant figures."""
    if v is None:
        return "—"
    if isinstance(v, float):
        # %g drops the trailing zeros that make a table of calibration offsets
        # unreadable, without rounding away a real difference at 1e-6.
        return f"{v:.6g}"
    return str(v)


# --- how much of the file could not be read -----------------------------------

class MeasuredULog(ULog):
    """A ULog that also reports HOW MUCH of the file the parser threw away.

    pyulog reports corruption as a single boolean (`file_corruption`), which
    cannot tell a log that lost one record from one that lost a third of the
    flight -- and the answer matters, because the first is ignorable and the
    second invalidates the analysis.

    The recovery path is the measurement.  On a bad record the parser seeks
    forward for the next SYNC marker (`_find_sync`), and the distance it covers
    is exactly the span it could not read.  Measured on log_53: 21065 bytes over
    3 events, which lines up with the 50 ms and 201 ms holes in sensor_combined.

    Counting is gated on `_file_corrupt` because `_find_sync` is ALSO how pyulog
    skips a message type it simply does not know -- a newer firmware adding a
    record type is not corruption, and both clean logs in the library report
    exactly 0 with this guard in place.  A file that corrupts early and then
    meets an unknown type can over-count; that errs toward flagging, which is the
    right way to be wrong here.
    """

    def __init__(self, *args, **kwargs):
        self.corrupt_bytes = 0
        self.corrupt_events = 0
        super().__init__(*args, **kwargs)

    def _find_sync(self, last_n_bytes=-1):
        fh = self._file_handle
        start = fh.tell()
        # last_n_bytes != -1 means "search backwards into the payload we just
        # read", so the span begins before the current position.
        base = start - last_n_bytes if last_n_bytes != -1 else start
        was_corrupt = self._file_corrupt
        result = super()._find_sync(last_n_bytes)
        skipped = fh.tell() - base
        if was_corrupt and skipped > 0:
            self.corrupt_bytes += skipped
            self.corrupt_events += 1
        return result


def corruption_of(ulog, path):
    """{corrupt_bytes, corrupt_events, corrupt_pct} for a parsed log."""
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    nbytes = getattr(ulog, "corrupt_bytes", 0)
    if not nbytes and getattr(ulog, "file_corruption", False):
        # Flagged but nothing measured: a plain ULog, or corruption found on a
        # path that does not resync.  Report it as unknown-size rather than as
        # clean -- "0.0%" would be a claim we cannot support.
        return {"corrupt_bytes": -1, "corrupt_events": -1, "corrupt_pct": -1.0}
    return {"corrupt_bytes": nbytes,
            "corrupt_events": getattr(ulog, "corrupt_events", 0),
            "corrupt_pct": (100.0 * nbytes / size) if size else 0.0}


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


def scan_log(path):
    """Everything the library columns need that requires reading the file.

    One parse, because pyulog walks the whole file whatever you filter -- asking
    separately for the date and for the corruption measurement would double a
    23 s library scan for no gain."""
    ulog = MeasuredULog(path, message_name_filter_list=GPS_TOPICS)
    facts = corruption_of(ulog, path)
    facts["started"] = _start_from_parsed(ulog) or 0.0
    facts["date_src"] = "gps" if facts["started"] else "none"
    return facts


class LibraryScanner(QtCore.QObject):
    """Fills in the columns that need the file read, one log at a time.

    A parse is 0.7-1.8 s on this project's logs, because pyulog walks the whole
    file whatever you filter -- so 45 logs is around 25 s.  Doing that at startup
    would mean an empty window while one column populates, so rows open with the
    name/mtime fallback and are corrected here as the answers arrive.

    Yields to the foreground parse: the user waiting on a plot they asked for
    outranks a column filling itself in.
    """
    found = QtCore.pyqtSignal(str, object)      # path, facts dict
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
                with contextlib.redirect_stdout(io.StringIO()):
                    facts = scan_log(path)
            except Exception:
                # Unreadable or truncated past recovery.  Recorded anyway, so the
                # next session does not re-parse it, and shown as unknown rather
                # than as clean.
                facts = {"started": 0.0, "date_src": "none",
                         "corrupt_bytes": -1, "corrupt_events": -1,
                         "corrupt_pct": -1.0}
            self.found.emit(path, facts)
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
            ulog = MeasuredULog(self.path, message_name_filter_list=self.topics)
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
        # No splitter: the library is a dropdown on the toolbar, so the plot page
        # owns the full window width.  These figures are ~15 inches of content
        # laid out in figure fractions, and a 470 px sidebar was costing every
        # panel a third of its horizontal resolution -- which is the axis the
        # time series actually needs.
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        cv = QtWidgets.QVBoxLayout(central)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)

        bar = QtWidgets.QWidget()
        bh = QtWidgets.QHBoxLayout(bar)
        bh.setContentsMargins(10, 6, 10, 6)

        bh.addWidget(QtWidgets.QLabel("log:"))
        self.picker = QtWidgets.QComboBox()
        # Monospace, because the entries carry the columns the tree used to:
        # name, duration, size, date, time, corruption.  Proportional type turns
        # those into ragged prose.
        self.picker.setStyleSheet("font-family: monospace;")
        self.picker.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLength)
        self.picker.setMinimumContentsLength(48)
        self.picker.activated.connect(self._picked)
        bh.addWidget(self.picker, 1)

        for label, slot in (("Open file…", self._open_file),
                            ("Add folder…", self._add_folder),
                            ("Refresh", lambda: self._populate()),
                            ("Rename…", self._rename_selected)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(slot)
            bh.addWidget(b)
            if label == "Rename…":
                self.btn_rename = b
        self.btn_params = QtWidgets.QPushButton("Compare params…")
        self.btn_params.setToolTip("Show every parameter that differs between "
                                   "the open log and another one")
        self.btn_params.setEnabled(False)      # needs a log open to compare FROM
        self.btn_params.clicked.connect(self._compare_params)
        bh.addWidget(self.btn_params)
        self.btn_pdf = QtWidgets.QPushButton("Export PDF…")
        self.btn_pdf.clicked.connect(self._export_pdf)
        bh.addWidget(self.btn_pdf)

        self.chk_link = QtWidgets.QCheckBox("Link time axes")
        self.chk_link.setToolTip(
            "Off: ctrl+wheel zooms only the plot under the pointer.\n"
            "On: every plot follows the same time window.")
        self.chk_link.setChecked(False)
        self.chk_link.toggled.connect(lambda on: self.page.set_link_time(on))
        bh.addWidget(self.chk_link)

        self.jump = QtWidgets.QComboBox()
        self.jump.addItem("jump to plot…")
        self.jump.activated.connect(self._jump)
        bh.addWidget(self.jump)

        self.busy = QtWidgets.QProgressBar()
        self.busy.setRange(0, 0)            # indeterminate
        self.busy.setFixedWidth(120)
        self.busy.hide()
        bh.addWidget(self.busy)
        cv.addWidget(bar)

        self.title = QtWidgets.QLabel("no log loaded")
        self.title.setStyleSheet(
            f"font-size: 13px; font-weight: 600; padding: 0 10px 6px 10px;")
        cv.addWidget(self.title)

        self.page = PlotPage()
        cv.addWidget(self.page, 1)

        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(110)
        self.console.setStyleSheet("font-family: monospace; font-size: 11px;")
        cv.addWidget(self.console)

        # The tree is still the MODEL -- it holds one row per log with the six
        # columns, the check states and the per-cell colours, and every method
        # that maintains them is unchanged.  It is simply never put in a layout;
        # _rebuild_picker projects it into the dropdown.  Keeping it beats
        # rewriting the population, scanning and rename bookkeeping against a
        # combo box that cannot express any of it.
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(6)

        # A selection change starts a 400 ms timer rather than loading at once,
        # so keyboard-scrolling the dropdown doesn't kick off a parse per
        # keystroke and leave the one you want behind a queue.
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(400)
        self._debounce.timeout.connect(self._load_selected)

    # -- the dropdown, projected from the tree
    def _row_text(self, item):
        """One dropdown line: the name, then the columns the tree used to show."""
        bits = [b for b in (item.text(1), item.text(2),
                            f"{item.text(3)} {item.text(4)}".strip(),
                            item.text(5)) if b and b != "—"]
        name = item.text(0)
        return f"{name}   ·   {'  ·  '.join(bits)}" if bits else name

    def _rebuild_picker(self):
        """Re-project the tree into the dropdown, preserving the selection."""
        keep = self._picker_path()
        self.picker.blockSignals(True)
        self.picker.clear()
        model = self.picker.model()
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            if not top.childCount():
                continue
            # Group headers stay as unselectable rows: the roots (Log Analysis /
            # HITL run folders / current directory) are how you know which
            # library a log came from, and a flat list of 46 entries loses that.
            self.picker.addItem(f"── {top.text(0)} ──")
            row = model.item(self.picker.count() - 1)
            row.setFlags(row.flags() & ~QtCore.Qt.ItemIsEnabled)
            for j in range(top.childCount()):
                child = top.child(j)
                self.picker.addItem(self._row_text(child))
                self.picker.setItemData(self.picker.count() - 1,
                                        child.data(0, QtCore.Qt.UserRole),
                                        QtCore.Qt.UserRole)
        self.picker.blockSignals(False)
        if keep and not self._select_path(keep):
            self.picker.setCurrentIndex(-1)
        elif not keep:
            self.picker.setCurrentIndex(-1)
        # The popup is free to be wider than the closed combo, and these lines
        # run past 90 characters.
        self.picker.view().setMinimumWidth(
            self.picker.fontMetrics().averageCharWidth() * 96)

    def _refresh_picker_row(self, path):
        """Re-render one dropdown line after its tree row changed."""
        target = os.path.abspath(path)
        for it in self._iter_items():
            p = it.data(0, QtCore.Qt.UserRole)
            if not p or os.path.abspath(p) != target:
                continue
            for k in range(self.picker.count()):
                q = self.picker.itemData(k, QtCore.Qt.UserRole)
                if q and os.path.abspath(q) == target:
                    self.picker.setItemText(k, self._row_text(it))
                    return

    def _picker_path(self):
        i = self.picker.currentIndex()
        return self.picker.itemData(i, QtCore.Qt.UserRole) if i >= 0 else None

    def _picked(self, _index):
        if self._picker_path():
            self._debounce.start()

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
            files = self._sorted_by_date(self._scan(root, is_tree))
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

        self._rebuild_picker()
        if current:
            self._select_path(current)
        self._start_library_scan()

    # -- log dates, filled in behind the library
    def _start_library_scan(self):
        """(Re)start the background scan over rows we have not read yet."""
        self._stop_library_scan()
        todo = []
        for it in self._iter_items():
            path = it.data(0, QtCore.Qt.UserRole)
            try:
                st = os.stat(path)
            except OSError:
                continue
            rec = self._cached_record(path, st) or {}
            # Version-stamped, so adding a column re-scans once instead of
            # leaving old rows permanently blank in the new column.
            if rec.get("scan_v") != SCAN_VERSION:
                todo.append(path)
        if not todo:
            return
        self._scan_thread = QtCore.QThread(self)
        self._scanner = LibraryScanner(todo, lambda: self._thread is not None)
        self._scanner.moveToThread(self._scan_thread)
        self._scan_thread.started.connect(self._scanner.run)
        self._scanner.found.connect(self._on_scanned)
        self._scanner.finished.connect(self._on_scan_finished)
        self._scan_thread.start()

    @QtCore.pyqtSlot()
    def _on_scan_finished(self):
        """Settle the order once, after the guesses have become real dates.

        Re-sorting on every result would make rows jump under the pointer for
        the whole scan.  `_populate` restarts the scan, but by now every row is
        cached at the current SCAN_VERSION so it finds nothing to do and stops
        immediately -- that is what keeps this from looping.
        """
        self._stop_library_scan()
        self._populate()

    def _stop_library_scan(self):
        if self._scanner is not None:
            self._scanner.stop()
        if self._scan_thread is not None:
            self._scan_thread.quit()
            self._scan_thread.wait()
        self._scan_thread = None
        self._scanner = None

    @QtCore.pyqtSlot(str, object)
    def _on_scanned(self, path, facts):
        try:
            self._update_record(path, scan_v=SCAN_VERSION, **facts)
        except OSError:
            return                  # the file went away mid-scan
        for it in self._iter_items():
            if it.data(0, QtCore.Qt.UserRole) == path:
                st = os.stat(path)
                self._set_date_cells(it, *self._log_date(path, st))
                self._set_corrupt_cell(it, self._cached_record(path, st) or {})
                self._refresh_picker_row(path)

    def _sorted_by_date(self, files):
        """Newest flight first, by the log's OWN date.

        Filename order is close to useless here: three of the four libraries use
        UUIDs or a per-session counter, and the HITL tree calls every log
        FC_log.ulg.  Newest-first because the log you want is almost always the
        one you just flew.

        Rows sort on whatever `_log_date` can supply right now -- GNSS time if it
        has been read, otherwise the name/mtime fallback -- so the list is in a
        sensible order immediately.  `_on_scan_finished` re-populates once the
        background scan has replaced the guesses with real flight times, which is
        the only point at which the order can still change.
        """
        def key(entry):
            try:
                return -self._log_date(entry[1], os.stat(entry[1]))[0]
            except OSError:
                return 0.0
        return sorted(files, key=key)

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
            "", "", "",
        ])
        self._set_date_cells(item, *self._log_date(path, st))
        self._set_corrupt_cell(item, self._cached_record(path, st) or {})
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

    def _set_corrupt_cell(self, item, rec):
        """Column 5: how much of the file the parser could not read.

        A percentage rather than a flag because the two failures it covers are
        not the same problem: 0.01% is three bad records in a 47-minute flight
        and changes nothing, while a percent or more means whole seconds are
        missing and any conclusion drawn across that gap is guesswork.  Real
        measured values on this library are 0.0131% and 0.0174%.
        """
        if rec.get("scan_v") != SCAN_VERSION:
            item.setText(5, "")
            item.setToolTip(5, "not read yet")
            return
        pct = rec.get("corrupt_pct", 0.0)
        nbytes = rec.get("corrupt_bytes", 0)
        if pct < 0:
            text, note = "?", ("flagged corrupt by pyulog, but the damaged span "
                               "could not be measured")
        elif nbytes <= 0:
            text, note = "ok", "parsed clean end to end"
        else:
            # Never round a corrupt file down to 0.00%: the whole point of the
            # column is that it is not clean.
            text = f"{pct:.3g}%" if pct >= 0.001 else "<0.001%"
            note = (f"{nbytes} bytes unreadable across "
                    f"{rec.get('corrupt_events', 0)} recovery point(s) -- the "
                    f"parser resynced and continued, so the rest of the log is "
                    f"good")
        item.setText(5, text)
        item.setToolTip(5, note)
        item.setForeground(5, QtGui.QColor(C_BAD if (pct > 0 or pct < 0) else C_MUTED))

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
        return self._picker_path()

    def _select_path(self, path):
        """Point the dropdown at `path`.  False if the library does not have it."""
        target = os.path.abspath(path)
        for k in range(self.picker.count()):
            p = self.picker.itemData(k, QtCore.Qt.UserRole)
            if p and os.path.abspath(p) == target:
                self.picker.blockSignals(True)
                self.picker.setCurrentIndex(k)
                self.picker.blockSignals(False)
                return True
        return False

    # -- loading
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
        started = _start_from_parsed(ulog) or 0.0
        facts = corruption_of(ulog, path)
        facts.update(started=started, date_src="gps" if started else "none",
                     scan_v=SCAN_VERSION)
        self._update_record(path, **facts)
        if facts["corrupt_bytes"]:
            self._log(f"  !! {facts['corrupt_bytes']} bytes unreadable "
                      f"({facts['corrupt_pct']:.3g}% of the file)")
        for it in self._iter_items():
            if it.data(0, QtCore.Qt.UserRole) == path:
                it.setText(1, f"{mins:.1f} min")
                self._set_date_cells(it, *self._log_date(path, os.stat(path)))
                self._set_corrupt_cell(it, self._cached_record(path, os.stat(path)) or {})
                self._refresh_picker_row(path)

        self.btn_params.setEnabled(True)
        self.page.clear()
        self.page.link_time = self.chk_link.isChecked()
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
            # A builder whose figure height depends on the log (the accel
            # plot's fault band is sized to its row count) states the pixels it
            # wants; spec.height is the fallback for the fixed-layout plots.
            self.page.add(spec.key, fig,
                          getattr(fig, "_page_height", spec.height))
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

    def keyPressEvent(self, event):
        # F2 used to be gated on the tree having focus; with the tree gone there
        # is no ambiguity about what it renames -- it is whatever is selected.
        if event.key() == QtCore.Qt.Key_F2:
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
    # -- parameter comparison
    def _choose_log(self, title, exclude=None):
        """Single-select log chooser.  None if cancelled.

        Same rows as the dropdown, so the date and duration are there to pick by
        -- with 36 identically-named HITL logs the name alone is not enough to
        choose the right one."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(820, 560)
        v = QtWidgets.QVBoxLayout(dlg)
        lst = QtWidgets.QListWidget()
        lst.setStyleSheet("font-family: monospace;")
        v.addWidget(lst, 1)
        skip = os.path.abspath(exclude) if exclude else None
        for it in self._iter_items():
            path = it.data(0, QtCore.Qt.UserRole)
            if not path or (skip and os.path.abspath(path) == skip):
                continue
            row = QtWidgets.QListWidgetItem(self._row_text(it))
            row.setData(QtCore.Qt.UserRole, path)
            lst.addItem(row)
        lst.itemDoubleClicked.connect(lambda *_: dlg.accept())
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        v.addWidget(btns)
        if dlg.exec_() != QtWidgets.QDialog.Accepted or not lst.currentItem():
            return None
        return lst.currentItem().data(QtCore.Qt.UserRole)

    def _compare_params(self):
        """Every parameter that differs between the open log and another."""
        if not self._current:
            return
        other = self._choose_log("Compare parameters with…", exclude=self._current)
        if not other:
            return
        try:
            pa, ia = params_of(self._current)
            pb, ib = params_of(other)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Compare parameters",
                                          f"{type(e).__name__}: {e}")
            return
        self._show_param_diff(self._current, other, pa, pb, ia, ib)

    def _show_param_diff(self, path_a, path_b, pa, pb, info_a, info_b):
        name_a, name_b = os.path.basename(path_a), os.path.basename(path_b)
        keys = sorted(set(pa) | set(pb))
        # A missing parameter is a difference, and usually the most informative
        # one -- it means the two logs are not even the same firmware build.
        diff = [k for k in keys if pa.get(k) != pb.get(k)]

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Parameter differences")
        dlg.resize(1000, 700)
        v = QtWidgets.QVBoxLayout(dlg)

        only_a = sum(1 for k in diff if k not in pb)
        only_b = sum(1 for k in diff if k not in pa)
        head = QtWidgets.QLabel(
            f"<b>{len(diff)}</b> of {len(keys)} parameters differ"
            + (f"  ·  {only_a} only in <b>{name_a}</b>" if only_a else "")
            + (f"  ·  {only_b} only in <b>{name_b}</b>" if only_b else ""))
        v.addWidget(head)

        fw_a = (info_a.get("ver_sw") or "")[:12]
        fw_b = (info_b.get("ver_sw") or "")[:12]
        if fw_a != fw_b:
            # Say this loudly: a firmware change explains a long diff list all by
            # itself, and reading those rows as configuration drift would be wrong.
            warn = QtWidgets.QLabel(
                f"⚠ different firmware: {name_a} on <tt>{fw_a}</tt>, "
                f"{name_b} on <tt>{fw_b}</tt> — some differences will be the "
                f"build, not the configuration")
            warn.setStyleSheet(f"color: {C_BAD};")
            warn.setWordWrap(True)
            v.addWidget(warn)

        filt = QtWidgets.QLineEdit()
        filt.setPlaceholderText("filter by name (e.g. EKF2, CAL_ACC, SDLOG)…")
        v.addWidget(filt)

        table = QtWidgets.QTableWidget(len(diff), 3)
        table.setHorizontalHeaderLabels(["parameter", name_a, name_b])
        table.setStyleSheet("font-family: monospace;")
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(False)          # rows are filled below; sort after
        for r, k in enumerate(diff):
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(k))
            for c, src in ((1, pa), (2, pb)):
                cell = QtWidgets.QTableWidgetItem(fmt_param(src.get(k)))
                if k not in src:
                    cell.setForeground(QtGui.QColor(C_MUTED))
                    cell.setToolTip("not present in this log")
                else:
                    cell.setForeground(QtGui.QColor(C_BAD))
                cell.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                table.setItem(r, c, cell)
        table.setSortingEnabled(True)
        # Enabling sorting applies the header's CURRENT indicator, which is not
        # ascending-by-column-0 until it is told to be -- without this the list
        # comes out reverse-alphabetical.
        table.sortItems(0, QtCore.Qt.AscendingOrder)
        table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents)
        for c in (1, 2):
            table.horizontalHeader().setSectionResizeMode(
                c, QtWidgets.QHeaderView.Stretch)
        v.addWidget(table, 1)

        def apply_filter(text):
            t = text.strip().upper()
            for r in range(table.rowCount()):
                table.setRowHidden(r, bool(t) and t not in table.item(r, 0).text())
        filt.textChanged.connect(apply_filter)

        row = QtWidgets.QHBoxLayout()
        def copy_all():
            # Walk the TABLE, not the source list: what lands on the clipboard is
            # then exactly what is on screen, in the order and with the filter
            # the reader is looking at.
            lines = [f"parameter\t{name_a}\t{name_b}"]
            for r in range(table.rowCount()):
                if table.isRowHidden(r):
                    continue
                lines.append("\t".join(table.item(r, c).text() for c in range(3)))
            QtWidgets.QApplication.clipboard().setText("\n".join(lines))
            self._log(f"copied {len(lines) - 1} parameter difference(s)")
        btn_copy = QtWidgets.QPushButton("Copy (tab-separated)")
        btn_copy.clicked.connect(copy_all)
        row.addWidget(btn_copy)
        row.addStretch(1)
        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        close.rejected.connect(dlg.reject)
        row.addWidget(close)
        v.addLayout(row)

        self._log(f"params: {len(diff)} difference(s) between {name_a} and {name_b}")
        dlg.exec_()

    def _pick_pdf_logs(self):
        """Choose which logs go in the report.  [] if the user cancelled.

        The tree's tick boxes went with the sidebar, so multi-select moved into a
        dialog rather than being dropped -- exporting a whole test session in one
        PDF is the reason the export exists.  Ticks are still stored on the tree
        rows, so a choice survives until the library is repopulated.
        """
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Export PDF - choose logs")
        dlg.resize(760, 520)
        v = QtWidgets.QVBoxLayout(dlg)
        v.addWidget(QtWidgets.QLabel("Tick the logs to include:"))
        lst = QtWidgets.QListWidget()
        lst.setStyleSheet("font-family: monospace;")
        v.addWidget(lst, 1)

        current = self._selected_path()
        checked = self._checked_paths()
        for it in self._iter_items():
            path = it.data(0, QtCore.Qt.UserRole)
            if not path:
                continue
            row = QtWidgets.QListWidgetItem(self._row_text(it))
            row.setData(QtCore.Qt.UserRole, path)
            row.setFlags(row.flags() | QtCore.Qt.ItemIsUserCheckable)
            # Default to the log on screen, so the common case -- "PDF of what I
            # am looking at" -- is one click through this dialog.
            on = (os.path.abspath(path) in checked or
                  (not checked and current and
                   os.path.abspath(path) == os.path.abspath(current)))
            row.setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)
            lst.addItem(row)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        v.addWidget(btns)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return []

        chosen = []
        for i in range(lst.count()):
            row = lst.item(i)
            path = row.data(QtCore.Qt.UserRole)
            on = row.checkState() == QtCore.Qt.Checked
            for it in self._iter_items():
                if it.data(0, QtCore.Qt.UserRole) == path:
                    it.setCheckState(0, QtCore.Qt.Checked if on
                                     else QtCore.Qt.Unchecked)
            if on:
                chosen.append(path)
        return sorted(chosen)

    def _export_pdf(self):
        paths = self._pick_pdf_logs()
        if not paths:
            return

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
        self._stop_library_scan()
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
