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
import faulthandler
import io
import json
import os
import re
import signal
import sys
import threading
import time
import traceback

import matplotlib
matplotlib.use("QtAgg")            # before any pyplot import, to match the shell

from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
import ulog_plots
from qt_common import NotesBox, PlotCanvas
from report_tab import ReportTab
from ulog_cache import (LogCache, MeasuredULog, corruption_of, parse_ulog,
                        start_epoch)
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


# --- crash log --------------------------------------------------------------
#
# Three different things kill this window, and each needs its own trap:
#
#   * a Python exception -- PyQt5 hands it to sys.excepthook and then calls
#     abort(), so when the browser was started from a launcher rather than a
#     terminal the traceback goes nowhere at all;
#   * a NATIVE crash inside Qt or matplotlib -- there is no Python exception to
#     catch, the process simply disappears;
#   * an out-of-memory kill -- SIGKILL, which by definition cannot be caught.
#
# sys.excepthook covers the first.  faulthandler covers the second: it dumps
# the Python stack from inside a signal handler, which is why it is given a
# real file object held open for the life of the process rather than something
# that buffers.  Nothing can cover the third, so the breadcrumbs below cover it
# instead -- each one carries the resident set size, so a log that stops mid-
# parse with RSS climbing past the free memory names the cause by itself.

CRASH_LOG = os.path.join(os.path.dirname(STATE_PATH), "log_browser_crash.log")
CRASH_LOG_MAX = 512 * 1024          # bytes; rotated to .1 past this
SESSION_MARK = "=== session "
EXIT_MARK = "=== clean exit "

_crash_fh = None


def _rss_mb():
    """This process's resident memory.  Cheap enough to stamp on every crumb."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1048576.0
    except (OSError, ValueError, IndexError):
        return float("nan")


def _mem_summary():
    """Free memory and swap at startup.

    Recorded because an out-of-memory kill leaves no other trace, and this box
    has 8 GB against ULogs that parse into hundreds of megabytes."""
    want = ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")
    vals = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in want:
                    vals[key] = int(rest.split()[0]) / 1048576.0
    except (OSError, ValueError):
        return "unknown"
    return "  ".join(f"{k}={vals[k]:.1f}GB" for k in want if k in vals)


def crumb(msg):
    """Record what the browser is ABOUT to do, in case it does not survive it.

    Flushed but not fsynced: a crashing process loses its Python buffer, not
    the kernel's page cache, so flush() is all a post-mortem needs."""
    if _crash_fh is None:
        return
    try:
        _crash_fh.write(f"{time.strftime('%H:%M:%S')} "
                        f"rss={_rss_mb():5.0f}MB  {msg}\n")
        _crash_fh.flush()
    except (OSError, ValueError):
        pass


def _previous_crash():
    """The last session's lines, if it never wrote its clean-exit marker."""
    try:
        with open(CRASH_LOG, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    starts = [i for i, ln in enumerate(lines) if ln.startswith(SESSION_MARK)]
    if not starts:
        return []
    last = lines[starts[-1]:]
    if any(ln.startswith(EXIT_MARK) for ln in last):
        return []                   # that session shut down cleanly
    return last


def install_crash_log():
    """Point every crash path at CRASH_LOG.

    Returns the previous session's lines if it died without writing its clean
    exit marker, so the browser can show them the moment it reopens."""
    global _crash_fh
    if _crash_fh is not None:
        return []
    try:
        os.makedirs(os.path.dirname(CRASH_LOG), exist_ok=True)
        if (os.path.exists(CRASH_LOG)
                and os.path.getsize(CRASH_LOG) > CRASH_LOG_MAX):
            os.replace(CRASH_LOG, CRASH_LOG + ".1")
        earlier = _previous_crash()
        _crash_fh = open(CRASH_LOG, "a", buffering=1, errors="replace")
    except OSError:
        return []                   # a browser that cannot log still runs

    faulthandler.enable(file=_crash_fh, all_threads=True)
    # SIGTERM is what a kill or a session logout sends.  SIGUSR1 is the manual
    # one: `kill -USR1 <pid>` dumps the stacks of a browser that has HUNG
    # rather than crashed, which none of the other traps here can see.
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        try:
            faulthandler.register(sig, file=_crash_fh, all_threads=True,
                                  chain=True)
        except (AttributeError, RuntimeError, OSError):
            pass

    def hook(etype, value, tb):
        try:
            _crash_fh.write(f"\n--- unhandled {etype.__name__} at "
                            f"{time.strftime('%H:%M:%S')} ---\n")
            traceback.print_exception(etype, value, tb, file=_crash_fh)
            _crash_fh.flush()
        except (OSError, ValueError):
            pass
        sys.__excepthook__(etype, value, tb)

    sys.excepthook = hook
    threading.excepthook = lambda arg: hook(arg.exc_type, arg.exc_value,
                                            arg.exc_traceback)

    _crash_fh.write(
        f"\n{SESSION_MARK}{time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"pid={os.getpid()} python={sys.version.split()[0]} "
        f"pyqt={QtCore.PYQT_VERSION_STR} qt={QtCore.QT_VERSION_STR} "
        f"mpl={matplotlib.__version__} ===\n"
        f"         {_mem_summary()}\n")
    _crash_fh.flush()
    return earlier


def close_crash_log():
    """Write the marker whose ABSENCE is how the next session detects a crash."""
    if _crash_fh is None:
        return
    try:
        _crash_fh.write(f"{EXIT_MARK}{time.strftime('%H:%M:%S')} ===\n")
        _crash_fh.flush()
    except (OSError, ValueError):
        pass


def _natural_key(name):
    """Digit runs compared as numbers, so log_9 sorts before log_10.

    Same rule as pull_log.natural_key; duplicated rather than imported because
    pull_log pulls in pymavlink, and the browser has no business requiring a
    MAVLink stack to list files on disk."""
    return [int(p) if p.isdigit() else p.lower()
            for p in re.split(r"(\d+)", name)]


def _fmt_size(n):
    return f"{n/1e6:.0f} MB" if n >= 1e6 else f"{n/1e3:.0f} kB"


# --- notes ------------------------------------------------------------------
# Free-text notes live in a sidecar beside the log rather than only in this
# browser's JSON state, for three reasons: they survive a wiped config, they
# travel with the folder when a run directory is copied off this machine, and
# they are readable (and greppable) without opening the GUI -- the same bargain
# FC_log_diag.txt already makes in the HITL run folders.  The rename path
# already knows how to carry sidecars along.
#
# The JSON state is the fallback for logs on read-only media (a mounted card, a
# share), so a note is never silently lost just because the folder said no.
NOTES_SUFFIX = "_notes.txt"


def notes_path_for(path):
    """<folder>/<name>.ulg -> <folder>/<name>_notes.txt"""
    stem = path[:-4] if path.lower().endswith(".ulg") else path
    return stem + NOTES_SUFFIX


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
                # Every builder makes its figure with plt.figure(), and pyplot
                # keeps a STRONG reference to each one in a global registry --
                # plus, on the Qt backend, a hidden window to manage it.
                # Dropping the canvas widget therefore frees NOTHING: the
                # figure, its artists and the arrays they close over stay alive
                # for the life of the process.  That is ~150 MB per log opened,
                # so a browsing session walks itself into the OOM killer, which
                # is SIGKILL and leaves no traceback.  plt.close() is what
                # actually releases it.
                fig = getattr(w, "figure", None)
                if fig is not None:
                    plt.close(fig)
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

    def figures(self):
        """Every figure currently on the page (spacer items excluded)."""
        out = []
        for i in range(self._box.count()):
            w = self._box.itemAt(i).widget()
            if w is not None and getattr(w, "figure", None) is not None:
                out.append(w.figure)
        return out

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


def scan_log(path):
    """Everything the library columns need that requires reading the file.

    One parse, because pyulog walks the whole file whatever you filter -- asking
    separately for the date and for the corruption measurement would double a
    23 s library scan for no gain."""
    ulog = MeasuredULog(path, message_name_filter_list=GPS_TOPICS)
    facts = corruption_of(ulog, path)
    facts["started"] = start_epoch(ulog) or 0.0
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
            ulog = parse_ulog(self.path, self.topics)
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
        self.state.setdefault("notes", {})        # fallback store, see notes_path_for
        self.state.setdefault("notes_open", False)
        self._notes_path = None       # which log the notes box currently holds
        self._thread = None
        self._worker = None
        self._current = None
        self._proc = None
        self._scan_thread = None
        self._scanner = None
        # One cache for the whole window.  The Report tab is handed the same
        # object, so a log opened in either tab is parsed once for both.
        self.cache = LogCache(log=lambda m: self._log(m))

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
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.tabs = QtWidgets.QTabWidget()
        outer.addWidget(self.tabs, 1)

        # Everything built below is the BROWSE tab -- one log, its seven plots.
        # The Report tab is a separate widget, added once the library tree it
        # reads its log list from exists.
        browse = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(browse)
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

        cv.addWidget(self._build_notes())

        self.page = PlotPage()
        cv.addWidget(self.page, 1)

        self.tabs.addTab(browse, "Browse")

        # One console under BOTH tabs rather than one each: the Report tab's
        # parses and saves are the same running commentary, and a second pane
        # would only halve the room each of them gets.
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(110)
        self.console.setStyleSheet("font-family: monospace; font-size: 11px;")
        outer.addWidget(self.console)

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

        # After the tree, because the Report tab asks it what logs exist.  It
        # shares this window's parse cache, so a log open in one tab is already
        # parsed for the other.
        self.report_tab = ReportTab(self.cache, self._library_entries, self._log,
                                    root=self._log_analysis_root())
        self.tabs.addTab(self.report_tab, "Report")

    def _library_entries(self):
        """[(path, one-line label)] for every log the library currently lists."""
        out = []
        for it in self._iter_items():
            path = it.data(0, QtCore.Qt.UserRole)
            if path:
                out.append((path, self._row_text(it)))
        return out

    @staticmethod
    def _log_analysis_root():
        """Where reports are kept: beside the logs, not in the config directory."""
        for label, path, _ in _default_roots():
            if label == "Log Analysis":
                return path
        return None

    # -- notes
    def _build_notes(self):
        """The collapsible free-text box that sits above the plots.

        Collapsed by default so it costs no vertical space on a log you are only
        skimming, but it opens itself for any log that already HAS a note -- a
        note you cannot see is a note you will not read.  When it is closed the
        header carries the first line, so the dropdown marker is not the only
        hint that something was written here.

        The behaviour lives in NotesBox now, because the Report tab wants the
        same box twice more (once per report, once per graph) and they differ
        only in where the text is persisted to."""
        def remember(on):
            self.state["notes_open"] = bool(on)
            _save_state(self.state)

        self.notes = NotesBox(
            label="Notes",
            placeholder="What you were testing, what went wrong, what to look "
                        "at next\u2026",
            tooltip="Free-text notes for the open log.  Saved automatically, "
                    "beside the .ulg.",
            on_save=self._save_log_notes,
            remember=(lambda: bool(self.state.get("notes_open")), remember))
        return self.notes

    def _save_log_notes(self, text):
        """Where this box's text goes: a sidecar beside the .ulg."""
        if self._notes_path is None:
            return
        self._write_notes(self._notes_path, text)
        self._refresh_picker_row(self._notes_path)   # the pencil may have changed

    def _flush_notes(self):
        """Write the buffer out if it changed.  Safe to call any number of times."""
        self.notes.flush()

    def _load_notes(self, path):
        """Point the box at another log.  Flushes the one it was holding first.

        Order matters: the flush has to happen while _notes_path still names the
        log the text belongs to, or the previous log's note lands on this one."""
        self.notes.flush()
        self._notes_path = path
        self.notes.set_text(self._read_notes(path) if path else "",
                            enabled=path is not None)

    def _has_note(self, path):
        """Cheap enough to ask once per dropdown row: one stat, or a dict hit."""
        if os.path.exists(notes_path_for(path)):
            return True
        return bool((self.state.get("notes") or {}).get(os.path.abspath(path)))

    def _read_notes(self, path):
        try:
            with open(notes_path_for(path), encoding="utf-8") as f:
                return f.read()
        except OSError:
            return (self.state.get("notes") or {}).get(os.path.abspath(path), "")

    def _write_notes(self, path, text):
        """Sidecar first, JSON state if the folder will not take it.

        An emptied note deletes the sidecar rather than leaving a 0-byte file
        next to the log."""
        side = notes_path_for(path)
        key = os.path.abspath(path)
        try:
            if text.strip():
                with open(side, "w", encoding="utf-8") as f:
                    f.write(text)
            elif os.path.exists(side):
                os.remove(side)
            self.state.get("notes", {}).pop(key, None)
        except OSError as e:
            # Read-only media, most likely.  Keep the note; say where it went,
            # once, so it is not a surprise when the folder is copied elsewhere.
            self.state.setdefault("notes", {})[key] = text
            self._log(f"  notes: {os.path.basename(side)} not writable ({e.strerror}); "
                      f"kept in {STATE_PATH}")
        _save_state(self.state)

    # -- the dropdown, projected from the tree
    def _row_text(self, item):
        """One dropdown line: the name, then the columns the tree used to show."""
        bits = [b for b in (item.text(1), item.text(2),
                            f"{item.text(3)} {item.text(4)}".strip(),
                            item.text(5)) if b and b != "—"]
        # Two columns of prefix on EVERY row, so the marked ones stand out
        # without knocking the names out of alignment in the monospace list.
        path = item.data(0, QtCore.Qt.UserRole)
        mark = "✎ " if path and self._has_note(path) else "  "
        name = mark + item.text(0)
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
    def report_crash(self, earlier):
        """Show what the LAST session was doing when it died, in the console.

        A crash log nobody reads is no better than no crash log, and the moment
        the window reopens is the only moment the user is certainly looking."""
        self._log(f"crash log: {CRASH_LOG}")
        if not earlier:
            return
        self._log("  !! the previous session did not exit cleanly -- its last "
                  "moments:")
        for line in earlier[-16:]:
            self._log(f"     {line}")

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
        self._notify_report_tab()

    # -- log dates, filled in behind the library
    def _notify_report_tab(self):
        """The library changed underneath a report -- re-resolve its logs.

        Renames are the reason: a report holds basenames, and the row it points
        at may now be called something else."""
        tab = getattr(self, "report_tab", None)
        if tab is not None:
            tab._refresh_logs_ui()

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
        self._load_notes(path)      # before the parse: notes are readable at once
        self.title.setText(f"{os.path.basename(path)}   —   reading…")
        self._log(f"reading {path}")
        topics = ulog_plots.all_topics(self.ctx)

        # A log the Report tab -- or a previous visit -- already read is redrawn
        # without touching the disk.  _on_parsed is reached the same way either
        # way, so everything downstream is unaware there was a cache at all.
        hit = self.cache.get(path, topics)
        if hit is not None:
            crumb(f"cached {os.path.basename(path)}")
            self._on_parsed(hit.ulog, path, 0.0)
            return

        self.busy.show()
        crumb(f"parse {os.path.basename(path)} "
              f"({os.path.getsize(path) / 1048576.0:.0f}MB)")

        self._thread = QtCore.QThread(self)
        self._worker = ParseWorker(path, topics)
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
        if secs:
            self.cache.put(path, ulog, ulog_plots.all_topics(self.ctx))
            self._log(f"  parsed in {secs:.1f}s")
        else:
            self._log("  from cache")
        crumb(f"parsed in {secs:.1f}s, building plots")
        mins = duration_min(ulog)
        self._remember_duration(path, mins)
        # The date comes free here: this parse already asked for the GPS topics
        # (the altitude plot needs them), so re-reading the file in the scanner
        # for a log the user just opened would be pure waste.
        started = start_epoch(ulog) or 0.0
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
            crumb(f"build {spec.key}")
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
        self._close_orphan_figures()
        crumb("page ready")

    def _close_orphan_figures(self):
        """Close figures pyplot is holding that no canvas on the page shows.

        The page closes the figures it displayed when the next log replaces
        them, but a builder that creates a figure and then returns None -- the
        "nothing plottable in this log" path, which several take -- leaves one
        behind that nothing else will ever reach."""
        shown = {id(f) for f in self.page.figures()}
        for num in plt.get_fignums():
            fig = plt.figure(num)
            if id(fig) not in shown:
                plt.close(fig)

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
        for suffix in (NOTES_SUFFIX, "_diag.txt", ".ulg:Zone.Identifier"):
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
        note = self.state.get("notes", {}).pop(os.path.abspath(path), None)
        if note:                    # only set for logs whose folder is read-only
            self.state["notes"][os.path.abspath(target)] = note
        _save_state(self.state)
        for src, dst in moves:
            self._log(f"renamed {os.path.basename(src)} -> {os.path.basename(dst)}")
        if self._current == path:
            self._current = target
        if self._notes_path == path:
            self._notes_path = target
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
        crumb("closing")
        self._flush_notes()
        if not self._offer_to_save_report():
            event.ignore()
            return
        self._stop_library_scan()
        self._teardown_thread()
        if getattr(self, "report_tab", None) is not None:
            self.report_tab.stop()
        if self._proc is not None:
            self._proc.kill()
        _save_state(self.state)
        super().closeEvent(event)


    def _offer_to_save_report(self):
        """True to carry on closing, False to stay open.

        Per-log notes autosave, so nothing else in this window can lose work on
        exit; a report is a document the user assembled and has to be asked
        about."""
        tab = getattr(self, "report_tab", None)
        if tab is None:
            return True
        tab.flush()
        if not tab.has_unsaved():
            return True
        where = os.path.basename(tab.report.path) if tab.report.path else None
        # Which way the overwrite would go is the whole question here, so say
        # it: the report file is shared with report_cli, and "unsaved changes"
        # alone does not tell you whether Save preserves work or destroys it.
        msg = (f"The report \u201c{tab.report.title or 'untitled'}\u201d has "
               f"changes that are not in "
               + (where or "any file") + ".\n\n")
        if where and not tab._disk_agrees():
            msg += (f"{where} has ALSO been rewritten on disk since this window "
                    f"opened it. Saving replaces that newer file with this "
                    f"window's version; discarding keeps it.\n\n")
        msg += "Save it before closing?"
        r = QtWidgets.QMessageBox.question(
            self, "logGraph", msg,
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard
            | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Save)
        if r == QtWidgets.QMessageBox.Cancel:
            return False
        if r == QtWidgets.QMessageBox.Save:
            tab._save_report()
        return True


def browse(paths=(), ctx=None):
    """Open the browser.  Blocks until the window is closed."""
    # Before the QApplication: a crash while Qt is coming up is still a crash,
    # and it is the one with the least other evidence.
    earlier = install_crash_log()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    crumb("qt up, building window")
    win = Browser(paths=[p for p in paths if p], ctx=ctx)
    win.show()
    win.report_crash(earlier)
    crumb("window shown")
    app.exec_()
    close_crash_log()
    return win


if __name__ == "__main__":
    browse(sys.argv[1:])
