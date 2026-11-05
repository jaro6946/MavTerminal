#!/usr/bin/env python3
"""bag_tab.py -- the browser's Bags tab: play a day's bag video and bookmark it.

A "bag" here is one folder under `Log Analysis/Bags/bag analysis/`, holding the
combined camera + data-panel MP4 that `Bags/make_bag_video/make_bag_video.sh`
renders from that day's ROS 2 bags.  The tab plays it and lets you drop
BOOKMARKS -- a point in time with a title and notes -- which list on the right;
selecting one parks the video, PAUSED, on that bookmark's frame.

Bookmarks live in a sidecar beside the video, `<video stem>_bookmarks.json`,
the same convention as the logs' `_notes.txt` / `_tags.txt`: the annotation
travels with the file it annotates, and a script can read it without the GUI.

A date folder under `Bags/` that has bags but no video yet is listed too, with
a "Make video" button that runs make_bag_video.sh and streams its output into
the console.

Why frames are painted by hand instead of with QVideoWidget: under WSLg the
widget renders through a native overlay that `grab()` cannot see (a screenshot
of it is solid black), so it could not be checked headlessly.  A
QAbstractVideoSurface receives every decoded frame as memory instead; painting
that is a few lines, and a paused seek delivers exactly one frame -- which is
precisely the "park on the bookmark" behaviour this tab needs.

Acronyms: GUI = graphical user interface, ROS = Robot Operating System,
WSLg = Windows Subsystem for Linux GUI support.
"""
import json
import os
import re
import sys

from PyQt5 import QtCore, QtGui, QtMultimedia, QtWidgets

from ulog_common import C_BAD, C_MUTED

__all__ = ["BagTab", "scan_bags", "script_env", "bookmarks_path_for",
           "load_bookmarks", "save_bookmarks", "fmt_time"]

ANALYSIS_DIR = "bag analysis"
SCRIPT_REL = os.path.join("make_bag_video", "make_bag_video.sh")
PREFERRED_VIDEO = "grazer10_all_bags_combined.mp4"
BOOKMARKS_SUFFIX = "_bookmarks.json"
DEFAULT_FPS = 10.0            # make_bag_video renders a fixed 10 fps grid

C_BOOKMARK_SEL = "#0b4fa8"    # side-tab label

# Bookmark colours, by NAME in the sidecar (readable, and a palette tweak here
# recolours old files).  Blue is the default; an unknown name falls back to it.
BOOKMARK_COLORS = {"blue": "#2a78d6", "red": "#d62f2f", "orange": "#e08a00",
                   "green": "#1e9e5a", "purple": "#8a3fb5", "grey": "#8c8c85"}
DEFAULT_COLOR = "blue"


def color_hex(name):
    return BOOKMARK_COLORS.get(name, BOOKMARK_COLORS[DEFAULT_COLOR])

_DATE_RE = re.compile(r"^(\d{4})_(\d{2})_(\d{2})$")
# An analysis folder covers one date, or a day range within a month:
# 2026_11_04, 2026_06_20-23.
_COVER_RE = re.compile(r"^(\d{4})_(\d{2})_(\d{2})(?:-(\d{2}))?$")


# --- model (no Qt) ---------------------------------------------------------

def fmt_time(seconds):
    """73.4 -> '1:13.4'; 3700 -> '1:01:40.0'.  Tenths, because the video is 10 fps."""
    if seconds is None:
        return "—"
    s = max(0.0, float(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h >= 1:
        return f"{int(h)}:{int(m):02d}:{sec:04.1f}"
    return f"{int(m)}:{sec:04.1f}"


def _covered_dates(name):
    """'2026_06_20-23' -> {'2026_06_20', ..., '2026_06_23'}; non-matching -> {name}."""
    m = _COVER_RE.match(name)
    if not m:
        return {name}
    y, mo, d0, d1 = m.groups()
    d1 = d1 or d0
    return {f"{y}_{mo}_{d:02d}" for d in range(int(d0), int(d1) + 1)}


def _pick_video(folder):
    try:
        mp4s = sorted(f for f in os.listdir(folder) if f.lower().endswith(".mp4"))
    except OSError:
        return None
    if not mp4s:
        return None
    return os.path.join(folder, PREFERRED_VIDEO if PREFERRED_VIDEO in mp4s else mp4s[0])


def _count_bags(folder):
    """Sub-folders that are real bags (make_bag_video skips ones without metadata)."""
    try:
        return sum(os.path.isfile(os.path.join(folder, d, "metadata.yaml"))
                   for d in os.listdir(folder))
    except OSError:
        return 0


def scan_bags(bags_root):
    """Everything the picker lists, as dicts sorted by name.

    {'name', 'video' (path or None), 'folder' (analysis folder or None),
     'date' (Bags/<date> name make_bag_video takes, or None), 'n_bags'}
    Analysis folders with a video come first in meaning; a date folder appears
    on its own only when no analysis folder covers it yet.
    """
    out, covered = [], set()
    adir = os.path.join(bags_root, ANALYSIS_DIR)
    if os.path.isdir(adir):
        for name in os.listdir(adir):
            folder = os.path.join(adir, name)
            if not os.path.isdir(folder):
                continue
            video = _pick_video(folder)
            if video is None:
                continue          # a half-made folder: let its date show as "no video"
            dates = _covered_dates(name)
            covered |= dates
            n = sum(_count_bags(os.path.join(bags_root, d)) for d in dates)
            out.append({"name": name, "video": video, "folder": folder,
                        "date": name if _DATE_RE.match(name) else None,
                        "n_bags": n})
    if os.path.isdir(bags_root):
        for name in os.listdir(bags_root):
            if not _DATE_RE.match(name) or name in covered:
                continue
            n = _count_bags(os.path.join(bags_root, name))
            if n:
                out.append({"name": name, "video": None, "folder": None,
                            "date": name, "n_bags": n})
    out.sort(key=lambda e: e["name"])
    return out


def bookmarks_path_for(video):
    return os.path.splitext(video)[0] + BOOKMARKS_SUFFIX


def load_bookmarks(video):
    """[{'title','t_s','notes'}] sorted by time; [] if none/unreadable."""
    try:
        with open(bookmarks_path_for(video)) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    out = []
    for b in data.get("bookmarks", []):
        try:
            out.append({"title": str(b.get("title", "")),
                        "t_s": float(b["t_s"]),
                        "notes": str(b.get("notes", "")),
                        "color": (b.get("color") if b.get("color") in BOOKMARK_COLORS
                                  else DEFAULT_COLOR)})
        except (KeyError, TypeError, ValueError):
            continue
    out.sort(key=lambda b: b["t_s"])
    return out


def save_bookmarks(video, bookmarks):
    """Atomic: write a temp file and rename over, so a crash never leaves half a JSON."""
    path = bookmarks_path_for(video)
    tmp = path + ".tmp"
    data = {"video": os.path.basename(video),
            "bookmarks": [{"title": b["title"], "t_s": round(b["t_s"], 3),
                           "notes": b["notes"],
                           "color": b.get("color", DEFAULT_COLOR)}
                          for b in sorted(bookmarks, key=lambda b: b["t_s"])]}
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    return path


def script_env():
    """The environment make_bag_video.sh runs in: this one minus the venv.

    The script wants the SYSTEM python3 -- it sources ROS 2 Jazzy, whose
    rosbag2_py is built against it, and its pylib relies on system numpy.  This
    browser runs from the rotorpy venv, whose bin/ would otherwise come first on
    PATH and hand the script the wrong interpreter.
    """
    env = QtCore.QProcessEnvironment.systemEnvironment()
    # VIRTUAL_ENV is only set by `activate`; the mavTerminal alias runs the
    # venv's python directly, so fall back to sys.prefix (the venv root when
    # running in one).  NOT realpath(sys.executable): a venv's python is a
    # symlink to /usr/bin/python3, which would strip /usr/bin itself.
    venv = env.value("VIRTUAL_ENV") or (
        sys.prefix if sys.prefix != sys.base_prefix else "")
    for k in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"):
        env.remove(k)
    keep = [p for p in env.value("PATH").split(os.pathsep)
            if p and not (venv and os.path.realpath(p).startswith(os.path.realpath(venv)))]
    env.insert("PATH", os.pathsep.join(keep))
    return env


# --- video plumbing ----------------------------------------------------------

class FrameSurface(QtMultimedia.QAbstractVideoSurface):
    """Receives decoded frames from QMediaPlayer and re-emits them as QImages."""

    frame = QtCore.pyqtSignal(QtGui.QImage)

    _FORMATS = [QtMultimedia.QVideoFrame.Format_RGB32,
                QtMultimedia.QVideoFrame.Format_ARGB32,
                QtMultimedia.QVideoFrame.Format_ARGB32_Premultiplied,
                QtMultimedia.QVideoFrame.Format_RGB24]

    def supportedPixelFormats(self, handle_type=QtMultimedia.QAbstractVideoBuffer.NoHandle):
        # Only formats QImage can wrap directly, so GStreamer does the colour
        # conversion and present() is a copy, not a YUV decode in Python.
        if handle_type == QtMultimedia.QAbstractVideoBuffer.NoHandle:
            return self._FORMATS
        return []

    def present(self, vframe):
        f = QtMultimedia.QVideoFrame(vframe)
        if not f.map(QtMultimedia.QAbstractVideoBuffer.ReadOnly):
            return False
        try:
            fmt = QtMultimedia.QVideoFrame.imageFormatFromPixelFormat(f.pixelFormat())
            # .copy(): the QImage would otherwise point into the frame's buffer,
            # which GStreamer recycles the moment we unmap.
            img = QtGui.QImage(f.bits(), f.width(), f.height(),
                               f.bytesPerLine(), fmt).copy()
        finally:
            f.unmap()
        self.frame.emit(img)
        return True


class VideoView(QtWidgets.QWidget):
    """Paints the latest frame, letterboxed, aspect kept.  Click = play/pause."""

    clicked = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._img = None
        self._msg = "pick a bag"
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)

    def set_frame(self, img):
        self._img = img
        self._msg = None
        self.update()

    def clear(self, msg):
        self._img, self._msg = None, msg
        self.update()

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtCore.Qt.black)
        if self._img is not None and not self._img.isNull():
            size = self._img.size().scaled(self.size(), QtCore.Qt.KeepAspectRatio)
            r = QtCore.QRect(QtCore.QPoint(0, 0), size)
            r.moveCenter(self.rect().center())
            p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
            p.drawImage(r, self._img)
        elif self._msg:
            p.setPen(QtGui.QColor("#bbbbbb"))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap,
                       self._msg)
        p.end()

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(ev)


class BookmarkSlider(QtWidgets.QSlider):
    """The scrubber, with a tick under the groove for every bookmark.

    Painted INTO the slider rather than as a strip beside it so the ticks line
    up with the handle exactly -- the groove's pixel extent comes from the
    style, and a separate widget would have to guess its margins.
    """

    def __init__(self, parent=None):
        super().__init__(QtCore.Qt.Horizontal, parent)
        self.ticks = []           # [(ms, hex colour)]
        self.selected = -1
        self.setMinimumHeight(28)

    def _x(self, ms, groove):
        span = max(1, self.maximum() - self.minimum())
        return groove.left() + (ms - self.minimum()) / span * groove.width()

    def mousePressEvent(self, ev):
        """A click on the groove jumps THERE (Qt's default is one page-step toward
        it), then the press carries on as a drag of the handle now under it."""
        if ev.button() == QtCore.Qt.LeftButton:
            opt = QtWidgets.QStyleOptionSlider()
            self.initStyleOption(opt)
            handle = self.style().subControlRect(
                QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderHandle, self)
            if not handle.contains(ev.pos()):
                half = handle.width() // 2
                val = QtWidgets.QStyle.sliderValueFromPosition(
                    self.minimum(), self.maximum(), ev.pos().x() - half,
                    max(1, self.width() - handle.width()))
                self.setValue(val)
                self.sliderMoved.emit(val)
        super().mousePressEvent(ev)

    def paintEvent(self, ev):
        super().paintEvent(ev)
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = self.style().subControlRect(
            QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderGroove, self)
        # Inset by half a handle so the extremes line up with the handle CENTRE,
        # which is where the slider's value sits.
        handle = self.style().subControlRect(
            QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderHandle, self)
        half = handle.width() // 2
        g = QtCore.QRect(groove.left() + half, groove.top(),
                         max(1, groove.width() - 2 * half), groove.height())
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtCore.Qt.NoPen)
        # An upward triangle under the groove, its point at the bookmark time.
        y0 = self.height() - 1
        # Selection is shown by size and a dark outline, not by colour -- the
        # colour belongs to the bookmark.
        for i, (ms, col) in enumerate(self.ticks):
            x = self._x(ms, g)
            sel = i == self.selected
            w, h = (6, 10) if sel else (4, 7)
            p.setPen(QtGui.QPen(QtGui.QColor("#202020"), 1.2) if sel else QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(col))
            p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(x, y0 - h),
                                           QtCore.QPointF(x - w, y0),
                                           QtCore.QPointF(x + w, y0)]))
        p.end()


class SideTab(QtWidgets.QPushButton):
    """A narrow full-height button with its label running bottom-to-top, like
    a docked-panel tab -- what is left of the bookmarks panel when collapsed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(30)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                           QtWidgets.QSizePolicy.Expanding)
        self.setCursor(QtCore.Qt.PointingHandCursor)

    def paintEvent(self, _ev):
        p = QtWidgets.QStylePainter(self)
        opt = QtWidgets.QStyleOptionButton()
        self.initStyleOption(opt)
        opt.text = ""
        p.drawControl(QtWidgets.QStyle.CE_PushButton, opt)
        f = p.font()
        f.setBold(True)
        p.setFont(f)
        p.setPen(QtGui.QColor(C_BOOKMARK_SEL))
        # The arrow is drawn UPRIGHT at the top: a glyph inside rotated text
        # turns with it, so a "◂" there would not point left.
        p.drawText(QtCore.QRect(0, 6, self.width(), 20), QtCore.Qt.AlignCenter, "◂")
        p.rotate(-90)
        # rotated by -90: x runs up the widget (negative), y runs right
        p.drawText(QtCore.QRect(-self.height(), 0, self.height(), self.width()),
                   QtCore.Qt.AlignCenter, self.text())
        p.end()


# --- the tab ---------------------------------------------------------------

class BagTab(QtWidgets.QWidget):
    """Bag picker + video player + bookmark list.  See the module docstring."""

    def __init__(self, bags_root, log, state=None, parent=None):
        super().__init__(parent)
        self.bags_root = bags_root
        self._log = log
        self.state = state if state is not None else {}
        self.entries = []
        self.video = None             # path of the loaded MP4
        self.bookmarks = []
        self._fps = DEFAULT_FPS
        self._scrubbing = False
        self._suppress_select = False
        self._proc = None
        self._pending_seek = None     # ms to park on once the media has loaded

        self.player = QtMultimedia.QMediaPlayer(
            self, QtMultimedia.QMediaPlayer.VideoSurface)
        self.surface = FrameSurface(self)
        self.player.setVideoOutput(self.surface)
        self.player.setNotifyInterval(50)
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self._on_duration)
        self.player.stateChanged.connect(self._on_state)
        self.player.mediaStatusChanged.connect(self._on_status)
        self.player.error.connect(self._on_error)

        # Title/notes typing is debounced into one write, not one per keystroke.
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(600)
        self._save_timer.timeout.connect(self._save)

        self._build_ui()
        self.surface.frame.connect(self.view.set_frame)
        self.refresh(select=self.state.get("bag_last"))

    # -- construction
    def _build_ui(self):
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        bar = QtWidgets.QWidget()
        bh = QtWidgets.QHBoxLayout(bar)
        bh.setContentsMargins(10, 6, 10, 6)
        bh.addWidget(QtWidgets.QLabel("bag:"))
        self.picker = QtWidgets.QComboBox()
        self.picker.setStyleSheet("font-family: monospace;")
        self.picker.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLength)
        self.picker.setMinimumContentsLength(40)
        self.picker.activated.connect(self._picked)
        bh.addWidget(self.picker, 1)
        b = QtWidgets.QPushButton("Refresh")
        b.clicked.connect(lambda: self.refresh(select=self._current_name()))
        bh.addWidget(b)
        self.btn_make = QtWidgets.QPushButton("Make video")
        self.btn_make.setToolTip(
            "Run make_bag_video.sh for this date (minutes; output streams to the "
            "console below)")
        self.btn_make.setEnabled(False)
        self.btn_make.clicked.connect(self._make_video)
        bh.addWidget(self.btn_make)
        self.busy = QtWidgets.QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setFixedWidth(120)
        self.busy.hide()
        bh.addWidget(self.busy)
        v.addWidget(bar)

        # Video on the left, bookmarks panel on the right.  The video is 1280 x
        # 1324 (data panel stacked over the camera), taller than wide, so it is
        # height-limited and a side panel mostly takes width the letterboxing
        # was wasting anyway.  The panel still collapses (Hide ▸) for when
        # the window is narrow.  The divider is draggable and remembered.
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setChildrenCollapsible(False)
        # The splitter shares a row with the collapsed panel's tab, which is the
        # only thing left on screen to bring the panel back.
        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(split, 1)
        self.btn_show = SideTab()
        self.btn_show.setToolTip("Show the bookmarks panel")
        self.btn_show.clicked.connect(lambda: self._set_strip_open(True))
        body.addWidget(self.btn_show)
        v.addLayout(body, 1)
        self.split = split

        # left: video + transport
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(10, 0, 4, 6)
        self.view = VideoView()
        self.view.clicked.connect(self._toggle_play)
        lv.addWidget(self.view, 1)

        self.slider = BookmarkSlider()
        self.slider.setRange(0, 0)
        self.slider.sliderPressed.connect(lambda: setattr(self, "_scrubbing", True))
        self.slider.sliderReleased.connect(self._scrub_done)
        self.slider.sliderMoved.connect(lambda ms: self.player.setPosition(ms))
        lv.addWidget(self.slider)

        tr = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("▶")
        self.btn_play.setToolTip("Play / pause (Space, or click the video)")
        self.btn_play.setFixedWidth(44)
        self.btn_play.clicked.connect(self._toggle_play)
        tr.addWidget(self.btn_play)
        for label, delta, tip in (("−5 s", -5.0, "Back 5 s and play"),
                                  ("−0.5 s", -0.5, "Back 0.5 s and play"),
                                  ("◀|", None, "Back one frame (,)"),
                                  ("|▶", None, "Forward one frame (.)"),
                                  ("+0.5 s", 0.5, "Forward 0.5 s and play"),
                                  ("+5 s", 5.0, "Forward 5 s and play")):
            btn = QtWidgets.QPushButton(label)
            btn.setToolTip(tip)
            btn.setFixedWidth(62 if "." in label else 52)
            if delta is not None:
                btn.clicked.connect(lambda _=False, d=delta: self._nudge(d))
            elif label == "◀|":
                btn.clicked.connect(lambda: self._step(-1))
            else:
                btn.clicked.connect(lambda: self._step(+1))
            tr.addWidget(btn)
        self.time_lbl = QtWidgets.QLabel("0:00.0 / 0:00.0")
        self.time_lbl.setStyleSheet("font-family: monospace; padding: 0 8px;")
        tr.addWidget(self.time_lbl)
        # On the transport row: one button no longer earns a row of its own,
        # and that row is video height.
        self.btn_add = QtWidgets.QPushButton("+ Add bookmark")
        self.btn_add.setToolTip("Bookmark the current position")
        self.btn_add.clicked.connect(self._add_bookmark)
        tr.addWidget(self.btn_add)
        tr.addStretch(1)
        tr.addWidget(QtWidgets.QLabel("speed"))
        self.speed = QtWidgets.QComboBox()
        for r in ("0.25", "0.5", "1", "2", "4"):
            self.speed.addItem(r + "×", float(r))
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(
            lambda i: self.player.setPlaybackRate(self.speed.itemData(i)))
        tr.addWidget(self.speed)
        lv.addLayout(tr)

        split.addWidget(left)

        # right: bookmark list over the editor
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(4, 0, 10, 6)
        hrow = QtWidgets.QHBoxLayout()
        hdr = QtWidgets.QLabel("Bookmarks")
        hdr.setStyleSheet("font-weight: 600;")
        hrow.addWidget(hdr)
        hrow.addStretch(1)
        btn_hide = QtWidgets.QPushButton("Hide ▸")
        btn_hide.setToolTip("Collapse this panel; the video takes the width")
        btn_hide.clicked.connect(lambda: self._set_strip_open(False))
        hrow.addWidget(btn_hide)
        rv.addLayout(hrow)
        self.list = QtWidgets.QListWidget()
        self.list.setStyleSheet("font-family: monospace;")
        self.list.currentRowChanged.connect(self._row_changed)
        # Clicking the row that is ALREADY selected changes no row, so it would
        # not re-seek; itemClicked covers "take me back to this one".
        self.list.itemClicked.connect(lambda _it: self._goto_selected())
        rv.addWidget(self.list, 1)

        self.editor = QtWidgets.QGroupBox("Selected bookmark")
        form = QtWidgets.QFormLayout(self.editor)
        self.ed_title = QtWidgets.QLineEdit()
        self.ed_title.setPlaceholderText("title")
        self.ed_title.textEdited.connect(self._edited)
        form.addRow("Title", self.ed_title)
        self.ed_time = QtWidgets.QLabel()
        self.ed_time.setStyleSheet("font-family: monospace;")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.ed_time, 1)
        b = QtWidgets.QPushButton("= playhead")
        b.setToolTip("Move this bookmark to the current position")
        b.clicked.connect(self._set_time)
        row.addWidget(b)
        form.addRow("Time", row)
        crow = QtWidgets.QHBoxLayout()
        crow.setSpacing(4)
        self.color_group = QtWidgets.QButtonGroup(self)
        self.color_group.setExclusive(True)
        self.color_btns = {}
        for name, hexc in BOOKMARK_COLORS.items():
            cb = QtWidgets.QToolButton()
            cb.setCheckable(True)
            cb.setToolTip(name)
            cb.setFixedSize(26, 22)
            # checked = thick dark ring, so the current colour reads at a glance
            cb.setStyleSheet(
                f"QToolButton {{ background: {hexc}; border: 2px solid #ffffff;"
                f" border-radius: 4px; }}"
                f"QToolButton:checked {{ border: 3px solid #202020; }}")
            cb.clicked.connect(lambda _=False, n=name: self._set_color(n))
            self.color_group.addButton(cb)
            self.color_btns[name] = cb
            crow.addWidget(cb)
        crow.addStretch(1)
        form.addRow("Color", crow)
        self.ed_notes = QtWidgets.QPlainTextEdit()
        self.ed_notes.setPlaceholderText("notes")
        self.ed_notes.textChanged.connect(self._notes_changed)
        form.addRow("Notes", self.ed_notes)
        self.btn_del = QtWidgets.QPushButton("Delete bookmark")
        self.btn_del.setStyleSheet(f"color: {C_BAD};")
        self.btn_del.clicked.connect(self._delete_bookmark)
        form.addRow("", self.btn_del)
        rv.addWidget(self.editor, 2)
        split.addWidget(right)
        self.strip = right
        split.setStretchFactor(0, 1)      # window resizes go to the video
        split.setStretchFactor(1, 0)
        # Own key: "bag_split" held the sizes of the short-lived vertical layout,
        # which would be nonsense applied to this one.
        self.state.pop("bag_split", None)
        sizes = self.state.get("bag_split_h")
        split.setSizes(sizes if isinstance(sizes, list) and len(sizes) == 2
                       else [1100, 420])
        split.splitterMoved.connect(
            lambda *_: self.state.__setitem__("bag_split_h", split.sizes()))

        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet(f"color: {C_MUTED}; padding: 2px 10px;")
        v.addWidget(self.status)

        # Shortcuts are scoped to this tab.  Text fields still get their own keys:
        # QLineEdit / QPlainTextEdit claim printable characters (incl. space,
        # ',') through ShortcutOverride before a shortcut can fire.
        for key, slot in (("Space", self._toggle_play),
                          (",", lambda: self._step(-1)),
                          (".", lambda: self._step(+1))):
            sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)

        self._fill_editor(None)
        self._set_media_enabled(False)
        # Own key: the panel used to be the "sections" one.
        self.state.pop("bag_sections_open", None)
        self._set_strip_open(bool(self.state.get("bag_bookmarks_open", True)))

    # -- picker
    def _current_name(self):
        i = self.picker.currentIndex()
        return self.entries[i]["name"] if 0 <= i < len(self.entries) else None

    def refresh(self, select=None):
        self.entries = scan_bags(self.bags_root)
        self.picker.clear()
        for e in self.entries:
            if e["video"]:
                n = len(load_bookmarks(e["video"]))
                extra = f"{n} bookmark{'s' if n != 1 else ''}" if n else ""
                txt = f"{e['name']:<16} {e['n_bags']:>2} bags   video   {extra}"
            else:
                txt = f"{e['name']:<16} {e['n_bags']:>2} bags   no video yet"
            self.picker.addItem(txt)
            if not e["video"]:
                self.picker.setItemData(self.picker.count() - 1,
                                        QtGui.QColor(C_MUTED), QtCore.Qt.ForegroundRole)
        if not self.entries:
            self.view.clear(f"no bags under {self.bags_root}")
            return
        names = [e["name"] for e in self.entries]
        idx = names.index(select) if select in names else len(names) - 1
        self.picker.setCurrentIndex(idx)
        if self.entries[idx]["video"] != self.video or not self.video:
            self._picked(idx)
        else:
            self._update_make_btn()

    def _picked(self, idx):
        if not 0 <= idx < len(self.entries):
            return
        e = self.entries[idx]
        self.state["bag_last"] = e["name"]
        self._update_make_btn()
        if e["video"] == self.video and self.video:
            return
        self._flush()
        self._load_video(e)

    def _update_make_btn(self):
        i = self.picker.currentIndex()
        e = self.entries[i] if 0 <= i < len(self.entries) else None
        self.btn_make.setEnabled(bool(e and not e["video"] and e["date"]
                                      and self._proc is None))

    # -- media
    def _load_video(self, e):
        self.player.stop()
        if not e["video"]:
            self.video = None
            self.bookmarks = []
            self._rebuild_list()
            self.player.setMedia(QtMultimedia.QMediaContent())
            self.view.clear(f"{e['name']}: {e['n_bags']} bags, no video yet.\n\n"
                            "Make video renders it with make_bag_video.sh.")
            self._set_media_enabled(False)
            self.status.setText("")
            return
        self.video = e["video"]
        self.bookmarks = load_bookmarks(self.video)
        self._rebuild_list()
        self.view.clear("loading…")
        self._pending_seek = 0
        self.player.setMedia(QtMultimedia.QMediaContent(
            QtCore.QUrl.fromLocalFile(self.video)))
        self._set_media_enabled(True)
        self.status.setText(f"{self.video}   ·   bookmarks → "
                            f"{os.path.basename(bookmarks_path_for(self.video))}")
        self._log(f"[bags] {e['name']}: {os.path.basename(self.video)}, "
                  f"{len(self.bookmarks)} bookmark(s)")

    def _set_media_enabled(self, on):
        for w in (self.slider, self.btn_play, self.btn_add, self.speed):
            w.setEnabled(on)

    def _on_status(self, status):
        MS = QtMultimedia.QMediaPlayer
        if status == MS.LoadedMedia and self._pending_seek is not None:
            fps = self.player.metaData("VideoFrameRate")
            try:
                self._fps = float(fps) if fps and float(fps) > 0 else DEFAULT_FPS
            except (TypeError, ValueError):
                self._fps = DEFAULT_FPS
            ms, self._pending_seek = self._pending_seek, None
            self._park(ms)
        elif status == MS.EndOfMedia:
            self.btn_play.setText("▶")
        elif status == MS.InvalidMedia:
            self.view.clear(f"cannot play {self.video}")

    def _on_error(self, _err):
        self._log(f"[bags] player error: {self.player.errorString()}")
        self.view.clear(f"player error: {self.player.errorString()}")

    def _on_duration(self, ms):
        self.slider.setRange(0, max(0, ms))
        self._update_time(self.player.position())

    def _on_position(self, ms):
        if not self._scrubbing:
            self.slider.blockSignals(True)
            self.slider.setValue(ms)
            self.slider.blockSignals(False)
        self._update_time(ms)

    def _on_state(self, st):
        self.btn_play.setText("❚❚" if st == QtMultimedia.QMediaPlayer.PlayingState
                              else "▶")

    def _update_time(self, ms):
        self.time_lbl.setText(f"{fmt_time(ms / 1000)} / "
                              f"{fmt_time(self.player.duration() / 1000)}")

    def _scrub_done(self):
        self._scrubbing = False
        self.player.setPosition(self.slider.value())

    def _park(self, ms):
        """Pause and show the frame at `ms` -- the 'ready to play' state.

        Order matters: pause() first puts GStreamer in PAUSED (prerolled), and a
        seek in PAUSED presents exactly one frame at the new position.  Seeking
        first from STOPPED shows nothing until play.
        """
        if not self.video:
            return
        dur = self.player.duration()
        ms = int(max(0, min(ms, dur if dur > 0 else ms)))
        self.player.pause()
        self.player.setPosition(ms)
        self._on_position(ms)

    def _toggle_play(self):
        if not self.video:
            return
        P = QtMultimedia.QMediaPlayer
        if self.player.state() == P.PlayingState:
            self.player.pause()
        else:
            if self.player.mediaStatus() == P.EndOfMedia:
                self.player.setPosition(0)
            self.player.play()

    def _nudge(self, seconds):
        """±5 s / ±0.5 s jump and PLAY from there -- a skip, not a frame step.  (Frame
        steps and bookmark selection park paused; those are for inspecting.)"""
        if not self.video:
            return
        dur = self.player.duration()
        ms = self.player.position() + seconds * 1000
        ms = int(max(0, min(ms, dur - 1 if dur > 0 else ms)))
        self.player.setPosition(ms)
        self._on_position(ms)
        self.player.play()

    def _step(self, n):
        self._park(self.player.position() + n * 1000.0 / self._fps)

    # -- bookmarks
    def _pos_s(self):
        return self.player.position() / 1000.0

    def _add_bookmark(self):
        if not self.video:
            return
        bm = {"title": f"Bookmark {len(self.bookmarks) + 1}",
              "t_s": self._pos_s(), "notes": "", "color": DEFAULT_COLOR}
        self.bookmarks.append(bm)
        self._save()
        self._rebuild_list(select=bm, seek=False)
        # Straight into naming it: the default title is a placeholder, so the
        # panel has to be open for that.
        self._set_strip_open(True)
        # Deferred: focus set in the same tick the panel is re-shown is lost.
        QtCore.QTimer.singleShot(0, self._focus_title)

    def _focus_title(self):
        self.ed_title.setFocus()
        self.ed_title.selectAll()

    def _row_text(self, b):
        title = b["title"] or "(untitled)"
        mark = "  ✎" if b["notes"].strip() else ""
        return f"{fmt_time(b['t_s']):>8}  {title}{mark}"

    def _rebuild_list(self, select=None, seek=False):
        self.bookmarks.sort(key=lambda b: b["t_s"])
        self._suppress_select = not seek
        self.list.clear()
        for b in self.bookmarks:
            it = QtWidgets.QListWidgetItem(self._row_text(b))
            it.setIcon(self._swatch(b.get("color")))
            if b["notes"].strip():
                it.setToolTip(b["notes"])
            self.list.addItem(it)
        row = -1
        if select is not None:
            row = next((i for i, b in enumerate(self.bookmarks) if b is select), -1)
        self.list.setCurrentRow(row)
        self._suppress_select = False
        self._fill_editor(self.bookmarks[row] if row >= 0 else None)
        self._update_ticks()

    def _update_ticks(self):
        self.slider.ticks = [(int(b["t_s"] * 1000), color_hex(b.get("color")))
                             for b in self.bookmarks]
        self.slider.selected = self.list.currentRow()
        self.slider.update()
        self._update_strip_btn()

    def _set_strip_open(self, open_):
        """Hide/show the bookmarks panel.  Hidden, the QSplitter gives its width
        to the video; shown again, it comes back at the size it had."""
        self.strip.setVisible(open_)
        self.btn_show.setVisible(not open_)
        self.state["bag_bookmarks_open"] = bool(open_)
        self._update_strip_btn()

    def _update_strip_btn(self):
        self.btn_show.setText(f"Bookmarks ({len(self.bookmarks)})")

    def _selected(self):
        r = self.list.currentRow()
        return self.bookmarks[r] if 0 <= r < len(self.bookmarks) else None

    def _row_changed(self, row):
        self._flush()
        bm = self.bookmarks[row] if 0 <= row < len(self.bookmarks) else None
        self._fill_editor(bm)
        self.slider.selected = row
        self.slider.update()
        if bm is not None and not self._suppress_select:
            self._park(bm["t_s"] * 1000)

    def _goto_selected(self):
        bm = self._selected()
        if bm is not None:
            self._park(bm["t_s"] * 1000)

    def _fill_editor(self, bm):
        self.editor.setEnabled(bm is not None)
        self.ed_title.blockSignals(True)
        self.ed_notes.blockSignals(True)
        self.ed_title.setText(bm["title"] if bm else "")
        self.ed_notes.setPlainText(bm["notes"] if bm else "")
        self.ed_title.blockSignals(False)
        self.ed_notes.blockSignals(False)
        self.ed_time.setText(fmt_time(bm["t_s"]) if bm else "—")
        btn = self.color_btns[bm.get("color", DEFAULT_COLOR) if bm else DEFAULT_COLOR]
        btn.setChecked(True)

    def _refresh_row(self):
        r = self.list.currentRow()
        bm = self._selected()
        if bm is None:
            return
        it = self.list.item(r)
        it.setText(self._row_text(bm))
        it.setIcon(self._swatch(bm.get("color")))
        it.setToolTip(bm["notes"] if bm["notes"].strip() else "")

    _swatch_cache = {}

    def _swatch(self, name):
        """A small filled circle in the bookmark's colour, for its list row."""
        hexc = color_hex(name)
        icon = self._swatch_cache.get(hexc)
        if icon is None:
            pm = QtGui.QPixmap(14, 14)
            pm.fill(QtCore.Qt.transparent)
            p = QtGui.QPainter(pm)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(hexc))
            p.drawEllipse(1, 1, 12, 12)
            p.end()
            icon = self._swatch_cache[hexc] = QtGui.QIcon(pm)
        return icon

    def _set_color(self, name):
        bm = self._selected()
        if bm is None:
            return
        bm["color"] = name
        self._refresh_row()
        self._update_ticks()
        self._save()

    def _edited(self, _text=None):
        bm = self._selected()
        if bm is None:
            return
        bm["title"] = self.ed_title.text()
        self._refresh_row()
        self._save_timer.start()

    def _notes_changed(self):
        bm = self._selected()
        if bm is None:
            return
        bm["notes"] = self.ed_notes.toPlainText()
        self._refresh_row()
        self._save_timer.start()

    def _set_time(self):
        bm = self._selected()
        if bm is None:
            return
        bm["t_s"] = self._pos_s()
        self._save()
        self._rebuild_list(select=bm, seek=False)

    def _delete_bookmark(self):
        bm = self._selected()
        if bm is None:
            return
        if QtWidgets.QMessageBox.question(
                self, "Delete bookmark",
                f"Delete “{bm['title'] or '(untitled)'}” at {fmt_time(bm['t_s'])}?"
        ) != QtWidgets.QMessageBox.Yes:
            return
        r = self.list.currentRow()
        self.bookmarks.remove(bm)
        self._save()
        nxt = (self.bookmarks[min(r, len(self.bookmarks) - 1)]
               if self.bookmarks else None)
        self._rebuild_list(select=nxt, seek=False)

    # -- persistence
    def _save(self):
        self._save_timer.stop()
        if not self.video:
            return
        try:
            save_bookmarks(self.video, self.bookmarks)
        except OSError as exc:
            self._log(f"[bags] could not save bookmarks: {exc}")
            QtWidgets.QMessageBox.warning(self, "Bookmarks", f"Could not save:\n{exc}")

    def _flush(self):
        """Write a pending debounced edit now (on switching, closing)."""
        if self._save_timer.isActive():
            self._save()

    # -- make video
    def _make_video(self):
        i = self.picker.currentIndex()
        if not 0 <= i < len(self.entries):
            return
        e = self.entries[i]
        script = os.path.join(self.bags_root, SCRIPT_REL)
        if not os.path.isfile(script):
            QtWidgets.QMessageBox.warning(self, "Make video", f"Not found:\n{script}")
            return
        self._proc = QtCore.QProcess(self)
        env = script_env()
        self._proc.setProcessEnvironment(env)
        self._proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._proc_output)
        self._proc.finished.connect(lambda code, _st, name=e["name"]:
                                    self._proc_done(name, code))
        self._log(f"[bags] make_bag_video.sh {e['date']} …")
        self.busy.show()
        self.btn_make.setEnabled(False)
        self.view.clear(f"rendering {e['name']}… (see the console)")
        self._proc.start("bash", [script, e["date"]])

    def _proc_output(self):
        data = bytes(self._proc.readAllStandardOutput()).decode(errors="replace")
        for line in data.splitlines():
            if line.strip():
                self._log("[bags] " + line)

    def _proc_done(self, name, code):
        self._proc = None
        self.busy.hide()
        self._log(f"[bags] make_bag_video.sh finished, exit {code}")
        self.refresh(select=name)
        if code != 0:
            self.view.clear(f"make_bag_video.sh failed (exit {code}) -- "
                            "see the console")

    # -- lifecycle
    def stop(self):
        """Called from the window's closeEvent."""
        self._flush()
        self.player.stop()
        if self._proc is not None:
            self._proc.kill()
            self._proc.waitForFinished(2000)
