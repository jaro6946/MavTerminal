#!/usr/bin/env python3
"""report_tab.py -- build a comparison report across several logs.

The browse tab answers "what happened in THIS log", with seven fixed plots.  This
one answers the other question the work keeps asking: "how do these logs differ
on the channel I care about" -- heat-sink A against heat-sink B, before and after
a calibration, four GPS antenna placements.

The shape is: pick the logs at the top, then stack up graphs, each choosing its
own subset of those logs and its own set of channels out of the ~3,300 fields a
PX4 log carries.  Below each graph, summary statistics for exactly the time
window on screen, and a note.  The whole thing saves next to the logs.

Three things here are load-bearing rather than cosmetic, and all three are about
not lying to the reader:

  * Lines are DECIMATED for drawing but statistics come from the full arrays.
    A mean computed off a min/max envelope is not the mean of the data.
  * Colour identifies the CHANNEL and line style identifies the LOG.  With three
    logs and six channels there are eighteen lines, and a legend of eighteen
    arbitrary colours is not readable by anyone.
  * A log that cannot satisfy the chosen time alignment -- no arming event, no
    GNSS fix -- is drawn and LABELLED as unaligned, never silently laid down at
    the wrong offset.

Acronyms: ULog = PX4's binary log format, GNSS = global navigation satellite
system, GUI = graphical user interface, LRU = least-recently-used.
"""
import os

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.figure import Figure

from log_browser_crumbs import crumb
from qt_common import FlowLayout, NotesBox, PlotCanvas, flow_holder
from report_model import (ALIGNMENTS, HEIGHT_RANGE, LogRef, Report,
                          list_reports, reports_dir)
from report_render import (DRAW_PX, STAT_COLS, assign_axes, build_figure,
                           fit_value_axes, fmt_stat, gather_series, plot_y,
                           short_ref, stats_of)
import ulog_cache
from ulog_cache import parse_ulog
from ulog_common import (C_INK, C_MUTED, C_SURFACE, VARY, add_mouse_navigation,
                         decimate, nav_hint, window_values)

__all__ = ["ReportTab"]

GRAPH_HEIGHT = 430          # px of plot per card

# Parse threads that outlived their tab (see ReportTab.stop).  Held only so the
# interpreter does not delete a QThread that is still inside a 275 MB parse.
_ORPHAN_THREADS = []


# --- loading -----------------------------------------------------------------

class LoadWorker(QtCore.QObject):
    """Parses logs for the report tab, unfiltered, one at a time off the GUI thread.

    Unfiltered because the field picker offers every channel in the file, and a
    parse restricted to the browse tab's topic list would silently offer a tenth
    of them.  The cache's superset rule means this parse then also serves the
    browse tab, so nothing is read twice.
    """
    one = QtCore.pyqtSignal(str, object)
    failed = QtCore.pyqtSignal(str, str)
    finished = QtCore.pyqtSignal()

    def __init__(self, paths):
        super().__init__()
        self.paths = list(paths)
        self.cancelled = False

    @QtCore.pyqtSlot()
    def run(self):
        for p in self.paths:
            # quit() only unwinds an event loop, and this is a straight-line
            # loop, so a window closed mid-parse would otherwise keep the thread
            # running through every remaining log while Qt tore its QThread down
            # underneath it -- a segfault at exit, no traceback.
            if self.cancelled:
                break
            try:
                self.one.emit(p, parse_ulog(p, None))
            except Exception as e:
                self.failed.emit(p, f"{type(e).__name__}: {e}")
        self.finished.emit()


# --- the field picker --------------------------------------------------------

class FieldPicker(QtWidgets.QWidget):
    """Choose channels out of the union of the selected logs' fields.

    A real log carries 1,300-3,300 plottable fields and 43% of them are constant
    or never filled in, which is invisible from the name alone.  So: grouped by
    topic, filtered by substring, and with the dead ones hidden by default.  The
    items are built once and hidden/shown by the filter rather than rebuilt,
    because rebuilding loses the ticks the user has already placed.
    """
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = {}            # ref -> QTreeWidgetItem
        self._building = False

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QtWidgets.QLabel("values:"))
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("filter — e.g. temp, gyro, innov…")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._apply_filter)
        row.addWidget(self.filter, 1)
        self.chk_flat = QtWidgets.QCheckBox("hide constant/empty")
        self.chk_flat.setToolTip(
            "PX4 logs many fields it never fills in for a given airframe.\n"
            "On a real log that is about 43% of them, and they are not\n"
            "distinguishable from useful channels by name.")
        self.chk_flat.setChecked(True)
        self.chk_flat.toggled.connect(self._apply_filter)
        row.addWidget(self.chk_flat)
        self.lbl_count = QtWidgets.QLabel("")
        self.lbl_count.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        row.addWidget(self.lbl_count)
        v.addLayout(row)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["channel", "in"])
        self.tree.setColumnWidth(0, 420)
        self.tree.setFixedHeight(190)
        self.tree.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.tree.setUniformRowHeights(True)
        self.tree.itemChanged.connect(self._item_changed)
        v.addWidget(self.tree)

    def populate(self, inventories, selected=()):
        """`inventories` is {log name: field_inventory(ulog)}.

        Fields are the UNION across the chosen logs, tagged with how many of them
        actually carry each one -- a channel present in two logs of three is
        still worth plotting, but you want to know before you read the gaps as
        data."""
        self._building = True
        self.tree.clear()
        self._items.clear()

        seen = {}                       # ref -> [kinds], count
        for inv in inventories.values():
            for ref, topic, mid, name, kind in inv:
                rec = seen.setdefault(ref, {"topic": topic, "n": 0, "kinds": set()})
                rec["n"] += 1
                rec["kinds"].add(kind)

        n_logs = max(1, len(inventories))
        by_topic = {}
        for ref, rec in seen.items():
            by_topic.setdefault(rec["topic"], []).append((ref, rec))

        selected = set(selected)
        for topic in sorted(by_topic):
            parent = QtWidgets.QTreeWidgetItem(self.tree, [topic, ""])
            parent.setFlags(parent.flags() & ~QtCore.Qt.ItemIsUserCheckable)
            parent.setForeground(0, QtGui.QColor(C_MUTED))
            for ref, rec in sorted(by_topic[topic]):
                # "Interesting" if it varies in ANY of the chosen logs: a channel
                # that is flat in the control run and moves in the test run is
                # the entire point of a comparison report.
                varies = VARY in rec["kinds"]
                label = ref[len(topic):].lstrip(".")
                if "[" in ref.split(".")[0]:
                    label = ref.split(".", 1)[1]
                    inst = ref.split("[")[1].split("]")[0]
                    if inst != "0":
                        label = f"{label}  [{inst}]"
                it = QtWidgets.QTreeWidgetItem(parent, [label, ""])
                it.setData(0, QtCore.Qt.UserRole, ref)
                it.setData(1, QtCore.Qt.UserRole, varies)
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(0, QtCore.Qt.Checked if ref in selected
                                 else QtCore.Qt.Unchecked)
                if rec["n"] < n_logs:
                    it.setText(1, f"{rec['n']}/{n_logs}")
                    it.setForeground(1, QtGui.QColor(C_MUTED))
                if not varies:
                    it.setForeground(0, QtGui.QColor(C_MUTED))
                self._items[ref] = it
        self._building = False
        self._apply_filter()

    def refs(self):
        """Every channel this picker is currently offering."""
        return list(self._items)

    def selected_refs(self):
        return [ref for ref, it in self._items.items()
                if it.checkState(0) == QtCore.Qt.Checked]

    def _item_changed(self, item, col):
        if self._building or col != 0:
            return
        # DEFERRED, and this is not defensive tidiness -- it is the fix for a
        # segfault.  This slot runs INSIDE Qt's itemChanged emission, and what
        # it triggers can clear() this tree; destroying the QTreeWidgetItem that
        # Qt is still holding a pointer to is a use-after-free in C++, which
        # surfaces as a crash in the event loop with no Python traceback at all.
        # A zero-delay timer lets the emission finish first.
        QtCore.QTimer.singleShot(0, self.changed.emit)

    def _apply_filter(self):
        text = self.filter.text().strip().lower()
        hide_flat = self.chk_flat.isChecked()
        shown = 0
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            any_shown = False
            for j in range(parent.childCount()):
                it = parent.child(j)
                ref = it.data(0, QtCore.Qt.UserRole) or ""
                varies = bool(it.data(1, QtCore.Qt.UserRole))
                checked = it.checkState(0) == QtCore.Qt.Checked
                # A ticked channel is never hidden by a filter: watching your own
                # selection disappear as you type reads as having lost it.
                ok = checked or ((text in ref.lower()) and (varies or not hide_flat))
                it.setHidden(not ok)
                any_shown |= ok
                shown += ok
            parent.setHidden(not any_shown)
            if any_shown and text:
                parent.setExpanded(True)
        self.lbl_count.setText(f"{shown} of {len(self._items)} shown")


# --- one graph ---------------------------------------------------------------

class GraphCard(QtWidgets.QFrame):
    """Title, log subset, channels, the plot, its statistics, and a note."""

    changed = QtCore.pyqtSignal()           # something worth saving
    removed = QtCore.pyqtSignal(str)        # graph id
    needs_logs = QtCore.pyqtSignal(str)     # graph id -- load my logs, then redraw

    def __init__(self, graph, tab, parent=None):
        super().__init__(parent)
        self.graph = graph
        self.tab = tab
        self._full = []             # [(label, t, y)] undecimated, for statistics
        self._lines = []
        self._ax = None
        self._canvas = None
        self._building = False
        self._load_tries = 0        # see refresh(): ask twice, then draw anyway

        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setStyleSheet(f"QFrame {{ background: {C_SURFACE}; }}")
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 8)
        v.setSpacing(5)

        # -- title row
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        self.title = QtWidgets.QLineEdit(graph.title)
        self.title.setPlaceholderText("graph title…")
        self.title.setStyleSheet("font-size: 13px; font-weight: 600;")
        self.title.textChanged.connect(self._title_changed)
        row.addWidget(self.title, 1)
        # The channel and log pickers are EDITING furniture: 600 px of it above
        # every plot, which is why a report opened to a screenful of check boxes
        # with the graph below the fold.  Folded away by default, so reading a
        # report shows the graphs and editing one is a click.
        self.btn_edit = QtWidgets.QToolButton()
        self.btn_edit.setText("logs + channels")
        self.btn_edit.setCheckable(True)
        self.btn_edit.setArrowType(QtCore.Qt.RightArrow)
        self.btn_edit.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.btn_edit.setToolTip("Show the log and channel pickers for this graph")
        self.btn_edit.toggled.connect(self._edit_toggled)
        row.addWidget(self.btn_edit)
        btn_del = QtWidgets.QToolButton()
        btn_del.setText("✕")
        btn_del.setToolTip("Remove this graph")
        btn_del.clicked.connect(lambda: self.removed.emit(self.graph.id))
        row.addWidget(btn_del)
        v.addLayout(row)

        # Everything between here and `self.editor` below goes in the fold.
        self.editor = QtWidgets.QWidget()
        ev = QtWidgets.QVBoxLayout(self.editor)
        ev.setContentsMargins(0, 0, 0, 0)
        ev.setSpacing(5)
        v.addWidget(self.editor)

        # -- which of the report's logs this graph draws
        # FlowLayout, not QHBoxLayout: one checkbox per log in a non-wrapping
        # row makes the card as wide as all the log names laid end to end, and
        # the graph inside it is then drawn to that width -- which is how the
        # plots ran off the side of the window.  See qt_common.FlowLayout.
        self.logs_row = FlowLayout(hspacing=10)
        self.logs_row.addWidget(QtWidgets.QLabel("logs:"))
        self._log_boxes = {}
        self._logs_holder = flow_holder(self.logs_row)
        ev.addWidget(self._logs_holder)

        # -- channels
        self.picker = FieldPicker()
        self.picker.changed.connect(self._fields_changed)
        ev.addWidget(self.picker)

        # -- selected channels, with their axis assignment
        self.chosen = QtWidgets.QTreeWidget()
        self.chosen.setHeaderLabels(["plotted channel", "axis"])
        self.chosen.setColumnWidth(0, 420)
        self.chosen.setMaximumHeight(110)
        self.chosen.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.chosen.setToolTip("Click a channel's axis cell to move it between "
                               "the left and right scales.")
        self.chosen.itemClicked.connect(self._axis_clicked)
        ev.addWidget(self.chosen)

        # -- alignment
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QtWidgets.QLabel("x-axis:"))
        self.cmb_align = QtWidgets.QComboBox()
        for key, label in ALIGNMENTS.items():
            self.cmb_align.addItem(label, key)
        i = self.cmb_align.findData(graph.align)
        self.cmb_align.setCurrentIndex(max(0, i))
        self.cmb_align.setToolTip(
            "How several logs are laid over one another in time.\n"
            "'since log start' suits bench runs; 'since first arm' suits\n"
            "comparing flights; 'absolute clock' needs a GNSS fix.")
        self.cmb_align.activated.connect(self._align_changed)
        row.addWidget(self.cmb_align)
        self.chk_norm = QtWidgets.QCheckBox("normalise")
        self.chk_norm.setToolTip("Scale every channel to 0-1 over its own range,\n"
                                 "to compare SHAPES rather than magnitudes.")
        self.chk_norm.setChecked(graph.normalise)
        self.chk_norm.toggled.connect(self._norm_changed)
        row.addWidget(self.chk_norm)
        self.chk_lanes = QtWidgets.QCheckBox("lanes")
        self.chk_lanes.setToolTip(
            "Stack 0/1 channels on the right scale as lanes -- one level per\n"
            "line, kept to the bottom of the frame.  Without it, every binary\n"
            "verdict draws on the same two values and only the last is visible.")
        self.chk_lanes.setChecked(getattr(graph, "lanes", False))
        self.chk_lanes.toggled.connect(self._lanes_changed)
        row.addWidget(self.chk_lanes)
        # Height lives on the graph rather than in the window because it is a
        # property of what is being drawn -- a lane per log runs out of room at
        # around eight -- so the PDF has to inherit the same choice.
        row.addWidget(QtWidgets.QLabel("height"))
        self.spin_height = QtWidgets.QDoubleSpinBox()
        self.spin_height.setRange(*HEIGHT_RANGE)
        self.spin_height.setSingleStep(0.1)
        self.spin_height.setDecimals(1)
        self.spin_height.setSuffix(" x")
        self.spin_height.setToolTip(
            "Vertical room for THIS graph, as a multiple of the standard plot\n"
            "height.  Applies to the card here and to the PDF page, so a graph\n"
            "that needs the room keeps it wherever it is rendered.")
        self.spin_height.setValue(getattr(graph, "height", 1.0))
        self.spin_height.valueChanged.connect(self._height_changed)
        row.addWidget(self.spin_height)
        self.lbl_warn = QtWidgets.QLabel("")
        self.lbl_warn.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        row.addWidget(self.lbl_warn, 1)
        v.addLayout(row)

        # -- the plot
        self.plot_holder = QtWidgets.QVBoxLayout()
        self.plot_holder.setContentsMargins(0, 0, 0, 0)
        self.plot_holder_w = QtWidgets.QWidget()
        self.plot_holder_w.setLayout(self.plot_holder)
        self.plot_holder_w.setFixedHeight(self._plot_px())
        v.addWidget(self.plot_holder_w)

        # -- statistics
        self.stats = QtWidgets.QTableWidget(0, 2 + len(STAT_COLS))
        self.stats.setHorizontalHeaderLabels(["log", "channel"] + STAT_COLS)
        self.stats.verticalHeader().setVisible(False)
        self.stats.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.stats.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.stats.horizontalHeader().setStretchLastSection(True)
        # No vertical scrolling: these are a handful of summary rows, and a table
        # that scrolls hides rows behind a gesture nobody expects to need for six
        # numbers.  _size_stats() grows the widget to fit them instead, and the
        # page it sits in does the scrolling.
        self.stats.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.stats.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                 QtWidgets.QSizePolicy.Fixed)
        v.addWidget(self.stats)
        self.lbl_stats = QtWidgets.QLabel("")
        self.lbl_stats.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        v.addWidget(self.lbl_stats)

        # -- note
        self.notes = NotesBox(
            label="Notes",
            placeholder="What this graph shows, and what you concluded from it…",
            tooltip="Saved with the report.",
            on_save=self._notes_saved)
        self.notes.set_text(graph.notes, enabled=True)
        v.addWidget(self.notes)

        # A graph with no channels yet has nothing to look at, so for that one
        # the pickers ARE the card -- open it rather than hiding the only thing
        # that would make it draw something.
        self.btn_edit.setChecked(not graph.fields)
        self._edit_toggled(self.btn_edit.isChecked())

    # -- report-level changes
    def set_available_logs(self, names):
        """Rebuild the per-graph log tick boxes from the report's log list."""
        self._building = True
        while self.logs_row.count() > 1:
            item = self.logs_row.takeAt(1)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._log_boxes.clear()
        for name in names:
            # Label shortened, identity kept: _log_boxes is keyed by the real
            # basename, so what the box SAYS is free to be short.  The full name
            # is a tooltip away, and a fourteen-log report otherwise spends four
            # wrapped rows on ".ulg" and repeated prefixes.
            label = os.path.splitext(name)[0]
            if len(label) > 30:
                label = label[:14] + "…" + label[-15:]
            cb = QtWidgets.QCheckBox(label)
            cb.setToolTip(name)
            cb.setChecked(name in self.graph.logs)
            cb.toggled.connect(self._logs_changed)
            self.logs_row.addWidget(cb)
            self._log_boxes[name] = cb
        # No addStretch: a flow layout packs left and wraps, so a stretch item
        # would just be an invisible box competing for a row.
        # Drop references to logs the report no longer has.
        self.graph.logs = [n for n in self.graph.logs if n in names]
        self._building = False

    def selected_logs(self):
        return [n for n, cb in self._log_boxes.items() if cb.isChecked()]

    # -- edits
    def _title_changed(self, text):
        self.graph.title = text
        if self._ax is not None:
            self._ax.set_title(text or "untitled graph", loc="left",
                               fontsize=11, color=C_INK)
            self._canvas.draw_idle()
        self.changed.emit()

    def _notes_saved(self, text):
        self.graph.notes = text
        self.changed.emit()

    def _logs_changed(self):
        if self._building:
            return
        self.graph.logs = self.selected_logs()
        self._load_tries = 0
        self.changed.emit()
        self.needs_logs.emit(self.graph.id)

    def _fields_changed(self):
        """Ticking a channel must not silently drop channels from other logs.

        The picker only offers what the CURRENTLY ticked logs carry, so a naive
        `fields = selected_refs()` would delete a channel that only the log you
        just unticked had -- a destructive edit nobody asked for.  Channels the
        picker cannot see are therefore kept, and order is preserved because the
        colour of a channel is its index in this list.
        """
        offered = set(self.picker.refs())
        chosen = set(self.picker.selected_refs())
        kept = [f for f in self.graph.fields if f in chosen or f not in offered]
        self.graph.fields = kept + [f for f in self.picker.selected_refs()
                                    if f not in self.graph.fields]
        self.changed.emit()
        self.refresh(repopulate=False)

    def _align_changed(self):
        self.graph.align = self.cmb_align.currentData()
        self.graph.xlim = None          # a new alignment invalidates the old zoom
        self.changed.emit()
        self.refresh(repopulate=False)

    def _norm_changed(self, on):
        self.graph.normalise = bool(on)
        self.changed.emit()
        self.refresh(repopulate=False)

    def _edit_toggled(self, on):
        self.editor.setVisible(bool(on))
        self.btn_edit.setArrowType(QtCore.Qt.DownArrow if on
                                   else QtCore.Qt.RightArrow)

    def _plot_px(self):
        """Card plot height in pixels, honouring Graph.height."""
        return int(round(GRAPH_HEIGHT * getattr(self.graph, "height", 1.0)))

    def _height_changed(self, value):
        self.graph.height = float(value)
        self.plot_holder_w.setFixedHeight(self._plot_px())
        self.changed.emit()
        self.refresh(repopulate=False)

    def _lanes_changed(self, on):
        self.graph.lanes = bool(on)
        self.changed.emit()
        self.refresh(repopulate=False)

    def _axis_clicked(self, item, col):
        ref = item.data(0, QtCore.Qt.UserRole)
        if col != 1 or not ref:
            return
        now = self.graph.axis.get(ref, item.data(1, QtCore.Qt.UserRole) or "left")
        self.graph.axis[ref] = "right" if now == "left" else "left"
        self.changed.emit()
        # Deferred for the same reason as FieldPicker._item_changed: the redraw
        # clears the very tree this click came from.
        QtCore.QTimer.singleShot(0, lambda: self.refresh(repopulate=False))

    # -- drawing
    def refresh(self, repopulate=True):
        """Rebuild the plot and statistics; optionally the channel list too.

        `repopulate` is False whenever only the SELECTION changed, not which logs
        are on offer: the tree's contents depend on the logs, so rebuilding it on
        every tick both risks the crash above and throws away the user's scroll
        position and expanded topics mid-click."""
        names = self.selected_logs()
        ulogs = self.tab.loaded_ulogs(names)
        absent = [n for n in names if n not in ulogs]
        if absent and self._load_tries < 2:
            # Ask once (twice at most) for the missing parses, then give up ASKING
            # -- but never give up DRAWING.  Looping here is how a graph whose
            # logs cannot all be resident at once stays permanently blank.
            self._load_tries += 1
            self.needs_logs.emit(self.graph.id)
            return
        self._load_tries = 0

        if repopulate or not self.picker.refs():
            inv = {n: self.tab.inventory_of(n) for n in names}
            self.picker.populate(inv, selected=self.graph.fields)

        crumb(f"report graph {self.graph.id}: {len(self.graph.fields)} field(s) "
              f"x {len(names)} log(s)")
        series, problems = gather_series(self.graph, ulogs, names)
        if absent:
            # One line, not one per log: with a dozen logs the per-log form ran
            # three names and a "+9 more" across the top of the figure.  And it
            # says WHICH cap, because "cache full?" left the reader guessing at
            # the one thing the program already knows.
            st = self.tab.cache.stats()
            need = self._graph_mb(self.graph)
            problems = ([f"{len(absent)} of {len(names)} log(s) not drawn: "
                         f"{', '.join(os.path.splitext(n)[0] for n in absent[:2])}"
                         + (f" +{len(absent) - 2}" if len(absent) > 2 else "")
                         + f" — this graph needs {need:.0f} MB, cache holds "
                           f"{self.tab.cache.max_mb} MB, "
                           f"{st['avail_mb']:.0f} MB free"] + problems)
        self._draw(series, problems)
        self._fill_chosen(series)
        self.update_stats()

    def _draw(self, series, problems):
        """Mount the figure report_render built.

        Everything about WHAT is drawn lives in report_render, so this card and
        the headless exporter cannot drift apart.  What is left here is the part
        that only makes sense with a window in front of it: navigation, the
        problem label, and swapping the canvas."""
        self._auto = assign_axes(self.graph, series)
        px = self._plot_px()
        fig, ax, axr, lines = build_figure(
            self.graph, series, problems,
            figsize=(13, px / 100.0), auto=self._auto)
        self._full = list(series)
        self._lines = lines

        canvas = PlotCanvas(fig)
        # Same three lines PlotPage.add explains: FigureCanvasQTAgg takes its
        # size hint from figsize * dpi, so a 13-inch figure demands 1300 px and
        # the card scrolls sideways unless the width is allowed to track the
        # viewport instead.  And show() because a widget added to a layout after
        # its parent is already visible stays hidden otherwise -- which is the
        # whole plot.
        canvas.setFixedHeight(px)
        canvas.setMinimumWidth(320)
        canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                             QtWidgets.QSizePolicy.Fixed)
        axes = [ax] + ([axr] if axr is not None else [])
        add_mouse_navigation(fig, axes, page_scroll=True,
                             on_xlim=self._xlim_changed, on_view=self._view_changed)
        fig.text(0.995, 0.008, nav_hint(True), ha="right", va="bottom",
                 fontsize=7, color=C_MUTED)

        note = "; ".join(list(problems)[:3])
        if len(problems) > 3:
            note += f"; +{len(problems) - 3} more"
        self.lbl_warn.setText(note)

        self._drop_canvas()
        self.plot_holder.addWidget(canvas)
        canvas.show()
        self._canvas, self._ax, self._axr = canvas, ax, axr
        self._rescale()

    def _drop_canvas(self):
        """Retire the previous canvas and its figure.

        The browse tab gets this for free from plt.close().  These figures are
        built with Figure() precisely so they never enter pyplot's registry, so
        nothing releases their artists for them and a card redrawn on every tick
        would accumulate them.

        Note what this does NOT do: reach into the canvas's CallbackRegistry.
        Clearing that dict directly desynchronises matplotlib's weakref
        bookkeeping and buries the console in ignored KeyErrors from
        cbook._remove_proxy.  Dropping the last reference to the canvas is what
        actually retires its callbacks, and Nav is only reachable through the
        figure (add_mouse_navigation parks it on fig._nav), so letting both go is
        both sufficient and correct.
        """
        while self.plot_holder.count():
            w = self.plot_holder.takeAt(0).widget()
            if w is None:
                continue
            fig = getattr(w, "figure", None)
            if fig is not None:
                fig.clear()
            w.setParent(None)
            w.deleteLater()

    def _fill_chosen(self, series):
        """The compact list under the picker: what is plotted, and on which scale."""
        self.chosen.clear()
        seen = {}
        for s in series:
            seen.setdefault(s["ref"], s)
        for ref, s in seen.items():
            it = QtWidgets.QTreeWidgetItem(self.chosen, [short_ref(ref), s["axis"]])
            it.setData(0, QtCore.Qt.UserRole, ref)
            it.setData(1, QtCore.Qt.UserRole, s["axis"])
            it.setForeground(0, QtGui.QColor(s["color"]))
            if ref not in self.graph.axis:
                it.setForeground(1, QtGui.QColor(C_MUTED))
                it.setToolTip(1, "assigned automatically — click to override")

    # -- zoom follow-through
    def _xlim_changed(self, lo, hi):
        self.graph.xlim = (float(lo), float(hi))
        self.changed.emit()

    def _view_changed(self):
        """After a zoom: redraw at the new resolution and restate the numbers.

        Re-decimating matters -- zoom into ten seconds of a 40 minute log and the
        original envelope has perhaps two points in the window, so without this
        the plot gets emptier the closer you look."""
        if self._ax is None:
            return
        if getattr(self.graph, "kind", "series") == "scatter":
            # A scatter has no time axis and no lines to re-decimate, and its
            # x limits are a fit to the points -- saving them as `xlim` would
            # mark the report modified for a pan that means nothing.
            return
        lo, hi = self._ax.get_xlim()
        # Recorded here as well as in _xlim_changed: Nav reports a view change
        # through two callbacks and not every gesture fires both, so the saved
        # window would depend on HOW you zoomed rather than on where you ended up.
        self.graph.xlim = (float(lo), float(hi))
        for line, s in self._lines:
            # plot_y, not s["y"]: a lane series is drawn at its own level, and
            # reading the raw channel here would drop it back onto 0/1 the first
            # time anyone zoomed.
            t, y = s["t"], plot_y(s)
            a, b = np.searchsorted(t, [min(lo, hi), max(lo, hi)])
            a, b = max(0, a - 1), min(t.size, b + 1)
            td, yd = decimate(t[a:b], y[a:b], DRAW_PX)
            line.set_data(td, yd)
        self._rescale()
        self.update_stats()

    def _rescale(self):
        """Fit each value axis to what is actually inside the time window."""
        fit_value_axes([getattr(self, "_ax", None), getattr(self, "_axr", None)],
                       self._lines, window_values_fn=window_values)
        if self._canvas is not None:
            self._canvas.draw_idle()

    def _size_stats(self):
        """Make the statistics table exactly as tall as its own rows.

        Qt sizes an item view to whatever box it is given and scrolls the rest,
        so the height has to be computed from the rows themselves -- header,
        every row, and the frame -- rather than left to a constant that is right
        for one graph and wrong for the next."""
        h = self.stats.horizontalHeader().height() + 2 * self.stats.frameWidth()
        for r in range(self.stats.rowCount()):
            h += self.stats.rowHeight(r)
        self.stats.setFixedHeight(max(h, 40))

    def update_stats(self):
        xlim = self._ax.get_xlim() if self._ax is not None else None
        self.stats.setRowCount(len(self._full))
        for r, s in enumerate(self._full):
            st = stats_of(s["t"], s["y"], xlim)
            cells = [os.path.splitext(s["log"])[0], short_ref(s["ref"])]
            cells += [fmt_stat(st[c]) for c in STAT_COLS]
            for c, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if c == 1:
                    item.setForeground(QtGui.QColor(s["color"]))
                if c >= 2:
                    item.setTextAlignment(QtCore.Qt.AlignRight
                                          | QtCore.Qt.AlignVCenter)
                self.stats.setItem(r, c, item)
        self.stats.resizeColumnsToContents()
        self._size_stats()
        if xlim and self._full:
            self.lbl_stats.setText(
                f"over the visible window {min(xlim):.2f} – {max(xlim):.2f} min "
                f"(zoom with ctrl+wheel to restrict it)")
        else:
            self.lbl_stats.setText("")


# --- the tab -----------------------------------------------------------------

class ReportTab(QtWidgets.QWidget):
    """Report picker and title at the top, then the report's graphs."""

    def __init__(self, cache, list_logs, log, root=None, parent=None):
        super().__init__(parent)
        self.cache = cache
        self._list_logs = list_logs     # () -> [(path, label)] from the library
        self._log = log                 # console line
        self.root = root
        self.report = Report()
        self._cards = {}
        self._thread = None
        self._worker = None
        self._pending = None
        self._dirty = False
        self._disk = None           # (mtime, size) of the file as we last knew it

        self._build_ui()
        self._sync_reports()
        self._refresh_logs_ui()

        # A report is a plain JSON file in a shared folder, and report_cli
        # writes the same files this tab does -- so the copy on disk can move
        # under an open window.  Left undetected that ends one of two bad ways:
        # the close-time "unsaved changes, save?" prompt silently overwrites the
        # newer file with this window's stale copy, or the user answers Discard
        # and loses their own edits without ever being told the two versions
        # differed.  Polling mtime is enough here (one small file, 2 s) and
        # avoids a QFileSystemWatcher's editor-rename blind spot -- Report.save
        # writes to .tmp and renames, which drops a watcher off the inode.
        self._watch = QtCore.QTimer(self)
        self._watch.setInterval(2000)
        self._watch.timeout.connect(self._check_disk)
        self._watch.start()

    # -- construction
    def _build_disk_bar(self):
        """The strip that appears when the file and the window disagree.

        Deliberately a banner and not a dialog: it turns up because something
        ELSE wrote the file, at a moment the user did not choose, and a modal
        box then steals a keystroke and forces an answer before they have read
        what happened.  The banner states which version is which and leaves both
        doors open."""
        self.disk_bar = QtWidgets.QFrame()
        self.disk_bar.setStyleSheet(
            "QFrame { background: #fdf3d8; border: 1px solid #e0c97f; }")
        h = QtWidgets.QHBoxLayout(self.disk_bar)
        h.setContentsMargins(10, 4, 10, 4)
        self.lbl_disk = QtWidgets.QLabel("")
        self.lbl_disk.setWordWrap(True)
        self.lbl_disk.setStyleSheet("color: #4a3b00; font-size: 11px; border: 0;")
        h.addWidget(self.lbl_disk, 1)
        self.btn_disk_reload = QtWidgets.QPushButton("Reload from disk")
        self.btn_disk_reload.setToolTip(
            "Throw away this window's edits and show the file as it now is.")
        self.btn_disk_reload.clicked.connect(lambda: self._reload_from_disk())
        h.addWidget(self.btn_disk_reload)
        self.btn_disk_keep = QtWidgets.QPushButton("Keep my version")
        self.btn_disk_keep.setToolTip(
            "Leave this window as it is.  Saving will then overwrite the file.")
        self.btn_disk_keep.clicked.connect(self._keep_mine)
        h.addWidget(self.btn_disk_keep)
        self.disk_bar.hide()
        return self.disk_bar

    def _build_ui(self):
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        bar = QtWidgets.QWidget()
        bh = QtWidgets.QHBoxLayout(bar)
        bh.setContentsMargins(10, 6, 10, 4)
        bh.addWidget(QtWidgets.QLabel("report:"))
        self.picker = QtWidgets.QComboBox()
        self.picker.setMinimumContentsLength(34)
        self.picker.activated.connect(self._picked_report)
        bh.addWidget(self.picker, 1)
        for label, slot, tip in (
                ("New", self._new_report, "Start an empty report"),
                ("Save", self._save_report, "Save to this report's file"),
                ("Save As…", self._save_report_as, "Save under a new name"),
                ("Delete…", self._delete_report, "Delete this report file")):
            b = QtWidgets.QPushButton(label)
            b.setToolTip(tip)
            # lambda, not the bound method: QPushButton.clicked carries a
            # `checked` bool, and PyQt hands it to any slot that will take an
            # argument -- so connecting _save_report directly called it with
            # path=False and Save died in os.path.dirname(False).
            b.clicked.connect(lambda _checked=False, fn=slot: fn())
            bh.addWidget(b)
        self.busy = QtWidgets.QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setFixedWidth(120)
        self.busy.hide()
        bh.addWidget(self.busy)
        v.addWidget(bar)
        v.addWidget(self._build_disk_bar())

        row = QtWidgets.QWidget()
        rh = QtWidgets.QHBoxLayout(row)
        rh.setContentsMargins(10, 0, 10, 4)
        rh.addWidget(QtWidgets.QLabel("title:"))
        self.title = QtWidgets.QLineEdit()
        self.title.setPlaceholderText("what this report is about…")
        self.title.setStyleSheet("font-size: 14px; font-weight: 600;")
        self.title.textChanged.connect(self._title_changed)
        rh.addWidget(self.title, 1)
        v.addWidget(row)

        self.notes = NotesBox(
            label="Notes",
            placeholder="What is being compared, the conditions, the conclusion…",
            tooltip="Notes for the whole report.  Saved with it.",
            on_save=self._notes_saved)
        self.notes.set_text("", enabled=True)
        v.addWidget(self.notes)

        row = QtWidgets.QWidget()
        rh = QtWidgets.QHBoxLayout(row)
        rh.setContentsMargins(10, 0, 10, 4)
        rh.setSpacing(8)
        rh.addWidget(QtWidgets.QLabel("logs:"))
        self.lbl_logs = QtWidgets.QLabel("none chosen")
        self.lbl_logs.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.lbl_logs.setWordWrap(True)
        rh.addWidget(self.lbl_logs, 1)
        b = QtWidgets.QPushButton("Choose logs…")
        b.clicked.connect(self._choose_logs)
        rh.addWidget(b)
        v.addWidget(row)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        inner.setStyleSheet(f"background: {C_SURFACE};")
        self.page = QtWidgets.QVBoxLayout(inner)
        self.page.setContentsMargins(8, 8, 8, 8)
        self.page.setSpacing(10)
        self.scroll.setWidget(inner)
        v.addWidget(self.scroll, 1)

        foot = QtWidgets.QWidget()
        fh = QtWidgets.QHBoxLayout(foot)
        fh.setContentsMargins(10, 4, 10, 6)
        b = QtWidgets.QPushButton("+ Add graph")
        b.clicked.connect(lambda: self._add_graph())
        fh.addWidget(b)
        fh.addStretch(1)
        self.lbl_state = QtWidgets.QLabel("")
        self.lbl_state.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        fh.addWidget(self.lbl_state)
        v.addWidget(foot)

    # -- report lifecycle
    def _sync_reports(self, keep=None):
        self.picker.blockSignals(True)
        self.picker.clear()
        self.picker.addItem("— unsaved report —", None)
        for path, title in list_reports(self.root):
            self.picker.addItem(title, path)
        target = keep or self.report.path
        i = self.picker.findData(target) if target else 0
        self.picker.setCurrentIndex(max(0, i))
        self.picker.blockSignals(False)

    def _picked_report(self, _i):
        path = self.picker.currentData()
        if path is None:
            return
        if not self._confirm_discard():
            self._sync_reports()
            return
        try:
            self.report = Report.load(path)
        except (OSError, ValueError) as e:
            QtWidgets.QMessageBox.warning(self, "Report",
                                          f"Could not open it:\n{e}")
            self._sync_reports()
            return
        self._log(f"report: opened {os.path.basename(path)}")
        self._load_into_ui()

    def _new_report(self):
        if not self._confirm_discard():
            return
        self.report = Report()
        self._load_into_ui()
        self._sync_reports()

    def _load_into_ui(self):
        self.title.blockSignals(True)
        self.title.setText(self.report.title)
        self.title.blockSignals(False)
        self.notes.set_text(self.report.notes, enabled=True)
        self._size_cache_for_report()
        for card in list(self._cards.values()):
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()
        for g in self.report.graphs:
            self._mount_card(g)
        self._refresh_logs_ui()
        self._dirty = False
        self._disk = self._disk_stamp()
        self.disk_bar.hide()
        self._show_state()
        self._reload_all()

    def _confirm_discard(self):
        if not self._dirty:
            return True
        where = os.path.basename(self.report.path) if self.report.path else None
        r = QtWidgets.QMessageBox.question(
            self, "Report",
            (f"Edits made here have not been written to "
             f"{where}." if where else
             "This report has never been saved.")
            + "\n\nDiscard them?",
            QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel)
        return r == QtWidgets.QMessageBox.Discard

    def _save_report(self, path=None):
        self.notes.flush()
        for card in self._cards.values():
            card.notes.flush()
        if path is False:               # a stray clicked(checked) argument
            path = None
        if path in (None, self.report.path) and not self._disk_agrees():
            r = QtWidgets.QMessageBox.question(
                self, "Report",
                f"{os.path.basename(self.report.path or '')} has changed on "
                f"disk since this window opened it — another tool wrote it.\n\n"
                f"Overwrite that file with the version in this window?",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Cancel)
            if r != QtWidgets.QMessageBox.Save:
                return
        if not self.report.title.strip():
            self.report.title = "untitled report"
            self.title.setText(self.report.title)
        try:
            p = self.report.save(path, root=self.root)
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Report", f"Could not save:\n{e}")
            return
        self._dirty = False
        self._disk = self._disk_stamp()
        self.disk_bar.hide()
        self._log(f"report: saved {p}")
        self._sync_reports(keep=p)
        self._show_state()

    def _save_report_as(self):
        d = reports_dir(self.root, create=True)
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save report as", os.path.join(d, "report.json"),
            "Reports (*.json)")
        if p:
            self._save_report(p)

    def _delete_report(self):
        path = self.report.path
        if not path or not os.path.exists(path):
            QtWidgets.QMessageBox.information(
                self, "Report", "This report has not been saved yet.")
            return
        r = QtWidgets.QMessageBox.question(
            self, "Report", f"Delete {os.path.basename(path)} permanently?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if r != QtWidgets.QMessageBox.Yes:
            return
        try:
            os.remove(path)
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Report", f"Could not delete:\n{e}")
            return
        self._log(f"report: deleted {os.path.basename(path)}")
        self.report.path = None
        self._dirty = True
        self._sync_reports()
        self._show_state()

    # -- edits
    def _title_changed(self, text):
        self.report.title = text
        self._touch()

    def _notes_saved(self, text):
        self.report.notes = text
        self._touch()

    def _touch(self):
        self._dirty = True
        self._show_state()

    def _show_state(self):
        where = os.path.basename(self.report.path) if self.report.path else "unsaved"
        self.lbl_state.setText(f"{where}{' — modified' if self._dirty else ''}   "
                               f"{len(self.report.graphs)} graph(s)")

    # -- the file underneath
    def _disk_stamp(self, path=None):
        """(mtime, size) of the report's file, or None if there isn't one.

        Size as well as mtime because a rewrite within the same filesystem
        timestamp tick is exactly what a scripted edit does."""
        path = path or self.report.path
        if not path:
            return None
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _disk_agrees(self):
        """True when the file is as this window last saw it (or isn't there)."""
        if not self.report.path or self._disk is None:
            return True
        return self._disk_stamp() == self._disk

    def _check_disk(self):
        """Poll for someone else writing this report, and say so plainly."""
        if not self.report.path:
            return
        now = self._disk_stamp()
        if now == self._disk:
            return
        self._disk = now
        name = os.path.basename(self.report.path)
        if now is None:
            self._log(f"report: {name} disappeared from disk")
            self._show_disk_bar(f"{name} is no longer on disk. Save to write "
                                f"this window's version back.", reload_ok=False)
            return
        # Push the notes editors out first: their autosave runs 700 ms behind
        # the keystroke, so without this a reload can land in that gap and take
        # a sentence with it while _dirty still reads False.
        self.flush()
        if not self._dirty:
            self._reload_from_disk(quiet=True)
            return
        self._show_disk_bar(
            f"{name} was rewritten on disk by another tool, and this window has "
            f"edits of its own. Reload to take the file's version, or keep this "
            f"one and overwrite it when you save.")

    def _show_disk_bar(self, text, reload_ok=True):
        self.lbl_disk.setText(text)
        self.btn_disk_reload.setVisible(reload_ok)
        self.disk_bar.show()

    def _keep_mine(self):
        """Dismiss the banner and accept this window's version as the future.

        Adopting the current stamp is the point: the user has now been told the
        file moved and chosen, so Save must not stop to ask the same question."""
        self._disk = self._disk_stamp()
        self.disk_bar.hide()
        self._touch()

    def _reload_from_disk(self, quiet=False):
        path = self.report.path
        if not path:
            return
        try:
            self.report = Report.load(path)
        except (OSError, ValueError) as e:
            QtWidgets.QMessageBox.warning(self, "Report",
                                          f"Could not re-read it:\n{e}")
            return
        name = os.path.basename(path)
        self._log(f"report: reloaded {name} (changed on disk"
                  + (")" if quiet else ", this window's edits dropped)"))
        self._load_into_ui()            # clears _dirty, re-stamps, hides the bar
        if quiet:
            self.lbl_state.setText(f"{name} — reloaded, changed on disk   "
                                   f"{len(self.report.graphs)} graph(s)")

    # -- logs
    def _choose_logs(self):
        """Tick the logs this report is about, out of the library."""
        entries = self._list_logs()
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Report — choose logs")
        dlg.resize(780, 520)
        v = QtWidgets.QVBoxLayout(dlg)
        v.addWidget(QtWidgets.QLabel("Tick the logs this report compares:"))
        lst = QtWidgets.QListWidget()
        lst.setStyleSheet("font-family: monospace;")
        v.addWidget(lst, 1)
        chosen = {r.resolve([p for p, _ in entries]) for r in self.report.logs}
        for path, label in entries:
            it = QtWidgets.QListWidgetItem(label)
            it.setData(QtCore.Qt.UserRole, path)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if path in chosen
                             else QtCore.Qt.Unchecked)
            lst.addItem(it)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok
                                          | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        v.addWidget(btns)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        picked = [lst.item(i).data(QtCore.Qt.UserRole) for i in range(lst.count())
                  if lst.item(i).checkState() == QtCore.Qt.Checked]
        self.report.logs = [LogRef.of(p) for p in picked]
        self._size_cache_for_report()
        self._touch()
        self._refresh_logs_ui()
        self._reload_all()

    def _graph_mb(self, graph):
        """Megabytes of log file one graph needs resident at the same time."""
        total = 0.0
        for name in graph.logs:
            p = self.path_for(name)
            try:
                total += os.path.getsize(p) / 1048576.0
            except (OSError, TypeError):
                pass
        return total

    def _size_cache_for_report(self):
        """Let the cache hold, simultaneously, whatever one graph needs.

        The default cap is 4 logs / 1200 MB, tuned for browsing one log at a
        time.  A report with six logs then evicts its own earlier graphs while
        loading its later ones, and those graphs can never be satisfied -- they
        ask for a reload, which evicts something else, forever.  A cap below the
        working set is not a cache, it is a treadmill.

        BOTH caps have to move, which is what this first got wrong: raising only
        the count left the megabyte cap at 1200, and a twelve-log graph totalling
        1207 MB quietly lost its oldest lane on every redraw -- a graph drawing
        eleven of twelve logs, with the twelfth reported as a cache miss rather
        than as the cap it actually was.  The working set is ONE GRAPH's logs,
        not the whole report's, because only one graph's logs must be resident
        together; 5% of headroom covers the parse being a little larger than
        the file.

        The MemAvailable floor is NOT raised, and it is the cap that actually
        protects the machine: if the box genuinely cannot hold the working set,
        eviction still happens and the graph still says what it is missing."""
        want = len(self.report.logs)
        self.cache.max_logs = max(ulog_cache.MAX_LOGS, want)
        need = max([self._graph_mb(g) for g in self.report.graphs] or [0.0])
        self.cache.max_mb = max(ulog_cache.MAX_MB, int(need * 1.05) + 1)

    def _refresh_logs_ui(self):
        known = [p for p, _ in self._list_logs()]
        res = self.report.resolved(known)
        bits = []
        for ref in self.report.logs:
            ok = res.get(ref.name)
            bits.append(ref.name if ok else f"{ref.name} (missing)")
        self.lbl_logs.setText("   ".join(bits) if bits else "none chosen")
        names = self.report.log_names()
        for card in self._cards.values():
            card.set_available_logs(names)

    def path_for(self, name):
        known = [p for p, _ in self._list_logs()]
        return self.report.resolved(known).get(name)

    def loaded_ulogs(self, names):
        """{name: ulog} for those already parsed.  Missing ones are simply absent."""
        out = {}
        for n in names:
            p = self.path_for(n)
            if not p:
                continue
            hit = self.cache.get(p, None)       # None = needs an unfiltered parse
            if hit is not None:
                out[n] = hit.ulog
        return out

    def inventory_of(self, name):
        p = self.path_for(name)
        hit = self.cache.get(p, None) if p else None
        return hit.inventory if hit is not None else []

    def _reload_all(self):
        self._ensure_loaded([g.id for g in self.report.graphs])

    def _ensure_loaded(self, graph_ids):
        """Parse whatever the named graphs still need, then refresh them."""
        want = []
        for gid in graph_ids:
            g = self.report.graph(gid)
            if g is None:
                continue
            for name in g.logs:
                p = self.path_for(name)
                if p and self.cache.get(p, None) is None and p not in want:
                    want.append(p)
        if not want:
            for gid in graph_ids:
                card = self._cards.get(gid)
                if card is not None:
                    card.refresh()
            return
        if self._thread is not None:            # a load is already running
            self._pending = list(dict.fromkeys((self._pending or [])
                                               + list(graph_ids)))
            return

        self._log(f"report: reading {len(want)} log(s)…")
        crumb(f"report parse {len(want)} log(s): "
              f"{', '.join(os.path.basename(p) for p in want)}")
        self.busy.show()
        self._pending_ids = list(dict.fromkeys(
            list(getattr(self, "_pending_ids", [])) + list(graph_ids)))
        self._thread = QtCore.QThread(self)
        self._worker = LoadWorker(want)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.one.connect(self._one_loaded)
        self._worker.failed.connect(self._one_failed)
        self._worker.finished.connect(self._load_finished)
        self._thread.start()

    @QtCore.pyqtSlot(str, object)
    def _one_loaded(self, path, ulog):
        # topics=None: this is the unfiltered parse, so it satisfies every
        # later request including the browse tab's.
        self.cache.put(path, ulog, None)
        self._log(f"  {os.path.basename(path)} ready")

    @QtCore.pyqtSlot(str, str)
    def _one_failed(self, path, msg):
        self._log(f"  !! {os.path.basename(path)}: {msg}")

    @QtCore.pyqtSlot()
    def _load_finished(self):
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None
        self.busy.hide()
        # Take the list and clear the field BEFORE refreshing anything.  A
        # refresh can discover it still lacks a log and queue itself again
        # through needs_logs -> _ensure_loaded, which writes _pending_ids; doing
        # the clear afterwards threw that request away and the graph then never
        # drew at all.
        todo = list(getattr(self, "_pending_ids", []))
        self._pending_ids = []
        for gid in todo:
            card = self._cards.get(gid)
            if card is not None:
                card.refresh()
        if self._pending:
            more, self._pending = self._pending, None
            self._ensure_loaded(more)

    # -- graphs
    def _add_graph(self):
        g = self.report.add_graph(logs=self.report.log_names())
        self._mount_card(g)
        self._touch()
        self._ensure_loaded([g.id])

    def _mount_card(self, graph):
        card = GraphCard(graph, self)
        card.changed.connect(self._touch)
        card.removed.connect(self._remove_graph)
        card.needs_logs.connect(lambda gid: self._ensure_loaded([gid]))
        card.set_available_logs(self.report.log_names())
        self.page.addWidget(card)
        self._cards[graph.id] = card
        return card

    def _remove_graph(self, gid):
        card = self._cards.pop(gid, None)
        if card is not None:
            card.setParent(None)
            card.deleteLater()
        self.report.remove_graph(gid)
        self._touch()

    # -- shutdown
    def stop(self):
        """Stop the parse thread.  Without this the window closes and the process
        then hangs waiting on a QThread nobody asked to quit."""
        self._watch.stop()
        self._pending = None
        if self._thread is not None:
            if self._worker is not None:
                self._worker.cancelled = True
            self._thread.quit()
            if not self._thread.wait(5000):
                # Still inside one big parse.  Dropping the last reference now
                # frees a QThread that is still running, which is the crash this
                # is avoiding; parking it costs one dead object per session.
                _ORPHAN_THREADS.append((self._thread, self._worker))
            self._thread = None
            self._worker = None

    def flush(self):
        """Push every editor's buffer into the model.  Does NOT write to disk."""
        self.notes.flush()
        for card in self._cards.values():
            card.notes.flush()

    def has_unsaved(self):
        return self._dirty
