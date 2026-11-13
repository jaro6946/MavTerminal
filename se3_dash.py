#!/usr/bin/env python3
"""se3_dash.py -- the Bags tab's SE3 session dashboard widget.

Sits UNDER the bag-video player (the video stays on top for when camera bags
exist): pick one session (bag) of the selected date folder, then a view --
the whole session, or a +-3 s window around any PX4<->SE3 handoff.  Drawing is
se3_bag_plots.build_dashboard; this file is only the Qt around it.

The figure is a plain matplotlib Figure, not a pyplot one, so dropping the
canvas really frees it (pyplot's registry is what leaked ~150 MB per log in the
browser before -- see log_browser.PlotPage.clear).
"""
import os

from PyQt5 import QtCore, QtWidgets
# matplotlib's Qt canvas is imported HERE, before se3_bag_plots can pull in
# rosbags: first-importing matplotlib (its pybind11 ft2font) after rosbags had
# loaded, from inside a Qt callback, aborted the process with an uncaught
# pybind11::attribute_error in headless tests (2026-11-13).  logGraph itself
# imports matplotlib long before this tab exists; this keeps that order for
# any other caller too.
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

import se3_bag_plots as P
from ulog_common import C_MUTED

__all__ = ["SE3Dashboard"]

ZOOM_HALF_S = 3.0


class SE3Dashboard(QtWidgets.QWidget):
    def __init__(self, log, state=None, parent=None):
        super().__init__(parent)
        self._log = log
        self.state = state if state is not None else {}
        self.folder = None
        self.sess = None
        self.canvas = None
        self.toolbar = None

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(10, 4, 10, 4)
        v.setSpacing(2)
        bar = QtWidgets.QHBoxLayout()
        hdr = QtWidgets.QLabel("SE3 session dashboard")
        hdr.setStyleSheet("font-weight: 600;")
        bar.addWidget(hdr)
        bar.addSpacing(10)
        bar.addWidget(QtWidgets.QLabel("session:"))
        self.bag_pick = QtWidgets.QComboBox()
        self.bag_pick.setStyleSheet("font-family: monospace;")
        self.bag_pick.setMinimumContentsLength(34)
        self.bag_pick.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLength)
        self.bag_pick.activated.connect(self._bag_picked)
        bar.addWidget(self.bag_pick, 1)
        bar.addWidget(QtWidgets.QLabel("view:"))
        self.view_pick = QtWidgets.QComboBox()
        self.view_pick.setMinimumContentsLength(28)
        self.view_pick.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLength)
        self.view_pick.activated.connect(lambda _i: self._draw())
        bar.addWidget(self.view_pick)
        b = QtWidgets.QPushButton("Reload")
        b.setToolTip("Re-read the bag and its logs/ (e.g. after copying a trace in)")
        b.clicked.connect(self._reload)
        bar.addWidget(b)
        v.addLayout(bar)

        self.tb_holder = QtWidgets.QHBoxLayout()
        v.addLayout(self.tb_holder)
        self.note = QtWidgets.QLabel()
        self.note.setStyleSheet(f"color: {C_MUTED};")
        self.note.setWordWrap(True)
        v.addWidget(self.note)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.page = QtWidgets.QWidget()
        self.page_lay = QtWidgets.QVBoxLayout(self.page)
        self.page_lay.setContentsMargins(0, 0, 0, 0)
        self.scroll.setWidget(self.page)
        v.addWidget(self.scroll, 1)

    # -- folder / bag ----------------------------------------------------------
    def set_folder(self, folder):
        """Show the sessions of one date folder (None = clear)."""
        if folder == self.folder and self.bag_pick.count():
            return
        self.folder = folder
        self.bag_pick.clear()
        bags = P.list_bags(folder) if folder and os.path.isdir(folder) else []
        ses = (P.session_numbers(os.path.dirname(os.path.abspath(folder)))
               if bags else {})
        for b in bags:
            name = os.path.basename(b)
            has_trace = os.path.isfile(os.path.join(b, "logs", "se3_trace.csv"))
            self.bag_pick.addItem(
                f"{ses.get(name, '------')}  {P.session_label_time(name):<16} "
                f"{P.missions_text(P.read_missions(b)):<14}"
                f"{'  + trace' if has_trace else ''}", b)
            self.bag_pick.setItemData(self.bag_pick.count() - 1, name,
                                      QtCore.Qt.ToolTipRole)
        if not bags:
            self._clear_canvas()
            self.view_pick.clear()
            self.note.setText(f"no bags in {folder}" if folder else "")
            return
        last = self.state.get("se3dash_last")
        idx = next((i for i, b in enumerate(bags) if b == last), len(bags) - 1)
        self.bag_pick.setCurrentIndex(idx)
        self._bag_picked(idx)

    def _bag_picked(self, idx):
        path = self.bag_pick.itemData(idx)
        if not path:
            return
        self.state["se3dash_last"] = path
        self._load(path)

    def _reload(self):
        path = self.bag_pick.currentData()
        if path:
            self._load(path)

    def _load(self, path):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.sess = P.load_session(path)
        except Exception as e:
            self.sess = None
            self._clear_canvas()
            self.view_pick.clear()
            self.note.setText(f"could not read {os.path.basename(path)}: "
                              f"{type(e).__name__}: {e}")
            self._log(f"[se3dash] {os.path.basename(path)}: {type(e).__name__}: {e}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.view_pick.clear()
        self.view_pick.addItem("Full session", None)
        for k, (t, lab) in enumerate(P.handoffs(self.sess), 1):
            self.view_pick.addItem(f"Handoff {k}: {lab} @ {t:.1f} s",
                                   (t - ZOOM_HALF_S, t + ZOOM_HALF_S))
        tr = self.sess.trace
        if tr is not None:
            # Also offer the moment the node entered recovery, if it did.
            rec = [i for i, p in enumerate(tr["phase"]) if "RECOVERY" in p]
            if rec:
                t = float(tr["t"][rec[0]])
                self.view_pick.addItem(f"SE3 RECOVERY entered @ {t:.1f} s",
                                       (t - ZOOM_HALF_S, t + ZOOM_HALF_S))
        self._log(f"[se3dash] {self.sess.name}: {self.sess.duration_s:.1f} s, "
                  f"{len(P.handoffs(self.sess))} handoff(s), "
                  f"{'trace' if tr is not None else self.sess.trace_note}")
        self._draw()

    # -- figure ------------------------------------------------------------------
    def _clear_canvas(self):
        for w in (self.toolbar, self.canvas):
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self.canvas = self.toolbar = None

    def _draw(self):
        if self.sess is None:
            return
        window = self.view_pick.currentData()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            fig = P.build_dashboard(self.sess, window=window)
        except Exception as e:
            self.note.setText(f"dashboard failed: {type(e).__name__}: {e}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._clear_canvas()
        self.canvas = FigureCanvasQTAgg(fig)
        # Fixed height, free width (log_browser.PlotPage.add): the figure's
        # inch size would otherwise force a sideways scroll.
        self.canvas.setFixedHeight(int(fig.get_figheight() * fig.dpi))
        self.canvas.setMinimumWidth(320)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                  QtWidgets.QSizePolicy.Fixed)
        self.page_lay.addWidget(self.canvas)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setIconSize(QtCore.QSize(16, 16))
        self.tb_holder.addWidget(self.toolbar)
        self.canvas.draw_idle()
        s = self.sess
        bits = [s.ses, P.session_label_time(s.name),
                P.missions_text(s.missions, short=False), f"{s.duration_s:.1f} s"]
        if s.truncated:
            bits.append("recorder was killed: bag read to its last complete chunk")
        bits.append(s.align or s.trace_note)
        self.note.setText("   ·   ".join(b for b in bits if b))

    def stop(self):
        self._clear_canvas()
