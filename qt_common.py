#!/usr/bin/env python3
"""qt_common.py -- Qt widgets shared by the browser's tabs.

Both tabs draw matplotlib into a scroll area and both offer a collapsible notes
box, so these two live here rather than in either tab: log_browser imports the
Report tab, which means the Report tab cannot import back out of it.

Acronyms: GUI = graphical user interface.
"""
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from ulog_common import C_MUTED

__all__ = ["PlotCanvas", "NotesBox", "FlowLayout", "first_line"]


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


class FlowLayout(QtWidgets.QLayout):
    """A left-to-right layout that WRAPS instead of growing forever.

    QHBoxLayout has no wrap: a row of one checkbox per log reports a minimum
    width of however wide all of them are side by side -- 5,047 px for a
    fourteen-log report -- and a QScrollArea honours that by giving its content
    that width.  Everything downstream then inherits it, so the graphs (which
    expand to the card) were drawn four screens wide and the whole page scrolled
    sideways.  A flow layout's minimum is its WIDEST SINGLE ITEM, so the card
    fits any window and the row grows downwards instead.

    The standard Qt flow-layout shape: heightForWidth reports the height the
    items need at a given width, and _lay does double duty as measurement (dry
    run) and placement.
    """

    def __init__(self, parent=None, margin=0, hspacing=8, vspacing=4):
        super().__init__(parent)
        self._items = []
        self._h, self._v = hspacing, vspacing
        self.setContentsMargins(margin, margin, margin, margin)

    # -- QLayout plumbing.  Qt calls takeAt until it returns None on teardown.
    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._lay(QtCore.QRect(0, 0, width, 0), measure=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._lay(rect, measure=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _lay(self, rect, measure):
        """Place (or just measure) the items; returns the height used."""
        m = self.contentsMargins()
        x, y = rect.x() + m.left(), rect.y() + m.top()
        right = rect.right() - m.right()
        line_h = 0
        for item in self._items:
            hint = item.sizeHint()
            if line_h and x + hint.width() > right:
                x = rect.x() + m.left()
                y += line_h + self._v
                line_h = 0
            if not measure:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x += hint.width() + self._h
            line_h = max(line_h, hint.height())
        return y + line_h + m.bottom() - rect.y()


def flow_holder(layout):
    """A widget carrying a FlowLayout that actually reports its wrapped height.

    A layout's heightForWidth only reaches the parent layout if the WIDGET's
    size policy says it has one -- otherwise the row is measured at its
    single-line height and the wrapped rows are drawn over whatever sits below.
    """
    w = QtWidgets.QWidget()
    w.setLayout(layout)
    policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                   QtWidgets.QSizePolicy.Minimum)
    policy.setHeightForWidth(True)
    w.setSizePolicy(policy)
    return w


def first_line(text, width=90):
    """The one-line preview shown next to a collapsed Notes header."""
    line = next((l.strip() for l in text.splitlines() if l.strip()), "")
    return line if len(line) <= width else line[:width - 1] + "…"


class NotesBox(QtWidgets.QWidget):
    """A collapsible free-text box that saves itself shortly after you stop typing.

    Three of these exist now -- one per log, one per report, one per graph -- and
    they differ ONLY in where the text ends up: a sidecar `.txt` beside the log,
    or a field in the report's JSON.  So this widget owns the parts that are the
    same (the disclosure header, the preview line, the debounce, the transient
    "saved" acknowledgement) and takes persistence as a callback.

    `remember` is an optional (read, write) pair of callables for the collapsed
    state.  Opening the box FOR the user, because the thing they just selected
    already has a note, is deliberately NOT written back through it: that is the
    widget being helpful once, not the user asking for every box to be open.
    """

    def __init__(self, label="Notes", placeholder="", tooltip="",
                 on_save=None, remember=None, min_height=64, max_height=150,
                 parent=None):
        super().__init__(parent)
        self._on_save = on_save or (lambda text: None)
        self._remember = remember
        self._dirty = False
        self._auto_open = False
        self._loaded = False        # nothing selected yet; stay disabled

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(10, 0, 10, 6)
        v.setSpacing(3)

        head = QtWidgets.QHBoxLayout()
        head.setSpacing(8)
        self.btn = QtWidgets.QToolButton()
        self.btn.setText(label)
        self.btn.setCheckable(True)
        self.btn.setArrowType(QtCore.Qt.RightArrow)
        self.btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        if tooltip:
            self.btn.setToolTip(tooltip)
        self.btn.toggled.connect(self._toggled)
        head.addWidget(self.btn)
        self.lbl = QtWidgets.QLabel("")
        self.lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        head.addWidget(self.lbl, 1)
        v.addLayout(head)

        self.edit = QtWidgets.QPlainTextEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.setStyleSheet("font-size: 12px;")
        self.edit.setMinimumHeight(min_height)
        self.edit.setMaximumHeight(max_height)
        self.edit.setEnabled(False)
        self.edit.textChanged.connect(self._changed)
        v.addWidget(self.edit)

        # Autosave: a keystroke restarts the timer, so the write happens once
        # you pause rather than once per character.  Every path that could lose
        # the buffer (switching logs, closing the window) flushes it first.
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(700)
        self._timer.timeout.connect(self.flush)

        # The "saved" tick is transient; this clears it without clearing a
        # collapsed header's preview line.
        self._ack = QtCore.QTimer(self)
        self._ack.setSingleShot(True)
        self._ack.setInterval(1600)
        self._ack.timeout.connect(lambda: self._update_header())

        if remember is not None:
            self.btn.setChecked(bool(remember[0]()))
        self._toggled(self.btn.isChecked())

    # -- content
    def text(self):
        return self.edit.toPlainText()

    def set_text(self, text, enabled=True):
        """Point the box at another subject.  Flushes what it was holding first."""
        self.flush()
        text = text or ""
        self.edit.blockSignals(True)
        self.edit.setPlainText(text)
        self.edit.blockSignals(False)
        self.edit.setEnabled(enabled)
        self._loaded = enabled
        self._dirty = False
        self._ack.stop()
        if text.strip() and not self.btn.isChecked():
            self._auto_open = True
            try:
                self.btn.setChecked(True)       # -> _toggled -> header
            finally:
                self._auto_open = False
        else:
            self._update_header()

    def clear(self):
        self.set_text("", enabled=False)

    def flush(self):
        """Write the buffer out if it changed.  Safe to call any number of times."""
        self._timer.stop()
        if not self._dirty or not self._loaded:
            return
        self._dirty = False
        self._on_save(self.edit.toPlainText())
        self._update_header("saved ✓")
        self._ack.start()

    # -- collapse
    def is_open(self):
        return self.btn.isChecked()

    def set_open(self, on):
        self.btn.setChecked(bool(on))

    def _toggled(self, on):
        self.edit.setVisible(bool(on))
        self.btn.setArrowType(QtCore.Qt.DownArrow if on else QtCore.Qt.RightArrow)
        if self._remember is not None and not self._auto_open:
            self._remember[1](bool(on))
        self._update_header()

    def _changed(self):
        self._dirty = True
        self._ack.stop()
        self._update_header("unsaved…")
        self._timer.start()

    def _update_header(self, status=""):
        """Right of the header: a save acknowledgement, or the collapsed preview."""
        if status:
            self.lbl.setText(status)
            return
        if not self._loaded:
            self.lbl.setText("")
            return
        text = self.edit.toPlainText().strip()
        if self.btn.isChecked():
            self.lbl.setText("" if text else "nothing written yet")
        else:
            self.lbl.setText(first_line(text) if text
                             else "(none) — click to add")
