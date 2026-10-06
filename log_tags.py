#!/usr/bin/env python3
"""log_tags.py -- short text tags on logs, and a dropdown filter built on them.

Notes answer "what happened in this log"; tags answer "which logs are the ones
I mean" -- bench vs flight, a firmware build, a test campaign -- so they are
kept short, reused across logs, and filterable.

Storage follows the notes box exactly, for the same reasons (survives a wiped
config, travels with a copied run folder, greppable without the GUI): a sidecar
`<stem>_tags.txt` beside the log, one tag per line, with the browser's JSON
state as the fallback for read-only folders.  The rename path carries the
sidecar along.  The VOCABULARY -- every tag ever used, so an old tag can be
re-applied from a list instead of retyped -- lives in the JSON state, since it
belongs to this library rather than to any one log.

This module holds the two widgets and the Qt-free normalisation; the Browser
owns persistence and passes it in as callbacks, like NotesBox.

Acronyms: GUI = graphical user interface, JSON = JavaScript Object Notation.
"""
import re

from PyQt5 import QtCore, QtWidgets

from qt_common import FlowLayout, flow_holder
from ulog_common import C_MUTED, C_SATS

__all__ = ["TAGS_SUFFIX", "tags_path_for", "normalize_tag", "merge_tags",
           "TagBar", "TagFilter"]

TAGS_SUFFIX = "_tags.txt"

# Pill styling shared by both widgets.  The filter's pills are checkable; the
# tag bar's are plain buttons whose click removes the tag.
_PILL = ("QToolButton { border: 1px solid #c9c8c3; border-radius: 9px;"
         " padding: 1px 8px; font-size: 11px; background: #efeeea; }"
         "QToolButton:hover { border-color: #9a9993; }"
         f"QToolButton:checked {{ background: {C_SATS}; color: white;"
         f" border-color: {C_SATS}; }}")


def tags_path_for(path):
    """<folder>/<name>.ulg -> <folder>/<name>_tags.txt"""
    stem = path[:-4] if path.lower().endswith(".ulg") else path
    return stem + TAGS_SUFFIX


def normalize_tag(text):
    """Trim and collapse whitespace.  Case is kept as typed; equality ignores it."""
    return re.sub(r"\s+", " ", text or "").strip()


def merge_tags(*lists):
    """Union of tag lists, first spelling wins, case-insensitive, sorted."""
    out = {}
    for tags in lists:
        for t in tags or ():
            t = normalize_tag(t)
            if t:
                out.setdefault(t.casefold(), t)
    return sorted(out.values(), key=str.casefold)


def _pill(text, tooltip="", checkable=False):
    b = QtWidgets.QToolButton()
    b.setText(text)
    b.setCheckable(checkable)
    b.setStyleSheet(_PILL)
    b.setCursor(QtCore.Qt.PointingHandCursor)
    if tooltip:
        b.setToolTip(tooltip)
    return b


def _clear_flow(layout, keep):
    """Remove (and delete) every widget after the first `keep` items."""
    while layout.count() > keep:
        item = layout.takeAt(keep)
        w = item.widget() if item else None
        if w is not None:
            w.setParent(None)       # gone NOW; deleteLater alone leaves it
            w.deleteLater()         # painted until the event loop next runs


class TagBar(QtWidgets.QWidget):
    """The open log's tags as removable pills, plus an add-tag box.

    The add box is an editable dropdown: its list is every known tag the log
    does not already carry, and typing a name that is not in it makes a new
    tag.  Either way Enter (or picking from the list) applies it at once --
    there is no save step, because a tag is one short write.

    `on_change(tags)` is called with the log's full new list; `vocabulary()`
    returns every known tag.  Both are the Browser's.
    """

    def __init__(self, on_change, vocabulary, parent=None):
        super().__init__(parent)
        self._on_change = on_change
        self._vocabulary = vocabulary
        self._tags = []
        self._loaded = False

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(10, 0, 10, 6)
        v.setSpacing(0)
        self.row = FlowLayout(hspacing=6, vspacing=4)
        head = QtWidgets.QLabel("Tags:")
        head.setStyleSheet("font-weight: 600;")
        self.row.addWidget(head)

        self.add = QtWidgets.QComboBox()
        self.add.setEditable(True)
        # NoInsert: a typed tag goes onto the LOG, not into this list as a
        # dangling entry; the list is rebuilt from the vocabulary afterwards.
        self.add.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.add.setMinimumContentsLength(18)
        self.add.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLength)
        self.add.lineEdit().setPlaceholderText("+ add tag…")
        self.add.setToolTip("Pick a tag you have used before, or type a new "
                            "one and press Enter.")
        comp = self.add.completer()
        comp.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        comp.setFilterMode(QtCore.Qt.MatchContains)
        comp.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.add.activated[int].connect(self._picked)
        self.add.lineEdit().returnPressed.connect(self._typed)

        self.row.addWidget(self.add)
        self._fixed = 1             # items before the pills: the "Tags:" label
        v.addWidget(flow_holder(self.row))
        self.set_tags([], enabled=False)

    # -- content
    def set_tags(self, tags, enabled=True):
        """Point the bar at another log (or at nothing, enabled=False)."""
        self._tags = merge_tags(tags)
        self._loaded = enabled
        self._render()

    def refresh_vocabulary(self):
        self._fill_add_list()

    def _render(self):
        # Pills sit between the label and the add box, so the box is taken out
        # and re-added at the end rather than rebuilt.
        self.row.removeWidget(self.add)
        _clear_flow(self.row, self._fixed)
        if not self._tags:
            hint = QtWidgets.QLabel("(none)" if self._loaded else "")
            hint.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
            self.row.addWidget(hint)
        for t in self._tags:
            b = _pill(f"{t}  ✕", f"Remove '{t}' from this log")
            b.clicked.connect(lambda _=False, t=t: self._remove(t))
            self.row.addWidget(b)
        self.row.addWidget(self.add)
        self.add.setEnabled(self._loaded)
        self._fill_add_list()
        self.row.invalidate()

    def _fill_add_list(self):
        have = {t.casefold() for t in self._tags}
        self.add.blockSignals(True)
        self.add.clear()
        self.add.addItems([t for t in self._vocabulary()
                           if t.casefold() not in have])
        self.add.setCurrentIndex(-1)
        self.add.clearEditText()
        self.add.blockSignals(False)

    # -- edits
    def _picked(self, index):
        if index >= 0:
            self._add_tag(self.add.itemText(index))

    def _typed(self):
        self._add_tag(self.add.currentText())

    def _add_tag(self, text):
        t = normalize_tag(text)
        if not t or not self._loaded:
            return
        if t.casefold() not in {x.casefold() for x in self._tags}:
            # A typed tag that matches a known one in another case takes the
            # known spelling, so "Bench" and "bench" never become two tags.
            known = {x.casefold(): x for x in self._vocabulary()}
            self._tags = merge_tags(self._tags, [known.get(t.casefold(), t)])
            self._on_change(list(self._tags))
        self._render()

    def _remove(self, tag):
        self._tags = [t for t in self._tags if t != tag]
        self._on_change(list(self._tags))
        self._render()


class TagFilter(QtWidgets.QWidget):
    """Collapsible "Filter by tag" section for the log dropdown.

    Closed by default, like the notes box.  Each known tag is a toggle pill
    (with how many library logs carry it); "all" keeps logs carrying every
    selected tag, "any" keeps logs carrying at least one.  When closed, the
    header still says whether a filter is active and how much it hides -- a
    filter you cannot see is a dropdown that silently lost half its logs.

    `on_change()` asks the Browser to re-project the dropdown.  Right-click a
    pill that no log uses to drop it from the vocabulary (typos).
    """

    def __init__(self, on_change, on_forget, parent=None):
        super().__init__(parent)
        self._on_change = on_change
        self._on_forget = on_forget
        self._selected = {}         # casefolded -> spelling shown
        self._counts = {}

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(10, 0, 10, 4)
        v.setSpacing(3)

        head = QtWidgets.QHBoxLayout()
        head.setSpacing(8)
        self.btn = QtWidgets.QToolButton()
        self.btn.setText("Filter by tag")
        self.btn.setCheckable(True)
        self.btn.setArrowType(QtCore.Qt.RightArrow)
        self.btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.btn.setToolTip("Limit the log dropdown to logs carrying the "
                            "selected tags")
        self.btn.toggled.connect(self._toggled)
        head.addWidget(self.btn)
        self.lbl = QtWidgets.QLabel("")
        self.lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        head.addWidget(self.lbl, 1)
        v.addLayout(head)

        self.body = QtWidgets.QWidget()
        bv = QtWidgets.QVBoxLayout(self.body)
        bv.setContentsMargins(18, 0, 0, 2)
        bv.setSpacing(4)
        ctl = QtWidgets.QHBoxLayout()
        ctl.addWidget(QtWidgets.QLabel("match:"))
        self.mode = QtWidgets.QComboBox()
        self.mode.addItem("all selected tags", "all")
        self.mode.addItem("any selected tag", "any")
        self.mode.currentIndexChanged.connect(lambda *_: self._changed())
        ctl.addWidget(self.mode)
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_selection)
        ctl.addWidget(self.btn_clear)
        ctl.addStretch(1)
        bv.addLayout(ctl)
        self.pills = FlowLayout(hspacing=6, vspacing=4)
        bv.addWidget(flow_holder(self.pills))
        v.addWidget(self.body)

        self._toggled(False)

    # -- the Browser's view of it
    def active(self):
        return bool(self._selected)

    def matches(self, tags):
        """Does a log carrying `tags` pass the filter?"""
        if not self._selected:
            return True
        have = {t.casefold() for t in tags}
        if self.mode.currentData() == "any":
            return bool(have & self._selected.keys())
        return self._selected.keys() <= have

    def set_vocabulary(self, tags, counts):
        """Rebuild the pills.  `counts` = {casefolded tag: logs carrying it}."""
        self._counts = counts
        live = {t.casefold() for t in tags}
        # A forgotten tag cannot stay selected.
        self._selected = {k: v for k, v in self._selected.items() if k in live}
        _clear_flow(self.pills, 0)
        if not tags:
            hint = QtWidgets.QLabel("No tags yet — add one to a log with the "
                                    "Tags row below the title.")
            hint.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
            self.pills.addWidget(hint)
        for t in tags:
            n = counts.get(t.casefold(), 0)
            b = _pill(f"{t}  ({n})", checkable=True,
                      tooltip=f"{n} log(s) in the library carry '{t}'"
                              + ("" if n else "\nRight-click to remove it from "
                                               "the tag list"))
            b.setChecked(t.casefold() in self._selected)
            b.toggled.connect(lambda on, t=t: self._pill_toggled(t, on))
            b.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            b.customContextMenuRequested.connect(
                lambda pos, b=b, t=t, n=n: self._pill_menu(b, pos, t, n))
            self.pills.addWidget(b)
        self.pills.invalidate()

    def set_summary(self, shown, total):
        if not self._selected:
            self.lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
            self.lbl.setText("" if self.btn.isChecked() else "off")
            return
        names = sorted(self._selected.values(), key=str.casefold)
        joiner = " AND " if self.mode.currentData() == "all" else " OR "
        self.lbl.setText(f"showing {shown} of {total} logs  ·  "
                         f"{joiner.join(names)}")
        # Colour the header while a filter hides logs, so it reads as "on" even
        # with the section closed.
        self.lbl.setStyleSheet(f"color: {C_SATS}; font-size: 11px; "
                               "font-weight: 600;")

    # -- interaction
    def clear_selection(self):
        if not self._selected:
            return
        self._selected.clear()
        for i in range(self.pills.count()):
            w = self.pills.itemAt(i).widget()
            if isinstance(w, QtWidgets.QToolButton):
                w.blockSignals(True)
                w.setChecked(False)
                w.blockSignals(False)
        self._changed()

    def _pill_toggled(self, tag, on):
        if on:
            self._selected[tag.casefold()] = tag
        else:
            self._selected.pop(tag.casefold(), None)
        self._changed()

    def _pill_menu(self, button, pos, tag, count):
        menu = QtWidgets.QMenu(self)
        act = menu.addAction(f"Remove '{tag}' from the tag list")
        act.setEnabled(count == 0)
        if count:
            act.setToolTip("Still on a log -- remove it there first")
        if menu.exec_(button.mapToGlobal(pos)) is act:
            self._on_forget(tag)

    def _changed(self):
        self._on_change()

    def _toggled(self, on):
        self.body.setVisible(bool(on))
        self.btn.setArrowType(QtCore.Qt.DownArrow if on else QtCore.Qt.RightArrow)
        self._on_change()
