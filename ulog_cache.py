#!/usr/bin/env python3
"""ulog_cache.py -- one parse of a ULog, shared by every tab that wants it.

The browse tab used to parse, draw, and drop the log; re-selecting it parsed it
again.  That was free, because nothing held the parse.  The Report tab changes
the arithmetic: it plots several logs at once and re-reads them on every field
you tick, so a cache stops being an optimisation and starts being the difference
between a responsive tab and one that stalls for seconds a click.

Two things about this cache are load-bearing rather than incidental:

  * It costs RESIDENT MEMORY that the old drop-it-immediately behaviour did not.
    A ULog holds roughly one megabyte per megabyte of file, so the library's
    biggest logs are 200 MB each and four of them is most of what this machine
    has free.  Hence the caps below, and hence the check against MemAvailable --
    getting OOM-killed loses the window and every unsaved note in it, and leaves
    no traceback to explain why.

  * A cached log remembers WHICH TOPICS it was parsed with.  The browse tab asks
    for the ~63 topics its plot registry needs; the Report tab needs all of them.
    Handing the former to the latter would silently show a tenth of the file's
    fields with no error anywhere -- so a hit requires the cached parse to be a
    superset of what the caller asked for, and an unfiltered parse serves
    everyone.

Acronyms: ULog = PX4's binary log format, LRU = least-recently-used,
RSS = resident set size.
"""
import os
import threading

import numpy as np
from pyulog import ULog

from ulog_common import field_inventory

__all__ = ["LogCache", "CacheEntry", "MeasuredULog", "corruption_of",
           "parse_ulog", "start_epoch"]

# Defaults sized for this machine: ~7.7 GB total, and in practice under 2 GB
# available with swap already full.  Four of the library's large logs is ~800 MB,
# which is a lot but survivable; the MemAvailable floor is what actually protects
# the process when something else on the box grows.
MAX_LOGS = 4
MAX_MB = 1200
MIN_AVAIL_MB = 600


def _mem_available_mb():
    """Free-ish memory, from MemAvailable -- the kernel's own estimate.

    MemFree is the wrong number here: it excludes reclaimable page cache, so it
    reads catastrophically low on a box that has merely been reading big files,
    which is exactly what this program does."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return float("inf")             # unknown: do not evict on a guess


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
        # pyulog keeps no reference to where the log came from, and callers that
        # are handed a parsed ULog (report_render, for one) then cannot get back
        # to the file -- to date it, say, when the log carries no GNSS fix.
        self.source_path = args[0] if args and isinstance(args[0], str) else None
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


def parse_ulog(path, topics=None):
    """The one place a ULog is read.  `topics=None` reads every topic.

    Always a MeasuredULog, whichever tab asked: the corruption measurement is
    per-parse state, so a log first read by the tab that does not care about it
    would otherwise arrive at the tab that does with nothing to report."""
    if topics is None:
        return MeasuredULog(path)
    return MeasuredULog(path, message_name_filter_list=list(topics))

def start_epoch(ulog):
    """Epoch seconds of the log's FIRST sample, from GNSS UTC.  None if no fix.

    `time_utc_usec` is absolute (microseconds since the Unix epoch) while
    `timestamp` is microseconds since boot, so one sample carrying both pins the
    whole log to wall clock:

        start_epoch = utc[i] - (timestamp[i] - ulog.start_timestamp)

    Samples before the first fix carry 0, hence the 1e15 floor (~year 2001) --
    without it the answer is 1970 and looks like a bug in this function rather
    than an absent fix.
    """
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


class CacheEntry:
    """One parsed log, plus what was derived from it at parse time."""

    def __init__(self, ulog, topics, size_mb):
        self.ulog = ulog
        # None means "parsed unfiltered" -- the superset of every request.
        self.topics = None if topics is None else frozenset(topics)
        self.size_mb = size_mb
        self._inventory = None

    def satisfies(self, topics):
        return self.topics is None or (topics is not None
                                       and set(topics) <= self.topics)

    @property
    def inventory(self):
        """field_inventory(), computed once and kept with the log.

        Lazy rather than eager: the browse tab never asks for it, and it is a
        second of work on the biggest logs."""
        if self._inventory is None:
            self._inventory = field_inventory(self.ulog)
        return self._inventory


class LogCache:
    """A small LRU of parsed logs, keyed by identity rather than by path alone.

    Thread-safe because the parse happens on a worker thread and the lookup on
    the GUI thread; the lock is only ever held for dictionary work, never across
    a parse.
    """

    def __init__(self, max_logs=MAX_LOGS, max_mb=MAX_MB,
                 min_avail_mb=MIN_AVAIL_MB, log=None):
        self.max_logs = max_logs
        self.max_mb = max_mb
        self.min_avail_mb = min_avail_mb
        self._log = log or (lambda msg: None)
        self._entries = {}          # key -> CacheEntry
        self._order = []            # keys, least-recently-used first
        self._lock = threading.Lock()

    # -- identity
    @staticmethod
    def key(path):
        """(abspath, mtime, size).

        Not the path on its own: logGraph can RENAME a log, and `log pull` can
        drop a different file at a name that was used before.  Including the
        stat means a changed file is a miss rather than a stale hit."""
        p = os.path.abspath(path)
        try:
            st = os.stat(p)
            return (p, int(st.st_mtime), st.st_size)
        except OSError:
            return (p, 0, 0)

    # -- lookup
    def get(self, path, topics=None):
        """The cached entry usable for this topic set, or None.

        `topics=None` means the caller wants everything, which only an
        unfiltered parse can satisfy."""
        k = self.key(path)
        with self._lock:
            entry = self._entries.get(k)
            if entry is None or not entry.satisfies(topics):
                return None
            self._touch(k)
            return entry

    def put(self, path, ulog, topics=None):
        """Cache a parse.  Replaces a narrower parse of the same file."""
        k = self.key(path)
        try:
            size_mb = os.path.getsize(path) / 1048576.0
        except OSError:
            size_mb = 0.0
        entry = CacheEntry(ulog, topics, size_mb)
        with self._lock:
            existing = self._entries.get(k)
            # Keep whichever parse is broader: a filtered parse arriving after an
            # unfiltered one must not narrow what later callers can see.
            if existing is not None and existing.satisfies(topics) and \
                    not entry.satisfies(existing.topics):
                self._touch(k)
                return existing
            self._entries[k] = entry
            self._touch(k)
            self._evict()
        return entry

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._order.clear()

    def stats(self):
        with self._lock:
            return {"logs": len(self._entries),
                    "mb": round(sum(e.size_mb for e in self._entries.values()), 1),
                    "avail_mb": round(_mem_available_mb(), 1)}

    # -- internals (call with the lock held)
    def _touch(self, k):
        if k in self._order:
            self._order.remove(k)
        self._order.append(k)

    def _evict(self):
        """Drop least-recently-used entries until every cap is satisfied.

        The memory-pressure check is deliberately last and deliberately keeps one
        entry: evicting the log the user is looking at to satisfy a floor that
        something else on the machine breached would just re-parse it a moment
        later."""
        def total_mb():
            return sum(e.size_mb for e in self._entries.values())

        while len(self._order) > self.max_logs or \
                (len(self._order) > 1 and total_mb() > self.max_mb):
            self._drop_oldest("cap")

        while len(self._order) > 1 and _mem_available_mb() < self.min_avail_mb:
            self._drop_oldest("memory pressure")

    def _drop_oldest(self, why):
        k = self._order.pop(0)
        entry = self._entries.pop(k, None)
        if entry is not None:
            self._log(f"  cache: released {os.path.basename(k[0])} "
                      f"({entry.size_mb:.0f} MB, {why})")
