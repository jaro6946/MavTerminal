#!/usr/bin/env python3
"""log_browser_crumbs.py -- the crash-log breadcrumb, importable without a cycle.

`crumb()` lives in log_browser, which imports report_tab, so report_tab cannot
import it back.  This thin shim resolves it at call time instead, so the Report
tab's activity lands in the same crash log as everything else -- the absence of
which is exactly why the 2026-09-04 segfault had to be diagnosed without it.
"""
import sys


def crumb(msg):
    mod = sys.modules.get("log_browser")
    fn = getattr(mod, "crumb", None) if mod is not None else None
    if fn is not None:
        fn(msg)
