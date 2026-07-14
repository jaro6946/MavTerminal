#!/usr/bin/env python3
"""pull_log.py -- pull PX4/ArduPilot .ulg flight logs off the FC over MAVLink FTP,
no SD-card removal and no QGroundControl.

Why: when a board "won't arm" / failsafes / terminates, the ONLY authoritative
record of WHY is the onboard .ulg (vehicle_status, failsafe_flags, battery_status,
actuator_motors ...).  MAVFTP pulls it over the same USB link mavshell uses -- a
~500 KB log comes down in ~5 s at 921600.  Analyse the result with `ulog_diag.py`.

Examples:
    # List the sessions/logs on the card (nothing downloaded):
    pull_log.py --list

    # Pull the NEWEST log from the newest session into the current dir:
    pull_log.py

    # Pull a specific one:
    pull_log.py --session sess112 --name log102.ulg -o /tmp

Auto-detects the FC port (same VID/keyword logic as mavshell); override with -p/-b.
Depends only on pymavlink (already in the agc_CTOL_SE3-rotopy venv).
"""
import argparse
import os
import sys
import time

from pymavlink import mavutil
from pymavlink.mavftp import MAVFTP

# ---- FC serial auto-detect (kept in sync with mavshell.py) ------------------
FC_VIDS = {0x26AC, 0x0483, 0x1209, 0x2DAE, 0x3162}
FC_KEYWORDS = ("px4", "pixhawk", "ardupilot", "fmu", "mro", "cube", "holybro",
               "control zero", "controlzero")
LOG_DIR = "/fs/microsd/log"


def autodetect_port():
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    for p in ports:
        if p.vid in FC_VIDS:
            return p.device
    for p in ports:
        text = f"{p.description} {p.manufacturer}".lower()
        if any(k in text for k in FC_KEYWORDS):
            return p.device
    return ports[0].device if ports else None


def listdir(ftp, path):
    """Return the DirectoryEntry list for a remote dir (cmd_list pumps internally)."""
    ftp.dir_offset = 0
    ftp.list_temp_result = []
    ftp.list_result = []
    ftp.cmd_list([path])
    return list(ftp.list_result)


def download(ftp, master, remote, local, stall_s=12.0):
    """Download one remote file to ``local``.  cmd_get only kicks off the burst
    read; we pump FILE_TRANSFER_PROTOCOL replies until it reports done."""
    ftp.done = False
    ftp.cmd_get([remote, local])
    pump = ftp._MAVFTP__mavlink_packet          # name-mangled private helpers
    idle = ftp._MAVFTP__idle_task
    deadline = time.time() + stall_s
    while not ftp.done and time.time() < deadline:
        msg = master.recv_match(type="FILE_TRANSFER_PROTOCOL", blocking=True, timeout=1.0)
        if msg is None:
            idle()
            continue
        deadline = time.time() + stall_s
        pump(msg)
        idle()
    return os.path.exists(local)


# ---- Reusable helpers (shared with mavshell's `log` command) -----------------
# These take an already-open ``master`` MAVLink connection and build the MAVFTP
# on top of it, so the mavTerminal shell can list/pull logs over the SAME serial
# port it already owns — no second port-owner (which would collide on the one FC
# link).  ``log`` is a print-like callback so callers can route output.

def make_ftp(master):
    return MAVFTP(master, master.target_system, master.target_component)


def list_sessions(ftp):
    """Return session dir entries under LOG_DIR, oldest→newest."""
    sessions = [e for e in listdir(ftp, LOG_DIR) if e.is_dir]
    sessions.sort(key=lambda e: e.name)
    return sessions


def list_logs(ftp, session):
    """Return the .ulg file entries in one session dir, oldest→newest."""
    logs = [e for e in listdir(ftp, f"{LOG_DIR}/{session}")
            if not e.is_dir and e.name.endswith(".ulg")]
    logs.sort(key=lambda e: e.name)
    return logs


def pull(ftp, master, outdir=".", session=None, name=None, log=print):
    """Resolve (session, name) — defaulting to the newest — and download it.

    Returns the local path on success, or None. Mirrors main()'s resolution so
    both the CLI and the shell's `log pull` behave identically."""
    sessions = list_sessions(ftp)
    if not sessions:
        log(f"no session dirs under {LOG_DIR}")
        return None
    sess = session or sessions[-1].name
    logs = list_logs(ftp, sess)
    if not logs:
        log(f"no .ulg in {sess}")
        return None
    target = next((e for e in logs if e.name == name), None) if name else logs[-1]
    if target is None:
        log(f"{name} not found in {sess} (have: {[e.name for e in logs]})")
        return None

    remote = f"{LOG_DIR}/{sess}/{target.name}"
    local = os.path.join(outdir, f"{sess}_{target.name}")
    log(f"downloading {remote} ({target.size_b} B) -> {local} ...")
    t0 = time.time()
    if not download(ftp, master, remote, local):
        log("download FAILED")
        return None
    log(f"wrote {os.path.getsize(local)} B in {time.time()-t0:.1f}s: {local}")
    return local


# ---- Deleting logs off the card (DESTRUCTIVE) --------------------------------
# cmd_rm / cmd_rmdir each drive their own FTP request/reply internally (they pump
# via process_ftp_reply), so they're the sole reader while running — the same
# single-owner requirement as download(); the shell pauses its recv thread first.

def remove_file(ftp, path):
    """Delete one remote file. Returns True on the FC's success ack."""
    return ftp.cmd_rm([path]).return_code == 0


def remove_dir(ftp, path):
    """Delete one remote directory (must already be empty). True on success."""
    return ftp.cmd_rmdir([path]).return_code == 0


def walk_logs(ftp, path=LOG_DIR):
    """Depth-first walk of everything under ``path``, yielding (entry, full_path,
    is_dir). Files are yielded before the directory that contains them, and
    ``path`` itself is never yielded — so a caller can delete bottom-up (files,
    then now-empty dirs) while keeping the top log directory intact. listdir()
    materializes each level into a list before we recurse, so deleting as you
    iterate is safe."""
    for e in listdir(ftp, path):
        if e.name in (".", "..", ""):
            continue
        child = f"{path}/{e.name}"
        if e.is_dir:
            yield from walk_logs(ftp, child)
            yield e, child, True
        else:
            yield e, child, False


def summarize_logs(ftp):
    """Count what a delete-all would remove: (n_files, n_dirs, total_bytes)."""
    n_files = n_dirs = total = 0
    for e, _path, is_dir in walk_logs(ftp):
        if is_dir:
            n_dirs += 1
        else:
            n_files += 1
            total += (e.size_b or 0)
    return n_files, n_dirs, total


def delete_all_logs(ftp, log=print):
    """Delete every file and session dir under LOG_DIR (LOG_DIR itself is kept).
    Returns (n_removed, n_failed)."""
    removed = failed = 0
    for e, path, is_dir in walk_logs(ftp):
        ok = remove_dir(ftp, path) if is_dir else remove_file(ftp, path)
        kind = "dir " if is_dir else ""
        if ok:
            removed += 1
            log(f"removed {kind}{path}")
        else:
            failed += 1
            log(f"FAILED  {kind}{path}")
    return removed, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--port", default=os.environ.get("MAV_PORT") or autodetect_port())
    ap.add_argument("-b", "--baud", type=int, default=int(os.environ.get("MAV_BAUD", "921600")))
    ap.add_argument("-o", "--outdir", default=".", help="where to save (default: cwd)")
    ap.add_argument("--list", action="store_true", help="list sessions/logs and exit")
    ap.add_argument("--session", help="session dir to pull from (default: newest)")
    ap.add_argument("--name", help="log file to pull (default: newest .ulg in the session)")
    ap.add_argument("--delete-all", action="store_true",
                    help="DELETE every log off the card (dry-run unless --yes is given)")
    ap.add_argument("--yes", action="store_true",
                    help="confirm a destructive --delete-all (skip the dry run)")
    args = ap.parse_args()

    if not args.port:
        print("no serial port found -- pass -p or attach the FC", file=sys.stderr)
        return 2

    print(f"connecting {args.port} @ {args.baud} ...", flush=True)
    master = mavutil.mavlink_connection(args.port, baud=args.baud)
    if master.wait_heartbeat(timeout=15) is None:
        print("no heartbeat -- is the FC attached (usbipd) and the sim stopped?",
              file=sys.stderr)
        return 1
    print(f"heartbeat: system {master.target_system} component {master.target_component}",
          flush=True)
    ftp = make_ftp(master)

    if args.list:
        if args.session:                    # one session -> its .ulg files (fast)
            for e in list_logs(ftp, args.session):
                print(f"  {args.session}/{e.name}  {e.size_b} B")
        else:                               # just the session names (one FTP call)
            sessions = list_sessions(ftp)
            if not sessions:
                print(f"no session dirs under {LOG_DIR}", file=sys.stderr)
                return 1
            for s in sessions:
                print(f"  {s.name}{'   <- newest' if s is sessions[-1] else ''}")
            print("\n(pass --session <name> --list to see its .ulg files, "
                  "or run with no args to pull the newest)")
        return 0

    if args.delete_all:
        n_files, n_dirs, total = summarize_logs(ftp)
        if n_files == 0 and n_dirs == 0:
            print(f"no logs under {LOG_DIR} -- nothing to delete")
            return 0
        print(f"{'DELETING' if args.yes else 'would delete'} {n_files} file(s) in "
              f"{n_dirs} session dir(s), {total/1e6:.1f} MB, under {LOG_DIR}")
        if not args.yes:
            print("this is DESTRUCTIVE and cannot be undone -- re-run with --yes to do it.")
            return 0
        removed, failed = delete_all_logs(ftp, log=lambda m: print(f"  {m}", flush=True))
        print(f"done: removed {removed}, failed {failed}")
        return 1 if failed else 0

    local = pull(ftp, master, args.outdir, args.session, args.name,
                 log=lambda m: print(m, flush=True))
    if local is None:
        return 1
    print(f"analyse it:  ulog_diag.py {local}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
