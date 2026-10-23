#!/usr/bin/env python3
"""pull_log.py -- pull PX4/ArduPilot .ulg flight logs off the FC over MAVLink,
no SD-card removal and no QGroundControl.  Downloads use MAVFTP; --log switches
to the MAVLink log protocol (the one QGC's Log Download uses).

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

    # Same, but through the companion (FC TELEM1 -> RB5 mavlink-routerd -> adb),
    # for when the FC's own USB link keeps wedging on big transfers:
    pull_log.py -p companion --session sess112 --name log102.ulg

Auto-detects the FC port (same VID/keyword logic as mavshell); override with -p/-b.
Depends only on pymavlink (already in the agc_CTOL_SE3-rotopy venv).
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

from pymavlink import mavutil
from pymavlink.mavftp import MAVFTP

# ---- FC serial auto-detect (kept in sync with mavshell.py) ------------------
FC_VIDS = {0x26AC, 0x0483, 0x1209, 0x2DAE, 0x3162}
FC_KEYWORDS = ("px4", "pixhawk", "ardupilot", "fmu", "mro", "cube", "holybro",
               "control zero", "controlzero")
LOG_DIR = "/fs/microsd/log"

# ---- Companion route: FC TELEM1 -> RB5 mavlink-routerd -> adb forward --------
# mavlink-routerd on the RB5 bridges the FC's TELEM1 (ttyUSB0) and serves TCP
# 5760; an adb forward brings that to the PC.  ~60 KiB/s, but it never touches
# the FC's USB port, which wedges FC->PC after a few MB on the grazer board.
COMPANION_PORT = "companion"
COMPANION_TCP = 5760                    # mavlink-routerd's TCP server on the RB5
COMPANION_LOCAL = 15760                 # PC end of the adb forward


def resolve_port(port):
    """Turn the name ``companion`` into a live TCP endpoint (setting up the adb
    forward); return any other port unchanged."""
    if port != COMPANION_PORT:
        return port
    r = subprocess.run(["adb", "forward", f"tcp:{COMPANION_LOCAL}",
                        f"tcp:{COMPANION_TCP}"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"adb forward failed ({(r.stderr or r.stdout).strip()}) "
                         "-- is the companion attached over adb?")
    return f"tcp:127.0.0.1:{COMPANION_LOCAL}"


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


# ---- MAVLink log protocol (what QGC's Log Download uses) ---------------------
# An alternative to download() (pull(method="log"), CLI --log).  Neither method
# survives a noisy USB link: one corrupted IN transaction (host sees -62 ETIME)
# wedges FC->PC until the FC's USB driver re-inits (2026-10-21: long cable routed
# past a noise source; a short/re-routed cable fixed it).  See usb_wedge_trial.py.
#
# Log ids carry no filename: they are the line index (0-based, both v1.14 and
# v1.18) of PX4's own list file, which it rewrites on every LOG_REQUEST_LIST.
# That file is small, so it is read over FTP to map session/name -> id exactly.
LOG_LIST_FILE = "/fs/microsd/logdata.txt"
LOG_DATA_LEN = 90                       # LOG_DATA payload bytes per message
LOG_CHUNK = 512 * LOG_DATA_LEN          # QGC's chunk: 512 bins = 46080 B


def log_entries(master, timeout=5.0):
    """LOG_REQUEST_LIST -> {id: (size_b, time_utc)}, empty if the FC has no logs
    or never answers.  Also makes PX4 rewrite LOG_LIST_FILE."""
    master.mav.log_request_list_send(master.target_system, master.target_component,
                                     0, 0xffff)
    entries, total = {}, None
    deadline = time.time() + timeout
    while time.time() < deadline:
        m = master.recv_match(type="LOG_ENTRY", blocking=True, timeout=0.5)
        if m is None:
            continue
        if m.num_logs == 0:
            return {}
        total = m.num_logs
        entries[m.id] = (m.size, m.time_utc)
        if len(entries) >= total:
            break
        deadline = time.time() + 2.0        # keep listening while entries arrive
    return entries


def log_id_for(ftp, master, remote, entries, size_hint=None):
    """The log-protocol id of ``remote`` (a full /fs/microsd/log/... path)."""
    fd, tmp = tempfile.mkstemp(suffix="_logdata.txt")
    os.close(fd)
    os.unlink(tmp)                          # download() reports success by existence
    try:
        if download(ftp, master, LOG_LIST_FILE, tmp):
            with open(tmp) as f:
                for i, line in enumerate(f):
                    parts = line.split()
                    if len(parts) >= 3 and parts[2] == remote and i in entries:
                        return i
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    # No list file (or no path match): fall back to a unique size match.
    if size_hint is not None:
        ids = [i for i, (size, _) in entries.items() if size == size_hint]
        if len(ids) == 1:
            return ids[0]
    return None


def download_logproto(master, log_id, size, local, log=print, stall_s=10.0):
    """Fetch log ``log_id`` (``size`` bytes, from LOG_ENTRY) into ``local``.

    Same shape as QGC: request one LOG_CHUNK at a time, track which 90-byte bins
    have arrived, and re-request the missing span of the chunk after a 0.5 s lull.
    Writes to ``local + '.part'`` and renames only once every bin is in."""
    nbins = (size + LOG_DATA_LEN - 1) // LOG_DATA_LEN
    have = bytearray(nbins)
    part = local + ".part"
    t0 = last_data = last_report = time.time()
    got = 0
    with open(part, "wb") as fh:
        fh.truncate(size)
        for c0 in range(0, nbins, LOG_CHUNK // LOG_DATA_LEN):
            c1 = min(c0 + LOG_CHUNK // LOG_DATA_LEN, nbins)
            while True:
                missing = [b for b in range(c0, c1) if not have[b]]
                if not missing:
                    break
                ofs = missing[0] * LOG_DATA_LEN
                end = min((missing[-1] + 1) * LOG_DATA_LEN, size)
                master.mav.log_request_data_send(master.target_system,
                                                 master.target_component,
                                                 log_id, ofs, end - ofs)
                lull = time.time() + 0.5
                while time.time() < lull:
                    m = master.recv_match(type="LOG_DATA", blocking=True, timeout=0.2)
                    now = time.time()
                    if m is None or m.id != log_id or m.count == 0:
                        if now - last_data > stall_s:
                            log(f"stalled: no LOG_DATA for {stall_s:.0f} s at "
                                f"{got}/{size} B")
                            master.mav.log_request_end_send(master.target_system,
                                                            master.target_component)
                            return False
                        continue
                    b = m.ofs // LOG_DATA_LEN
                    if b < nbins and not have[b]:
                        fh.seek(m.ofs)
                        fh.write(bytes(m.data[:m.count]))
                        have[b] = 1
                        got += m.count
                    last_data = now
                    lull = now + 0.5
                    if all(have[c0:c1]):
                        break
                    if now - last_report >= 5.0:
                        log(f"{got / 1e6:6.1f} / {size / 1e6:.1f} MB  "
                            f"{got / 1024 / (now - t0):5.0f} KiB/s")
                        last_report = now
    master.mav.log_request_end_send(master.target_system, master.target_component)
    os.replace(part, local)
    return True


# ---- Reusable helpers (shared with mavshell's `log` command) -----------------
# These take an already-open ``master`` MAVLink connection and build the MAVFTP
# on top of it, so the mavTerminal shell can list/pull logs over the SAME serial
# port it already owns — no second port-owner (which would collide on the one FC
# link).  ``log`` is a print-like callback so callers can route output.

def make_ftp(master):
    ftp = MAVFTP(master, master.target_system, master.target_component)
    # pymavlink hard-codes /tmp/temp_mavftp_file, which does not exist on Windows.
    ftp.temp_filename = os.path.join(tempfile.gettempdir(), "temp_mavftp_file")
    return ftp


def natural_key(name):
    """Sort key treating digit runs as numbers, so sess99 < sess100 < sess101
    (plain string sort puts sess100 BEFORE sess99, mis-picking the "newest"
    session/log at every digit-count rollover)."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", name)]


def list_sessions(ftp):
    """Return session dir entries under LOG_DIR, oldest→newest."""
    sessions = [e for e in listdir(ftp, LOG_DIR) if e.is_dir]
    sessions.sort(key=lambda e: natural_key(e.name))
    return sessions


def list_logs(ftp, session):
    """Return the .ulg file entries in one session dir, oldest→newest."""
    logs = [e for e in listdir(ftp, f"{LOG_DIR}/{session}")
            if not e.is_dir and e.name.endswith(".ulg")]
    logs.sort(key=lambda e: natural_key(e.name))
    return logs


def pull(ftp, master, outdir=".", session=None, name=None, log=print,
         dest_name=None, method="ftp"):
    """Resolve (session, name) — defaulting to the newest — and download it.

    Returns the local path on success, or None. Mirrors main()'s resolution so
    both the CLI and the shell's `log pull` behave identically.
    ``dest_name`` overrides the local filename (default ``<sess>_<name>``).
    ``method`` is "ftp" (MAVFTP burst, the default) or "log" (MAVLink log
    protocol, QGC's way).  FTP lists the card either way."""
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
    local = os.path.join(outdir, dest_name or f"{sess}_{target.name}")
    log(f"downloading {remote} ({target.size_b} B) -> {local} "
        f"[{'MAVFTP' if method == 'ftp' else 'log protocol'}] ...")
    t0 = time.time()
    if method == "ftp":
        ok = download(ftp, master, remote, local)
    else:
        entries = log_entries(master)
        if not entries:
            log("FC did not answer LOG_REQUEST_LIST")
            return None
        log_id = log_id_for(ftp, master, remote, entries, target.size_b)
        if log_id is None:
            log(f"could not map {remote} to a log id")
            return None
        # The size comes from the FC's list, taken just now: for a log that is
        # still open it can be a little past the FTP listing's.
        ok = download_logproto(master, log_id, entries[log_id][0], local, log=log)
    if not ok:
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
    ap.add_argument("-p", "--port", default=os.environ.get("MAV_PORT") or autodetect_port(),
                    help="serial port, or 'companion' to go through the RB5's "
                         "mavlink-routerd over adb (FC TELEM1, no FC USB)")
    ap.add_argument("-b", "--baud", type=int, default=int(os.environ.get("MAV_BAUD", "921600")))
    ap.add_argument("-o", "--outdir", default=".", help="where to save (default: cwd)")
    ap.add_argument("--list", action="store_true", help="list sessions/logs and exit")
    ap.add_argument("--session", help="session dir to pull from (default: newest)")
    ap.add_argument("--name", help="log file to pull (default: newest .ulg in the session)")
    ap.add_argument("--delete-all", action="store_true",
                    help="DELETE every log off the card (dry-run unless --yes is given)")
    ap.add_argument("--yes", action="store_true",
                    help="confirm a destructive --delete-all (skip the dry run)")
    ap.add_argument("--log", action="store_true",
                    help="download with the MAVLink log protocol (QGC's) instead "
                         "of MAVFTP")
    args = ap.parse_args()

    if not args.port:
        print("no serial port found -- pass -p or attach the FC", file=sys.stderr)
        return 2

    args.port = resolve_port(args.port)
    print(f"connecting {args.port} @ {args.baud} ...", flush=True)
    master = mavutil.mavlink_connection(args.port, baud=args.baud)
    # PX4's USB link stays silent until it hears a GCS heartbeat (same wake-up
    # as mavshell), so send ours while waiting for theirs.
    hb = None
    for _ in range(30):
        master.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                  mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        hb = master.wait_heartbeat(timeout=0.5)
        if hb is not None:
            break
    if hb is None:
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
                 log=lambda m: print(m, flush=True),
                 method="log" if args.log else "ftp")
    if local is None:
        return 1
    print(f"analyse it:  ulog_diag.py {local}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
