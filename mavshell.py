#!/usr/bin/env python3
"""Simple MAVLink terminal shell.

Two ways to run it:
  * Interactive REPL (default when run from a terminal with no commands).
  * Non-interactive batch mode (for scripts / automation): pass commands with
    -c/--cmd, or pipe them on stdin. The tool connects, runs the commands,
    prints the results, and exits — so it never blocks waiting on a keyboard.

Examples:
    mavshell.py                                 # interactive REPL
    mavshell.py -c temp                          # run one command and exit
    mavshell.py -c "param get SYS_AUTOSTART" -c temp
    printf 'temp\\nshow ATTITUDE\\n' | mavshell.py
"""
import argparse
import os
import shlex
import struct
import sys
import threading
import time
os.environ["MAVLINK20"] = "1"
from pymavlink import mavutil
from serial.tools import list_ports

# --- Flight-controller serial-port auto-detection (OS-agnostic) ------------
# We identify the FC by its USB vendor ID (most reliable), then fall back to
# description/manufacturer keywords, then to any CDC-ACM / COM device. This
# works identically on Windows (COM*) and Linux/WSL (/dev/ttyACM*), so the tool
# needs no per-machine port configuration.
FC_VIDS = {
    0x26AC,  # mRo / PX4 / 3DR  (this board: mRo ControlZero H7, 0x26AC:0x1024)
    0x0483,  # STMicroelectronics Virtual COM Port (Pixhawk/PX4 FMU)
    0x1209,  # pid.codes / PX4
    0x2DAE,  # Cube / Hex
    0x3162,  # Holybro
}
FC_KEYWORDS = ("px4", "pixhawk", "ardupilot", "fmu", "mro", "cube",
               "holybro", "mavlink", "control zero")


# PX4 integer param types.  PX4 (like QGC) transmits/receives integer parameters
# by byte-reinterpreting the int into the float `param_value` field rather than
# value-casting it, so both decode (read) and encode (set) key off this set.
PARAM_INT_TYPES = {
    mavutil.mavlink.MAV_PARAM_TYPE_UINT8,  mavutil.mavlink.MAV_PARAM_TYPE_INT8,
    mavutil.mavlink.MAV_PARAM_TYPE_UINT16, mavutil.mavlink.MAV_PARAM_TYPE_INT16,
    mavutil.mavlink.MAV_PARAM_TYPE_UINT32, mavutil.mavlink.MAV_PARAM_TYPE_INT32,
}


# --- Raw NSH (NuttShell) console over MAVLink --------------------------------
# PX4 exposes its onboard NuttShell over MAVLink SERIAL_CONTROL (#126) with
# device == SHELL: bytes we put in `data` are fed to nsh's stdin, and PX4 streams
# the console output back in SERIAL_CONTROL messages of the same device. This is
# exactly how QGroundControl's "MAVLink Console" works. getattr fallbacks keep it
# working on a dialect that happens to omit a constant.
SERIAL_CONTROL_DEV_SHELL = getattr(mavutil.mavlink, "SERIAL_CONTROL_DEV_SHELL", 10)
# RESPOND: ask PX4 to send the output back.  EXCLUSIVE: take the shell for this
# GCS.  MULTI: allow >1 SERIAL_CONTROL reply per request (long output).
NSH_FLAGS = (getattr(mavutil.mavlink, "SERIAL_CONTROL_FLAG_RESPOND", 2)
             | getattr(mavutil.mavlink, "SERIAL_CONTROL_FLAG_EXCLUSIVE", 4)
             | getattr(mavutil.mavlink, "SERIAL_CONTROL_FLAG_MULTI", 16))
NSH_DATA_LEN = 70   # SERIAL_CONTROL.data is a fixed 70-byte field


def decode_param(msg):
    """Return a PARAM_VALUE's real value.

    PX4 (like QGC) transmits integer parameters by byte-reinterpreting the int
    into the float `param_value` field, rather than value-casting it. So an
    INT32 of 1 arrives as the float 1.4e-45 (the float with bit pattern 0x1).
    Reinterpret the 4 bytes back to an int for the integer param types; leave
    real (float) params as-is.
    """
    if getattr(msg, "param_type", None) in PARAM_INT_TYPES:
        return struct.unpack("<i", struct.pack("<f", msg.param_value))[0]
    return msg.param_value


def autodetect_port():
    """Return the FC serial port, OS-agnostic (Windows COM* / Linux /dev/tty*)."""
    ports = list(list_ports.comports())
    for p in ports:                                   # 1) known FC vendor ID (most reliable)
        if p.vid in FC_VIDS:
            return p.device
    for p in ports:                                   # 2) description/manufacturer keyword
        text = " ".join(filter(None, [p.description, p.manufacturer, p.product])).lower()
        if any(k in text for k in FC_KEYWORDS):
            return p.device
    for p in ports:                                   # 3) platform fallback: first CDC-ACM/COM
        if p.device.startswith(("/dev/ttyACM", "/dev/ttyUSB")) or p.device.upper().startswith("COM"):
            return p.device
    return ports[0].device if ports else None         # 4) last resort: first port, or None


# --- Arguments -------------------------------------------------------------
ap = argparse.ArgumentParser(description="MAVLink terminal / batch tool")
ap.add_argument("-p", "--port", default=os.environ.get("MAV_PORT") or autodetect_port(),
                help="serial port (default: $MAV_PORT or auto-detected FC)")
ap.add_argument("-b", "--baud", type=int, default=int(os.environ.get("MAV_BAUD", "57600")),
                help="baud rate (default: $MAV_BAUD or 57600)")
ap.add_argument("-c", "--cmd", action="append", default=[],
                help="run a command non-interactively (repeatable); implies batch mode")
ap.add_argument("-t", "--heartbeat-timeout", type=int, default=10,
                help="seconds to wait for the first heartbeat before giving up")
ap.add_argument("--settle", type=float, default=2.0,
                help="seconds to let telemetry streams populate before running batch commands")
args = ap.parse_args()
PORT, BAUD = args.port, args.baud

# Decide the mode up front so prints/prompts adapt. Batch mode is triggered by
# -c commands, or by piped (non-TTY) stdin — either way we run and exit.
batch_cmds = list(args.cmd)
if not sys.stdin.isatty() and not batch_cmds:
    batch_cmds = [ln.strip() for ln in sys.stdin if ln.strip()]
INTERACTIVE = not batch_cmds


def _is_offline_cmd(line):
    """True if ``line`` only reads a local .ulg and never touches the FC.

    ``log diag`` / ``log graph`` analyze a file already on disk, so requiring a
    plugged-in flight controller just to look at one would be silly. Anything
    not on this list is assumed to need the link."""
    parts = line.split()
    if not parts:
        return False
    verb = parts[0].lower()
    if verb in ("loggraph", "graph", "help"):
        return True
    return verb == "log" and len(parts) > 1 and parts[1].lower() in ("graph", "diag")


# Skip the whole connect path when EVERY batch command is offline analysis.
# Interactive sessions always connect — you can't know in advance what will be
# typed, and a REPL is only useful with a live link anyway.
OFFLINE = bool(batch_cmds) and all(_is_offline_cmd(c) for c in batch_cmds)

if not OFFLINE:
    if not PORT:
        sys.exit("No flight-controller serial port found. "
                 "Plug in the FC or pass --port <PORT>.")

    print(f"Connecting to {PORT} @ {BAUD}...")
    try:
        mav = mavutil.mavlink_connection(PORT, baud=BAUD)
    except Exception as e:
        sys.exit(f"Could not open port '{PORT}': {e}\n"
                 "Check the FC is plugged in / passed through to WSL, or pass --port <PORT>.")

    # Send heartbeats to wake the FC before waiting for its response
    print("Sending heartbeats to wake FC...")
    for _ in range(5):
        mav.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0
        )
        time.sleep(0.5)

    print("Waiting for heartbeat...")
    hb = mav.wait_heartbeat(timeout=args.heartbeat_timeout)
    if hb is None:
        print("No heartbeat received. Check connection and power.")
        sys.exit(1)

    print(f"Connected! System {mav.target_system}, Component {mav.target_component}")

    # Request all data streams so we don't need QGC to prime the FC
    mav.mav.request_data_stream_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL,
        10,  # 10 Hz
        1    # start
    )
    print("Data streams requested.")
if INTERACTIVE:
    print("Type 'help' for commands. Ctrl+C to exit.\n")

streaming = False
statustext_on = True  # always print STATUSTEXT by default
last_msgs = {}
params = {}       # name -> latest decoded value, populated by the recv thread
param_types = {}  # name -> MAVLink param_type (so `param set` can encode correctly)
recv_paused = False  # when True the recv thread stops reading so a MAVFTP log
                     # transfer can be the SOLE reader of the one FC serial port
# Raw-NSH session state (mutated in place so recv_loop needs no `global`):
#   echo    -- while True, recv_loop writes SERIAL_CONTROL shell bytes straight
#              to stdout (this is the live console output).
#   last_rx -- wall-clock of the most recent shell byte, so a one-shot command
#              can tell when PX4 has gone quiet and stop draining.
nsh_state = {'echo': False, 'last_rx': 0.0}

def recv_loop():
    while True:
        try:
            if recv_paused:                # a `log` FTP transfer owns the port now
                time.sleep(0.05)
                continue
            msg = mav.recv_match(blocking=True, timeout=1)
            if msg and msg.get_type() != "BAD_DATA":
                mtype = msg.get_type()
                last_msgs[mtype] = msg
                # PARAM_VALUE shares one message type across all params, so index
                # it by name here — param get/set poll this dict instead of doing
                # their own recv_match (two readers on one serial port collide).
                if mtype == "PARAM_VALUE":
                    pname = msg.param_id.rstrip('\x00')
                    params[pname] = decode_param(msg)
                    param_types[pname] = getattr(msg, "param_type", None)
                # Raw-NSH console output: PX4 streams shell bytes back as
                # SERIAL_CONTROL.  While an nsh session is active, write them
                # verbatim to stdout (this IS the console) and remember when the
                # last byte landed so a one-shot command knows when it's done.
                elif mtype == "SERIAL_CONTROL" and nsh_state['echo']:
                    n = min(int(getattr(msg, "count", 0)), NSH_DATA_LEN)
                    if n:
                        raw = bytes(bytearray(msg.data[:n]))
                        sys.stdout.write(raw.decode("utf-8", "replace"))
                        sys.stdout.flush()
                        nsh_state['last_rx'] = time.time()
                    continue  # never echo the prompt into the console stream
                # Only redraw the "> " prompt when a human is watching; in batch
                # mode (or mid-NSH session) it would just litter the output.
                prompt = "" if (nsh_state['echo'] or not INTERACTIVE) else "> "
                if mtype == "STATUSTEXT" and statustext_on:
                    print(f"\n[FC] {msg.text.strip()}\n{prompt}", end="", flush=True)
                elif streaming:
                    print(f"\r[{mtype}] {msg}\n{prompt}", end="", flush=True)
        except Exception:
            break

if not OFFLINE:                 # no link open, so there is nothing to read
    t = threading.Thread(target=recv_loop, daemon=True)
    t.start()

def cmd_help():
    print("""Commands:
  stream on/off           toggle live message stream
  statustext on/off       toggle FC status messages (on by default)
  show <TYPE>             print latest message of given type (e.g. show ATTITUDE)
  list                    list received message types
  temp                    show current sensor temperatures
  param get <NAME>        read a parameter
  param set <NAME> <VAL>  set a parameter
  log list [session]      list onboard .ulg flight logs (newest marked)
  log pull [sess] [name]  download newest (or named) .ulg and diagnose it
  log diag <file.ulg>     diagnose an already-downloaded log
  log graph [file.ulg]    open the log browser: pick a log, scroll its plots
                          (thermal/GPS + altitude estimation), rename logs,
                          export a PDF.  Also as: logGraph [file.ulg]
                            --list            just list the channels
                            --classic         one window, thermal plot only
                            --save out.png    headless PNG of the thermal plot
                            --pdf out.pdf     headless report (several logs ok)
                          in a plot: wheel=scroll, ctrl+wheel=zoom time,
                          ctrl+shift+wheel=zoom values, drag=pan, dbl-click=reset
  log delete [yes]        DELETE all logs off the card (dry-run without 'yes')
  nsh <command>           run one NuttShell command on the FC and print output
  nsh                     open an interactive raw NSH shell (TTY only)
  tcal                    trigger thermal calibration
  heartbeat               show connection info
  arm / disarm / reboot   send commands
  quit / exit             disconnect""")

def cmd_show(args):
    if not args:
        print("Usage: show <MSG_TYPE>")
        return
    key = args[0].upper()
    if key in last_msgs:
        print(last_msgs[key])
    else:
        print(f"No message of type '{key}' received yet. Try 'list'.")

def cmd_list():
    if not last_msgs:
        print("No messages received yet.")
    else:
        print("  ".join(sorted(last_msgs.keys())))

def cmd_temp():
    # Request specific IMU messages in case they haven't arrived yet
    for msg_id, name in [
        (mavutil.mavlink.MAVLINK_MSG_ID_SCALED_IMU,  "SCALED_IMU"),
        (mavutil.mavlink.MAVLINK_MSG_ID_SCALED_IMU2, "SCALED_IMU2"),
        (mavutil.mavlink.MAVLINK_MSG_ID_SCALED_IMU3, "SCALED_IMU3"),
        (mavutil.mavlink.MAVLINK_MSG_ID_HIGHRES_IMU, "HIGHRES_IMU"),
    ]:
        if name not in last_msgs:
            mav.mav.message_interval_send(msg_id, 100000)  # 10 Hz

    # Wait briefly for any missing messages
    deadline = time.time() + 2
    needed = {"SCALED_IMU", "SCALED_IMU2", "SCALED_IMU3", "HIGHRES_IMU"}
    while time.time() < deadline and not needed.issubset(last_msgs.keys()):
        time.sleep(0.1)

    imu = last_msgs.get("HIGHRES_IMU")
    sc1 = last_msgs.get("SCALED_IMU")
    sc2 = last_msgs.get("SCALED_IMU2")
    sc3 = last_msgs.get("SCALED_IMU3")
    if imu:
        print(f"  Board (HIGHRES_IMU)     : {imu.temperature:.1f} °C")
    if sc1 and hasattr(sc1, 'temperature'):
        print(f"  IMU1 accel+gyro         : {sc1.temperature / 100:.1f} °C")
    if sc2 and hasattr(sc2, 'temperature'):
        print(f"  IMU2 accel+gyro         : {sc2.temperature / 100:.1f} °C")
    if sc3 and hasattr(sc3, 'temperature'):
        print(f"  IMU3 accel+gyro         : {sc3.temperature / 100:.1f} °C")
    if not any([imu, sc1, sc2, sc3]):
        print("No IMU messages received yet.")

def cmd_param_get(name):
    params.pop(name, None)  # force a fresh value from the FC
    mav.mav.param_request_read_send(
        mav.target_system, mav.target_component,
        name.encode("utf-8"), -1
    )
    deadline = time.time() + 3
    while time.time() < deadline:
        if name in params:                 # filled in by recv_loop (single reader)
            print(f"  {name} = {params[name]}")
            return
        time.sleep(0.05)
    print(f"  No response for '{name}' (timeout).")

def _learn_param_type(name):
    """Return the FC's registered param_type for ``name`` (reading it if we don't
    already know it), or None if the FC doesn't answer.  We need the type to set
    the value with the right encoding — PX4 checks it and stores int params
    bytewise."""
    if name in param_types:
        return param_types[name]
    mav.mav.param_request_read_send(
        mav.target_system, mav.target_component, name.encode("utf-8"), -1)
    deadline = time.time() + 2
    while time.time() < deadline:
        if name in param_types:
            return param_types[name]
        time.sleep(0.05)
    return None

def cmd_param_set(name, value):
    # PX4 stores int params BYTEWISE in the float field and validates the sent
    # param_type against the param's registered type, so an int param MUST be sent
    # as its int type with the bits packed into the float slot; sending REAL32
    # would write garbage.  Learn the type first, then encode accordingly.
    ptype = _learn_param_type(name)
    if ptype is None:
        print(f"  Could not read type of '{name}' (no PARAM_VALUE) — not setting.")
        return
    if ptype in PARAM_INT_TYPES:
        ival = int(round(float(value)))
        pv = struct.unpack("<f", struct.pack("<i", ival))[0]  # int bits -> float slot
    else:
        pv = float(value)

    params.pop(name, None)
    mav.mav.param_set_send(
        mav.target_system, mav.target_component,
        name.encode("utf-8"), pv, ptype,
    )
    deadline = time.time() + 3
    while time.time() < deadline:
        if name in params:                 # FC echoes a PARAM_VALUE on accept (decoded)
            print(f"  {name} set to {params[name]}")
            return
        time.sleep(0.05)
    print(f"  No confirmation for '{name}' (timeout).")

def cmd_tcal():
    print("Sending thermal calibration command...")
    # MAV_CMD_PREFLIGHT_CALIBRATION: param5=3 triggers thermal cal on PX4
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_PREFLIGHT_CALIBRATION,
        0,   # confirmation
        0, 0, 0, 0, 3, 0, 0  # param5=3 = thermal cal
    )
    print("Command sent. Watch for [FC] messages...")

def _pause_recv():
    """Stop the background reader and wait for its in-flight recv_match to
    return, so the caller becomes the SOLE reader of the FC serial port.
    recv_match blocks up to 1 s, so we wait a hair longer than that."""
    global recv_paused
    recv_paused = True
    time.sleep(1.15)


def _resume_recv():
    global recv_paused
    recv_paused = False


def cmd_log(args):
    """Pull & diagnose the FC's onboard .ulg flight logs over the SAME serial
    link this shell already owns (MAVLink FTP). Sub-commands:

        log list                 list sessions on the SD card (newest marked)
        log list <session>       list the .ulg files in one session
        log pull                 download the NEWEST log, then diagnose it
        log pull <session>       download the newest log in <session>, diagnose
        log pull <session> <name.ulg>   download that exact log, diagnose
        log diag <file.ulg>      diagnose an already-downloaded local .ulg
        log delete               DRY-RUN: show what deleting all logs would remove
        log delete yes           DESTRUCTIVE: delete EVERY log off the SD card

    Downloads land in $MAV_LOG_DIR (default: cwd). This is the batch-drivable
    equivalent of pull_log.py + ulog_diag.py, but reusing this connection so
    there's never a second owner fighting over the one FC port."""
    # Import lazily: listing/pulling need only pymavlink (always present), and we
    # want `log` usable even if pyulog (the analyzer's dep) isn't installed.
    import pull_log
    sub = (args[0].lower() if args else "")
    outdir = os.environ.get("MAV_LOG_DIR", ".")

    if sub == "diag":
        if len(args) < 2:
            print("Usage: log diag <file.ulg>")
            return
        try:
            from ulog_diag import diagnose
        except ImportError as e:
            print(f"  analyzer needs pyulog: {e}  (pip install pyulog)")
            return
        diagnose(args[1])
        return

    if sub == "graph":
        # Purely offline, like `diag` — handled BEFORE the MAVFTP block below,
        # which is the only part of `log` that needs the serial port.
        try:
            import ulog_graph
        except ImportError as e:
            print(f"  grapher needs pyulog + matplotlib: {e}")
            return

        rest = args[1:]
        # A bare `log graph` is no longer an error: it opens the browser, which
        # has its own log picker.  Everything that is not a flag is a log path,
        # so several can be given (a multi-log --pdf report is the point).
        flags = {"--list", "--abs", "--no-show", "--classic"}
        takes_value = {"--smooth", "--rate-src", "--add", "--save", "--pdf"}
        paths, opts, skip = [], [], False
        for i, a in enumerate(rest):
            if skip:
                skip = False
                continue
            if a in takes_value:
                opts.append(a)
                if i + 1 < len(rest):
                    opts.append(rest[i + 1])
                    skip = True
            elif a in flags or a.startswith("--"):
                opts.append(a)
            else:
                paths.append(a)

        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            for p in missing:
                print(f"  no such file: {p}")
            return
        if "--list" in opts:
            if not paths:
                print("  --list needs a log file")
                return
            ulog_graph.list_channels(paths[0])
            return

        def _opt(name, cast=str, default=None):
            return cast(opts[opts.index(name) + 1]) if name in opts else default

        try:
            ulog_graph.graph(
                paths[0] if paths else None,
                smooth=_opt("--smooth", float, 31.0),
                use_abs="--abs" in opts,
                rate_src=_opt("--rate-src"),
                adds=[opts[i + 1] for i, o in enumerate(opts) if o == "--add"],
                save=_opt("--save"),
                show="--no-show" not in opts,
                classic="--classic" in opts,
                pdf=_opt("--pdf"),
                extra_paths=paths[1:],
            )
        except (IndexError, ValueError) as e:
            print(f"  bad options: {e}")
        except SystemExit as e:
            # ulog_graph.graph uses sys.exit for user errors ("no display").  In
            # the REPL that would kill the whole shell mid-session.
            if e.code:
                print(f"  {e.code}")
        return

    # list / pull both drive MAVFTP, which must be the only reader of the port.
    _pause_recv()
    try:
        ftp = pull_log.make_ftp(mav)
        if sub == "" or sub == "list":
            session = args[1] if len(args) > 1 else None
            if session:
                logs = pull_log.list_logs(ftp, session)
                if not logs:
                    print(f"  no .ulg in {session}")
                for e in logs:
                    print(f"  {session}/{e.name}  {e.size_b} B")
            else:
                sessions = pull_log.list_sessions(ftp)
                if not sessions:
                    print("  no log sessions on the card")
                for s in sessions:
                    print(f"  {s.name}{'   <- newest' if s is sessions[-1] else ''}")
                print("  (log list <session> to see its files; log pull to grab the newest)")
        elif sub == "pull":
            session = args[1] if len(args) > 1 else None
            name = args[2] if len(args) > 2 else None
            local = pull_log.pull(ftp, mav, outdir, session, name,
                                  log=lambda m: print(f"  {m}"))
            if local:
                try:
                    from ulog_diag import diagnose
                    print()
                    diagnose(local)
                except ImportError as e:
                    print(f"  downloaded ok; analyzer needs pyulog: {e}")
        elif sub in ("delete", "rm", "clear"):
            # DESTRUCTIVE: wipe every log off the card. Gated so a bare
            # `log delete` only reports what WOULD go — you must add an explicit
            # `yes`/`confirm` to actually delete (safe for batch/automation).
            confirmed = len(args) > 1 and args[1].lower() in ("yes", "confirm", "force", "-y")
            n_files, n_dirs, total = pull_log.summarize_logs(ftp)
            if n_files == 0 and n_dirs == 0:
                print(f"  no logs under {pull_log.LOG_DIR} — nothing to delete")
            elif not confirmed:
                print(f"  would delete {n_files} file(s) in {n_dirs} session dir(s), "
                      f"{total/1e6:.1f} MB, under {pull_log.LOG_DIR}")
                print("  this is DESTRUCTIVE and cannot be undone.")
                print("  re-run  log delete yes  to actually delete.")
            else:
                print(f"  deleting {n_files} file(s) + {n_dirs} dir(s) under "
                      f"{pull_log.LOG_DIR} ...")
                removed, failed = pull_log.delete_all_logs(ftp, log=lambda m: print(f"  {m}"))
                print(f"  done: removed {removed}, failed {failed}")
        else:
            print("Usage: log list | log list <session> | log pull [session] [name] | "
                  "log diag <file.ulg> | log graph <file.ulg> | log delete [yes]")
    finally:
        _resume_recv()


def _nsh_send(text):
    """Feed ``text`` to the FC's NuttShell over SERIAL_CONTROL.

    The data field is a fixed 70 bytes, so anything longer is split across
    several messages (``count`` carries the valid length of each).  Always sends
    at least one message even for empty text, so a bare send can be used to POLL
    for output (e.g. to fetch the initial ``nsh> `` prompt)."""
    data = text.encode("utf-8", "replace")
    idx, first = 0, True
    while first or idx < len(data):
        first = False
        chunk = data[idx:idx + NSH_DATA_LEN]
        idx += NSH_DATA_LEN
        payload = list(chunk) + [0] * (NSH_DATA_LEN - len(chunk))
        # device, flags, timeout, baudrate, count, data  (no target fields in
        # this dialect -> PX4 treats it as broadcast and accepts it).
        mav.mav.serial_control_send(SERIAL_CONTROL_DEV_SHELL, NSH_FLAGS,
                                    0, 0, len(chunk), payload)


def _nsh_drain(quiet=0.35, cap=3.0, min_wait=0.5):
    """Block while NSH output is still streaming back, so a one-shot command
    prints its full result before we return.

    Caller sets ``nsh_state['last_rx'] = 0`` right before sending.  We then wait
    until output has arrived AND has been quiet for ``quiet`` seconds (command
    finished), or ``min_wait`` elapsed with no output at all (silent command),
    or the hard ``cap`` is hit (never hang a batch run).  recv_loop does the
    actual printing; this only watches the clock."""
    start = time.time()
    while True:
        el = time.time() - start
        if el >= cap:
            return
        last = nsh_state['last_rx']
        if last and (time.time() - last) >= quiet:
            return
        if not last and el >= min_wait:
            return
        time.sleep(0.03)


def cmd_nsh(args):
    """Raw NuttShell (NSH) console on the FC over MAVLink SERIAL_CONTROL.

        nsh <command...>   run one shell command, print its output, return
        nsh                open an interactive NSH sub-shell (TTY only)

    One-shot form is batch-drivable: ``mavTerminal -c "nsh ver all"`` /
    ``-c "nsh dmesg"`` / ``-c "nsh top once"``.  The interactive form drops you
    into the FC's live shell (PX4 echoes its own ``nsh> `` prompt); leave it with
    ``exit``, ``~.``, or Ctrl-C."""
    nsh_state['echo'] = True
    try:
        if args:                                   # one-shot command
            nsh_state['last_rx'] = 0.0
            _nsh_send(" ".join(args) + "\n")
            _nsh_drain()
            print()                                # clean line before the prompt
            return
        if not INTERACTIVE:
            print("`nsh` with no command needs an interactive terminal; "
                  "for batch use, pass the command: nsh <command>")
            return
        print("Entering NSH shell. Type 'exit', '~.', or Ctrl-C to leave.\n")
        nsh_state['last_rx'] = 0.0
        _nsh_send("\n")                            # elicit the initial prompt
        _nsh_drain(min_wait=0.6)
        while True:
            try:
                line = input("")                  # PX4 echoes its own nsh> prompt
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip() in ("exit", "quit", "~."):
                break
            nsh_state['last_rx'] = 0.0
            _nsh_send(line + "\n")
            _nsh_drain()
        print("\nLeft NSH shell.")
    finally:
        nsh_state['echo'] = False


def run_command(raw):
    """Execute a single command line. Returns True if the user asked to quit.

    Shared by the interactive REPL and batch mode so both paths behave
    identically.
    """
    global streaming, statustext_on
    raw = raw.strip()
    if not raw:
        return False
    # shlex, not str.split: log paths routinely contain spaces (e.g. a
    # "Log Analysis" folder), and a plain split would hand `log graph` half a
    # filename. Fall back to a naive split if the quoting is unbalanced so a
    # stray quote can't kill the REPL.
    try:
        parts = shlex.split(raw)
    except ValueError:
        parts = raw.split()
    if not parts:
        return False
    cmd, args = parts[0].lower(), parts[1:]

    # `logGraph <file>` is an alias for `log graph <file>` (run_command has
    # already lowercased the verb, so either capitalization works).
    if cmd in ("loggraph", "graph"):
        cmd, args = "log", ["graph"] + args

    if cmd in ("quit", "exit", "q"):
        return True
    elif cmd == "help":
        cmd_help()
    elif cmd == "stream":
        streaming = args[0].lower() == "on" if args else True
        print(f"Stream {'on' if streaming else 'off'}.")
    elif cmd == "statustext":
        statustext_on = (not args) or args[0].lower() == "on"
        print(f"STATUSTEXT {'on' if statustext_on else 'off'}.")
    elif cmd == "show":
        cmd_show(args)
    elif cmd == "list":
        cmd_list()
    elif cmd == "temp":
        cmd_temp()
    elif cmd == "param":
        if len(args) >= 2 and args[0].lower() == "get":
            cmd_param_get(args[1].upper())
        elif len(args) >= 3 and args[0].lower() == "set":
            cmd_param_set(args[1].upper(), args[2])
        else:
            print("Usage: param get <NAME>  |  param set <NAME> <VALUE>")
    elif cmd == "tcal":
        cmd_tcal()
    elif cmd == "log":
        cmd_log(args)
    elif cmd == "nsh":
        cmd_nsh(args)
    elif cmd == "heartbeat":
        print(f"System {mav.target_system}, Component {mav.target_component}")
    elif cmd == "arm":
        mav.arducopter_arm()
        print("Arm command sent.")
    elif cmd == "disarm":
        mav.arducopter_disarm()
        print("Disarm command sent.")
    elif cmd == "reboot":
        mav.reboot_autopilot()
        print("Reboot command sent.")
    else:
        print(f"Unknown command '{cmd}'. Type 'help'.")
    return False

if batch_cmds:
    # Give the just-requested data streams a moment to populate last_msgs so
    # commands like `temp` / `show` have data to report, then run and exit.
    # Offline runs have no streams to wait for, so they skip straight through.
    if not OFFLINE:
        time.sleep(args.settle)
    for c in batch_cmds:
        print(f"> {c}")
        if run_command(c):   # honor an explicit quit/exit in the batch
            break
    sys.exit(0)

# Interactive REPL
while True:
    try:
        raw = input("> ").strip()
        if not raw:
            continue
        if run_command(raw):
            break
    except (KeyboardInterrupt, EOFError):
        break

print("\nDisconnected.")
