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

if not PORT:
    sys.exit("No flight-controller serial port found. Plug in the FC or pass --port <PORT>.")

# Decide the mode up front so prints/prompts adapt. Batch mode is triggered by
# -c commands, or by piped (non-TTY) stdin — either way we run and exit.
batch_cmds = list(args.cmd)
if not sys.stdin.isatty() and not batch_cmds:
    batch_cmds = [ln.strip() for ln in sys.stdin if ln.strip()]
INTERACTIVE = not batch_cmds

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

def recv_loop():
    while True:
        try:
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
                # Only redraw the "> " prompt when a human is watching; in batch
                # mode it would just litter the captured output.
                prompt = "> " if INTERACTIVE else ""
                if mtype == "STATUSTEXT" and statustext_on:
                    print(f"\n[FC] {msg.text.strip()}\n{prompt}", end="", flush=True)
                elif streaming:
                    print(f"\r[{mtype}] {msg}\n{prompt}", end="", flush=True)
        except Exception:
            break

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

def run_command(raw):
    """Execute a single command line. Returns True if the user asked to quit.

    Shared by the interactive REPL and batch mode so both paths behave
    identically.
    """
    global streaming, statustext_on
    raw = raw.strip()
    if not raw:
        return False
    parts = raw.split()
    cmd, args = parts[0].lower(), parts[1:]

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
