#!/usr/bin/env python3
"""Simple MAVLink terminal shell."""
import os
import sys
import threading
import time
os.environ["MAVLINK20"] = "1"
from pymavlink import mavutil

PORT = "COM6"
BAUD = 57600

print(f"Connecting to {PORT} @ {BAUD}...")
mav = mavutil.mavlink_connection(PORT, baud=BAUD)

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
hb = mav.wait_heartbeat(timeout=10)
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
print("Type 'help' for commands. Ctrl+C to exit.\n")

streaming = False
statustext_on = True  # always print STATUSTEXT by default
last_msgs = {}

def recv_loop():
    while True:
        try:
            msg = mav.recv_match(blocking=True, timeout=1)
            if msg and msg.get_type() != "BAD_DATA":
                mtype = msg.get_type()
                last_msgs[mtype] = msg
                if mtype == "STATUSTEXT" and statustext_on:
                    print(f"\n[FC] {msg.text.strip()}\n> ", end="", flush=True)
                elif streaming:
                    print(f"\r[{mtype}] {msg}\n> ", end="", flush=True)
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
    mav.mav.param_request_read_send(
        mav.target_system, mav.target_component,
        name.encode("utf-8"), -1
    )
    deadline = time.time() + 3
    while time.time() < deadline:
        msg = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
        if msg and msg.param_id.rstrip('\x00') == name:
            print(f"  {name} = {msg.param_value}")
            return
    print(f"  No response for '{name}' (timeout).")

def cmd_param_set(name, value):
    mav.mav.param_set_send(
        mav.target_system, mav.target_component,
        name.encode("utf-8"), float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )
    deadline = time.time() + 3
    while time.time() < deadline:
        msg = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
        if msg and msg.param_id.rstrip('\x00') == name:
            print(f"  {name} set to {msg.param_value}")
            return
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

while True:
    try:
        raw = input("> ").strip()
        if not raw:
            continue
        parts = raw.split()
        cmd, args = parts[0].lower(), parts[1:]

        if cmd in ("quit", "exit", "q"):
            break
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
    except (KeyboardInterrupt, EOFError):
        break

print("\nDisconnected.")
