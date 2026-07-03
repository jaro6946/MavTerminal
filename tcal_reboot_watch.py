#!/usr/bin/env python3
"""Set cal flags, reboot FC, and immediately reconnect to catch boot messages."""
import os, sys, time
os.environ["MAVLINK20"] = "1"
from pymavlink import mavutil

PORT = "COM6"
BAUD = 57600

def connect():
    mav = mavutil.mavlink_connection(PORT, baud=BAUD)
    for _ in range(5):
        mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        time.sleep(0.5)
    if not mav.wait_heartbeat(timeout=15):
        return None
    return mav

def get_param(mav, name):
    mav.mav.param_request_read_send(mav.target_system, mav.target_component, name.encode(), -1)
    deadline = time.time() + 3
    while time.time() < deadline:
        msg = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if msg and msg.param_id.rstrip('\x00') == name:
            return msg.param_value, msg.param_type
    return None, None

def set_param(mav, name, value):
    # Read current type first so we match it exactly
    _, ptype = get_param(mav, name)
    if ptype is None:
        ptype = mavutil.mavlink.MAV_PARAM_TYPE_INT32
    mav.mav.param_set_send(
        mav.target_system, mav.target_component,
        name.encode(), float(value), ptype
    )
    deadline = time.time() + 3
    while time.time() < deadline:
        msg = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if msg and msg.param_id.rstrip('\x00') == name:
            return msg.param_value
    return None

# Initial connection
print("Connecting...")
mav = connect()
if not mav:
    print("No heartbeat."); sys.exit(1)
print(f"Connected. System {mav.target_system}\n")

# Set cal flags and verify they stuck
print("Setting SYS_CAL_ACCEL=1 and SYS_CAL_GYRO=1...")
v1 = set_param(mav, "SYS_CAL_ACCEL", 1)
v2 = set_param(mav, "SYS_CAL_GYRO", 1)
print(f"  SYS_CAL_ACCEL confirmed = {v1}")
print(f"  SYS_CAL_GYRO  confirmed = {v2}")
if v1 != 1.0 or v2 != 1.0:
    print("ERROR: flags did not set correctly. Aborting."); sys.exit(1)
time.sleep(1)  # give FC time to flush params to storage
print("Flags set and verified. Rebooting...")

mav.reboot_autopilot()
mav.close()
time.sleep(1)

# Wait for COM6 to reappear after reboot
print("Waiting for COM6 to reappear...")
import serial.tools.list_ports
deadline_port = time.time() + 30
while time.time() < deadline_port:
    ports = [p.device for p in serial.tools.list_ports.comports()]
    if PORT in ports:
        print(f"COM6 back. Reconnecting...")
        break
    time.sleep(0.5)
else:
    print("COM6 did not reappear within 30s."); sys.exit(1)

time.sleep(1)
# Reconnect immediately and catch boot messages
print("Reconnecting to catch boot messages...\n")
mav2 = mavutil.mavlink_connection(PORT, baud=BAUD)
for _ in range(10):
    mav2.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
    time.sleep(0.5)
hb = mav2.wait_heartbeat(timeout=20)
if not hb:
    print("No heartbeat after reboot."); sys.exit(1)
print(f"Reconnected. Listening for 30 seconds...\n")

deadline = time.time() + 30
while time.time() < deadline:
    msg = mav2.recv_match(blocking=True, timeout=0.5)
    if not msg or msg.get_type() == "BAD_DATA":
        continue
    if msg.get_type() == "STATUSTEXT":
        print(f"  [FC] {msg.text.strip()}")
    elif msg.get_type() == "HIGHRES_IMU":
        remaining = int(deadline - time.time())
        print(f"  [TEMP] {msg.temperature:.1f}°C  ({remaining}s remaining)", end="\r")

print("\n\nDone.")
