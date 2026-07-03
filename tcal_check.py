#!/usr/bin/env python3
"""Check if thermal calibration is running."""
import sys
import time
from pymavlink import mavutil

PORT = "COM6"
BAUD = 57600

print(f"Connecting to {PORT} @ {BAUD}...")
mav = mavutil.mavlink_connection(PORT, baud=BAUD)
hb = mav.wait_heartbeat(timeout=10)
if hb is None:
    print("ERROR: No heartbeat.")
    sys.exit(1)
print(f"Connected. Listening for 8 seconds...\n")

def get_param(name):
    mav.mav.param_request_read_send(
        mav.target_system, mav.target_component,
        name.encode("utf-8"), -1
    )
    deadline = time.time() + 3
    while time.time() < deadline:
        msg = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if msg and msg.param_id.rstrip('\x00') == name:
            return msg.param_value
    return None

# Check SYS_CAL flags
print("=== Calibration Flags ===")
for p in ["SYS_CAL_ACCEL", "SYS_CAL_GYRO", "SYS_CAL_BARO"]:
    val = get_param(p)
    status = "SCHEDULED" if val and val > 0 else "off"
    print(f"  {p:<20} = {val}  ({status})")

# Current temps
print("\n=== Current Temperatures ===")
msgs = {}
deadline = time.time() + 4
while time.time() < deadline:
    msg = mav.recv_match(blocking=True, timeout=0.5)
    if not msg or msg.get_type() == "BAD_DATA":
        continue
    msgs[msg.get_type()] = msg

imu = msgs.get("HIGHRES_IMU")
if imu:
    print(f"  HIGHRES_IMU  : {imu.temperature:.1f} °C")
for label, key in [("IMU1","SCALED_IMU"),("IMU2","SCALED_IMU2"),("IMU3","SCALED_IMU3")]:
    m = msgs.get(key)
    if m and hasattr(m, 'temperature') and m.temperature != 0:
        print(f"  {label}         : {m.temperature/100:.1f} °C")

# STATUSTEXT
print("\n=== FC Status Messages ===")
found = [t for t in msgs.values() if t.get_type() == "STATUSTEXT"]
status_texts = []
deadline = time.time() + 4
while time.time() < deadline:
    msg = mav.recv_match(type="STATUSTEXT", blocking=True, timeout=0.5)
    if msg:
        print(f"  [FC] {msg.text.strip()}")
        status_texts.append(msg.text.strip())

if not status_texts:
    print("  (no STATUSTEXT — calibration does not appear to be running)")
