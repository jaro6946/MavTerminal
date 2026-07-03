#!/usr/bin/env python3
"""Check whether thermal calibration ran and if it's currently in progress."""
import os, sys, time
os.environ["MAVLINK20"] = "1"
from pymavlink import mavutil

PORT = "COM6"
BAUD = 57600

mav = mavutil.mavlink_connection(PORT, baud=BAUD)
for _ in range(5):
    mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
    time.sleep(0.5)
if not mav.wait_heartbeat(timeout=10):
    print("No heartbeat."); sys.exit(1)
print(f"Connected.\n")

def get_param(name):
    mav.mav.param_request_read_send(mav.target_system, mav.target_component, name.encode(), -1)
    deadline = time.time() + 3
    while time.time() < deadline:
        msg = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if msg and msg.param_id.rstrip('\x00') == name:
            return msg.param_value
    return None

def int_val(v):
    if v is None: return None
    import struct
    return struct.unpack('<i', struct.pack('<f', v))[0]

# Check TC coefficient params — non-zero means calibration data exists
print("=== TC Coefficient Params (non-zero = cal data exists) ===")
TC_COEFF_PARAMS = [
    "TC_A0_ID", "TC_A0_TMIN", "TC_A0_TMAX", "TC_A0_TREF",
    "TC_A0_X0_0", "TC_A0_X0_1", "TC_A0_X0_2",
    "TC_G0_ID", "TC_G0_TMIN", "TC_G0_TMAX", "TC_G0_TREF",
    "TC_G0_X0_0", "TC_G0_X0_1", "TC_G0_X0_2",
    "TC_B0_ID", "TC_B0_X0", "TC_B0_X1", "TC_B0_X2",
]
any_set = False
for p in TC_COEFF_PARAMS:
    val = get_param(p)
    if val is not None and val != 0.0:
        print(f"  {p:<25} = {val}")
        any_set = True
if not any_set:
    print("  (all zero — no calibration data stored yet)")

# Listen for STATUSTEXT for 8 seconds to catch any cal-in-progress messages
print("\n=== Listening for FC status messages (8s) ===")
print("(calibration in progress would show messages here)\n")
deadline = time.time() + 8
found = False
while time.time() < deadline:
    msg = mav.recv_match(type="STATUSTEXT", blocking=True, timeout=0.5)
    if msg:
        print(f"  [FC] {msg.text.strip()}")
        found = True
if not found:
    print("  (no STATUSTEXT — calibration is not actively running)")

# Current temps
print("\n=== Current Temperatures ===")
last = {}
deadline = time.time() + 3
while time.time() < deadline:
    msg = mav.recv_match(blocking=True, timeout=0.5)
    if msg and msg.get_type() != "BAD_DATA":
        last[msg.get_type()] = msg
imu = last.get("HIGHRES_IMU")
if imu: print(f"  Board        : {imu.temperature:.1f} °C  (TMIN=50, need to reach {50+int_val(get_param('SYS_CAL_TDEL'))}°C)")
for label, key in [("IMU1","SCALED_IMU"),("IMU2","SCALED_IMU2"),("IMU3","SCALED_IMU3")]:
    m = last.get(key)
    if m and hasattr(m, 'temperature'):
        print(f"  {label}         : {m.temperature/100:.1f} °C")
