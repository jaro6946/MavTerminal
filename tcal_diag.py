#!/usr/bin/env python3
"""Thermal calibration diagnostic — connects, collects info, reports, exits."""
import sys
import time
from pymavlink import mavutil

PORT = "COM6"
BAUD = 57600
PARAM_TIMEOUT = 3

print(f"Connecting to {PORT} @ {BAUD}...")
mav = mavutil.mavlink_connection(PORT, baud=BAUD)
hb = mav.wait_heartbeat(timeout=10)
if hb is None:
    print("ERROR: No heartbeat. Check connection.")
    sys.exit(1)
print(f"Connected. System {mav.target_system}, Component {mav.target_component}\n")

# Collect incoming messages for a few seconds
print("Collecting messages for 4 seconds...")
status_texts = []
msgs = {}
deadline = time.time() + 4
while time.time() < deadline:
    msg = mav.recv_match(blocking=True, timeout=0.5)
    if not msg or msg.get_type() == "BAD_DATA":
        continue
    msgs[msg.get_type()] = msg
    if msg.get_type() == "STATUSTEXT":
        status_texts.append(msg.text.strip())

# --- Temperatures ---
print("=== Sensor Temperatures ===")
imu = msgs.get("HIGHRES_IMU")
if imu:
    print(f"  HIGHRES_IMU  : {imu.temperature:.1f} °C")
for name, key in [("IMU1","SCALED_IMU"),("IMU2","SCALED_IMU2"),("IMU3","SCALED_IMU3")]:
    m = msgs.get(key)
    if m and hasattr(m, 'temperature') and m.temperature != 0:
        print(f"  {name}         : {m.temperature/100:.1f} °C")

# --- STATUSTEXT seen so far ---
print("\n=== FC Status Messages (last 4s) ===")
if status_texts:
    for s in status_texts:
        print(f"  {s}")
else:
    print("  (none)")

# --- Key parameters ---
TC_PARAMS = [
    "CAL_ACC_TC_EN",
    "CAL_GYRO_TC_EN",
    "SYS_CAL_ACCEL",
    "SYS_CAL_GYRO",
    "SYS_CAL_BARO",
    "SYS_CAL_TDEL",
    "SYS_CAL_TMAX",
    "SYS_CAL_TMIN",
    "SENS_BOARD_TC_T",
    "TC_A0_ID",
    "TC_G0_ID",
]

def get_param(name):
    mav.mav.param_request_read_send(
        mav.target_system, mav.target_component,
        name.encode("utf-8"), -1
    )
    deadline = time.time() + PARAM_TIMEOUT
    while time.time() < deadline:
        msg = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if msg and msg.param_id.rstrip('\x00') == name:
            return msg.param_value
    return None

print("\n=== Thermal Calibration Parameters ===")
for p in TC_PARAMS:
    val = get_param(p)
    if val is not None:
        print(f"  {p:<25} = {val}")
    else:
        print(f"  {p:<25} = (no response)")

# --- Trigger tcal and watch for response ---
print("\n=== Triggering Thermal Calibration ===")
mav.mav.command_long_send(
    mav.target_system, mav.target_component,
    mavutil.mavlink.MAV_CMD_PREFLIGHT_CALIBRATION,
    0, 0, 0, 0, 0, 3, 0, 0  # param5=3 = thermal cal
)
print("Command sent. Watching for FC response for 5 seconds...")
deadline = time.time() + 5
responses = []
while time.time() < deadline:
    msg = mav.recv_match(blocking=True, timeout=0.5)
    if not msg or msg.get_type() == "BAD_DATA":
        continue
    if msg.get_type() == "STATUSTEXT":
        responses.append(msg.text.strip())
        print(f"  [FC] {msg.text.strip()}")
    if msg.get_type() == "COMMAND_ACK":
        result = msg.result
        result_names = {0:"ACCEPTED",1:"TEMPORARILY_REJECTED",2:"DENIED",3:"UNSUPPORTED",4:"FAILED",5:"IN_PROGRESS"}
        print(f"  [ACK] result={result_names.get(result, result)}")

if not responses:
    print("  (no STATUSTEXT response — command may have been silently ignored)")

print("\n=== Done ===")
