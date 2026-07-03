#!/usr/bin/env python3
"""Read all thermal calibration relevant parameters."""
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
print(f"Connected.\n")

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

def int_val(v):
    """Parameters stored as INT32 but sent as float — decode raw bits."""
    if v is None:
        return None
    import struct
    raw = struct.pack('<f', v)
    return struct.unpack('<i', raw)[0]

PARAMS = [
    "SYS_CAL_ACCEL",
    "SYS_CAL_GYRO",
    "SYS_CAL_BARO",
    "SYS_CAL_TDEL",
    "SYS_CAL_TMAX",
    "SYS_CAL_TMIN",
    "CAL_ACC_TC_EN",
    "CAL_GYRO_TC_EN",
    "TC_A0_ID",
    "TC_A1_ID",
    "TC_G0_ID",
    "TC_G1_ID",
    "SENS_BOARD_TC_T",
    "SER_TEL1_BAUD",
]

print(f"{'Parameter':<25} {'Raw Float':<20} {'As Int'}")
print("-" * 60)
for p in PARAMS:
    val = get_param(p)
    if val is not None:
        iv = int_val(val)
        print(f"  {p:<23} {val:<20} {iv}")
    else:
        print(f"  {p:<23} (no response)")
