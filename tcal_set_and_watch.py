#!/usr/bin/env python3
"""Set recommended TC parameters, set cal flags, reboot, and watch boot messages."""
import os, sys, time, struct
os.environ["MAVLINK20"] = "1"
from pymavlink import mavutil
import serial.tools.list_ports

PORT = "COM6"
BAUD = 57600

def connect(timeout=20):
    mav = mavutil.mavlink_connection(PORT, baud=BAUD)
    for _ in range(10):
        mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        time.sleep(0.5)
    if not mav.wait_heartbeat(timeout=timeout):
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
    _, ptype = get_param(mav, name)
    if ptype is None:
        ptype = mavutil.mavlink.MAV_PARAM_TYPE_INT32
    # INT32 params must be sent as their int bits reinterpreted as float
    if ptype == mavutil.mavlink.MAV_PARAM_TYPE_INT32:
        send_val = struct.unpack('<f', struct.pack('<i', int(value)))[0]
    else:
        send_val = float(value)
    mav.mav.param_set_send(mav.target_system, mav.target_component, name.encode(), send_val, ptype)
    deadline = time.time() + 3
    while time.time() < deadline:
        msg = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if msg and msg.param_id.rstrip('\x00') == name:
            return msg.param_value, msg.param_type
    return None, None

def int_val(v):
    if v is None: return None
    return struct.unpack('<i', struct.pack('<f', v))[0]

print("Connecting...")
mav = connect()
if not mav:
    print("No heartbeat."); sys.exit(1)
print(f"Connected. System {mav.target_system}\n")

# Set recommended window parameters
SETTINGS = {
    "SYS_CAL_TMIN": 30,
    "SYS_CAL_TMAX": 50,
    "SYS_CAL_TDEL": 15,
    "SYS_CAL_ACCEL": 1,
    "SYS_CAL_GYRO":  1,
}
print("Setting parameters...")
for name, value in SETTINGS.items():
    result, rtype = set_param(mav, name, value)
    confirmed = int_val(result) if result is not None else None
    status = "OK" if confirmed == value else f"MISMATCH (got {confirmed})"
    print(f"  {name:<20} = {value}  [{status}]")

time.sleep(1)  # let FC flush to storage
print("\nRebooting...")
mav.reboot_autopilot()
mav.close()
time.sleep(1)

# Wait for COM6 to reappear
print("Waiting for COM6...")
deadline_port = time.time() + 30
while time.time() < deadline_port:
    if PORT in [p.device for p in serial.tools.list_ports.comports()]:
        print("COM6 back. Reconnecting...\n")
        break
    time.sleep(0.5)
else:
    print("COM6 did not reappear."); sys.exit(1)

time.sleep(1)
mav2 = connect(timeout=20)
if not mav2:
    print("No heartbeat after reboot."); sys.exit(1)
print(f"Reconnected. Listening for 30 seconds...\n")

deadline = time.time() + 30
last_temp_print = 0
while time.time() < deadline:
    msg = mav2.recv_match(blocking=True, timeout=0.5)
    if not msg or msg.get_type() == "BAD_DATA":
        continue
    if msg.get_type() == "STATUSTEXT":
        print(f"\n  [FC] {msg.text.strip()}")
    elif msg.get_type() == "HIGHRES_IMU" and time.time() - last_temp_print > 2:
        remaining = int(deadline - time.time())
        print(f"  [TEMP] {msg.temperature:.1f}°C  ({remaining}s left)   ", end="\r")
        last_temp_print = time.time()

print("\n\nDone.")
