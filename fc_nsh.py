#!/usr/bin/env python3
"""Run NuttShell commands on the FC over any MAVLink link and return the text.

Library helper for scripts that need the FC's shell over a *second* link
(e.g. the companion's TELEM1 route while USB is wedged).  CLI:

    python3 fc_nsh.py -p companion "mavlink status" "top once"
"""
import argparse
import time

from pymavlink import mavutil

SHELL_DEV = mavutil.mavlink.SERIAL_CONTROL_DEV_SHELL
FLAGS = (mavutil.mavlink.SERIAL_CONTROL_FLAG_EXCLUSIVE
         | mavutil.mavlink.SERIAL_CONTROL_FLAG_RESPOND)
DATA_LEN = 70


def connect(port, baud=921600, timeout=10.0):
    import pull_log
    port = pull_log.resolve_port(port)
    m = mavutil.mavlink_connection(port, baud=baud, source_system=254,
                                   source_component=191)
    deadline = time.time() + timeout
    while time.time() < deadline:
        m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                             mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
        if hb and hb.get_srcComponent() == 1:
            return m
    raise TimeoutError(f"no autopilot heartbeat on {port}")


def _send(m, text):
    data = text.encode()
    idx, first = 0, True
    while first or idx < len(data):
        first = False
        chunk = data[idx:idx + DATA_LEN]
        idx += DATA_LEN
        m.mav.serial_control_send(SHELL_DEV, FLAGS, 0, 0, len(chunk),
                                  list(chunk) + [0] * (DATA_LEN - len(chunk)))


def run(m, cmd, quiet=0.6, cap=8.0):
    """Send one command line, return everything the shell printed back."""
    out, last = [], 0.0
    _send(m, cmd + "\n")
    start = time.time()
    while time.time() - start < cap:
        msg = m.recv_match(type="SERIAL_CONTROL", blocking=True, timeout=0.1)
        now = time.time()
        if msg is not None and msg.count:
            out.append(bytes(msg.data[:msg.count]).decode("utf-8", "replace"))
            last = now
            continue
        if last and now - last > quiet:
            break
        if not last and now - start > 2.0:
            _send(m, "")                 # poll: PX4 only replies to requests
    return "".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--port", default="companion")
    ap.add_argument("-b", "--baud", type=int, default=921600)
    ap.add_argument("cmds", nargs="+")
    a = ap.parse_args()
    m = connect(a.port, a.baud)
    for c in a.cmds:
        print(f"### {c}\n{run(m, c)}")
