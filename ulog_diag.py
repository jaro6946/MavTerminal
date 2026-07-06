#!/usr/bin/env python3
"""ulog_diag.py -- read a PX4 .ulg (pulled with pull_log.py) and print WHY the
vehicle failsafed / wouldn't arm / wouldn't fly.

This is the payoff of pulling logs: PX4's HEARTBEAT/STATUSTEXT over MAVLink do NOT
tell you the real failsafe reason (the reason lives in structured events whose IDs
are firmware hashes).  But the internal uORB topics logged in the .ulg do -- so we
read them directly:
  * logged_messages   -- the [FC] STATUSTEXT stream (what you'd see live).
  * vehicle_status    -- arming_state / nav_state / failsafe timeline.
  * failsafe_flags    -- EXACTLY which condition is set at the failsafe onset
                         (manual_control_signal_lost, battery_warning, *_invalid ...).
  * battery_status    -- voltage / connected / warning per instance.
  * actuator_motors   -- did PX4 actually command thrust? (0 => it never tried.)

Usage:  ulog_diag.py <log.ulg>
Requires pyulog  (pip install pyulog  -- already in the agc_CTOL_SE3-rotopy venv).

nav_state cheatsheet (PX4, version-dependent): 0 MANUAL 2 POSCTL 3 AUTO_MISSION
4 AUTO_LOITER 5 AUTO_RTL 14 OFFBOARD 17 AUTO_PRECLAND.  arming_state: 1 STANDBY
2 ARMED.  HEARTBEAT.system_status: 3 STANDBY 4 ACTIVE 5 CRITICAL 8 FLIGHT_TERMINATION.
"""
import sys

try:
    import numpy as np
    from pyulog import ULog
except ImportError as e:
    sys.exit(f"needs numpy + pyulog: {e}  (pip install pyulog)")


def get(u, name):
    for d in u.data_list:
        if d.name == name:
            return d
    return None


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: ulog_diag.py <log.ulg>")
    u = ULog(sys.argv[1])

    print("=== logged messages (the [FC] STATUSTEXT stream) ===")
    for m in u.logged_messages:
        print(f"  [{m.timestamp/1e6:8.2f}] L{m.log_level} {m.message}")

    vs = get(u, "vehicle_status")
    if vs:
        print("\n=== vehicle_status timeline (on change) ===")
        prev = None
        first_fs = None
        for i in range(len(vs.data["timestamp"])):
            cur = (int(vs.data["nav_state"][i]), int(vs.data["failsafe"][i]),
                   int(vs.data["arming_state"][i]))
            if cur != prev:
                prev = cur
                print(f"  [{vs.data['timestamp'][i]/1e6:8.2f}] "
                      f"nav_state={cur[0]} failsafe={cur[1]} arming_state={cur[2]}")
            if first_fs is None and cur[1] == 1:
                first_fs = vs.data["timestamp"][i]

        ff = get(u, "failsafe_flags")
        if ff and first_fs is not None:
            j = min(int(np.searchsorted(np.array(ff.data["timestamp"]), first_fs)),
                    len(ff.data["timestamp"]) - 1)
            print(f"\n=== failsafe_flags AT failsafe onset (t={ff.data['timestamp'][j]/1e6:.2f}s) "
                  "-- these are the CAUSE ===")
            for k in sorted(ff.data):
                if k == "timestamp" or k.startswith("mode_req"):
                    continue
                if ff.data[k][j] != 0:
                    print(f"    {k} = {ff.data[k][j]}")

        if ff:
            print("\n=== failsafe_flags that are EVER set (whole log) ===")
            for k in sorted(ff.data):
                if k == "timestamp" or k.startswith("mode_req"):
                    continue
                mx = max(ff.data[k])
                if mx != 0:
                    print(f"    {k}: max={mx}")

    for bs in [d for d in u.data_list if d.name == "battery_status"]:
        v = bs.data.get("voltage_v", [0])
        print(f"\n=== battery_status inst {bs.multi_id} ===")
        print(f"    V {min(v):.2f}..{max(v):.2f}"
              + (f"  connected {min(bs.data['connected'])}..{max(bs.data['connected'])}"
                 if "connected" in bs.data else "")
              + (f"  warning max={max(bs.data['warning'])}" if "warning" in bs.data else ""))

    am = get(u, "actuator_motors")
    if am:
        cols = [am.data[f"control[{i}]"] for i in range(8) if f"control[{i}]" in am.data]
        allv = np.concatenate([np.asarray(c) for c in cols])
        allv = allv[~np.isnan(allv)]
        if allv.size:
            print(f"\n=== actuator_motors === range {allv.min():.3f}..{allv.max():.3f} "
                  f"mean {allv.mean():.3f}   (0 => PX4 never commanded thrust)")


if __name__ == "__main__":
    main()
