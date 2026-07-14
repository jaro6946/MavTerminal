# MavTerminal

A tiny, dependency-light MAVLink terminal for talking to a USB-attached PX4/ArduPilot
flight controller (FC) **without QGroundControl**. It's a MAVLink (Micro Air Vehicle
Link) REPL plus a non-interactive batch mode, built for quick param reads/sets and
watching the FC's status text from the command line or a script.

The core tool is one file: [`mavshell.py`](mavshell.py). Alongside it are standalone helpers
that share its port auto-detect: [`pull_log.py`](pull_log.py) / [`ulog_diag.py`](ulog_diag.py)
for [pulling & diagnosing flight logs](#pulling--diagnosing-flight-logs-find-out-why-it-wont-fly),
and the `tcal_*.py` [thermal-calibration scripts](#thermal-calibration-helpers).

---

## Quick start

```bash
# Convenience alias (add to ~/.bashrc). Uses the agc_CTOL_SE3-rotopy venv, which
# already has pymavlink + pyserial:
alias mavTerminal='/home/rober/jacobAtGar/agc_CTOL_SE3-rotopy/venv/bin/python /home/rober/jacobAtGar/mavterminal/MavTerminal/mavshell.py'

# Interactive REPL:
mavTerminal

# Batch mode — connect, run commands, print results, exit (great for scripts/CI):
mavTerminal -c "param get SYS_HITL"
mavTerminal -c "param get BAT1_SOURCE" -c "param get SENS_IMU_MODE"   # -c is repeatable
printf 'temp\nshow ATTITUDE\n' | mavTerminal                          # or pipe on stdin
```

> **Aliases don't load in non-interactive shells.** If a script or another tool can't
> find `mavTerminal`, call the full path:
> `/home/rober/jacobAtGar/agc_CTOL_SE3-rotopy/venv/bin/python .../mavshell.py`.

---

## How it connects

- **Port auto-detection** (OS-agnostic, no config): it scans serial ports and picks the
  FC by USB **vendor ID** first (mRo `0x26AC`, ST `0x0483`, pid.codes `0x1209`, Cube
  `0x2DAE`, Holybro `0x3162`), then by description/manufacturer keyword
  (`px4`, `pixhawk`, `ardupilot`, `fmu`, `mro`, `cube`, `holybro`, `control zero`, …),
  then any CDC-ACM/`COM*` device. Works the same on Linux/WSL (`/dev/ttyACM*`) and
  Windows (`COM*`). Override with `-p/--port` or `$MAV_PORT`.
- **Baud defaults to `57600`** (`-b/--baud` or `$MAV_BAUD`). This is fine for params and
  status text even though the HITL sim streams at `921600` — the FC's USB CDC link isn't
  really baud-limited.
- **It wakes the FC itself.** PX4 only starts streaming on a serial link once it sees a
  GCS heartbeat, so the tool sends 5 heartbeats, waits for the FC's heartbeat
  (`-t/--heartbeat-timeout`, default 10 s), then requests `MAV_DATA_STREAM_ALL` at 10 Hz.
  No QGC needed.
- **One serial owner at a time.** The FC's USB link can't be shared. You **cannot** run
  MavTerminal while the HITL sim (or QGC) holds the same port — stop the sim first.

---

## Commands

From `help` inside the REPL (all also work in `-c` batch mode):

| Command | Description |
|---|---|
| `param get <NAME>` | Read one parameter (decoded, see below). |
| `param set <NAME> <VAL>` | Write one parameter (encoded with the FC's registered type). |
| `show <TYPE>` | Print the latest message of a type, e.g. `show ATTITUDE`, `show SYS_STATUS`. |
| `list` | List message types received so far. |
| `stream on/off` | Toggle a live dump of every incoming message. |
| `statustext on/off` | Toggle printing FC `STATUSTEXT` (`[FC] …`); **on** by default. |
| `temp` | Show board/IMU temperatures (HIGHRES_IMU + SCALED_IMU1/2/3). |
| `tcal` | Trigger PX4 thermal calibration (`MAV_CMD_PREFLIGHT_CALIBRATION` param5=3). |
| `heartbeat` | Print connection info (system/component id). |
| `arm` / `disarm` / `reboot` | Send arm / disarm / reboot-autopilot commands. |
| `quit` / `exit` / `q` | Disconnect. |

### Gotchas (learned the hard way)

- **There is no `param show`.** Read one at a time with `param get <NAME>`. To read
  several, repeat `-c "param get X"` (batch echoes each as `> param get X`). Param names
  are upper-cased for you.
- **Integer params are bytewise-encoded.** PX4/QGC transmit integer params by copying the
  raw int **bytes** into the float `param_value` field rather than value-casting, so an
  `INT32` of `1` arrives as the float `1.4e-45` (bit pattern `0x00000001`). MavTerminal
  decodes this for you on `get` and re-encodes it on `set` (it first reads the param's
  registered type so it doesn't write garbage). Values you see/enter are the real ints.
- **`[FC]` messages on connect may be *stale replays*.** PX4 re-streams its recent
  critical `STATUSTEXT` to a newly-connected GCS. So a burst of criticals right after
  "Data streams requested." (e.g. `Flight termination active`, `low battery`) can be the
  *last* session replayed, not the live state. A truly **live** condition keeps
  re-emitting every ~1–2 s (e.g. `MAG #0 failed: TIMEOUT!` when no sensor feed is running).
  Judge live-vs-stale by whether a message *repeats* after the initial connect burst.
- **`param get` forces a fresh read** (it drops any cached value and re-requests), and
  `param set` waits for the FC to echo the accepted `PARAM_VALUE` — so a printed result is
  confirmed by the board, not just sent.

### Options

```
-p, --port                 serial port (default: $MAV_PORT or auto-detected)
-b, --baud                 baud rate (default: $MAV_BAUD or 57600)
-c, --cmd                  run a command non-interactively (repeatable) -> batch mode
-t, --heartbeat-timeout    seconds to wait for first heartbeat (default 10)
    --settle               seconds to let streams populate before batch cmds (default 2.0)
```

Batch mode is entered automatically by `-c` **or** by piping commands on non-TTY stdin.

---

## HITL param cheat-sheet (this bench: mRo ControlZero H7, sys id 17)

These are the flight-controller params the RotorPy HITL (Hardware-In-The-Loop) sim needs so
PX4 arms and **holds OFFBOARD** on a bench (no RC, no battery). `SYS_HITL`, `BAT1_SOURCE`,
`SENS_IMU_MODE`, and `COM_RC_IN_MODE` are **reboot-required** — set, then `reboot`, then
re-attach USB. Read any with `mavTerminal -c "param get <NAME>"`:

| Param | Bench value | Why |
|---|---|---|
| `SYS_HITL` | `1` | Route actuator outputs to the HIL interface (else the sim gets no motor commands). |
| `BAT1_SOURCE` | `-1` (disabled) | **No battery on the bench.** Injecting one over MAVLink was unreliable on the saturated single USB link — the `connected` flag flickered to 0, PX4 read "Battery disconnected" and RTL-failsafed. Disabling the source means PX4 monitors *no* battery and the sim injects none. |
| `SENS_IMU_MODE` | `1` (single) | Single-EKF mode; clears the standing `ekf2 missing data` preflight fail (multi-EKF needs `estimator_selector_status`, which HIL doesn't provide). |
| `COM_RC_IN_MODE` | `1` (joystick) | Accept a MAVLink joystick and drop RC-hardware checks. The sim injects a neutral virtual joystick so PX4 never sees manual control as "lost" (else it RTL-failsafes out of OFFBOARD the instant it arms). |
| `COM_RCL_EXCEPT` | `7` | Except **all** modes (mission+hold+offboard) from the RC-loss failsafe. |
| `COM_RC_LOSS_T` | `2.0` | RC-loss timeout 2 s — tolerates the virtual-joystick injection's packet-drop gaps on the shared link. |
| `COM_DISARM_PRFLT` | `0` | Don't auto-disarm while armed-but-not-airborne. The 15 s default silently kills the readiness gate mid-arm. |
| `FD_ACT_EN` / `FD_ESCS_EN` | `0` | Actuator/ESC failure detection off — meaningless on a bench with no real motors/ESCs, and it false-triggers. |
| `GF_ACTION` / `NAV_DLL_ACT` / `NAV_RCL_ACT` | `0` | Geofence / data-link-loss / RC-loss failsafe actions off. |
| `CBRK_FLIGHTTERM` | `121212` | Disables **new** flight terminations (bench only). |

The RotorPy GUI's **Utilities → HITL setup** panel writes this whole set with one button (see
`_HITL_PARAMS` in `gui/rotorpysim.py`). Props **OFF** for any bench arm.

**Notes worth remembering:**
- `param get` reads **ground truth off the board's flash** — the actual persisted value,
  independent of any config file. That's how you confirm a reboot-required param actually stuck.
- **Termination re-latches if you kill an armed run.** Feed-loss while armed makes PX4 terminate,
  and it survives into the next session until a **both-power** reboot (USB *and* any bench/battery
  supply — a USB-only unplug won't reset the FMU if it's powered elsewhere). Check the true state
  with `show HEARTBEAT` → `system_status` (8 = flight-termination) — see the log-diagnosis section.
- `CBRK_FLIGHTTERM=121212` prevents *new* terminations but does **not un-latch one already
  active** — only a reboot clears it. If `Flight termination active` keeps firing on a fresh
  boot with a good feed, it's usually a failsafe escalating (no RC / no battery / no position),
  not this breaker — `ulog_diag.py` will name the real trigger.
- With `SYS_HITL=1` the real sensors are off, so from power-on until the sim's HIL feed starts
  the FC sees **no** sensors — expect a transient blast of `No valid data from
  Accel/Gyro/Baro/Compass` and `ekf2 missing data` during that window (benign).

The board's full parameter dump lives in [`mav_17_1.parm`](mav_17_1.parm)
(`NAME  VALUE`, one per line — same format QGC exports).

---

## Pulling & diagnosing flight logs (find out *why* it won't fly)

When a board **won't arm, keeps failsafing, or silently won't fly**, MAVLink itself won't
tell you the real reason: the `HEARTBEAT`/`STATUSTEXT` stream is a lagging, lossy summary, and
the actual failsafe reason lives in PX4 *events* whose IDs are firmware hashes you can't decode
offline. The ground truth is the onboard `.ulg` log — and you can pull it over the **same USB
link**, no SD-card removal and no QGroundControl.

### The easy way: the built-in `log` command (one invocation, batch-friendly)

The whole pull→diagnose flow is built into `mavTerminal` as a `log` command, so you don't
juggle two scripts or two port owners — it reuses the connection the shell already holds:

```bash
mavTerminal -b 921600 -c "log pull"          # download the NEWEST log AND print the diagnosis
mavTerminal -c "log list"                     # list sessions on the card (newest marked)
mavTerminal -c "log list sess114"             # list that session's .ulg files
mavTerminal -c "log pull sess112 log102.ulg"  # pull a specific log, then diagnose
mavTerminal -c "log diag sess114_log101.ulg"  # re-diagnose an already-downloaded local .ulg
```

`log pull` is the one-shot "why won't it fly" button: a single non-interactive command that
downloads the newest `.ulg` and prints the full `ulog_diag` breakdown. Downloads land in
`$MAV_LOG_DIR` (default: current dir). Use `-b 921600` for the fast transfer baud (the default
57600 works but a ~1 MB log takes minutes). Internally the shell pauses its background reader so
the MAVFTP transfer is the sole reader of the one FC serial port.

### The scripts underneath (if you want them standalone)

The `log` command wraps two standalone scripts (run with the same venv python), still usable on
their own — e.g. when the shell isn't running:

| Script | Purpose |
|---|---|
| [`pull_log.py`](pull_log.py) | Pull a `.ulg` off the FC over **MAVLink FTP**. ~500 KB comes down in ~5 s (a 3 MB log in ~9 s) at 921600. |
| [`ulog_diag.py`](ulog_diag.py) | Parse a `.ulg` with `pyulog` and print **why it failsafed**: the STATUSTEXT stream, the `vehicle_status` nav_state/arming/failsafe timeline, the **`failsafe_flags` set at the failsafe onset** (the actual cause), the per-instance `battery_status`, and whether PX4 ever commanded thrust (`actuator_motors`). |

```bash
alias mavPy='/home/rober/jacobAtGar/agc_CTOL_SE3-rotopy/venv/bin/python'

mavPy pull_log.py --list -b 921600            # list sessions on the card (newest marked)
mavPy pull_log.py --list --session sess114    # list that session's .ulg files
mavPy pull_log.py -o /tmp -b 921600           # pull the NEWEST log (the usual case)
mavPy pull_log.py --session sess112 --name log102.ulg   # pull a specific one
mavPy ulog_diag.py /tmp/sess114_log101.ulg    # ...then read why it failsafed
```

`ulog_diag.py` output pinpoints the cause. Example (a bench with no RC and a flickering
injected battery): `failsafe_flags AT failsafe onset` shows `manual_control_signal_lost = 1`
and `battery_warning = 2` — i.e. it RTL-failsafed on RC-loss and a phantom low battery, *not*
on anything in the STATUSTEXT. `actuator_motors range 0.000..0.000` means PX4 never even tried
to spin the motors (it was locked out); a non-zero range means it *is* commanding thrust and
the problem is downstream (e.g. the HIL actuator feedback isn't reaching the sim).

> **The stop-guessing move:** the moment a HITL/bench board misbehaves, `pull_log.py` then
> `ulog_diag.py`. It replaces a dozen speculative param pokes with the one true reason.

**Reading the FC's live failsafe/termination state without a log:** the `HEARTBEAT` carries
it directly — `mavTerminal -c "show HEARTBEAT"` and read `system_status` (8 =
`MAV_STATE_FLIGHT_TERMINATION`, 5 = `CRITICAL`/failsafe, 4 = `ACTIVE`, 3 = `STANDBY`). This is
**authoritative** — unlike a `Flight termination active` STATUSTEXT, which PX4 does *not*
re-emit reliably, so a gate that keys off STATUSTEXT can miss a terminated board.

`pull_log.py` also uses the shared FC **port auto-detect** (same VID/keyword logic as
`mavshell.py`), so `-p` is usually unnecessary; it needs `pyulog` for the analyzer
(`pip install pyulog`, already in the venv).

---

## Thermal-calibration helpers

Standalone scripts (run with the same venv python) for the PX4 thermal-cal workflow:

| Script | Purpose |
|---|---|
| `tcal_status.py` | Check whether thermal cal ran and if it's currently in progress. |
| `tcal_check.py` | Check if thermal calibration is running. |
| `tcal_params.py` | Read all thermal-cal-relevant parameters. |
| `tcal_diag.py` | Connect, collect thermal-cal info, report, exit. |
| `tcal_set_and_watch.py` | Set recommended TC params + cal flags, reboot, watch boot messages. |
| `tcal_reboot_watch.py` | Set cal flags, reboot, reconnect immediately to catch boot messages. |

---

## Requirements

- Python 3 with `pymavlink` and `pyserial` (both present in the
  `agc_CTOL_SE3-rotopy` venv the alias points at). `ulog_diag.py` also needs
  `pyulog` (`pip install pyulog` — already in that venv).
- The FC attached to this host. Under WSL, pass it through with
  `usbipd attach --wsl --busid <id>` on the Windows side first.
