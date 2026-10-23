#!/usr/bin/env python3
"""Repeatable trials for the grazer FC's USB wedge (FC->PC stops dead until the
device re-enumerates).  One trial = load the USB link until either --target-mb
arrives (PASS) or no byte arrives for --stall-s (WEDGE).

On a wedge the FC is interrogated over the SECOND link (companion -> TELEM1),
which still works, and then recovery is attempted from the gentlest layer up --
the first rung that brings bytes back says where the stuck state lives:

    1 reopen   close + reopen the port on the PC  -> PC app / tty layer
    2 mavlink  restart PX4's USB mavlink instance -> PX4 mavlink / cdcacm buffer
    3 usb      serdis + sercon on the FC          -> NuttX USB driver state
    4 reboot   FC reboot (ESC startup tune!)      -> MCU / hardware state

Loads:
    logdata  LOG_REQUEST_DATA for a whole big log, re-requested on lulls
             (pure FC->PC bulk, SD read on the FC side)
    idle     default streams + heartbeats only (is it time- or load-driven?)

Each trial appends a row to <out>/trials.csv; wedge snapshots go to <out>/<n>_*.txt.

    python3 usb_wedge_trial.py --tag baseline --trials 3
"""
import argparse
import csv
import os
import subprocess
import threading
import time

from pymavlink import mavutil

import fc_nsh
import pull_log

USBIPD = "/mnt/c/Program Files/usbipd-win/usbipd.exe"
OUT = os.path.expanduser("~/jacobAtGar/Log Analysis/usb_wedge")
FIELDS = ["trial", "start", "tag", "load", "rate_kib", "result", "mb", "secs", "kib_s",
          "fc_uptime", "fc_tx_Bps", "fc_txerr_Bps", "fc_rx_Bps", "recovered_by"]


class UsbLink:
    """The PC end of the FC's USB CDC port, with a GCS heartbeat thread and a
    byte counter that the wedge detector watches."""

    def __init__(self, port, baud):
        self.port, self.baud = port, baud
        self.m = None
        self.bytes = 0
        self.last_rx = time.time()
        self._stop = threading.Event()

    def open(self):
        self.m = mavutil.mavlink_connection(self.port, baud=self.baud,
                                            source_system=255, source_component=190)
        self.bytes, self.last_rx = 0, time.time()
        self._stop.clear()
        threading.Thread(target=self._hb, daemon=True).start()

    def close(self):
        self._stop.set()
        try:
            self.m.close()
        except Exception:
            pass

    def _hb(self):
        while not self._stop.wait(1.0):
            try:
                self.m.mav.heartbeat_send(6, 8, 0, 0, 0)
            except Exception:
                pass

    def recv(self, timeout=0.2):
        try:
            msg = self.m.recv_match(blocking=True, timeout=timeout)
        except TypeError:
            # pymavlink's per-instance message cache trips on some messages
            # (add_message: _instances is None); the bytes did arrive.
            msg = None
            self.last_rx = time.time()
        if msg is not None:
            self.bytes += len(msg.get_msgbuf())
            self.last_rx = time.time()
        return msg

    def alive_within(self, secs):
        """Pump for ``secs``; True if any byte arrived."""
        n0, end = self.bytes, time.time() + secs
        while time.time() < end:
            self.recv()
        return self.bytes > n0


def wait_for_tty(port, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if os.path.exists(port):
            return True
        time.sleep(0.5)
    return False


def parse_usb_status(text):
    """tx / txerr / rx (B/s) of the USB CDC instance from `mavlink status`."""
    blk = text[text.find("USB CDC"):] if "USB CDC" in text else ""
    vals = {}
    for key in ("tx", "txerr", "rx"):
        for line in blk.splitlines():
            s = line.strip()
            if s.startswith(key + ":"):
                vals[key] = s.split(":")[1].split()[0]
                break
    return vals.get("tx", ""), vals.get("txerr", ""), vals.get("rx", "")


def companion_snapshot(comp, path):
    """FC state over TELEM1.  `mavlink status` twice, 2 s apart, so the 1 s
    rate windows cover the wedged period rather than a session start."""
    parts = []
    for cmd in ("mavlink status", "sleep 2", "mavlink status", "uorb top -1",
                "top once", "dmesg", "perf", "free"):
        if cmd.startswith("sleep"):
            time.sleep(2)
            continue
        parts.append(f"### {cmd}\n{fc_nsh.run(comp, cmd, cap=15)}")
    text = "\n".join(parts)
    with open(path, "w") as f:
        f.write(text)
    return text


class Monitor(threading.Thread):
    """Samples the FC's USB-instance counters over the companion every few
    seconds during a load, so the run-up to a wedge is on record."""

    def __init__(self, comp, path, link, period=3.0):
        super().__init__(daemon=True)
        self.comp, self.path, self.link, self.period = comp, path, link, period
        self.halt = threading.Event()
        self.last = ("", "", "")

    def run(self):
        t0 = time.time()
        with open(self.path, "w") as f:
            f.write("t_s,pc_bytes,fc_tx_Bps,fc_txerr_Bps,fc_rx_Bps\n")
            while not self.halt.is_set():
                self.last = parse_usb_status(fc_nsh.run(self.comp, "mavlink status"))
                f.write(f"{time.time()-t0:.1f},{self.link.bytes},{','.join(self.last)}\n")
                f.flush()
                self.halt.wait(self.period)


def fc_uptime(comp):
    """FC boot time (min) from SYSTEM_TIME on the companion link."""
    msg = comp.recv_match(type="SYSTEM_TIME", blocking=True, timeout=5.0)
    return f"{msg.time_boot_ms/60000:.1f}min" if msg else ""


def pick_log(link):
    """Largest log the FC lists (id, size)."""
    entries = pull_log.log_entries(link.m)
    lid = max(entries, key=lambda k: entries[k][0])
    return lid, entries[lid][0]


def load_logdata(link, target, stall_s, log, rate_kib=0):
    """Bulk FC->PC via LOG_DATA.  ``rate_kib`` > 0 paces the requests (chunks of
    50 messages) to that average rate instead of letting PX4 send flat out."""
    lid, size = pick_log(link)
    log(f"  logdata: id {lid}, {size/1e6:.1f} MB, "
        f"{'unpaced' if not rate_kib else f'{rate_kib} KiB/s'}")
    chunk = 50 * pull_log.LOG_DATA_LEN if rate_kib else 0xFFFFFFFF
    got, ofs, t0 = 0, 0, time.time()
    req_end = ofs + chunk
    link.m.mav.log_request_data_send(1, 1, lid, ofs, chunk)
    last_req = time.time()
    while got < target:
        msg = link.recv(timeout=0.05)
        now = time.time()
        if msg is not None and msg.get_type() == "LOG_DATA" and msg.id == lid:
            got += msg.count
            ofs = msg.ofs + msg.count
            if ofs >= size:                       # wrap: read it again
                ofs = 0
                req_end = 0
        if now - link.last_rx > stall_s:
            return "WEDGE", got, now - t0
        due = (not rate_kib) or now >= t0 + got / (rate_kib * 1024)
        done = ofs >= req_end
        lull = now - last_req > 1.0 and now - link.last_rx > 0.5
        if (rate_kib and done and due) or lull or (not rate_kib and req_end == 0):
            req_end = ofs + chunk
            link.m.mav.log_request_data_send(1, 1, lid, ofs, chunk)
            last_req = now
    link.m.mav.log_request_end_send(1, 1)
    return "PASS", got, time.time() - t0


def load_idle(link, secs, stall_s, log):
    t0 = time.time()
    while time.time() - t0 < secs:
        link.recv()
        if time.time() - link.last_rx > stall_s:
            return "WEDGE", link.bytes, time.time() - t0
    return "PASS", link.bytes, time.time() - t0


def recover(link, comp, busid, log, allow_reboot=False):
    link.close()
    time.sleep(1)
    link.open()
    if link.alive_within(4):
        return "reopen"
    log("  rung 1 (reopen) failed")
    link.close()
    fc_nsh.run(comp, "mavlink stop -d /dev/ttyACM0")
    time.sleep(1)
    fc_nsh.run(comp, "mavlink start -d /dev/ttyACM0 -m 2")
    time.sleep(1)
    link.open()
    if link.alive_within(4):
        return "mavlink"
    log("  rung 2 (mavlink restart) failed")
    link.close()
    # Software USB disconnect/reconnect on the FC: serdis drops the pull-up and
    # tears down the NuttX CDC/ACM driver, the autostart's sercon rebuilds it and
    # the host re-enumerates.  No MCU reset, so no ESC startup tune.
    fc_nsh.run(comp, "cdcacm_autostart stop")
    fc_nsh.run(comp, "serdis")
    end = time.time() + 8
    while os.path.exists(link.port) and time.time() < end:
        time.sleep(0.2)
    time.sleep(1)
    fc_nsh.run(comp, "cdcacm_autostart start")
    if wait_for_tty(link.port, 40):
        time.sleep(3)
        link.open()
        if link.alive_within(8):
            return "usb_reinit"
    log("  rung 3 (USB driver re-init) failed")
    link.close()
    if not allow_reboot:
        log("  not rebooting (ESC startup tune) -- pass --allow-reboot to")
        return "NONE"
    # usbipd detach/attach does NOT reset the FC's side (Windows keeps the
    # device enumerated), so the real re-enumeration is an FC reboot.
    if fc_armed(comp):
        log("  FC is ARMED -- not rebooting")
        return "NONE"
    fc_nsh.run(comp, "reboot", cap=2)
    end = time.time() + 10
    while os.path.exists(link.port) and time.time() < end:
        time.sleep(0.2)
    if wait_for_tty(link.port, 60):
        time.sleep(3)
        link.open()
        if link.alive_within(8):
            return "fc_reboot"
    return "NONE"


def fc_armed(comp):
    for _ in range(10):
        hb = comp.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
        if hb is not None and hb.get_srcComponent() == 1:
            return bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    return True                                   # unknown -> treat as armed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="condition label for the CSV")
    ap.add_argument("--load", choices=("logdata", "idle"), default="logdata")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--target-mb", type=float, default=20.0)
    ap.add_argument("--rate-kib", type=float, default=0,
                    help="pace logdata to this average rate (0 = flat out)")
    ap.add_argument("--idle-s", type=float, default=300.0)
    ap.add_argument("--stall-s", type=float, default=3.0)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=2000000)
    ap.add_argument("--busid", default="2-2")
    ap.add_argument("--no-companion", action="store_true",
                    help="companion unplugged/off: no FC snapshots, rung 2 skipped")
    ap.add_argument("--fc-pre", action="append", default=[],
                    help="FC shell command run over the companion before every "
                         "trial (repeatable), e.g. 'logger stop' -- reboots undo them")
    ap.add_argument("--allow-reboot", action="store_true",
                    help="last-resort FC reboot when the USB re-init fails "
                         "(makes the ESCs play their startup tune)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, "trials.csv")
    new = not os.path.exists(csv_path)
    n0 = 0 if new else sum(1 for _ in open(csv_path)) - 1

    def log(s):
        print(time.strftime("%H:%M:%S"), s, flush=True)

    comp = None if a.no_companion else fc_nsh.connect("companion")
    link = UsbLink(a.port, a.baud)
    link.open()
    if not link.alive_within(4):
        raise SystemExit("USB link is not live at start -- recover it first")

    for i in range(a.trials):
        n = n0 + i + 1
        start = time.strftime("%Y-%m-%d %H:%M:%S")
        log(f"trial {n} [{a.tag}] load={a.load}")
        for cmd in a.fc_pre:
            log(f"  fc: {cmd} -> {fc_nsh.run(comp, cmd).strip().splitlines()[-2:]}")
        mon = None
        if comp is not None:
            mon = Monitor(comp, os.path.join(a.out, f"{n}_monitor.csv"), link)
            mon.start()
        if a.load == "logdata":
            res, got, secs = load_logdata(link, a.target_mb * 1e6, a.stall_s, log, a.rate_kib)
        else:
            res, got, secs = load_idle(link, a.idle_s, a.stall_s, log)
        row = dict(trial=n, start=start, tag=a.tag, load=a.load, rate_kib=a.rate_kib, result=res,
                   mb=f"{got/1e6:.2f}", secs=f"{secs:.1f}",
                   kib_s=f"{got/1024/max(secs, 1e-3):.0f}", recovered_by="")
        log(f"  {res} after {got/1e6:.2f} MB / {secs:.1f} s")
        if mon is not None:
            mon.halt.set()
            mon.join(timeout=15)
        if comp is not None:
            row["fc_uptime"] = fc_uptime(comp)
        if res == "WEDGE":
            if comp is not None:
                snap = companion_snapshot(comp, os.path.join(a.out, f"{n}_wedge.txt"))
                row["fc_tx_Bps"], row["fc_txerr_Bps"], row["fc_rx_Bps"] = \
                    parse_usb_status(snap[snap.rfind("### mavlink status"):])
                log(f"  FC USB instance while wedged: tx {row['fc_tx_Bps']} "
                    f"txerr {row['fc_txerr_Bps']} rx {row['fc_rx_Bps']} B/s")
                row["recovered_by"] = recover(link, comp, a.busid, log, a.allow_reboot)
            else:
                link.close()
                subprocess.run([USBIPD, "detach", "--busid", a.busid], capture_output=True)
                time.sleep(3)
                if not wait_for_tty(a.port, 5):
                    subprocess.run([USBIPD, "attach", "--wsl", "--busid", a.busid],
                                   capture_output=True)
                ok = wait_for_tty(a.port, 30)
                time.sleep(2)
                link.open()
                row["recovered_by"] = "usbip" if ok and link.alive_within(5) else "NONE"
            log(f"  recovered by: {row['recovered_by']}")
            if row["recovered_by"] == "NONE":
                with open(csv_path, "a", newline="") as f:
                    w = csv.DictWriter(f, FIELDS)
                    if new:
                        w.writeheader()
                    w.writerow(row)
                raise SystemExit("could not recover the USB link -- stopping")
        elif mon is not None:
            # healthy-load reference: the counters as last sampled under load
            row["fc_tx_Bps"], row["fc_txerr_Bps"], row["fc_rx_Bps"] = mon.last
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            if new:
                w.writeheader()
                new = False
            w.writerow(row)
    link.close()


if __name__ == "__main__":
    main()
