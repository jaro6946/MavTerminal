"""Stage 1 (rosbags + the bag's embedded schemas): dump per-bag overlay series.

Everything is raw, as the controller saw it:
  * odom z      = odom.pose.z where recorded, else odom->base_link TF z
                  (the bridge publishes both from the same value). EKF reset
                  jumps are left in.
  * measured v  = odom.twist.linear where recorded (the EKF velocity the
                  controller consumes); else a 0.2 s central difference of
                  the TF position, reset jumps NOT removed.
  * log lines   = every line in the bag's launch.log inside the bag window.
  * GPS height  = reference only (controller does not use it).
"""
from pathlib import Path
from rosbags.highlevel import AnyReader
import yaml, glob, re, json, sys, numpy as np
from scipy.spatial.transform import Rotation as Rot

# usage: overlay_data.py <date folder of bags> <output .json>
BAG_DIR, OUT_JSON = sys.argv[1], sys.argv[2]

NAV = {0: 'MANUAL', 1: 'ALTCTL', 2: 'POSCTL', 3: 'MISSION', 4: 'LOITER', 5: 'RTL', 12: 'DESCEND',
       13: 'TERMINATION', 14: 'OFFBOARD', 15: 'STABILIZED', 17: 'TAKEOFF', 18: 'LAND', 20: 'PRECLAND'}
LEVEL = re.compile(r'\[(DEBUG|INFO|WARN|ERROR|FATAL)\]')
TAG_POSE = '/grazer10/precision_landing/rover_tracker/pose'   # tag pose in grazer10/hires_optical
TAG_PIXEL = '/grazer10/precision_landing/rover_tracker/pixel' # tag centre, RECTIFIED image px (what the centred gate reads)
CAM_INFO = '/grazer10/hires/camera_info'
# controller's "camera is steering" line, throttled to one per 0.5 s while the marker servo overrides the target
SERVO_RE = re.compile(r'override p_selected\.xy')
# same line, with both targets: GPS (rover odom) target -> camera-implied pad, in the controller's nav frame
OVR_RE = re.compile(r'override p_selected\.xy \(([-\d.]+), ([-\d.]+)\) -> \(([-\d.]+), ([-\d.]+)\)')
ROVER = 'rover2'
# RoverPursuitVisual_rover2 marker_mount_in_pad (mav.yaml): tag position relative to the pad centre, rover frame
MOUNT_IN_PAD = np.array([0.85, 0.0])
out = {}
for d in sorted(glob.glob(f"{BAG_DIR}/grazer10_*")):
    d = Path(d)
    # skip bags that never closed cleanly (no metadata.yaml) or were only partly copied
    if not (d / "metadata.yaml").exists():
        print(f"{d.name}: SKIPPED (no metadata.yaml)")
        continue
    meta = yaml.safe_load(open(d / "metadata.yaml"))['rosbag2_bagfile_information']
    b0 = meta['starting_time']['nanoseconds_since_epoch'] / 1e9
    dur = meta['duration']['nanoseconds'] / 1e9
    status, tf, gps, odom, att, tagp, static, rover, pix = [], [], [], [], [], [], {}, [], []
    cam = None
    with AnyReader([d]) as r:
        con = [c for c in r.connections
               if c.topic in ('/grazer10/status', '/grazer10/tf', '/grazer10/gps', '/grazer10/odom',
                              TAG_POSE, TAG_PIXEL, CAM_INFO, '/tf_static', '/agents/status')]
        for c, t, raw in r.messages(connections=con):
            tt = t / 1e9
            m = r.deserialize(raw, c.msgtype)
            if c.topic == '/grazer10/status':
                status.append((tt, NAV.get(int(m.nav_state), str(m.nav_state)), int(m.arming_state) == 2))
            elif c.topic.endswith('gps'):
                gps.append((tt, float(m.altitude)))
            elif c.topic == '/agents/status':
                if m.robot_name == ROVER:
                    p, q = m.pose.position, m.pose.orientation
                    rover.append((tt, p.x, p.y, float(Rot.from_quat([q.x, q.y, q.z, q.w]).as_euler('ZYX')[0])))
            elif c.topic == TAG_PIXEL:
                # stamp = image capture time, the age the controller uses
                st_ = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                pix.append((st_, float(m.point.x), float(m.point.y)))
            elif c.topic == TAG_POSE:
                p, q = m.pose.position, m.pose.orientation
                tagp.append((tt, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
            elif c.topic == CAM_INFO:
                if cam is None:
                    cam = dict(w=int(m.width), h=int(m.height), K=[float(k) for k in m.k], D=[float(k) for k in m.d])
            elif c.topic == '/tf_static':
                for x in m.transforms:
                    p, q = x.transform.translation, x.transform.rotation
                    static[(x.header.frame_id, x.child_frame_id)] = ([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
            elif c.topic.endswith('odom'):
                v = m.twist.twist.linear
                odom.append((tt, m.pose.pose.position.z, v.x, v.y, v.z))
            else:
                for x in m.transforms:
                    if x.header.frame_id == 'grazer10/odom' and x.child_frame_id == 'grazer10/base_link':
                        p, q = x.transform.translation, x.transform.rotation
                        tf.append((tt, p.x, p.y, p.z))
                        att.append((tt, q.x, q.y, q.z, q.w))

    if odom:
        o = np.array(sorted(odom))
        z = dict(t=o[:, 0].tolist(), z=o[:, 1].tolist(), src='odom')
        vel = dict(t=o[:, 0].tolist(), vh=np.hypot(o[:, 2], o[:, 3]).tolist(), vz=o[:, 4].tolist(),
                   src='odom twist (EKF velocity)')
    else:
        a = np.array(sorted(tf))
        _, u = np.unique(a[:, 0], return_index=True)
        a = a[u]
        z = dict(t=a[:, 0].tolist(), z=a[:, 3].tolist(), src='odom->base_link TF')
        t = a[:, 0]
        grid = np.arange(t[0], t[-1], 0.05)
        H = 0.1
        P = lambda tq: np.column_stack([np.interp(tq, t, a[:, k]) for k in (1, 2, 3)])
        V = (P(grid + H) - P(grid - H)) / (2 * H)
        vel = dict(t=grid.tolist(), vh=np.hypot(V[:, 0], V[:, 1]).tolist(), vz=V[:, 2].tolist(),
                   src='d/dt of TF position (odom not recorded)')

    # every launch.log line in the window: "<recv epoch> [proc-N] <text>"
    logs = []
    for line in open(d / 'logs' / 'launch.log', errors='ignore'):
        m = re.match(r'(\d+\.\d+) \[([^\]]+)\] (.*)', line.rstrip('\n'))
        if not m:
            continue
        tt = float(m.group(1))
        if not (b0 - 0.5 <= tt <= b0 + dur + 0.5):
            continue
        text = m.group(3)
        lv = LEVEL.search(text)
        # strip the duplicated inner "[LEVEL] [epoch] " prefix, keep [node] name
        text = re.sub(r'^\[(DEBUG|INFO|WARN|ERROR|FATAL)\] \[\d+\.\d+\] ', '', text)
        logs.append((tt, lv.group(1) if lv else 'OUT', m.group(2), text))

    # the controller's own reported inputs, parsed from its log lines
    ctrl = []
    for tt, lv, proc, text in logs:
        if 'RoverPursuit' not in text:
            continue
        f = {}
        for k, pat in [('source', r'source=(\S+)'), ('dxy', r'dxy=([\d.]+)'),
                       ('alt_above_target', r'alt_above_target=([\d.]+)'), ('z_offset', r'z_offset=([\d.]+)'),
                       ('marker_alt', r'marker_alt=(\S+) m'), ('alt_err', r'alt_err=([\d.]+)'),
                       ('yaw_err', r'yaw_err=([-\d.]+)'), ('mav_alt_above_pad', r'mav_alt_above_pad=([\d.]+)'),
                       ('z_cmd', r'z_cmd ([-\d.]+) -> ([-\d.]+)'), ('state', r'state=(\w+)')]:
            mm = re.search(pat, text)
            if mm:
                f[k] = mm.group(mm.lastindex)
        if f:
            ctrl.append((tt, f))

    trees = []
    for tt, lv, proc, text in [l for l in []]:
        pass
    # tree starts/ends anywhere in launch.log (incl. before the bag) so the first frame knows the active tree
    for line in open(d / 'logs' / 'launch.log', errors='ignore'):
        m = re.match(r'(\d+\.\d+) .*bt_navigator\]: (Starting (\S+) tree|Navigation (failed|canceled|succeeded))', line)
        if m:
            trees.append((float(m.group(1)), m.group(3), m.group(4)))

    # ---- tag as seen from the drone ----
    # The tracker gives the tag pose in the camera optical frame. Chain it through the static camera
    # mount (base_link -> hires -> hires_optical) and the drone attitude at that instant, then drop the
    # drone's yaw: the result is in a level frame that turns with the drone's nose (fwd, left, up).
    #   bearing = atan2(left, fwd)   (+ = tag is left of the nose)
    #   horiz   = horizontal distance, range = 3-D distance from the camera, height = drone above tag
    tag = []
    mount = [static.get(('grazer10/base_link', 'grazer10/hires')), static.get(('grazer10/hires', 'grazer10/hires_optical'))]
    if tagp and att and all(mount):
        t_bh, q_bh = mount[0]
        R_bo = Rot.from_quat(q_bh) * Rot.from_quat(mount[1][1])
        at = np.array(sorted(att)); _, u = np.unique(at[:, 0], return_index=True); at = at[u]
        tfa = np.array(sorted(tf))
        for tt, x, y, zc, *q in tagp:
            i = min(max(np.searchsorted(at[:, 0], tt), 0), len(at) - 1)
            R_wb = Rot.from_quat(at[i, 1:5])
            yaw = R_wb.as_euler('ZYX')[0]
            v_lvl = Rot.from_euler('Z', -yaw).apply(R_wb.apply(np.array(t_bh) + R_bo.apply([x, y, zc])))
            fwd, left, up = v_lvl
            j = min(max(np.searchsorted(tfa[:, 0], tt), 0), len(tfa) - 1)
            nav = (tfa[j, 1:3] + R_wb.apply(np.array(t_bh) + R_bo.apply([x, y, zc]))[:2]).tolist()
            tag.append(dict(t=tt, p=[x, y, zc], q=list(q), range=float(np.linalg.norm([x, y, zc])),
                            horiz=float(np.hypot(fwd, left)), bearing=float(np.degrees(np.arctan2(left, fwd))),
                            height=float(-up), nav=nav))
    servo = [lt for lt, lv, proc, text in logs if SERVO_RE.search(text)]
    ovr = []
    for lt, lv, proc, text in logs:
        mm = OVR_RE.search(text)
        if mm:
            ovr.append([lt] + [float(g) for g in mm.groups()])
    # drone pose in the nav (odom) frame at 20 Hz: x, y, yaw
    pose = []
    if att and tf:
        at2 = np.array(sorted(att)); tf2 = np.array(sorted(tf))
        for tq in np.arange(tf2[0, 0], tf2[-1, 0], 0.05):
            i = min(np.searchsorted(tf2[:, 0], tq), len(tf2) - 1); k = min(np.searchsorted(at2[:, 0], tq), len(at2) - 1)
            pose.append((float(tq), float(tf2[i, 1]), float(tf2[i, 2]), float(Rot.from_quat(at2[k, 1:5]).as_euler('ZYX')[0])))

    gps = np.array(sorted(gps))
    out[d.name] = dict(b0=b0, dur=dur, status=status, z=z, vel=vel,
                       gps=dict(t=gps[:, 0].tolist(), h=(gps[:, 1] - gps[0, 1]).tolist()),
                       logs=logs, ctrl=ctrl, trees=sorted(trees),
                       tag=tag, cam=cam, servo=servo, ovr=ovr, rover=sorted(rover), pose=pose,
                       pix=sorted(pix))
    print(f"{d.name}: z from {z['src']}, vel from {vel['src']}, {len(logs)} log lines, "
          f"{len(ctrl)} controller lines, {len(trees)} tree events, {len(status)} status, "
          f"{len(tag)} tag poses, {len(servo)} camera-steering lines")
for v in out.values():
    v['mount_in_pad'] = MOUNT_IN_PAD.tolist()
json.dump(out, open(OUT_JSON, 'w'))
