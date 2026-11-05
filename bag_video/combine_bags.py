#!/usr/bin/env python3
"""Stage 2: concatenate every bag's hires camera into one MP4 with a data
panel under the image (nothing drawn over the camera picture).

Panel = PX4 mode, active BT tree, who is steering (camera tag servo vs GPS
target), the tag's distance/bearing/height as seen from the drone, odom z
(controller input, raw), measured velocity (raw), commanded velocity (not
recorded), the controller's last logged inputs, GPS height (reference), and a
pane with every launch.log line, unfiltered. The tag outline is drawn on the
camera image, reprojected from the tracker's pose (the bags carry no corner
points). Also writes all log lines, untruncated, to a .txt file.
"""
import glob, json, os, sys, textwrap
from bisect import bisect_right
from datetime import datetime, timezone
import numpy as np, cv2, yaml
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage

BAG_ROOT, OUT, DATA = sys.argv[1], sys.argv[2], sys.argv[3]
FPS, W, IMG_H = 10.0, 1280, 768
PANEL_H = 520
HDR_H = 36
H = HDR_H + PANEL_H + IMG_H
TOPIC = '/grazer10/hires/image_raw/compressed'
FONT = cv2.FONT_HERSHEY_SIMPLEX
AGE_GREY = 3.0  # s: controller-status values fade from green to grey over this age
SERVO_HOLD = 0.75  # s: the controller logs its camera override at most every 0.5 s; older than this = GPS target
# RoverPursuitVisual_rover2 (mav.yaml): servo at any altitude (marker_servo_z_threshold 100 m), keeps steering on a
# marker pose up to marker_pose_age_max_sec = 5.0 s old, but only extrapolates the first 1.0 s with rover velocity
SERVO_AGE_MAX = 5.0
SERVO_AGE_COMP = 1.0
TAG_FRESH = 0.15   # s: tag pose this close to the frame time is drawn as a live detection
TAG_STALE = 1.0    # s: older than this the tag is reported as not seen
TAG_SIZE = 0.665   # m: fractal marker outer size, from the tracker's startup log
# RoverPursuitVisual_rover2 centred gate (markerCentredInImage): |pixel.x - image_cx| < frac/2 * image_width, on the
# tracker's RECTIFIED pixel, fail-closed once the pixel is older than pixel_age_max_sec
CENTRE_CX, CENTRE_FRAC, CENTRE_W, PIXEL_AGE_MAX = 634.3, 0.20, 1280.0, 5.0
D = json.load(open(DATA))

WHITE, GREY, GREEN, YELLOW, ORANGE, RED, CYAN = (255, 255, 255), (170, 170, 170), (80, 220, 80), (0, 230, 255), (0, 165, 255), (60, 60, 255), (255, 220, 120)
LV_COLOR = {'ERROR': RED, 'FATAL': RED, 'WARN': ORANGE, 'INFO': WHITE, 'DEBUG': GREY, 'OUT': GREY}


ASCII = {'\u00b0': ' deg', '\u00b1': '+/-', '\u2014': '--', '\u2013': '-', '\u2192': '->', '\u2264': '<=', '\u2265': '>='}


def put(img, s, x, y, color=WHITE, scale=0.5, thick=1):
    # Hershey fonts are ASCII-only: spell out the few symbols the logs use
    for k, r in ASCII.items():
        s = s.replace(k, r)
    s = s.encode('ascii', 'replace').decode()
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def last_before(ts, vals, t):
    i = bisect_right(ts, t) - 1
    return (vals[i], ts[i]) if i >= 0 else (None, None)


def tag_at(v, t):
    """Latest tag sample at or before t, and its age."""
    ts = v.get('_tag_t')
    if ts is None:
        ts = v['_tag_t'] = [g['t'] for g in v.get('tag', [])]
    i = bisect_right(ts, t) - 1
    return (v['tag'][i], t - ts[i]) if i >= 0 else (None, None)


# FRACTAL_3L_6 (aruco fractalmarkerset.cpp): three concentric markers. Corners in the config are +/-1, +/-3/7 and
# +/-1/7 of the outer half-size; bit grids 12x12, 10x10 and 6x6 (+1-cell border each side). The tracker fits ONE
# pose for the whole set from the corners of every level it finds; the bag does not record which levels those were.
LEVELS = [(1.0, 'outer', 14), (3 / 7, 'mid', 12), (1 / 7, 'inner', 8)]
MIN_PX_PER_CELL = 2.0   # below this a level's bits can't be read


def _K(cam):
    K = np.array(cam['K']).reshape(3, 3).copy()
    K[0] *= W / cam['w']; K[1] *= IMG_H / cam['h']            # video frame may be resized from native
    return K


def draw_tag(img, v, t):
    """Outline each fractal level on the raw (distorted) camera image, projected from the tracker's pose with
    the camera's K and distortion. A level is drawn green when it is whole in the frame and big enough to read in
    the tracker's rectified image (so it can contribute to the pose); dim when it can't. Thickest = largest usable
    level. Grey outer outline = last detection, up to TAG_STALE old."""
    from scipy.spatial.transform import Rotation as Rot
    g, age = tag_at(v, t)
    cam = v.get('cam')
    if g is None or cam is None or age > TAG_STALE:
        return
    K, Dist = _K(cam), np.array(cam['D'])
    rv, tv = Rot.from_quat(g['q']).as_rotvec(), np.array(g['p'])
    live = age <= TAG_FRESH
    usable, first = [], True
    for frac, name, cells in LEVELS:
        h = TAG_SIZE / 2 * frac
        sq = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)
        raw, _ = cv2.projectPoints(sq, rv, tv, K, Dist)
        rect, _ = cv2.projectPoints(sq, rv, tv, K, None)            # tracker detects on the rectified image
        raw, rect = raw.reshape(-1, 2), rect.reshape(-1, 2)
        if not np.all(np.isfinite(raw)) or np.abs(raw).max() > 5 * W:
            continue
        side = np.mean(np.linalg.norm(rect - np.roll(rect, 1, 0), axis=1))
        whole = np.all((rect[:, 0] > 1) & (rect[:, 0] < W - 2) & (rect[:, 1] > 1) & (rect[:, 1] < IMG_H - 2))
        ok = whole and side / cells >= MIN_PX_PER_CELL
        if not live:
            if frac == 1.0:
                cv2.polylines(img, [raw.round().astype(np.int32)], True, GREY, 1, cv2.LINE_AA)
            continue
        if ok:
            usable.append(name)
            cv2.polylines(img, [raw.round().astype(np.int32)], True, GREEN, 3 if first else 2, cv2.LINE_AA)
            first = False
        else:
            cv2.polylines(img, [raw.round().astype(np.int32)], True, (90, 90, 90), 1, cv2.LINE_AA)
    c, _ = cv2.projectPoints(np.zeros((1, 3)), rv, tv, K, Dist)
    c = tuple(int(round(k)) for k in c.reshape(2))
    col = GREEN if live else GREY
    cv2.drawMarker(img, c, col, cv2.MARKER_CROSS, 22, 2 if live else 1)
    # same units as the top-left table (fwd / L-R from the drone, m) -- this is the raw detection, the table's
    # Camera row is that detection carried forward to the current frame, so they can differ by a few tenths
    b = np.radians(g['bearing'])
    lat = g['horiz'] * np.sin(b)
    lbl = f"{g['horiz'] * np.cos(b):+.1f} fwd  {abs(lat):.1f}{'L' if lat >= 0 else 'R'}"
    lbl += '' if live else f"  ({age:.1f}s old)"
    h = TAG_SIZE / 2
    top, _ = cv2.projectPoints(np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float), rv, tv, K, Dist)
    top = top.reshape(-1, 2)
    x0, y0 = int(np.clip(top[:, 0].min(), 0, W - 300)), int(top[:, 1].min()) - 10
    cv2.putText(img, lbl, (x0, max(y0, 20)), FONT, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, lbl, (x0, max(y0, 20)), FONT, 0.7, col, 2, cv2.LINE_AA)


def _interp(ts, xs, t):
    return float(np.interp(t, ts, xs))


def tag_estimates(v, t):
    """Where each source puts the TAG at time t, relative to the drone (fwd, left) in metres.
    camera: the latest tracker pose, fixed in the nav frame at detection, carried forward with the rover's velocity
            for at most SERVO_AGE_COMP s (as the controller does), valid while younger than SERVO_AGE_MAX.
    gps:    where the rover's GPS puts the tag NOW: rover pose + the rover->GPS-target offset measured on the
            override lines where the logged GPS target had just refreshed (same bag only), + the tag mount offset.
            The controller's own GPS target is this point as of its last plan refresh (every ~1.5-4 s), so while the
            rover moves the target it actually steers to trails this by up to speed x refresh interval."""
    ps = v.get('pose') or []
    if not ps:
        return None, None
    if '_pose' not in v:
        v['_pose'] = np.array(ps)
        r = np.array(v['rover']) if v.get('rover') else None
        v['_rover'] = r
        if r is not None and len(r) > 2:
            v['_rvel'] = np.c_[np.gradient(r[:, 1], r[:, 0]), np.gradient(r[:, 2], r[:, 0])]
        o = np.array(v['ovr']) if v.get('ovr') and r is not None else None
        if o is not None:
            # The GPS target only moves when the plan is refreshed (~every 1.5-4 s) and is held in between, so
            # the rover->target offset is only meaningful on lines where the logged GPS target just jumped.
            jump = np.r_[True, np.hypot(np.diff(o[:, 1]), np.diff(o[:, 2])) > 0.05]
            o = o[jump]
            v['_ovr_d'] = np.c_[o[:, 0], o[:, 1] - np.interp(o[:, 0], r[:, 0], r[:, 1]),
                                o[:, 2] - np.interp(o[:, 0], r[:, 0], r[:, 2])]
    P = v['_pose']
    dx_, dy_, yaw = _interp(P[:, 0], P[:, 1], t), _interp(P[:, 0], P[:, 2], t), _interp(P[:, 0], np.unwrap(P[:, 3]), t)
    cs, sn = np.cos(yaw), np.sin(yaw)
    rel = lambda x, y: (cs * (x - dx_) + sn * (y - dy_), -sn * (x - dx_) + cs * (y - dy_))
    r = v.get('_rover')
    mount = np.array(v.get('mount_in_pad', [0.85, 0.0]))
    cam = gps = None
    g, age = tag_at(v, t)
    if g is not None and age <= SERVO_AGE_MAX and 'nav' in g:
        x, y = g['nav']
        if r is not None and '_rvel' in v:
            k = min(age, SERVO_AGE_COMP)
            x += _interp(r[:, 0], v['_rvel'][:, 0], t) * k; y += _interp(r[:, 0], v['_rvel'][:, 1], t) * k
        cam = rel(x, y) + (age,)
    if r is not None and '_ovr_d' in v:
        od = v['_ovr_d']
        ryaw = _interp(r[:, 0], r[:, 3], t)
        px = _interp(r[:, 0], r[:, 1], t) + _interp(od[:, 0], od[:, 1], t)
        py = _interp(r[:, 0], r[:, 2], t) + _interp(od[:, 0], od[:, 2], t)
        tx = px + np.cos(ryaw) * mount[0] - np.sin(ryaw) * mount[1]
        ty = py + np.sin(ryaw) * mount[0] + np.cos(ryaw) * mount[1]
        gps = rel(tx, ty)
    return cam, gps


def age_colour(age):
    """Tag-steering colour by marker age: green while fresh (<= SERVO_AGE_COMP, the part the controller
    extrapolates), then yellow -> red as it approaches SERVO_AGE_MAX, where the controller drops to GPS."""
    if age is None or age <= SERVO_AGE_COMP:
        return GREEN
    a = min((age - SERVO_AGE_COMP) / (SERVO_AGE_MAX - SERVO_AGE_COMP), 1.0)
    mid = 0.5
    lo, hi, f = (GREEN, YELLOW, a / mid) if a < mid else (YELLOW, RED, (a - mid) / mid)
    return tuple(int(round(p + (q - p) * f)) for p, q in zip(lo, hi))


NAV_DOWN_RE = ('IS DOWN after not receiving a heartbeat', 'navigation_bringup', 'process has died')


def nav_state(v, t):
    """None if a navigation task is running at t, else why not: the last BT tree event ended a task, or the
    navigation stack was declared down / its process died (until the next tree starts)."""
    if '_down' not in v:
        down = []
        for lt, lv, proc, text in v['logs']:
            if NAV_DOWN_RE[0] in text:
                down.append((lt, 'nav stack DOWN'))
            elif NAV_DOWN_RE[1] in text and NAV_DOWN_RE[2] in text:
                down.append((lt, 'nav process CRASHED'))
        v['_down'] = down
    ev, et = last_before([e[0] for e in v['trees']], v['trees'], t)
    dn, dt_ = last_before([d[0] for d in v['_down']], v['_down'], t)
    # a crash also cancels the task a moment later: report the crash, not the cancel
    if dn is not None and (et is None or dt_ >= et or (not ev[1] and et - dt_ < 2.0)):
        return dn[1]
    if ev is not None and not ev[1]:
        return f"task {ev[2]}"
    # only the rover-pursuit landing trees run RoverPursuit; anything else (idle hover, takeoff...) isn't GPS/tag
    if ev is not None and not any(k in ev[1] for k in ('land_on', 'pursu')):
        return f"{ev[1]} task"
    return None


def steering(v, t):
    """Who sets the horizontal target at time t: (label, colour, source key, tag age or None)."""
    st, _ = last_before([s[0] for s in v['status']], v['status'], t)
    sv, _ = last_before(v.get('servo', []), v.get('servo', []), t)
    if not st or st[1] != 'OFFBOARD':
        return 'PX4/pilot', GREY, None, None
    why = nav_state(v, t)
    if why is not None:
        return f"none ({why})", RED, None, None
    if sv is not None and t - sv <= SERVO_HOLD:
        g, age = tag_at(v, t)
        return 'Tag', age_colour(age), 'cam', age
    return 'GPS', ORANGE, 'gps', None


def draw_steering(img, v, t):
    """Narrow box in the top-left of the camera image: who is steering, then a small table of where each
    source puts the tag relative to the drone. The steering source's row is coloured, the other is grey."""
    lbl, col, src, sage = steering(v, t)
    cam, gps = tag_estimates(v, t)
    big, sm = 0.85, 0.55
    head = f"Steering - {lbl}" + (f"  {sage:.1f}s" if src == 'cam' and sage is not None and sage > SERVO_AGE_COMP else '')
    # column x offsets inside the box
    X = dict(name=0, fwd=95, lr=165, dist=235, age=295)
    rows = [(['Tag pos', 'fwd', 'L/R', 'dist', 'age'], GREY)]
    for key, e, name in (('cam', cam, 'Camera'), ('gps', gps, 'GPS*')):
        c = col if key == src else GREY
        if e is None:
            rows.append(([name, '-', '-', '-', ''], c))
        else:
            lr = f"{abs(e[1]):.1f}{'L' if e[1] >= 0 else 'R'}"
            ag = f"{e[2]:.1f}" if key == 'cam' else ''
            rows.append(([name, f"{e[0]:+.1f}", lr, f"{np.hypot(e[0], e[1]):.1f}", ag], c))
    if cam is not None and gps is not None:
        rows.append((['Diff', '', '', f"{np.hypot(cam[0] - gps[0], cam[1] - gps[1]):.1f}", ''], GREY))
    rows.append((['m, from drone  *rover now', '', '', '', ''], (110, 110, 110)))
    (hw, hh), _ = cv2.getTextSize(head, FONT, big, 2)
    rh = 24
    bw = max(hw, X['age'] + 40) + 20
    bh = hh + 22 + rh * len(rows)
    cv2.rectangle(img, (8, 8), (8 + bw, 8 + bh), (0, 0, 0), -1)
    cv2.putText(img, head, (18, 18 + hh), FONT, big, col, 2, cv2.LINE_AA)
    y = 18 + hh + 8
    for cells, c in rows:
        y += rh
        for (k, x), txt in zip(X.items(), cells):
            if txt:
                cv2.putText(img, txt, (18 + x, y), FONT, sm, c, 1 if c == GREY or c == (110, 110, 110) else 2, cv2.LINE_AA)


def draw_centre_band(img, v, t):
    """The controller's 'centred' strip, drawn as a box: rectified x in cx +/- frac/2*width over the full height,
    pushed through the lens distortion onto the raw image. Green = last tag pixel inside it, red = outside,
    both fading to grey as that pixel ages to PIXEL_AGE_MAX (after which the gate fails closed: plain grey)."""
    cam = v.get('cam')
    if cam is None:
        return
    if '_band' not in v:
        K = _K(cam)
        half = CENTRE_FRAC * 0.5 * CENTRE_W
        edges = []
        for x in (CENTRE_CX - half, CENTRE_CX + half):
            ys = np.linspace(0, cam['h'], 40)
            und = np.c_[(x - K[0, 2]) / K[0, 0] * np.ones_like(ys), (ys * IMG_H / cam['h'] - K[1, 2]) / K[1, 1],
                        np.ones_like(ys)]
            p, _ = cv2.projectPoints(und, np.zeros(3), np.zeros(3), K, np.array(cam['D']))
            edges.append(p.reshape(-1, 2))
        v['_band'] = np.vstack([edges[0], edges[1][::-1]]).round().astype(np.int32)
        v['_pix_t'] = [p[0] for p in v.get('pix', [])]
    i = bisect_right(v['_pix_t'], t) - 1
    if i < 0:
        col, thick = (90, 90, 90), 1
    else:
        _, u, _ = v['pix'][i]
        age = t - v['_pix_t'][i]
        if age > PIXEL_AGE_MAX:
            col, thick = (90, 90, 90), 1
        else:
            base = GREEN if abs(u - CENTRE_CX) < CENTRE_FRAC * 0.5 * CENTRE_W else RED
            a = min(max(age / PIXEL_AGE_MAX, 0.0), 1.0)
            col = tuple(int(round(p + (q - p) * a)) for p, q in zip(base, (90, 90, 90)))
            thick = 2
    cv2.polylines(img, [v['_band']], True, col, thick, cv2.LINE_AA)


def compose(img, v, t):
    """Camera image with the per-frame overlays (tag outline, steering), on a copy so held frames stay clean."""
    out = img.copy()
    draw_centre_band(out, v, t)
    draw_tag(out, v, t)
    draw_steering(out, v, t)
    return out


def render_panel(v, t, b0):
    p = np.full((PANEL_H, W, 3), 18, np.uint8)
    rel = t - b0
    y = 24
    # PX4 mode
    st, _ = last_before([s[0] for s in v['status']], v['status'], t)
    if st:
        mode, armed = st[1], st[2]
        put(p, 'PX4 mode:', 12, y, GREY)
        put(p, f"{mode}  {'ARMED' if armed else 'disarmed'}", 130, y, GREEN if mode == 'OFFBOARD' else WHITE, 0.6, 2)
    # active tree
    ev, et = last_before([e[0] for e in v['trees']], v['trees'], t)
    put(p, 'BT tree:', 640, y, GREY)
    if ev is None:
        put(p, '(none logged yet)', 730, y, GREY, 0.6)
    elif ev[1]:
        put(p, f"{ev[1]}  (since t{et - b0:+.1f}s)", 730, y, YELLOW, 0.6, 2)
    else:
        prev = [e for e in v['trees'] if e[0] <= et and e[1]]
        name = prev[-1][1] if prev else '?'
        put(p, f"{name} ended: {ev[2]} at t{et - b0:+.1f}s", 730, y, ORANGE, 0.6, 2)
    y += 30
    # who is steering the horizontal target: the controller logs a marker-servo override line
    # (throttled to 2 Hz) whenever the camera replaces the GPS/rover-odom target
    put(p, 'Steering:', 12, y, GREY)
    lbl, col, src, sage = steering(v, t)
    lbl = lbl + (f'  (tag {sage:.1f} s old)' if src == 'cam' and sage is not None else '')
    put(p, lbl, 130, y, col, 0.6, 2)
    y += 30
    # the tag as seen from the drone (level frame that turns with the nose)
    put(p, 'Tag from drone:', 12, y, GREY)
    g, age = tag_at(v, t)
    if g is None:
        put(p, 'not seen yet in this bag', 160, y, GREY, 0.6)
    else:
        side = 'LEFT' if g['bearing'] >= 0 else 'RIGHT'
        s = (f"horiz {g['horiz']:5.2f} m   bearing {abs(g['bearing']):4.1f} deg {side} of nose   "
             f"3-D {g['range']:5.2f} m   height {g['height']:+5.2f} m (unverified: TF mount pitch)")
        if age <= TAG_STALE:
            put(p, s, 160, y, CYAN, 0.6, 2)
        else:
            put(p, f"NOT SEEN for {age:4.1f} s   (last: {s})", 160, y, GREY, 0.5)
    y += 30
    # odom z (controller input)
    zt, zz = v['z']['t'], v['z']['z']
    zval, _ = last_before(zt, zz, t)
    put(p, f"odom z (EKF local, controller input; src {v['z']['src']}):", 12, y, GREY)
    if zval is not None:
        put(p, f"{zval:+8.2f} m", 520, y, CYAN, 0.6, 2)
    g, _ = last_before(v['gps']['t'], v['gps']['h'], t)
    put(p, 'GPS height above arm point (reference only):', 700, y, GREY)
    if g is not None:
        put(p, f"{g:5.1f} m", 1110, y, WHITE, 0.6)
    y += 30
    # measured / commanded velocity
    vi = bisect_right(v['vel']['t'], t) - 1
    put(p, f"measured vel ({v['vel']['src']}):", 12, y, GREY)
    if vi >= 0:
        put(p, f"horiz {v['vel']['vh'][vi]:5.2f} m/s   vert {v['vel']['vz'][vi]:+5.2f} m/s", 520, y, CYAN, 0.6, 2)
    y += 30
    put(p, 'commanded vel:', 12, y, GREY)
    put(p, 'NOT RECORDED in these bags (setpoint topics missing from bag topic list)', 160, y, ORANGE, 0.5)
    y += 30
    # controller's own logged inputs: latest value of every field, with its age
    put(p, 'RoverPursuit status (last value it logged for each field; colour = age: green just logged -> grey 3 s+ old):',
        12, y, GREY)
    latest = {}
    for ct, f in v['ctrl']:
        if ct > t:
            break
        for k, val in f.items():
            latest[k] = (val, ct)
    order = ('source', 'state', 'dxy', 'alt_above_target', 'z_offset', 'marker_alt', 'mav_alt_above_pad',
             'alt_err', 'yaw_err', 'z_cmd')
    items = [(f"{k}={latest[k][0]}", t - latest[k][1]) for k in order if k in latest]
    if not items:
        put(p, '(nothing logged yet)', 12, y + 22, GREY, 0.45)
    x, yy0 = 12, y + 22
    for s, age in items:
        # linear fade: age 0 -> green, age >= AGE_GREY -> grey
        a = min(max(age / AGE_GREY, 0.0), 1.0)
        col = tuple(int(round(g + (q - g) * a)) for g, q in zip(GREEN, GREY))
        tw = cv2.getTextSize(s, FONT, 0.5, 1)[0][0]
        if x + tw > W - 12:
            x, yy0 = 12, yy0 + 22
        put(p, s, x, yy0, col, 0.5, 2 if a < 1.0 else 1)
        x += tw + 26
    y = yy0
    y += 16
    cv2.line(p, (0, y), (W, y), (70, 70, 70), 1)
    # unfiltered log pane: newest at the bottom, long lines wrapped
    rows = []
    for lt, lv, proc, text in v['logs']:
        if lt > t:
            break
        head = f"t{lt - b0:+7.2f} {lv:5s} "
        wrapped = textwrap.wrap(text, 150) or ['']
        for k, part in enumerate(wrapped):
            rows.append(((head if k == 0 else ' ' * len(head)) + part, LV_COLOR.get(lv, WHITE), t - lt < 1.0))
    nrows = (PANEL_H - y - 8) // 19
    yy = y + 19
    for s, col, fresh in rows[-nrows:]:
        if fresh:
            cv2.rectangle(p, (6, yy - 14), (W - 6, yy + 5), (55, 55, 55), -1)
        put(p, s, 12, yy, col, 0.45)
        yy += 19
    return p


def header(img, name, bi, n, t, b0):
    wall = datetime.fromtimestamp(t, timezone.utc)
    cv2.rectangle(img, (0, 0), (W, HDR_H), (0, 0, 0), -1)
    put(img, f"{name}  (bag {bi}/{n})    {wall:%Y-%m-%d %H:%M:%S}.{wall.microsecond // 100000} UTC (grazer10 clock)    t+{t - b0:6.1f} s",
        12, 24, WHITE, 0.6, 2)


# only the bags overlay_data.py accepted (it skips incomplete ones)
bags = sorted(os.path.join(BAG_ROOT, n) for n in D)
writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'avc1'), FPS, (W, H))
if not writer.isOpened():
    sys.exit('could not open H.264 writer')
txt = open(os.path.splitext(OUT)[0] + '_logs.txt', 'w')
written = 0          # frames written so far = video position, for bookmarks
crashes = []         # (video seconds, notes) for every nav stack crash

for bi, d in enumerate(bags, 1):
    name = os.path.basename(d)
    v = D[name]
    b0, dur = v['b0'], v['dur']
    txt.write(f"===== bag {bi}/{len(bags)} {name}  start {datetime.fromtimestamp(b0, timezone.utc):%Y-%m-%d %H:%M:%S} UTC  ({dur:.0f} s)\n")
    for lt, lv, proc, text in v['logs']:
        txt.write(f"t{lt - b0:+8.2f}  {lv:5s} [{proc}] {text}\n")
    card = np.zeros((H, W, 3), np.uint8)
    for k, s in enumerate([f'Bag {bi} of {len(bags)}: {name}',
                           f'start {datetime.fromtimestamp(b0, timezone.utc):%Y-%m-%d %H:%M:%S} UTC   duration {dur:.0f} s',
                           'times = grazer10 (RB5) system clock',
                           f"altitude/velocity source: {v['z']['src']} / {v['vel']['src']}"]):
        (tw, _), _ = cv2.getTextSize(s, FONT, 0.9, 2)
        put(card, s, (W - tw) // 2, H // 2 - 60 + k * 45, WHITE, 0.9, 2)
    for _ in range(int(2 * FPS)):
        writer.write(card)
    written += int(2 * FPS)
    # nav stack crashes in this bag: the lifecycle manager declaring a server down (the moment navigation stops);
    # a process death within a few seconds is the same crash
    events, last = [], None
    for lt, lv, proc, text in v['logs']:
        hit = NAV_DOWN_RE[0] in text or (NAV_DOWN_RE[1] in text and NAV_DOWN_RE[2] in text)
        if hit:
            if last is None or lt - last > 10.0:
                events.append([lt, []])
            events[-1][1].append(f"t{lt - b0:+.2f} s  {text[:200]}")
            last = lt

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=d, storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    r.set_filter(rosbag2_py.StorageFilter(topics=[TOPIC]))
    grid_t, last_img, n = None, None, 0
    while r.has_next():
        _, data, t_ns = r.read_next()
        img = cv2.imdecode(np.frombuffer(bytes(deserialize_message(data, CompressedImage).data), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (W, IMG_H), interpolation=cv2.INTER_AREA)
        t = t_ns / 1e9
        if grid_t is None:
            grid_t = t
            for lt, lines in events:   # video time = frames before this bag's first image + offset into the bag
                crashes.append((written / FPS + max(lt - t, 0.0),
                                f"Bag {bi}/{len(bags)} ({name[-8:]}) t+{lt - b0:.2f} s, "
                                f"{datetime.fromtimestamp(lt, timezone.utc):%Y-%m-%d %H:%M:%S} UTC (grazer10 clock).\n"
                                + "\n".join(lines)))
        # fixed 10 fps grid: hold the previous camera image across gaps, but keep the panel live
        while last_img is not None and grid_t + 1e-6 < t:
            frame = np.vstack([np.zeros((HDR_H, W, 3), np.uint8), render_panel(v, grid_t, b0), compose(last_img, v, grid_t)])
            header(frame, name, bi, len(bags), grid_t, b0)
            writer.write(frame); grid_t += 1 / FPS; n += 1; written += 1
        frame = np.vstack([np.zeros((HDR_H, W, 3), np.uint8), render_panel(v, t, b0), compose(img, v, t)])
        header(frame, name, bi, len(bags), t, b0)
        writer.write(frame); grid_t += 1 / FPS; n += 1; written += 1
        last_img = img
    print(f'{name}: {n} frames', flush=True)

writer.release()
txt.close()

# Bookmarks for mavTerminal's Bags tab (<video stem>_bookmarks.json, same schema as bag_tab.save_bookmarks).
# Merged, never overwritten: hand-made bookmarks stay; an existing "Nav stack crash n" is moved to the new time.
bpath = os.path.splitext(OUT)[0] + '_bookmarks.json'
try:
    bdata = json.load(open(bpath))
except (OSError, ValueError):
    bdata = {'video': os.path.basename(OUT), 'bookmarks': []}
bms = [b for b in bdata.get('bookmarks', []) if not b.get('title', '').lower().startswith('nav stack crash ')]
for k, (vt, notes) in enumerate(sorted(crashes), 1):
    bms.append({'title': f'Nav stack crash {k}', 't_s': round(vt, 3), 'notes': notes})
bdata['bookmarks'] = sorted(bms, key=lambda b: b['t_s'])
with open(bpath + '.tmp', 'w') as f:
    json.dump(bdata, f, indent=2)
    f.write('\n')
os.replace(bpath + '.tmp', bpath)
print(f'{len(crashes)} nav stack crash bookmark(s) -> {os.path.basename(bpath)}')
print('done ->', OUT)
