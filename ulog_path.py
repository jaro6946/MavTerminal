#!/usr/bin/env python3
"""ulog_path.py -- where the vehicle actually went, in 3D, coloured by time.

Every other plot in this toolkit is a time series: it answers "what was this
value at 4.2 minutes".  This one answers the question a time series cannot --
"where was the vehicle when that happened" -- by putting the flight in the space
it happened in and using COLOUR for time instead of an axis.

Three views of the same track:

  1. a 3D path  -- east / north / up, coloured dark-to-bright with time
  2. a top-down plan view -- the ground track, equal aspect, and the place a
     satellite image goes (see "Basemaps" below)
  3. a colour bar -- the legend for time, since time is no longer an axis

The frame: re-anchored, not raw local position
----------------------------------------------
The obvious thing is to plot `vehicle_local_position` x/y/z straight.  Don't.
PX4 (the flight stack) publishes ONE estimator instance's local position --
whichever the EKF (Extended Kalman Filter) selector currently has primary -- and
every instance anchors its own local origin.  A handover therefore republishes
the same physical point against a different datum, and the path gets a step in it
that the vehicle never flew.  Measured on d05a88e3: `-z` jumps 22.76 m at a
handover while the vehicle is holding position.

So each sample is re-anchored onto ONE fixed origin before it is plotted, using
the reference origin published alongside it:

    east  = (ref_lon_i - lon0) * m_per_deg_lon + y_i     (y is East in NED)
    north = (ref_lat_i - lat0) * m_per_deg_lat + x_i     (x is North in NED)
    up    = (ref_alt_i - z_i) - alt0                     (z is DOWN-positive)

`ref_alt_i - z_i` is altitude above mean sea level, which is origin-independent,
so subtracting one fixed `alt0` gives a height that survives a handover.  On
d05a88e3 that takes the worst vertical step from 22.76 m to 0.88 m.  It does NOT
fix a genuine estimator reset -- SquareWaypointMission_1 still steps 59 m,
because there the filter really did reset its height state -- and those get
marked rather than smoothed, since a reset is a finding, not a rendering
artefact.

The by-product is that the frame is GEOREFERENCED: the origin is a real WGS84
(World Geodetic System 1984) latitude and longitude, and every point on the plot
is a known number of metres east and north of it.  That is what makes a satellite
image possible at all.

Basemaps (the satellite image)
------------------------------
This tool does not fetch tiles.  It has no network dependency, it runs on logs
pulled off a bench, and a plot that silently phones a tile server is not a thing
you want in a flight-test loop.

What it does instead: if a georeferenced image is on disk, it is drawn under the
ground track in the plan view and on the floor of the 3D box.  Drop a pair of
files into ~/.logGraph/basemaps (or $LOGGRAPH_BASEMAP_DIR):

    site.png
    site.json     {"image": "site.png", "bounds": [south, west, north, east]}

`bounds` is in degrees, the same order Leaflet and folium use.  Any image library
matplotlib can read will do.  The one whose bounds contain the log's origin wins.

When there is no basemap, the plan view says so and prints the origin's latitude
and longitude, which is the thing you would need to go and grab a tile for.  The
projection, the extent maths and both draw paths are already here -- adding a
downloader later is a function that returns (rgb, bounds), nothing more.

Acronyms: EKF = extended Kalman filter, ENU = East-North-Up, NED =
North-East-Down, AMSL = above mean sea level, GPS = Global Positioning System,
WGS84 = World Geodetic System 1984.
"""
import glob
import json
import os

import numpy as np

from ulog_common import (C_BAD, C_GRID, C_INK, C_MUTED, C_SURFACE, PlotCtx,
                         Series, _get, _time_min, add_view_navigation,
                         armed_spans, check_panel, duration_min, field,
                         has_topic, mode_changes, mode_color, mode_key,
                         nav_state_name, resample_to, view_nav_hint)

PATH_TOPICS = [
    "vehicle_local_position",           # the track, and the reference origin
    "vehicle_local_position_setpoint",  # where it was ASKED to be
    "vehicle_gps_position",             # the independent, unfiltered track
    "home_position",
    "actuator_armed",
    "vehicle_status",                   # mode-change markers
]

# --- colour -----------------------------------------------------------------
# Time is the colour axis, so it gets a perceptually uniform sequential map:
# viridis is monotonic in lightness, which is what makes "later" readable as
# "brighter" without a lookup, and it survives being printed in grey.
TIME_CMAP = "viridis"
C_TRACK_KEY = "#3b7f8c"     # a mid-viridis tone, for the checkbox label only
C_SETPOINT = "#d81b60"      # magenta -- commanded, deliberately not on the ramp
C_GPS = "#8d6e63"           # brown -- raw, unfiltered, a supporting witness
C_SHADOW = "#9a9a94"
C_START = "#2e7d32"
C_END = "#c0392b"
C_HOME = "#20222b"

# Draw at most this many points.  A 19 k-sample track is 19 k line segments,
# and mplot3d re-sorts every segment by depth on EVERY redraw -- including the
# ones a rotate-drag fires continuously.  6 k keeps a rotate interactive; the
# decimation is a plain stride, so the shape is preserved and only the sample
# density drops.
MAX_POINTS = 6000

# Break the line across a sample gap longer than this many times the median
# spacing.  A dropout in vehicle_local_position is not a straight-line flight
# between the samples either side of it, and drawing one invents a path.
GAP_FACTOR = 8.0

# Below this the vertical extent is stretched so the profile is readable, and
# the factor is stated on the figure.  A 300 m square flown at 30 m altitude is
# a flat ribbon at true scale, and "true scale" that shows nothing is not more
# honest than a labelled exaggeration.
MIN_VERT_FRAC = 0.35

BASEMAP_ENV = "LOGGRAPH_BASEMAP_DIR"
BASEMAP_HOME = os.path.expanduser("~/.logGraph/basemaps")


# --- the geographic frame ----------------------------------------------------

def _meters_per_degree(lat_deg):
    """(m per degree latitude, m per degree longitude) at this latitude.

    The standard WGS84 series expansion.  Good to well under a metre over the
    few kilometres a multirotor log covers, and it is a LOCAL TANGENT PLANE --
    the same thing a web-mercator tile approximates over one screen, which is
    why a basemap can be laid on it with a plain affine extent.
    """
    phi = np.radians(float(lat_deg))
    m_lat = 111132.92 - 559.82 * np.cos(2 * phi) + 1.175 * np.cos(4 * phi)
    m_lon = 111412.84 * np.cos(phi) - 93.5 * np.cos(3 * phi)
    return m_lat, m_lon


class GeoFrame:
    """A local ENU tangent plane anchored on one WGS84 point.

    `valid` is False for a log with no global reference at all -- a HITL
    (Hardware In The Loop) log flown without GPS, say.  The plot still works:
    the axes become raw local metres and the basemap is skipped, and the caller
    says so rather than quietly implying a geographic frame it does not have.
    """

    def __init__(self, lat0=0.0, lon0=0.0, alt0=0.0, valid=True):
        self.lat0, self.lon0, self.alt0 = float(lat0), float(lon0), float(alt0)
        self.valid = bool(valid)
        self.m_lat, self.m_lon = _meters_per_degree(lat0 if valid else 0.0)

    def to_local(self, lat, lon):
        return ((np.asarray(lon, float) - self.lon0) * self.m_lon,
                (np.asarray(lat, float) - self.lat0) * self.m_lat)

    def to_geo(self, east, north):
        return (self.lat0 + np.asarray(north, float) / self.m_lat,
                self.lon0 + np.asarray(east, float) / self.m_lon)


def geo_frame(ulog):
    """The frame everything is plotted in: the log's FIRST valid global origin.

    First rather than most-common on purpose.  The origins move (that is the
    whole reason for re-anchoring), and picking one arbitrarily would put the
    plot in a frame that no part of the log actually used; the first is the one
    the flight started in, so "0, 0" is where the vehicle was sitting.
    """
    d = _get(ulog, "vehicle_local_position")
    if d is None:
        return GeoFrame(valid=False)
    lat = np.asarray(d.data.get("ref_lat", []), dtype=float)
    lon = np.asarray(d.data.get("ref_lon", []), dtype=float)
    alt = np.asarray(d.data.get("ref_alt", []), dtype=float)
    if lat.size == 0:
        return GeoFrame(valid=False)
    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt) & (np.abs(lat) > 1e-6)
    if not ok.any():
        return GeoFrame(valid=False)
    i = int(np.flatnonzero(ok)[0])
    return GeoFrame(lat[i], lon[i], alt[i], valid=True)


# --- tracks ------------------------------------------------------------------

class Track:
    """One path through the frame: time, east, north, up, all in metres."""

    def __init__(self, tid, label, color, t, e, n, u, visible=True, lw=2.0,
                 ls="-", ramp=False, breaks=()):
        self.id = tid
        self.label = label
        self.color = color
        self.t, self.e, self.n, self.u = t, e, n, u
        self.visible = visible
        self.lw, self.ls = lw, ls
        self.ramp = ramp          # coloured by time rather than one flat colour
        # Instants the line must NOT be drawn across -- see _reset_times.
        self.breaks = np.asarray(breaks, dtype=float)
        self.artists = []

    @property
    def n_points(self):
        return int(np.isfinite(self.e).sum())


def _decimate(*arrays):
    """Stride every array down to MAX_POINTS, keeping the last sample.

    Keeping the last sample matters: the end marker is placed on it, and a
    stride that drops it would put "landed here" at wherever the stride
    happened to stop.
    """
    n = len(arrays[0])
    if n <= MAX_POINTS:
        return arrays
    step = int(np.ceil(n / MAX_POINTS))
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return tuple(np.asarray(a)[idx] for a in arrays)


def _anchor(geo, ref_lat, ref_lon, ref_alt, x, y, z):
    """Re-anchor NED local position onto the frame's fixed origin.

    This is the function the module docstring is about.  Where the reference is
    missing (before the estimator has a global origin) the sample falls back to
    the raw local value, which is correct for the pre-flight portion of a log:
    the vehicle is sitting at the origin and both frames agree there.
    """
    e = np.asarray(y, float).copy()
    n = np.asarray(x, float).copy()
    u = -np.asarray(z, float).copy()
    if not geo.valid:
        return e, n, u
    ok = (np.isfinite(ref_lat) & np.isfinite(ref_lon) & np.isfinite(ref_alt)
          & (np.abs(np.asarray(ref_lat, float)) > 1e-6))
    de, dn = geo.to_local(ref_lat, ref_lon)
    e[ok] += de[ok]
    n[ok] += dn[ok]
    # AMSL minus one fixed origin altitude -- the term that survives a handover.
    u[ok] = (np.asarray(ref_alt, float)[ok] - np.asarray(z, float)[ok]) - geo.alt0
    return e, n, u


def _reset_times(ulog):
    """When the estimator reset its position or height state, in minutes.

    Re-anchoring fixes the frame, not the state.  When a filter RESETS -- throws
    its position away and re-initialises it -- the vehicle really did not move,
    but the estimate really did jump, and no choice of origin makes those two
    facts agree.  Measured on SquareWaypointMission_1: a 59 m vertical step at
    1.276 min, exactly on a z_reset_counter increment.

    So the line is broken at every reset and the jump is marked.  Drawing
    through one would put a 59 m climb on the plot that no motor ever flew.
    """
    d = _get(ulog, "vehicle_local_position")
    if d is None:
        return np.empty(0)
    t = _time_min(ulog, d)
    out = []
    for key in ("xy_reset_counter", "z_reset_counter"):
        c = np.asarray(d.data.get(key, []), dtype=float)
        if c.size == t.size and c.size > 1:
            out.append(t[np.flatnonzero(np.diff(c) != 0) + 1])
    if not out:
        return np.empty(0)
    return np.unique(np.concatenate(out))


def _ramp_window(ulog, track):
    """(lo, hi) for the time colour ramp -- the ARMED window when there is one.

    Normalising over the whole log wastes most of the ramp on the ground.  On
    Hexi_log100 the vehicle is armed for 4.9 of 11.5 minutes and parked at the
    origin for the rest, so a full-log ramp renders the entire flight in one
    shade of green and the colour axis says nothing.  The pre- and post-flight
    samples clip to the ramp's end colours, which is the right reading: they are
    all one place anyway.
    """
    lo, hi = float(np.nanmin(track.t)), float(np.nanmax(track.t))
    spans = armed_spans(ulog)
    if not spans:
        return lo, hi, False
    a, b = spans[0][0], spans[-1][1]
    if not np.isfinite(a) or not np.isfinite(b) or (b - a) < 0.02 * (hi - lo):
        return lo, hi, False
    return float(a), float(b), True


def _positions_at(track, times):
    """The track's own sample at or just after each instant.

    Interpolating would be wrong here: these instants are exactly the ones the
    line is BROKEN at, and interpolating across a break returns a point halfway
    along a jump that never happened.
    """
    ok = np.isfinite(track.e) & np.isfinite(track.n) & np.isfinite(track.u)
    if not ok.any() or len(times) == 0:
        return np.empty((0, 3))
    t, e, n, u = track.t[ok], track.e[ok], track.n[ok], track.u[ok]
    i = np.clip(np.searchsorted(t, np.asarray(times, float)), 0, t.size - 1)
    return np.column_stack([e[i], n[i], u[i]])


def _fused_track(ulog, geo):
    """The published estimate: the path the vehicle believed it flew."""
    d = _get(ulog, "vehicle_local_position")
    if d is None:
        return None
    D = d.data
    t = _time_min(ulog, d)
    x, y, z = (np.asarray(D.get(k, []), float) for k in ("x", "y", "z"))
    if x.size == 0:
        return None
    valid = np.asarray(D.get("xy_valid", np.ones_like(x)), float) > 0.5
    e, n, u = _anchor(geo, np.asarray(D.get("ref_lat", np.zeros_like(x)), float),
                      np.asarray(D.get("ref_lon", np.zeros_like(x)), float),
                      np.asarray(D.get("ref_alt", np.zeros_like(x)), float),
                      x, y, z)
    # An invalid xy sample is not a position, so it becomes a BREAK in the line
    # rather than a point at the last believed location.
    bad = ~(valid & np.isfinite(e) & np.isfinite(n) & np.isfinite(u))
    e[bad] = n[bad] = u[bad] = np.nan
    t, e, n, u = _decimate(t, e, n, u)
    return Track("fused", "estimate (coloured by time)", C_TRACK_KEY, t, e, n, u,
                 visible=True, lw=2.0, ramp=True, breaks=_reset_times(ulog))


def _setpoint_track(ulog, geo):
    """Where the controller was ASKED to be.

    Setpoints live in the same local frame as the estimate, so they are
    re-anchored with the reference resampled onto their own timestamps -- using
    the estimate's reference at the wrong instant would offset the commanded
    path by an origin move and invent a tracking error.

    Most of these samples are NaN by design: PX4 publishes a position setpoint
    only in the modes that have one, and fills the rest with NaN (24% finite on
    SquareWaypointMission_1).  The NaNs break the line, which is right -- the
    gaps are the manual and velocity-controlled stretches.
    """
    d = _get(ulog, "vehicle_local_position_setpoint")
    if d is None:
        return None
    D = d.data
    t = _time_min(ulog, d)
    x, y, z = (np.asarray(D.get(k, []), float) for k in ("x", "y", "z"))
    if x.size == 0 or not np.isfinite(x).any():
        return None
    t_ref, ref_lat = field(ulog, "vehicle_local_position", "ref_lat")
    _, ref_lon = field(ulog, "vehicle_local_position", "ref_lon")
    _, ref_alt = field(ulog, "vehicle_local_position", "ref_alt")
    e, n, u = _anchor(geo, resample_to(t, t_ref, ref_lat),
                      resample_to(t, t_ref, ref_lon),
                      resample_to(t, t_ref, ref_alt), x, y, z)
    t, e, n, u = _decimate(t, e, n, u)
    return Track("setpoint", "position setpoint", C_SETPOINT, t, e, n, u,
                 visible=True, lw=1.4, ls="--")


def _gps_latlonalt(d):
    """(lat_deg, lon_deg, alt_m) across both field namings.

    PX4 renamed these between the firmwares in play here: the older one carries
    `lat`/`lon` as int32 1e-7 degrees and `alt` as int32 millimetres, the newer
    one carries `latitude_deg`/`longitude_deg`/`altitude_msl_m` as floats in
    degrees and metres.  Reading the wrong one does not raise -- it silently
    plots a track 10^7 times too big -- so the scale is inferred from the value
    rather than assumed from the name.
    """
    D = d.data
    if "latitude_deg" in D:
        return (np.asarray(D["latitude_deg"], float),
                np.asarray(D["longitude_deg"], float),
                np.asarray(D.get("altitude_msl_m", np.zeros(1)), float))
    if "lat" not in D:
        return None
    lat = np.asarray(D["lat"], float)
    lon = np.asarray(D["lon"], float)
    alt = np.asarray(D.get("alt", np.zeros_like(lat)), float)
    # Degrees already, or 1e-7 degrees?  A latitude is bounded by 90, so a value
    # past that can only be the scaled integer form.
    if np.nanmax(np.abs(lat)) > 90.0:
        lat, lon, alt = lat * 1e-7, lon * 1e-7, alt * 1e-3
    return lat, lon, alt


def _gps_track(ulog, geo):
    """The raw receiver fix -- the one track the estimator did not touch."""
    d = _get(ulog, "vehicle_gps_position")
    if d is None or not geo.valid:
        return None
    got = _gps_latlonalt(d)
    if got is None:
        return None
    lat, lon, alt = got
    t = _time_min(ulog, d)
    fix = np.asarray(d.data.get("fix_type", np.full_like(lat, 3)), float)
    e, n = geo.to_local(lat, lon)
    u = alt - geo.alt0
    # fix_type < 3 is no 3D fix, and the reported position then is whatever the
    # receiver last had -- not a place the vehicle was.
    bad = ~(np.isfinite(e) & np.isfinite(n) & (fix >= 3))
    e, n, u = e.copy(), n.copy(), u.copy()
    e[bad] = n[bad] = u[bad] = np.nan
    t, e, n, u = _decimate(t, e, n, u)
    return Track("gps", "raw GPS fix", C_GPS, t, e, n, u, visible=False, lw=1.2)


# --- geometry helpers --------------------------------------------------------

def _segments(track, gap_min):
    """(segments, segment_time) for a LineCollection, broken across gaps.

    matplotlib draws a straight line between two points however far apart in
    time they are, so the break has to be made here: a segment is emitted only
    when both endpoints are finite, close enough in time to be one flight, and
    with no estimator reset between them.
    """
    e, n, u, t = track.e, track.n, track.u, track.t
    pts = np.column_stack([e, n, u])
    ok = np.isfinite(pts).all(axis=1)
    joins = ok[:-1] & ok[1:] & (np.diff(t) <= gap_min)
    if track.breaks.size:
        # A pair is joinable only if no reset instant falls inside it.
        crossed = (np.searchsorted(track.breaks, t[1:], side="right")
                   - np.searchsorted(track.breaks, t[:-1], side="right"))
        joins &= (crossed == 0)
    if not joins.any():
        return np.empty((0, 2, 3)), np.empty(0)
    i = np.flatnonzero(joins)
    seg = np.stack([pts[i], pts[i + 1]], axis=1)
    return seg, 0.5 * (t[i] + t[i + 1])


def _gap_threshold(track):
    """GAP_FACTOR x the median sample spacing, floored so a fast topic with a
    couple of jittery samples does not shatter into dots."""
    dt = np.diff(track.t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return np.inf
    return max(float(np.median(dt)) * GAP_FACTOR, 1.0 / 60.0)


def _bounds(tracks, pad=0.06):
    """(e0,e1,n0,n1,u0,u1) over the VISIBLE tracks, east/north made square.

    East and north share one scale unconditionally: a ground track drawn with
    different metres-per-pixel on the two axes is a different SHAPE, and the
    shape is the entire content of a flight path.  The vertical is handled
    separately (see MIN_VERT_FRAC).
    """
    def _pts(sel):
        out = [np.column_stack([tr.e, tr.n, tr.u]) for tr in sel]
        out = [p[np.isfinite(p).all(axis=1)] for p in out]
        return [p for p in out if p.size]

    # Everything switched off leaves the box undefined, so fall back to the
    # full set rather than to an arbitrary unit cube -- switching a track back
    # on then returns to the same view instead of jumping.
    pts = _pts([tr for tr in tracks if tr.visible]) or _pts(tracks)
    if not pts:
        return (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    allp = np.vstack(pts)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    # Square up east/north about their own midpoints.
    h = max(span[0], span[1])
    for k in (0, 1):
        mid = 0.5 * (lo[k] + hi[k])
        lo[k], hi[k] = mid - h / 2, mid + h / 2
    m = np.maximum((hi - lo) * pad, 0.5)
    lo, hi = lo - m, hi + m
    return (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])


def _vertical_exaggeration(b):
    """1.0 when the flight is as tall as it is wide, more when it is a pancake."""
    dh = max(b[1] - b[0], b[3] - b[2])
    dv = b[5] - b[4]
    if dv <= 0 or dh <= 0:
        return 1.0
    want = dh * MIN_VERT_FRAC
    return max(1.0, want / dv)


def _path_stats(track, gap_min):
    """(length_3d_m, ground_length_m, max_up_m, max_radius_m) over one track."""
    e, n, u, t = track.e, track.n, track.u, track.t
    pts = np.column_stack([e, n, u])
    ok = np.isfinite(pts).all(axis=1)
    joins = ok[:-1] & ok[1:] & (np.diff(t) <= gap_min)
    if track.breaks.size:
        # Counting a reset jump as distance flown would add 59 m of "path" to
        # SquareWaypointMission_1 that the vehicle never travelled.
        crossed = (np.searchsorted(track.breaks, t[1:], side="right")
                   - np.searchsorted(track.breaks, t[:-1], side="right"))
        joins &= (crossed == 0)
    if not joins.any():
        return 0.0, 0.0, 0.0, 0.0
    i = np.flatnonzero(joins)
    d = pts[i + 1] - pts[i]
    l3 = float(np.sqrt((d ** 2).sum(axis=1)).sum())
    lg = float(np.sqrt((d[:, :2] ** 2).sum(axis=1)).sum())
    good = pts[ok]
    return (l3, lg, float(good[:, 2].max()),
            float(np.sqrt((good[:, :2] ** 2).sum(axis=1)).max()))


def _at_times(track, times):
    """Interpolate the track to a list of instants; NaN outside its range.

    Used for the mode-change and arm/disarm markers, which know WHEN they
    happened and need to be placed WHERE.
    """
    ok = np.isfinite(track.e) & np.isfinite(track.n) & np.isfinite(track.u)
    if ok.sum() < 2:
        return None
    t = np.asarray(times, float)
    out = np.column_stack([resample_to(t, track.t[ok], track.e[ok]),
                           resample_to(t, track.t[ok], track.n[ok]),
                           resample_to(t, track.t[ok], track.u[ok])])
    return out


# --- basemaps ----------------------------------------------------------------

def basemap_dirs():
    d = [BASEMAP_HOME]
    env = os.environ.get(BASEMAP_ENV)
    if env:
        d.insert(0, os.path.expanduser(env))
    return d


def load_basemap(geo):
    """(rgb, (e0, e1, n0, n1)) for a georeferenced image covering the origin.

    The whole satellite-image hook, and deliberately the only part of it: read a
    sidecar, project its bounds into the plot's own metre frame, hand back an
    array.  A tile downloader added later needs to return exactly this pair and
    nothing else in this module changes.
    """
    if not geo.valid:
        return None
    import matplotlib.image as mpimg
    for d in basemap_dirs():
        for meta_path in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                with open(meta_path) as fh:
                    meta = json.load(fh)
                south, west, north, east = (float(v) for v in meta["bounds"])
                if not (south <= geo.lat0 <= north and west <= geo.lon0 <= east):
                    continue
                img = os.path.join(d, meta.get(
                    "image", os.path.splitext(os.path.basename(meta_path))[0] + ".png"))
                rgb = mpimg.imread(img)
            except Exception:
                continue        # a malformed basemap must never break the plot
            e0, n0 = geo.to_local(south, west)
            e1, n1 = geo.to_local(north, east)
            return rgb, (float(e0), float(e1), float(n0), float(n1))
    return None


def _resize_rgb(rgb, ny, nx):
    """Nearest-neighbour resample -- no scipy/PIL dependency for a floor tile."""
    iy = np.clip((np.arange(ny) * rgb.shape[0] / ny).astype(int), 0, rgb.shape[0] - 1)
    ix = np.clip((np.arange(nx) * rgb.shape[1] / nx).astype(int), 0, rgb.shape[1] - 1)
    return rgb[np.ix_(iy, ix)]


def _crop_rgb(rgb, extent, want):
    """Crop a georeferenced image to `want` (e0, e1, n0, n1); None if disjoint.

    Image row 0 is the NORTH edge (the convention `imshow(origin="upper")` and
    every tile server use), which is why the row index counts down from n1.
    """
    e0, e1, n0, n1 = extent
    ce0, ce1 = max(e0, want[0]), min(e1, want[1])
    cn0, cn1 = max(n0, want[2]), min(n1, want[3])
    if ce1 <= ce0 or cn1 <= cn0:
        return None
    h, w = rgb.shape[:2]
    c0 = int(np.floor((ce0 - e0) / (e1 - e0) * w))
    c1 = int(np.ceil((ce1 - e0) / (e1 - e0) * w))
    r0 = int(np.floor((n1 - cn1) / (n1 - n0) * h))
    r1 = int(np.ceil((n1 - cn0) / (n1 - n0) * h))
    c0, c1 = max(c0, 0), min(max(c1, c0 + 1), w)
    r0, r1 = max(r0, 0), min(max(r1, r0 + 1), h)
    # Report the extent of the pixels actually kept, not the requested one --
    # rounding to whole pixels moves the edges by up to one pixel and pinning
    # the image to the request would shear it by that much.
    ke0 = e0 + c0 / w * (e1 - e0)
    ke1 = e0 + c1 / w * (e1 - e0)
    kn1 = n1 - r0 / h * (n1 - n0)
    kn0 = n1 - r1 / h * (n1 - n0)
    return rgb[r0:r1, c0:c1], (ke0, ke1, kn0, kn1)


def _draw_floor_image(ax, rgb, extent, z, clip):
    """Lay the basemap flat on the floor of the 3D box.

    plot_surface rather than imshow because a 3D axis has no image primitive.

    The crop is not an optimisation -- it is the difference between a working
    plot and a blank one.  mplot3d does NOT clip a surface to the axis limits,
    so a tile covering 400 x 570 m drawn into a box 100 m across renders as an
    opaque sheet over the entire figure with the flight path somewhere behind
    it.  `clip` is the visible box, widened, so a modest zoom-out still lands on
    image rather than on the crop edge.
    """
    got = _crop_rgb(np.asarray(rgb), extent, clip)
    if got is None:
        return None
    tile, (e0, e1, n0, n1) = got
    ny, nx = 160, 160
    # Rows flipped: image row 0 is north, the surface's Y increases northward.
    tile = _resize_rgb(tile, ny, nx)[::-1]
    if tile.ndim == 2:                       # greyscale -> RGB
        tile = np.repeat(tile[:, :, None], 3, axis=2)
    if np.issubdtype(tile.dtype, np.integer) or tile.max() > 1.0:
        tile = tile / 255.0
    X, Y = np.meshgrid(np.linspace(e0, e1, nx), np.linspace(n0, n1, ny))
    return ax.plot_surface(X, Y, np.full_like(X, z), facecolors=tile[:, :, :3],
                           shade=False, rstride=1, cstride=1, linewidth=0,
                           antialiased=False, zorder=0)


# --- the figure --------------------------------------------------------------

def _style_3d(ax):
    """Make an mplot3d axis look like the rest of this toolkit."""
    ax.set_facecolor(C_SURFACE)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(C_SURFACE)
        pane.pane.set_edgecolor(C_GRID)
        pane.pane.set_alpha(1.0)
        pane._axinfo["grid"]["color"] = C_GRID
        pane._axinfo["grid"]["linewidth"] = 0.6
    ax.tick_params(colors=C_MUTED, labelsize=7)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.label.set_color(C_MUTED)
        a.label.set_fontsize(8)


def build_path(ulog, ctx=None, path=""):
    """The 3D flight-path figure.  Same signature as every plot builder."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    import mpl_toolkits.mplot3d  # noqa: F401  (registers the '3d' projection)

    ctx = ctx or PlotCtx()

    if not has_topic(ulog, "vehicle_local_position"):
        ctx.note("no vehicle_local_position in this log -- no path to plot")
        return None

    geo = geo_frame(ulog)
    if not geo.valid:
        ctx.note("no global reference in this log -- axes are raw local metres "
                 "about the estimator origin, and no basemap is possible")

    fused = _fused_track(ulog, geo)
    if fused is None or fused.n_points < 2:
        ctx.note("vehicle_local_position carries no valid xy -- nothing to plot")
        return None
    tracks = [t for t in (fused, _setpoint_track(ulog, geo),
                          _gps_track(ulog, geo)) if t is not None]

    gap_min = _gap_threshold(fused)
    ramp_lo, ramp_hi, ramp_armed = _ramp_window(ulog, fused)
    norm = Normalize(vmin=ramp_lo, vmax=ramp_hi)

    # --- layout -------------------------------------------------------------
    fig = plt.figure(figsize=(15, 9.5), facecolor=C_SURFACE)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(
            f"logGraph flight path - {os.path.basename(path)}")

    # mplot3d reserves a generous margin inside its own rectangle, so the box
    # renders noticeably smaller than the rect asks for -- hence the wide rect.
    # The gap between the two has to hold the 3D box's z tick labels AND its
    # axis label, which mplot3d places outside the rect and will not let you
    # move.  At the PDF page size (14 in) a narrower gap puts "m above origin"
    # straight through the plan view's "north (m)".
    ax3d = fig.add_axes([0.145, 0.155, 0.545, 0.725], projection="3d",
                        facecolor=C_SURFACE)
    ax_plan = fig.add_axes([0.755, 0.315, 0.225, 0.50], facecolor=C_SURFACE)
    cax = fig.add_axes([0.245, 0.115, 0.36, 0.016])
    _style_3d(ax3d)

    # --- basemap, under everything -----------------------------------------
    bm = load_basemap(geo)
    bm_art = []

    # --- the tracks ---------------------------------------------------------
    for tr in tracks:
        seg3, seg_t = _segments(tr, gap_min)
        if seg3.shape[0] == 0:
            continue
        if tr.ramp:
            lc3 = Line3DCollection(seg3, cmap=TIME_CMAP, norm=norm, lw=tr.lw,
                                   zorder=4)
            lc3.set_array(seg_t)
            lc2 = LineCollection(seg3[:, :, :2], cmap=TIME_CMAP, norm=norm,
                                 lw=tr.lw, zorder=4)
            lc2.set_array(seg_t)
            ramp_mappable = lc2
        else:
            lc3 = Line3DCollection(seg3, colors=tr.color, lw=tr.lw,
                                   linestyles=tr.ls, zorder=3)
            lc2 = LineCollection(seg3[:, :, :2], colors=tr.color, lw=tr.lw,
                                 linestyles=tr.ls, zorder=3)
        ax3d.add_collection3d(lc3)
        ax_plan.add_collection(lc2)
        tr.artists = [lc3, lc2]
        for art in tr.artists:
            art.set_visible(tr.visible)

    b = _bounds(tracks)
    z_floor = b[4]

    if bm is not None:
        rgb, extent = bm
        bm_art.append(ax_plan.imshow(rgb, extent=list(extent), origin="upper",
                                     zorder=0, interpolation="bilinear"))
        # Cropped to EXACTLY the box, not a margin around it: mplot3d does not
        # clip a surface to the axis limits, so every metre of overhang is
        # painted outside the box -- and, being nearer the camera than the axes,
        # over the flight path itself.  Zooming out past this shows a floor
        # smaller than the box, which is the harmless failure of the two.
        try:
            floor = _draw_floor_image(ax3d, rgb, extent, z_floor,
                                      (b[0], b[1], b[2], b[3]))
            if floor is not None:
                bm_art.append(floor)
                # mplot3d depth-sorts whole ARTISTS by their mean z, so one
                # surface spanning the floor sorts as "the middle of the box"
                # and paints over every part of the path below that -- the
                # takeoff column disappears behind the ground.  Switching to
                # explicit zorder is exact here precisely because the floor IS
                # below everything: there is no case where it should occlude.
                ax3d.computed_zorder = False
        except Exception as e:      # a bad image must not cost the whole plot
            ctx.note(f"basemap floor could not be drawn: {type(e).__name__}: {e}")

    # --- ground projection and drop lines ----------------------------------
    # The shadow is what makes a 3D line readable as a position: without it the
    # eye cannot tell a high near point from a low far one, because a projection
    # of a line has no depth cue of its own.
    ok = np.isfinite(fused.e) & np.isfinite(fused.n) & np.isfinite(fused.u)
    shadow_art = []
    seg3, _ = _segments(fused, gap_min)
    if seg3.shape[0]:
        flat = seg3.copy()
        flat[:, :, 2] = z_floor
        sh = Line3DCollection(flat, colors=C_SHADOW, lw=1.0, alpha=0.85, zorder=2)
        ax3d.add_collection3d(sh)
        shadow_art.append(sh)

    # One drop line every so often, tying the path to its shadow.  Capped by
    # count rather than by interval: the point is a readable ladder, and 40 rungs
    # is that whether the flight is 40 seconds or 40 minutes.
    drop_art = []
    idx = np.flatnonzero(ok)
    if idx.size > 4:
        pick = idx[np.linspace(0, idx.size - 1, min(40, idx.size)).astype(int)]
        drops = np.stack([
            np.column_stack([fused.e[pick], fused.n[pick],
                             np.full(pick.size, z_floor)]),
            np.column_stack([fused.e[pick], fused.n[pick], fused.u[pick]])],
            axis=1)
        dl = Line3DCollection(drops, colors=C_SHADOW, lw=0.6, alpha=0.5, zorder=2)
        ax3d.add_collection3d(dl)
        drop_art.append(dl)

    # --- markers ------------------------------------------------------------
    ends_art = []
    if idx.size:
        i0, i1 = idx[0], idx[-1]
        # Offsets in opposite directions: a vehicle that lands where it took
        # off puts these two labels on the same pixel otherwise.
        for i, color, marker, label, dy in ((i0, C_START, "o", "start", 6),
                                            (i1, C_END, "X", "end", -11)):
            ends_art.append(ax3d.scatter([fused.e[i]], [fused.n[i]], [fused.u[i]],
                                         c=color, marker=marker, s=55,
                                         depthshade=False, zorder=6))
            ends_art.append(ax_plan.scatter([fused.e[i]], [fused.n[i]], c=color,
                                            marker=marker, s=45, zorder=6))
            ends_art.append(ax_plan.annotate(
                label, (fused.e[i], fused.n[i]), textcoords="offset points",
                xytext=(6, dy), fontsize=7, color=color))

    home_art = []
    hp = _get(ulog, "home_position")
    if hp is not None and geo.valid and "lat" in hp.data:
        hlat = np.asarray(hp.data["lat"], float)
        hlon = np.asarray(hp.data["lon"], float)
        halt = np.asarray(hp.data.get("alt", np.zeros_like(hlat)), float)
        good = np.isfinite(hlat) & (np.abs(hlat) > 1e-6)
        if good.any():
            j = int(np.flatnonzero(good)[-1])    # the home the flight ended with
            he, hn = geo.to_local(hlat[j], hlon[j])
            hu = float(halt[j]) - geo.alt0
            home_art.append(ax3d.scatter([he], [hn], [hu], c=C_HOME, marker="^",
                                         s=45, depthshade=False, zorder=6))
            home_art.append(ax_plan.scatter([he], [hn], c=C_HOME, marker="^",
                                            s=40, zorder=6))

    # Arm / disarm, placed on the path.  On a log with several flights in it
    # these are the only marks that say where each one began.
    arm_art = []
    spans = armed_spans(ulog)
    if spans:
        times = [s[0] for s in spans] + [s[1] for s in spans]
        pos = _at_times(fused, times)
        if pos is not None:
            half = len(spans)
            for k, (marker, color) in enumerate((("^", C_START), ("v", C_END))):
                p = pos[k * half:(k + 1) * half]
                p = p[np.isfinite(p).all(axis=1)]
                if p.size:
                    arm_art.append(ax3d.scatter(p[:, 0], p[:, 1], p[:, 2],
                                                c=color, marker=marker, s=28,
                                                depthshade=False, zorder=6,
                                                edgecolors="none", alpha=0.9))
                    arm_art.append(ax_plan.scatter(p[:, 0], p[:, 1], c=color,
                                                   marker=marker, s=24, zorder=6,
                                                   edgecolors="none", alpha=0.9))

    # Estimator resets: where the line is broken, and why.  These are the only
    # red marks on the figure, which is the convention everywhere in this
    # toolkit -- red means something went wrong, not "another data series".
    reset_art = []
    reset_pos = _positions_at(fused, fused.breaks)
    if reset_pos.size:
        reset_art.append(ax3d.scatter(reset_pos[:, 0], reset_pos[:, 1],
                                      reset_pos[:, 2], c=C_BAD, marker="o",
                                      s=26, depthshade=False, zorder=7,
                                      edgecolors="none", alpha=0.9))
        reset_art.append(ax_plan.scatter(reset_pos[:, 0], reset_pos[:, 1],
                                         c=C_BAD, marker="o", s=22, zorder=7,
                                         edgecolors="none", alpha=0.9))

    # Mode changes, in the same colours the time-series plots use for their
    # rules -- so "the Hold at 6.1 min" on the altitude plot and "the orange dot
    # at the far corner" here are recognisably the same event.
    mode_art, mode_codes = [], []
    changes = mode_changes(ulog)
    if changes:
        pos = _at_times(fused, [c[0] for c in changes])
        if pos is not None:
            for (t_c, code), p in zip(changes, pos):
                if not np.isfinite(p).all():
                    continue
                if int(code) not in mode_codes:
                    mode_codes.append(int(code))
                col = mode_color(code)
                mode_art.append(ax3d.scatter([p[0]], [p[1]], [p[2]], c=col,
                                             marker="s", s=22, depthshade=False,
                                             zorder=7, edgecolors="none"))
                mode_art.append(ax_plan.scatter([p[0]], [p[1]], c=col, marker="s",
                                                s=18, zorder=7, edgecolors="none"))
    # Its own row BELOW the nav hint: at 0.030 the two rows are 13 px apart at
    # this figure size and the key's "modes:" label lands on the hint's tail.
    mode_art += mode_key(fig, 0.985, 0.012, mode_codes)
    for art in mode_art:
        art.set_visible(True)

    # --- axis furniture -----------------------------------------------------
    unit = "m east of origin" if geo.valid else "m east (local)"
    ax3d.set_xlabel(unit)
    ax3d.set_ylabel("m north of origin" if geo.valid else "m north (local)")
    ax3d.set_zlabel("m above origin")
    ax_plan.set_xlabel("east (m)", fontsize=8, color=C_MUTED)
    ax_plan.set_ylabel("north (m)", fontsize=8, color=C_MUTED)
    ax_plan.tick_params(colors=C_MUTED, labelsize=7)
    ax_plan.grid(True, color=C_GRID, lw=0.7)
    ax_plan.set_axisbelow(True)
    for spine in ax_plan.spines.values():
        spine.set_color(C_GRID)
    ax_plan.set_title("plan view (top-down)", fontsize=9, color=C_MUTED,
                      loc="left")
    ax_plan.set_aspect("equal", adjustable="datalim")

    # A viewpoint that shows a ground track as a ground track: high enough to
    # read the plan, low enough for altitude to be a visible dimension.
    ax3d.view_init(elev=26, azim=-58)

    exagg = {"k": 1.0}

    def apply_bounds():
        bb = _bounds(tracks)
        ax3d.set_xlim(bb[0], bb[1])
        ax3d.set_ylim(bb[2], bb[3])
        ax3d.set_zlim(bb[4], bb[5])
        k = _vertical_exaggeration(bb)
        exagg["k"] = k
        ax3d.set_box_aspect((bb[1] - bb[0], bb[3] - bb[2], (bb[5] - bb[4]) * k))
        ax_plan.set_xlim(bb[0], bb[1])
        ax_plan.set_ylim(bb[2], bb[3])
        vnote.set_text("true scale on all three axes" if k <= 1.001 else
                       f"vertical exaggerated {k:.1f}x for readability")

    vnote = fig.text(0.175, 0.888, "", color=C_MUTED, fontsize=8, ha="left")
    apply_bounds()

    # --- colour bar: the legend for time ------------------------------------
    sm = plt.cm.ScalarMappable(cmap=TIME_CMAP, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("time (min from log start)"
                 + ("  --  ramp spans the armed window" if ramp_armed else ""),
                 fontsize=8, color=C_MUTED)
    cb.ax.tick_params(colors=C_MUTED, labelsize=7)
    cb.outline.set_edgecolor(C_GRID)

    # --- titles and the numbers ---------------------------------------------
    fig.text(0.175, 0.955, "Flight path", color=C_INK, fontsize=13,
             fontweight="bold", ha="left")
    who = f"{os.path.basename(path)}   |   " if path else ""
    if geo.valid:
        frame = (f"origin {geo.lat0:.6f}, {geo.lon0:.6f} at {geo.alt0:.1f} m AMSL"
                 f"   |   re-anchored per sample")
    else:
        frame = "no global origin -- raw local frame, handovers NOT re-anchored"
    fig.text(0.175, 0.925, f"{who}{duration_min(ulog):.1f} min   |   {frame}",
             color=C_MUTED, fontsize=9, ha="left")

    l3, lg, mx_u, mx_r = _path_stats(fused, gap_min)
    lines = [f"path length      {l3:8.0f} m",
             f"ground track     {lg:8.0f} m",
             f"max height       {mx_u:8.1f} m",
             f"max radius       {mx_r:8.1f} m",
             f"points drawn     {fused.n_points:8d}"]
    if fused.breaks.size:
        lines.append(f"estimator resets {fused.breaks.size:8d}  (line broken)")
    if bm is not None:
        lines.append("basemap          loaded")
    fig.text(0.755, 0.255, "\n".join(lines), color=C_MUTED, fontsize=8,
             ha="left", va="top", family="monospace")

    if bm is None and geo.valid:
        ax_plan.text(0.5, 0.02,
                     "no basemap for this site", transform=ax_plan.transAxes,
                     ha="center", va="bottom", fontsize=7, color=C_MUTED)
        fig.text(0.755, 0.135,
                 "satellite image: drop <name>.png + <name>.json\n"
                 '{"image": "<name>.png", "bounds": [S, W, N, E]}\n'
                 f"into {BASEMAP_HOME}\n"
                 f"covering {geo.lat0:.5f}, {geo.lon0:.5f}",
                 color=C_MUTED, fontsize=7, ha="left", va="top")

    # --- toggles ------------------------------------------------------------
    # Each track owns TWO artists (the 3D line and its plan-view twin) and one
    # checkbox has to drive both, so the Series the panel is built from carries a
    # fan-out proxy in place of a single line.
    class _Fan:
        def __init__(self, artists):
            self.artists = list(artists)
            self._v = True

        def set_visible(self, v):
            self._v = bool(v)
            for a in self.artists:
                a.set_visible(self._v)

        def get_visible(self):
            return self._v

    series = []
    for tr in tracks:
        if not tr.artists:
            continue
        s = Series(tr.id, tr.label, tr.t, tr.u, "track", tr.color,
                   visible=tr.visible)
        s.line = _Fan(tr.artists)
        s.line.set_visible(tr.visible)
        s.track = tr
        series.append(s)

    def refresh():
        for s in series:
            s.track.visible = s.line.get_visible()
        apply_bounds()

    extra = []
    if shadow_art:
        extra.append(("ground shadow", shadow_art, True))
    if drop_art:
        extra.append(("drop lines", drop_art, True))
    if ends_art:
        extra.append(("start / end", ends_art, True))
    if home_art:
        extra.append(("home position", home_art, True))
    if arm_art:
        extra.append(("arm / disarm", arm_art, True))
    if reset_art:
        extra.append(("estimator resets", reset_art, True))
    if mode_art:
        extra.append(("mode changes", mode_art, True))
    if bm_art:
        extra.append(("satellite basemap", bm_art, True))

    h = min(0.60, 0.035 * (len(series) + len(extra) + 2) + 0.05)
    check_panel(fig, [0.012, 0.86 - h, 0.145, h], series,
                [("track", "ALL tracks")], extra=extra, on_change=refresh,
                title="tracks")

    add_view_navigation(fig, plan_axes=[ax_plan], view_axes=[ax3d],
                        page_scroll=ctx.page_scroll)
    fig.text(0.175, 0.045, view_nav_hint(ctx.page_scroll), color=C_MUTED,
             fontsize=8, ha="left")
    fig._page_height = 860
    return fig
