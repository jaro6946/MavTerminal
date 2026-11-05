#!/usr/bin/env bash
# Build the combined hires-camera video (data panel + launch.log pane + tag/steering overlays) for one day of
# grazer10 ROS 2 bags.  The Bags tab's "Make video" button runs this; it also runs fine by hand:
#
#   bag_video/make_bag_video.sh 2026_11_04 ["<bags root>"]
#
# <bags root> defaults to $BAGS_ROOT, else ~/jacobAtGar/Log Analysis/Bags.
# Reads   <bags root>/<date>/grazer10_*            (bags without metadata.yaml are skipped)
# Writes  <bags root>/bag analysis/<date>/grazer10_all_bags_combined.mp4
#         ..._logs.txt         every log line, untruncated
#         ..._bookmarks.json   "Nav stack crash n" bookmarks merged in (hand-made ones are kept)
#         overlay_data.json    stage-1 series, reusable
#
# Stage 1 (overlay_data.py) reads the bags with the pure-python `rosbags` package, kept in ./pylib
# (gitignored; installed here on first run).  Stage 2 (combine_bags.py) needs ROS 2 Jazzy's rosbag2_py + OpenCV,
# so both stages run on the SYSTEM python3, never a venv's.
set -euo pipefail
DATE="${1:?usage: $0 <date folder, e.g. 2026_11_04> [bags root]}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BAGS="${2:-${BAGS_ROOT:-$HOME/jacobAtGar/Log Analysis/Bags}}"
OUT="$BAGS/bag analysis/$DATE"
[ -d "$BAGS/$DATE" ] || { echo "no bag folder: $BAGS/$DATE" >&2; exit 2; }

PY=/usr/bin/python3
if ! PYTHONPATH="$HERE/pylib" "$PY" -c "import rosbags.highlevel" 2>/dev/null; then
    echo "installing rosbags into $HERE/pylib ..."
    "$PY" -m pip install --quiet --target "$HERE/pylib" rosbags
    # pip drags in its own numpy; drop it so the system numpy (which ROS's OpenCV is built against) is used
    rm -rf "$HERE"/pylib/numpy "$HERE"/pylib/numpy-* "$HERE"/pylib/numpy.libs "$HERE"/pylib/bin
fi
mkdir -p "$OUT"

PYTHONPATH="$HERE/pylib" "$PY" "$HERE/overlay_data.py" "$BAGS/$DATE" "$OUT/overlay_data.json"

set +u; source /opt/ros/jazzy/setup.bash; set -u
"$PY" "$HERE/combine_bags.py" "$BAGS/$DATE" "$OUT/grazer10_all_bags_combined.mp4" "$OUT/overlay_data.json" 2>&1 \
    | grep -v '^\[INFO\]'
ls -la "$OUT"
