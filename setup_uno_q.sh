#!/usr/bin/env bash
# One-time setup of the sign demo on an Arduino UNO Q (any 64-bit ARM Debian board).
# Copy the demo folder to the board, open a terminal there, then:   bash setup_uno_q.sh
# Why a special setup: the board ships Python 3.13, but TensorFlow 2.14 and MediaPipe 0.10.14 need 3.11 or older.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HOME/demo_env}"
TOOLS="${TOOLS:-$HOME/tools}"
MAX_LEN="${MAX_LEN:-128}"                     # must equal the CFG_MAX_LEN the model was trained with

echo "== 0. Board check =="
echo "architecture : $(uname -m)"
echo "system python: $(python3 --version 2>&1)"
MEM_MB=$(awk '/MemTotal/ {printf "%d", $2/1024}' /proc/meminfo 2>/dev/null || echo 0)
echo "RAM          : ${MEM_MB} MB"
echo "free disk    : $(df -h "$HOME" | awk 'NR==2 {print $4}') in $HOME"
[ "$(uname -m)" = "aarch64" ] || echo "NOTE: not an aarch64 board; continuing anyway."
[ "$MEM_MB" -ge 1700 ] || echo "WARNING: under 2 GB of RAM visible. TensorFlow + MediaPipe may not fit."
[ "$MEM_MB" -ge 3500 ] || echo "NOTE: with 2 GB boards, close other programs; adding swap space can help if you see 'Killed'."

echo; echo "== 1. Tools environment (uv) =="
if [ ! -x "$TOOLS/bin/uv" ]; then
  if ! python3 -m venv "$TOOLS"; then
    echo "ERROR: could not create a virtual environment. Run:  sudo apt install -y python3-venv python3-pip"
    echo "then run this script again."; exit 1
  fi
  "$TOOLS/bin/python" -m pip install --quiet uv || { echo "ERROR: could not install uv (is the board online?)"; exit 1; }
fi
echo "uv ready: $("$TOOLS/bin/uv" --version)"

echo; echo "== 2. Python 3.11 environment at $VENV =="
if [ ! -x "$VENV/bin/python" ]; then
  "$TOOLS/bin/uv" venv --seed --python 3.11 "$VENV" || { echo "ERROR: could not create the Python 3.11 environment"; exit 1; }
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
case "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" in
  3.9|3.10|3.11) echo "using $(python --version)";;
  *) echo "ERROR: $VENV uses $(python --version); delete it (rm -rf $VENV) and re-run."; exit 1;;
esac

echo; echo "== 3. Packages (this takes a while on the board) =="
python -m pip install --upgrade pip wheel
FAILED=()
# mediapipe 0.10.14 leaves jax/jaxlib unpinned; their latest pull ml-dtypes>=0.5 (numpy 2 only), which breaks
# TF 2.14 (needs ml-dtypes==0.2.0), so the last step pins the whole set back to versions that agree.
for pkg in "numpy<2" "tensorflow==2.14.1" "mediapipe==0.10.14" "opencv-contrib-python==4.9.0.80" "numpy<2 ml-dtypes==0.2.0 jax==0.4.20 jaxlib==0.4.20"; do
  echo "--- $pkg"
  # shellcheck disable=SC2086  # the last entry is several packages that must resolve together
  python -m pip install $pkg || FAILED+=("$pkg")
done
if [ ${#FAILED[@]} -gt 0 ]; then
  echo; echo "!! These packages could not be installed: ${FAILED[*]}"
  echo "!! Send me the error text above for the first one; it probably has no build for this board."
fi

echo; echo "== 4. Checks =="
python - <<'PY' || echo "!! Import check failed -- see the error above."
import numpy, cv2, mediapipe as mp
print("numpy", numpy.__version__, "| opencv", cv2.__version__, "| mediapipe", mp.__version__)
assert hasattr(mp, "solutions"), "this MediaPipe has no solutions.holistic; need mediapipe==0.10.14"
import tensorflow as tf
print("tensorflow", tf.__version__)
PY
echo "cameras      : $(ls /dev/video* 2>/dev/null | tr '\n' ' ')"
[ -n "$(ls /dev/video* 2>/dev/null)" ] || echo "NOTE: no /dev/video* found. Plug the USB webcam into the hub and re-check with: ls /dev/video*"

W="$(ls "$HERE"/*.h5 2>/dev/null | head -n 1 || true)"
if [ -n "$W" ] && [ -f "$HERE/sign_to_prediction_index_map.json" ]; then
  echo; echo "== 5. Model check with $(basename "$W") (max_len=$MAX_LEN) =="
  ( cd "$HERE" && python demo_webcam.py --weights "$W" --signmap sign_to_prediction_index_map.json --max-len "$MAX_LEN" --dry-run )
else
  echo; echo "(Skipping the model check: put your .h5 weights and sign_to_prediction_index_map.json next to this script.)"
fi

cat <<MSG

== Done. Next, activate the environment in every new terminal:   source $VENV/bin/activate
   1) Speed + memory test (no screen needed):
        python demo_webcam.py --benchmark --complexity 0 --weights <your>.h5 --max-len $MAX_LEN
   2) Live predictions without a window:
        python demo_webcam.py --headless --complexity 0 --weights <your>.h5 --max-len $MAX_LEN
   Send me the benchmark output and I will tell you whether this setup is good enough.
MSG
