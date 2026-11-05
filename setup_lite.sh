#!/usr/bin/env bash
# Slim runtime for the sign speaker on the UNO Q: numpy + OpenCV + tflite-runtime only (no TensorFlow / MediaPipe / jax).
# Needs ~/tools/bin/uv (created by setup_uno_q.sh) and, once, the full ~/demo_env to provide the MediaPipe models and
# to run convert_sign_model.py.   Usage:  bash setup_lite.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HOME/sign_env}"
UV="${UV:-$HOME/tools/bin/uv}"
FULL="${FULL:-$HOME/demo_env}"

[ -x "$VENV/bin/python" ] || "$UV" venv --python 3.11 "$VENV"
"$UV" pip install --python "$VENV/bin/python" -r "$HERE/requirements-lite.txt"

# the landmark models ship inside the mediapipe wheel; copy the four the tracker uses next to the code
MP="$("$FULL/bin/python" -c 'import os, mediapipe; print(os.path.dirname(mediapipe.__file__))' 2>/dev/null)"
mkdir -p "$HERE/models"
for m in palm_detection/palm_detection_full hand_landmark/hand_landmark_full hand_landmark/hand_landmark_lite \
         face_detection/face_detection_short_range face_landmark/face_landmark; do
  cp "$MP/modules/$m.tflite" "$HERE/models/"
done
ls -la "$HERE/models"

if [ ! -f "$HERE/sign_model.tflite" ]; then
  ( cd "$HERE" && "$FULL/bin/python" convert_sign_model.py --weights "$(ls "$HERE"/*.h5 | head -n 1)" --out sign_model.tflite )
fi
"$VENV/bin/python" -c 'import numpy, cv2, tflite_runtime; print("numpy", numpy.__version__, "| opencv", cv2.__version__, "| tflite-runtime", tflite_runtime.__version__)'
