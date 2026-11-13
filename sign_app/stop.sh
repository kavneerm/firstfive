#!/bin/bash
# Stop the sign-to-speech program (frees the camera and port 5000).
pids=$(pgrep -f "^/home/arduino/demo_env/bin/python -u board_live.py")
[ -z "$pids" ] && { echo "not running"; exit 0; }
kill -CONT $pids 2>/dev/null; kill $pids
for i in $(seq 1 10); do pgrep -f "^/home/arduino/demo_env/bin/python -u board_live.py" >/dev/null || { echo "stopped"; exit 0; }; sleep 1; done
kill -9 $(pgrep -f "^/home/arduino/demo_env/bin/python -u board_live.py") 2>/dev/null; echo "stopped (forced)"
