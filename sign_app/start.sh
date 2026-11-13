#!/bin/bash
# Start the sign-to-speech program in the background. Preview: http://<board-address>:5000/  Log: ~/sign_app/live.log
# Extra sign_speaker.py options pass through, e.g.  ./start.sh --volume 80   or   ./start.sh --threshold 2  (silent)
cd "$(dirname "$0")"
if pgrep -f "^/home/arduino/demo_env/bin/python -u board_live.py" >/dev/null; then echo "already running (./stop.sh first)"; exit 1; fi
[ "$#" -eq 0 ] && set -- --volume 50
SIGN_NO_XNNPACK=1 setsid nohup ~/demo_env/bin/python -u board_live.py "$@" > live.log 2>&1 < /dev/null &
echo "started (pid $!); log: $(pwd)/live.log"
