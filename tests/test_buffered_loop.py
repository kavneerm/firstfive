#!/usr/bin/env python3
"""
Runs sign_speaker.run() for real in --buffered mode with a fake camera, a forced sign end and a preview viewer,
and checks the loop survives the pass where the buffered result arrives (it used to crash right there).
Silent: the speaker is replaced. Needs a free port 5098.

    cd ~/sign_demo && ~/sign_env/bin/python tests/test_buffered_loop.py
"""
import io, os, sys, threading, time, urllib.request
from contextlib import redirect_stdout
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import sign_speaker as S

N_FRAMES = 150                                          # 10 s of fake camera at 15 fps


class FakeCam:
    def __init__(self):
        self.i = 0

    def read(self):
        time.sleep(1 / 15)
        self.i += 1
        if self.i > N_FRAMES:
            return False, None
        return True, np.random.RandomState(self.i).randint(0, 255, (480, 640, 3), np.uint8)

    def release(self):
        pass


class ForcedSegmenter(S.TimedSegmenter):
    """Ends one 'sign' on the 10th frame it sees, regardless of hands."""

    def update(self, lm, t, seen=None):
        self.calls = getattr(self, 'calls', 0) + 1
        if self.calls == 10:
            self.last_span = (t - 1.0, t)
            return np.zeros((15, 543, 3), np.float32)
        return None


class SilentSpeaker:
    said = []

    def say(self, word):
        self.said.append(word)


S.open_capture = lambda cv2, source, width, height: (FakeCam(), True)
S.TimedSegmenter = ForcedSegmenter
S.landmarks_for_frames = lambda a, frames: np.full((len(frames), 543, 3), 0.5, np.float32)   # fast stand-in

# record=... makes the overlay draw on EVERY pass, so the pass where the result arrives always hits the overlay code
# (with only the preview it is drawn every 2nd frame while watched, and the crash depended on timing)
a = SimpleNamespace(camera='/dev/fake', width=640, height=480, buffered=True, max_sign=4.0, sign_end=0.4,
                    threshold=0.0, cooldown=1.0, record=os.path.join(__import__('tempfile').mkdtemp(), 'loop.mp4'),
                    record_fps=15.0, flip=False, landmarker='custom',
                    complexity=0, hand_complexity=1, face_every=2, hand_model='full', palm_every=6,
                    preview_width=400, buffer_preroll=0.3, port=5098, status_every=2.0, rest_line=0.8)


class FakePredictor:
    def predict(self, clip, topk=5):
        time.sleep(0.5)                                 # long enough for several loop passes while 'busy'
        return [('apple', 0.93), ('zipper', 0.01), ('tomorrow', 0.01), ('ear', 0.01), ('dry', 0.01)]


preview = S.Preview(a.port)
stop = threading.Event()


def viewer():                                           # keeps the preview 'watched' so the overlay path runs
    seq = 0
    while not stop.is_set():
        try:
            r = urllib.request.urlopen(f'http://127.0.0.1:{a.port}/frame.jpg?after={seq}', timeout=3)
            if r.status == 200:
                seq = int(r.headers['X-Seq'])
        except Exception:
            time.sleep(0.1)


threading.Thread(target=viewer, daemon=True).start()
out, err = io.StringIO(), None
try:
    with redirect_stdout(out):
        S.run(a, FakePredictor(), SilentSpeaker(), preview)
except Exception as e:
    err = e
stop.set()
log = out.getvalue()
ok = err is None and 'buffered (' in log and 'speaking "apple"' in log and 'Camera returned no frame' in log
print(log.strip().replace('\n', '\n    '))
print(f'{"PASS" if ok else "FAIL"}  buffered loop survives the result pass and runs to the end of the camera'
      f'  (exception: {err!r}, spoken: {SilentSpeaker.said})')
sys.exit(0 if ok else 1)
