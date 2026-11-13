"""Checks for position.classify and the watcher's hold time (no camera, no MediaPipe). Exit 1 on failure."""
import os, sys, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import position as P


def pose(width, mid=0.5, nose_y=0.3):
    a = np.zeros((33, 3), np.float32)
    a[P.L_SHOULDER, 0], a[P.R_SHOULDER, 0] = mid + width / 2, mid - width / 2
    a[P.NOSE] = (mid, nose_y, 1.0)
    return a


cases = [
    ('nobody',                 None,                    0, 'NONE'),
    ('face only',              None,                    1, 'EDGE'),
    ('good',                   pose(0.35),              1, 'OK'),
    ('good, face missed',      pose(0.35),              0, 'OK'),
    ('too close',              pose(0.70),              1, 'CLOSE'),
    ('too far',                pose(0.10),              1, 'FAR'),
    ('just inside near limit', pose(0.54),              1, 'OK'),
    ('just inside far limit',  pose(0.19),              1, 'OK'),
    ('off to the side',        pose(0.30, mid=0.85),    1, 'EDGE'),
    ('face cut off at top',    pose(0.35, nose_y=0.01), 1, 'EDGE'),
    ('two people',             pose(0.35),              2, 'MANY'),
    ('two faces, no pose',     None,                    3, 'MANY'),
]
ok = True
for name, p, n, want in cases:
    got, _ = P.classify(p, n)
    ok &= got == want
    print(f'{"PASS" if got == want else "FAIL"} {name:24s} -> {got}' + ('' if got == want else f' (wanted {want})'))
for state, (l1, l2, rgb) in P.MESSAGES.items():
    good = len(l1) <= 16 and len(l2) <= 16 and all(0 <= c <= 255 for c in rgb)
    ok &= good
    print(f'{"PASS" if good else "FAIL"} message {state}: "{l1}" / "{l2}" fits 16 characters')


class FakeLCD:
    def __init__(self): self.sent = []
    def show(self, *a): self.sent.append(a[0]); return True
    def close(self): pass


class FakeInner:
    def __init__(self): self.res = None
    def process(self, rgb): return self.res
    def close(self): pass


def lm(p):
    pts = [types.SimpleNamespace(x=float(x), y=float(y), visibility=float(v)) for x, y, v in p]
    return types.SimpleNamespace(pose_landmarks=types.SimpleNamespace(landmark=pts))


fake_mp = types.SimpleNamespace(solutions=types.SimpleNamespace(face_detection=types.SimpleNamespace(
    FaceDetection=lambda **k: types.SimpleNamespace(process=lambda rgb: types.SimpleNamespace(detections=[1]), close=lambda: None))))
inner, lcd = FakeInner(), FakeLCD()
w = P.PositionWatcher(inner, fake_mp, hold=0.2, lcd=lcd)
inner.res = lm(pose(0.35)); w.process(None)
first = list(lcd.sent)                                     # nothing yet: the state has not held for 0.2 s
time.sleep(0.25); w.process(None)
inner.res = lm(pose(0.70)); w.process(None)                # a one-frame flicker to CLOSE must not show
inner.res = lm(pose(0.35)); w.process(None)
good = first == [] and lcd.sent == ['IN POSITION']
ok &= good
print(f'{"PASS" if good else "FAIL"} hold time: shown {lcd.sent} (a one-frame change is ignored)')
print('ALL PASS' if ok else 'SOME FAILED')
sys.exit(0 if ok else 1)
