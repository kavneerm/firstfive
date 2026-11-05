#!/usr/bin/env python3
"""
Checks for the demo-pipeline pieces that need no camera: TimedSegmenter timing at different frame rates, the
Recorder's constant-rate output, and the preview's never-serve-stale rule. Plain asserts (no pytest on the board).

    cd ~/sign_demo && ~/sign_env/bin/python tests/test_demo_parts.py
"""
import os, sys, tempfile, threading, time, urllib.request
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import cv2
import sign_speaker
from sign_speaker import LatestFrame, Preview, Recorder, Speaker, TimedSegmenter, hands_up

FAILED = []


def check(name, cond, detail=''):
    print(f'{"PASS" if cond else "FAIL"}  {name}  {detail}')
    if not cond:
        FAILED.append(name)


def frame(hand):
    lm = np.full((543, 3), np.nan, np.float32)
    lm[:468] = 0.5                                     # face always present
    if hand:
        lm[522:543] = 0.4
    return lm


def feed(seg, pattern, fps, last_hand=None):
    """pattern: list of (seconds, hands_visible), turned into whole frames. Returns [(t_emitted, clip_len)];
    if `last_hand` (a list) is given, the time of the last frame with a hand is appended to it."""
    out, t = [], 0.0
    for secs, hand in pattern:
        for _ in range(int(round(secs * fps))):
            clip = seg.update(frame(hand), t)
            if hand and last_hand is not None:
                last_hand[:] = [t]
            if clip is not None:
                out.append((round(t, 3), len(clip)))
            t += 1.0 / fps
    return out


# --- TimedSegmenter: one sign, same timing at 7 and 15 fps
for fps in (7, 15):
    last = []
    got = feed(TimedSegmenter(), [(1.0, False), (1.5, True), (1.0, False)], fps, last)
    check(f'one sign -> one clip @ {fps} fps', len(got) == 1, got)
    if got:
        t_emit, n = got[0]
        gap = t_emit - last[0]                         # from the last frame that actually had a hand
        check(f'clip ends 0.4 s (+ at most 1 frame) after the last hand @ {fps} fps',
              0.4 - 1e-6 <= gap <= 0.4 + 1.0 / fps + 1e-6, f'gap {gap:.3f} s')
        lo, hi = int(1.5 * fps), int((1.5 + 0.3 + 0.1 + 0.4) * fps) + 3
        check(f'clip length plausible @ {fps} fps', lo <= n <= hi, f'{n} frames, expected {lo}..{hi}')

# --- hands never drop (someone resting in frame, not signing): nothing is returned, so nothing is guessed/spoken
seg = TimedSegmenter(max_len=4.0)
got = feed(seg, [(10.0, True)], 10)
# one drop at 4 s; after that it waits for the hands to leave, so no second clip even starts
check('continuous hands -> no clips (dropped, not guessed)', got == [] and seg.discarded == 1, f'{got}, discarded {seg.discarded}')

# --- ...and a real sign right after such a stretch is still picked up
got = feed(TimedSegmenter(max_len=4.0), [(5.0, True), (1.0, False), (1.2, True), (1.0, False)], 10)
check('real sign after a resting stretch -> one clip', len(got) == 1, got)

# --- a one-frame flicker of a hand is not a sign
got = feed(TimedSegmenter(), [(1.0, False), (1 / 7, True), (1.0, False)], 7)
check('single-frame flicker ignored', got == [], got)

# --- two signs separated by a short drop give two clips
got = feed(TimedSegmenter(), [(0.5, False), (1.0, True), (0.6, False), (1.0, True), (1.0, False)], 10)
check('two signs with a 0.6 s gap -> two clips', len(got) == 2, got)

# --- hands_up: hands resting at the bottom edge count as down; hands at chest/face height count as up
def at_height(y_right=None, y_left=None):
    lm = frame(False)
    if y_right is not None:
        lm[522:543, 1] = y_right; lm[522:543, 0] = 0.5
    if y_left is not None:
        lm[468:489, 1] = y_left; lm[468:489, 0] = 0.5
    return lm
check('no hands -> down', not hands_up(at_height()))
check('hand resting at bottom edge (y=0.95) -> down', not hands_up(at_height(0.95)))
check('hand signing at chest (y=0.6) -> up', hands_up(at_height(0.6)))
check('one hand resting, one signing -> up', hands_up(at_height(0.95, 0.5)))
# a sign made from a resting position: hands rest in view, rise, sign, rest again -> exactly one clip
seg, t, got = TimedSegmenter(), 0.0, []
for secs, y in ((2.0, 0.95), (1.5, 0.55), (2.0, 0.95)):
    for _ in range(int(round(secs * 10))):
        lm = at_height(y)
        c = seg.update(lm, t, hands_up(lm))
        if c is not None:
            got.append((round(t, 2), len(c)))
        t += 0.1
check('sign from and back to resting-in-view -> one clip, not dropped', len(got) == 1 and seg.discarded == 0,
      f'{got}, dropped {seg.discarded}')

# --- last_span: the capture-time window of the returned clip (used by --buffered to pull every camera frame)
seg = TimedSegmenter()
got = feed(seg, [(1.0, False), (1.5, True), (1.0, False)], 10)
t0, t1 = seg.last_span if seg.last_span else (None, None)
check('last_span covers preroll .. 2 frames after the hands drop',
      got and t0 is not None and 0.6 <= t0 <= 1.0 and 2.5 <= t1 <= 2.65, f'span {seg.last_span}, clips {got}')


# --- LatestFrame history: every camera frame in a time window, in order, none drawn on by the live loop
class FakeCam:
    def __init__(self):
        self.i = 0
    def read(self):
        time.sleep(1 / 30)
        self.i += 1
        return True, np.full((4, 4, 3), self.i % 256, np.uint8)
    def release(self):
        pass

lf = LatestFrame(FakeCam(), history_s=2.0)
time.sleep(1.0)
t_a = time.time(); time.sleep(0.5); t_b = time.time()
ok, live = lf.read()
live[:] = 0                                            # simulate drawing on a frame the live loop got (should copy!)
time.sleep(0.2)
frames = lf.frames_between(t_a, t_b)
ids = [int(f[0, 0, 0]) for f in frames]
lf.release()
check('history returns ~15 frames for 0.5 s at 30 fps', 12 <= len(frames) <= 17, f'{len(frames)} frames')
check('history frames are consecutive', ids == list(range(ids[0], ids[0] + len(ids))) if ids else False, ids)
time.sleep(0.1)
lf2 = LatestFrame(FakeCam(), history_s=0.5)
time.sleep(1.2)
n_kept = len(lf2.history)
lf2.release()
check('history keeps only history_s seconds', 12 <= n_kept <= 17, f'{n_kept} kept for 0.5 s')

# --- Recorder: irregular processing times still give a constant-rate, real-time-length video
path = os.path.join(tempfile.mkdtemp(), 'rec.mp4')
rec = Recorder(cv2, path, fps=15.0)
rng, t = np.random.RandomState(0), 100.0
for i in range(25):                                     # ~7 fps with jitter, about 3.6 s
    img = np.full((480, 640, 3), (i * 10) % 255, np.uint8)
    rec.add(img, t)
    t += 1 / 7 + rng.uniform(-0.03, 0.03)
duration = t - 1 / 7 - 100.0
rec.close()
cap = cv2.VideoCapture(path)
n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
cap.release()
check('recording is 15 fps', abs(fps - 15.0) < 0.01, f'{fps}')
check('recording length matches real time', abs(n / 15.0 - duration) <= 2 / 15.0, f'{n} frames = {n / 15:.2f} s vs {duration:.2f} s')

# --- Preview: a frame left from an earlier viewer is never served; a fresh one is
p = Preview(5099)
p.publish(b'\xff\xd8OLD', time.time() - 30)            # stale: captured 30 s ago
t0 = time.time()
try:
    r = urllib.request.urlopen('http://127.0.0.1:5099/frame.jpg?after=0', timeout=5)
    status = r.status
except Exception as e:
    status = getattr(e, 'code', repr(e))
check('stale frame not served (204 after wait)', status == 204 and time.time() - t0 >= 1.9, f'status {status}')
threading.Timer(0.3, lambda: p.publish(b'\xff\xd8NEW', time.time())).start()
r = urllib.request.urlopen('http://127.0.0.1:5099/frame.jpg?after=0', timeout=5)
body = r.read()
check('fresh frame served', r.status == 200 and body.endswith(b'NEW') and r.headers['X-Seq'] == '2', f'{r.status} {body[-3:]}')
page = urllib.request.urlopen('http://127.0.0.1:5099/').read()
check('page pulls /frame.jpg', b'/frame.jpg?after=' in page and b'info.textContent' in page)

# --- Speaker sets the card volume right before each word (silently: the real commands are intercepted)
calls = []
real_run, real_popen = sign_speaker.subprocess.run, sign_speaker.subprocess.Popen
sign_speaker.subprocess.run = lambda cmd, **kw: calls.append(('run', cmd))
sign_speaker.subprocess.Popen = lambda cmd, **kw: calls.append(('popen', cmd)) or SimpleNamespace(poll=lambda: 0)
try:
    sp = Speaker('/home/arduino/audio_loud', 'plughw:CARD=UACDemoV10,DEV=0', volume=100)
    sp.say('apple')
    Speaker('/x', 'plughw:CARD=UACDemoV10,DEV=0', volume=None).say('apple')
finally:
    sign_speaker.subprocess.run, sign_speaker.subprocess.Popen = real_run, real_popen
check('speaker card parsed from device', sp.card == 'UACDemoV10', sp.card)
check('volume set before the word plays',
      [c[0] for c in calls[:2]] == ['run', 'popen'] and calls[0][1] == ['amixer', '-q', '-c', 'UACDemoV10', 'sset', 'PCM', '100%', 'unmute']
      and calls[1][1][-1].endswith('audio_loud/apple.wav'), calls[:2])
check('volume=None leaves the mixer alone', [c[0] for c in calls[2:]] == ['popen'], calls[2:])

print('ALL PASS' if not FAILED else f'FAILED: {FAILED}')
sys.exit(1 if FAILED else 0)
