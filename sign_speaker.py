#!/usr/bin/env python3
"""
Sign-to-speech for the Arduino UNO Q:

  Arducam --> MediaPipe Holistic --> AutoSegmenter (hands in / hands out) --> model --> aplay <word>.wav

Words are spoken only when the top guess is at least --threshold confident. A live preview (landmarks, REC state,
top-5 bars) is served as MJPEG at http://<board>:<port>/ so it can be watched from a laptop browser.

Run on the board (after setup_lite.sh; no TensorFlow or MediaPipe needed):
    cd ~/sign_demo && ~/sign_env/bin/python sign_speaker.py

Other modes:
    --test-speaker hello   play one word through the speaker and exit (exit code = aplay's)
    --benchmark            measure camera + landmark speed and model latency, no preview, no audio
    --record demo.mp4      also write the annotated video on the board (smooth regardless of the network)
    --buffered             keep every camera frame; when a sign ends, run the landmarks on ALL of its frames
                           (laptop-like input: dozens of frames per sign) before guessing. Slower to answer.

Landmarks come from fast_tracker.FastTracker by default (MediaPipe's models run directly with tflite-runtime;
detectors only when tracking is lost). --landmarker holistic / split need the full ~/demo_env.
The sign model is sign_model.tflite (from convert_sign_model.py); --weights x.h5 uses the Keras model instead.
"""
import argparse, json, os, resource, subprocess, sys, threading, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from demo_webcam import Predictor, draw_overlay, hand_visible, pack_landmarks, print_result
from fast_landmarks import make_landmarker

ARDUCAM = '/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam-5648-5MP-S-HDR_SN0001-video-index0'
AUDIO_DIR = '/home/arduino/audio'
AUDIO_DEVICE = 'plughw:CARD=UACDemoV10,DEV=0'


# ----------------------------------------------------------------------------- audio
class Speaker:
    """Plays <audio_dir>/<word>.wav with aplay, like ~/speak.py. A new word cuts off one that is still playing,
    so a backlog of clips never builds up behind the signer. With `volume`, the card's PCM level is set right before
    each word: something on the board kept putting it back to 40% between plays."""

    def __init__(self, audio_dir, device, volume=None):
        self.audio_dir, self.device, self.proc, self.volume = audio_dir, device, None, volume
        card = [kv.split('=', 1)[1] for kv in device.split(':', 1)[-1].split(',') if kv.startswith('CARD=')]
        self.card = card[0] if card else None

    def path(self, word):
        return os.path.join(self.audio_dir, f"{word.replace(' ', '_')}.wav")

    def missing(self, words):
        return [w for w in words if not os.path.isfile(self.path(w))]

    def say(self, word):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        if self.volume is not None and self.card:
            subprocess.run(['amixer', '-q', '-c', self.card, 'sset', 'PCM', f'{self.volume}%', 'unmute'], check=False)
        self.proc = subprocess.Popen(['aplay', '-q', '-D', self.device, self.path(word)])
        return self.proc


class SpeechGate:
    """Decides whether a prediction gets spoken: confident enough, and not the same word again within `cooldown`."""

    def __init__(self, threshold, cooldown):
        self.threshold, self.cooldown = threshold, cooldown
        self.last_word, self.last_time = None, -1e9

    def check(self, result, now):
        if not result:
            return None
        word, p = result[0]
        if p < self.threshold:
            return None
        if word == self.last_word and now - self.last_time < self.cooldown:
            return None
        self.last_word, self.last_time = word, now
        return word


def hands_up(lm, rest_line=0.8):
    """A hand counts as up (signing) when detected AND its centre is above `rest_line` (fraction of frame height,
    0 = top). With the camera at table height, hands resting on the table stay in view at the bottom edge; without
    this, they never 'leave' and every sign looks like resting hands."""
    for sl in (slice(468, 489), slice(522, 543)):
        y = lm[sl, 1]
        if not np.isnan(y).all() and np.nanmean(y) < rest_line:
            return True
    return False


class TimedSegmenter:
    """demo_webcam.AutoSegmenter's hands-in / hands-out rule measured in seconds instead of frames, so it behaves the
    same at 7 or 15 fps (8 frames is 0.27 s at 30 fps but over 1 s at 7 fps; its 300-frame cap was 30 s here).
    A sign starts after `start` s of visible hands and ends `end` s after they drop. Only a clip that ended that way
    is returned: if hands stay in view for `max_len` s (someone resting their hands in frame, not signing) the clip
    is dropped, so nothing is guessed or spoken for it."""

    def __init__(self, start=0.1, end=0.4, max_len=4.0, preroll=0.3, min_frames=4):
        self.start, self.end, self.max_len, self.preroll, self.min_frames = start, end, max_len, preroll, min_frames
        self.recent = deque()                         # (t, lm) before a sign starts
        self.clip, self.clip_t, self.active, self.hand_since = [], [], False, None
        self.t_start = self.last_hand = 0.0
        self.last_span = None                         # (first, last) capture time of the last returned clip
        self.trailing = 0                             # frames without hands at the end of the clip
        self.discarded = 0                            # clips dropped because hands never left the frame
        self.need_drop = False                        # after a drop: wait for hands to leave before a new sign
        self.no_hand_since = None

    @property
    def recording(self):
        return self.active

    def update(self, lm, t, seen=None):
        """Feed one frame captured at time t (seconds). Returns a finished clip (T, 543, 3) or None.
        `seen` overrides whether hands count as up for this frame (default: any hand detected)."""
        seen = hand_visible(lm) if seen is None else seen
        if self.need_drop:                            # don't start the tail of a resting stretch as a new sign
            if seen:
                self.no_hand_since = None
                return None
            self.no_hand_since = self.no_hand_since if self.no_hand_since is not None else t
            if t - self.no_hand_since < self.end:
                return None
            self.need_drop, self.no_hand_since = False, None
        if not self.active:
            self.recent.append((t, lm))
            while self.recent and t - self.recent[0][0] > self.preroll + self.start:
                self.recent.popleft()
            self.hand_since = (self.hand_since if self.hand_since is not None else t) if seen else None
            if seen and t - self.hand_since >= self.start:
                self.active, self.clip = True, [f for _, f in self.recent]
                self.clip_t = [tt for tt, _ in self.recent]
                self.t_start, self.last_hand, self.trailing = t, t, 0
            return None
        self.clip.append(lm); self.clip_t.append(t)
        if seen:
            self.last_hand, self.trailing = t, 0
        else:
            self.trailing += 1
        ended = t - self.last_hand >= self.end
        if ended or t - self.t_start >= self.max_len:
            keep = len(self.clip) - max(0, self.trailing - 2)               # keep 2 no-hand frames, like AutoSegmenter
            clip, span = self.clip[:keep], (self.clip_t[0], self.clip_t[keep - 1])
            self.clip, self.clip_t, self.active, self.hand_since = [], [], False, None
            self.recent.clear()
            if not ended:
                self.discarded += 1
                self.need_drop = True
                return None
            if len(clip) < self.min_frames:
                return None
            self.last_span = span
            return np.stack(clip)
        return None


class Recorder:
    """Writes annotated frames to an MP4 at a constant `fps`, repeating the latest frame to fill the gaps, so the file
    plays smoothly in real time whatever rate the board processes at."""

    def __init__(self, cv2, path, fps=15.0):
        self.cv2, self.path, self.fps, self.w, self.t0, self.written, self.last = cv2, path, fps, None, None, 0, None

    def add(self, frame, t):
        if self.w is None:                               # size from the first real frame, not the requested one
            h, w = frame.shape[:2]
            self.w = self.cv2.VideoWriter(self.path, self.cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
            if not self.w.isOpened():
                sys.exit(f'could not open {self.path} for writing')
            self.t0 = t
        due = int((t - self.t0) * self.fps) + 1          # output frames that should exist by time t
        while self.written < due:
            self.w.write(self.last if self.last is not None and self.written < due - 1 else frame)
            self.written += 1
        self.last = frame

    def close(self):
        if self.w is None:
            return
        self.w.release()
        print(f'recorded {self.written} frames ({self.written / self.fps:.1f} s at {self.fps:g} fps) to {self.path}', flush=True)


# ----------------------------------------------------------------------------- preview
class Preview:
    """Holds the newest annotated JPEG. The page at / pulls one frame at a time from /frame.jpg and asks for the
    next only after showing it, so frames can never queue up between board and browser (an MJPEG stream can: some
    browsers buffer it and drift tens of seconds behind). /video still serves MJPEG for tools. Frames are only
    encoded while someone is watching, and nothing older than MAX_AGE seconds is ever served."""

    MAX_AGE = 1.0
    PAGE = b"""<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sign speaker</title>
<body style="margin:0;background:#111;color:#bbb;font:13px system-ui,sans-serif">
<img id="v" alt="" style="display:block;max-width:100%;margin:auto">
<div id="s" style="text-align:center;padding:6px">connecting...</div>
<script>
let seq = 0, shown = 0, t0 = performance.now();
const img = document.getElementById('v'), info = document.getElementById('s');
async function next() {
  try {
    const r = await fetch('/frame.jpg?after=' + seq, {cache: 'no-store'});
    if (r.status === 200) {
      seq = Number(r.headers.get('X-Seq'));
      const url = URL.createObjectURL(await r.blob());
      await new Promise(done => { img.onload = img.onerror = done; img.src = url; });
      URL.revokeObjectURL(url);
      shown++;
    }
  } catch (e) {
    info.textContent = 'reconnecting...';
    await new Promise(done => setTimeout(done, 1000));
  }
  const dt = (performance.now() - t0) / 1000;
  if (dt >= 2) { info.textContent = (shown / dt).toFixed(1) + ' frames/s shown'; shown = 0; t0 = performance.now(); }
  next();
}
next();
</script></body>"""

    def __init__(self, port):
        self.cond, self.jpeg, self.seq, self.viewers, self.t_capture = threading.Condition(), None, 0, 0, 0.0
        self.last_poll = 0.0
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                path, _, query = self.path.partition('?')
                if path == '/':
                    self.send_response(200); self.send_header('Content-Type', 'text/html')
                    self.end_headers(); self.wfile.write(Preview.PAGE)
                elif path == '/frame.jpg':
                    after = dict(kv.partition('=')[::2] for kv in query.split('&') if kv).get('after', '0')
                    preview.frame(self, int(after) if after.isdigit() else 0)
                elif path == '/video':
                    preview.stream(self)
                else:
                    self.send_error(404)

        self.server = ThreadingHTTPServer(('0.0.0.0', port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def watched(self):
        return self.viewers > 0 or time.time() - self.last_poll < 2.0

    def _fresh(self, after):
        return self.jpeg is not None and self.seq > after and time.time() - self.t_capture < self.MAX_AGE

    def frame(self, h, after):
        """One frame newer than `after` and captured < MAX_AGE ago; 204 if none arrives within 2 s."""
        self.last_poll = time.time()                  # turns encoding on, so a fresh frame follows shortly
        with self.cond:
            if after > self.seq:                      # the board restarted since the page last asked
                after = 0
            ok = self.cond.wait_for(lambda: self._fresh(after), timeout=2.0)
            jpeg, seq, t_cap = self.jpeg, self.seq, self.t_capture
        try:
            if not ok:
                h.send_response(204); h.send_header('Cache-Control', 'no-store'); h.end_headers()
                return
            h.send_response(200)
            for k, v in (('Content-Type', 'image/jpeg'), ('Content-Length', str(len(jpeg))), ('Cache-Control', 'no-store'),
                         ('X-Seq', str(seq)), ('X-Capture-Time', f'{t_cap:.3f}')):
                h.send_header(k, v)
            h.end_headers(); h.wfile.write(jpeg)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def publish(self, jpeg, t_capture=0.0):
        with self.cond:
            self.jpeg, self.seq, self.t_capture = jpeg, self.seq + 1, t_capture
            self.cond.notify_all()

    def stream(self, h):
        h.send_response(200)
        h.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        h.send_header('Cache-Control', 'no-cache')
        h.end_headers()
        with self.cond:
            self.viewers += 1
        seen = -1
        try:
            while True:
                with self.cond:
                    if not self.cond.wait_for(lambda: self._fresh(seen), timeout=2.0):
                        continue                      # never send a frame left over from an earlier viewer
                    jpeg, seen, t_cap = self.jpeg, self.seq, self.t_capture
                # X-Capture-Time (board clock) lets a client measure how stale each frame is on arrival
                h.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nX-Capture-Time: %.3f\r\nContent-Length: %d\r\n\r\n'
                              % (t_cap, len(jpeg)))
                h.wfile.write(jpeg + b'\r\n')
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with self.cond:
                self.viewers -= 1


# ----------------------------------------------------------------------------- main loop
def open_capture(cv2, source, width, height):
    """The Arducam by-id path (or any /dev/video*) is opened with V4L2 + MJPG like ~/stream.py; anything else
    is treated as a video file."""
    is_device = source.startswith('/dev/')
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2) if is_device else cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f'Could not open {source}. Is another program (e.g. ~/stream.py) using the camera?')
    if is_device:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # keep latency low when MediaPipe runs slower than the camera
    return cap, is_device


class LatestFrame:
    """Reads the camera on its own thread and hands out only the newest frame, so the landmark loop never waits on
    cap.read() and never processes a stale buffered frame. Also counts the rate the camera really delivers."""

    def __init__(self, cap, history_s=0.0):
        self.cap, self.cond = cap, threading.Condition()
        self.frame, self.seq, self.taken, self.ok = None, 0, 0, True
        self.history_s, self.history = history_s, deque()   # (capture time, frame) for --buffered
        self.stop = threading.Event()           # separate from `ok`, which each read overwrites
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.ok and not self.stop.is_set():
            ok, frame = self.cap.read()
            with self.cond:
                self.ok = ok
                if ok:
                    self.frame, self.seq = frame, self.seq + 1
                    if self.history_s:
                        t = time.time()
                        self.history.append((t, frame))
                        while self.history and t - self.history[0][0] > self.history_s:
                            self.history.popleft()
                self.cond.notify_all()

    def read(self):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != self.taken or not self.ok, timeout=5.0)
            if self.seq == self.taken:
                return False, None
            self.taken = self.seq
            # with a history, hand out a copy: drawing on it must not change the frames kept for --buffered
            return True, (self.frame.copy() if self.history_s else self.frame)

    def frames_between(self, t0, t1):
        """Every kept camera frame captured in [t0, t1] (callers must not draw on them)."""
        with self.cond:
            return [f for t, f in self.history if t0 <= t <= t1]

    def release(self):
        self.stop.set()
        self.thread.join(timeout=2.0)           # the reader may be inside cap.read(); releasing under it crashes OpenCV
        self.cap.release()


def landmarker(a):
    return make_landmarker(a.landmarker, a.complexity, a.hand_complexity, a.face_every, a.hand_model, a.palm_every)


def make_predictor(a):
    if a.weights:
        return Predictor(a.weights, a.signmap, a.max_len, a.dim)
    from sign_infer_lite import LitePredictor
    return LitePredictor(a.model, a.signmap, a.max_len, num_threads=a.model_threads)


def benchmark(a, predictor):
    """Headless speed check of camera read + landmarks (+ model latency with --weights)."""
    import cv2, numpy as np
    _, lm = landmarker(a)
    raw, live = open_capture(cv2, a.camera, a.width, a.height)
    cap = LatestFrame(raw) if live else raw
    t_read = t_lm = 0.0; n = 0; hands = 0
    print(f'benchmarking {a.bench_frames} frames (landmarker={a.landmarker}) ... keep your hands in view', flush=True)
    cpu0, wall0, seq0 = None, None, 0
    while n < a.bench_frames:
        t0 = time.time(); ok, frame = cap.read(); t1 = time.time()
        if not ok:
            break
        packed = pack_landmarks(lm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))); t2 = time.time()
        if n == 3:
            c = os.times(); cpu0, wall0 = c.user + c.system, time.time()
            seq0 = cap.seq if live else 0
        if n >= 3:                                   # skip warm-up frames
            t_read += t1 - t0; t_lm += t2 - t1
            hands += not (np.isnan(packed[468:489]).all() and np.isnan(packed[522:543]).all())
        n += 1
    wall = time.time() - wall0 if wall0 is not None else float('nan')
    c = os.times(); cpu = 100 * (c.user + c.system - cpu0) / max(wall, 1e-6) if cpu0 is not None else float('nan')
    cam_fps = (cap.seq - seq0) / max(wall, 1e-6) if live else float('nan')     # same window as the processed rate
    cap.release(); lm.close()
    m = max(n - 3, 1)
    print(f'camera delivers     : {cam_fps:6.1f} frames/second')
    print(f'waiting for a frame : {1000 * t_read / m:6.1f} ms/frame')
    print(f'landmarks           : {1000 * t_lm / m:6.1f} ms/frame   (hands seen in {hands}/{m} frames)')
    for stage, (tot, cnt) in sorted(getattr(lm, 'timing', {}).items()):
        print(f'   {stage:16s} : {1000 * tot / max(cnt, 1):6.1f} ms per run, ran on {cnt} frames')
    print(f'=> processed rate   : {m / max(wall, 1e-6):6.1f} frames/second (wall clock)')
    print(f'CPU while running   : {cpu:6.0f} % of one core (400 % = all four)')
    if predictor is not None:
        fake = np.random.RandomState(0).rand(30, 543, 3).astype(np.float32)
        predictor.predict(fake)
        t = time.time()
        for _ in range(5): predictor.predict(fake)
        print(f'model + preprocess  : {1000 * (time.time() - t) / 5:6.1f} ms per sign (30-frame clip)')
    print(f'peak memory (RSS)   : {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:6.0f} MB')


def landmarks_for_frames(a, frames):
    """--buffered: run a fresh landmarker over every frame of one sign, like the laptop demo's recording."""
    import cv2
    _, lm = landmarker(a)
    try:
        packed = [pack_landmarks(lm.process(cv2.cvtColor(cv2.flip(f, 1) if a.flip else f, cv2.COLOR_BGR2RGB)))
                  for f in frames]
    finally:
        lm.close()
    return np.stack(packed)


def run(a, predictor, speaker, preview):
    import cv2
    mp, holistic = landmarker(a)
    raw, live = open_capture(cv2, a.camera, a.width, a.height)
    if a.buffered and not live:
        sys.exit('--buffered needs a live camera')
    cap = LatestFrame(raw, history_s=a.max_sign + 3.0 if a.buffered else 0.0) if live else raw
    seg, gate = TimedSegmenter(end=a.sign_end, max_len=a.max_sign), SpeechGate(a.threshold, a.cooldown)
    worker, pending = ThreadPoolExecutor(max_workers=1), None     # the sign model runs off the frame loop
    rec = Recorder(cv2, a.record, a.record_fps) if a.record else None
    state = dict(auto=True, flip=a.flip, recording=False, result=None, fps=0.0, nframes=0)
    t_prev, n, avg_dt = time.time(), 0, 0.1
    t_status, seen_n, seen_hands = time.time(), 0, 0                 # for the periodic status line
    where = f' -- preview on port {a.port}' if preview else ''
    print(f'Running{where}. Sign with your hands in view, drop them to finish a sign. Ctrl+C to stop.', flush=True)
    try:
        while True:
            ok, frame = cap.read()
            t_cap = time.time()
            if not ok:
                if live:
                    print('Camera returned no frame; stopping.')
                break
            n += 1
            busy = a.buffered and pending is not None
            busy_frames = pending[3] if busy else 0          # read now: `pending` may be cleared later this pass
            if a.flip:
                frame = cv2.flip(frame, 1)
            res = None
            if not busy:                                       # while a buffered sign is processed, leave it the CPU
                res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                packed = pack_landmarks(res)
                up = hands_up(packed, a.rest_line)
                seen_n += 1; seen_hands += up
                clip = seg.update(packed, t_cap, up)
            else:
                clip = None
            state['recording'], state['nframes'] = (seg.recording, len(seg.clip)) if not busy else (False, 0)

            if clip is not None:
                if pending is not None:
                    pending[1].result()                        # one sign at a time keeps the spoken order right
                if a.buffered:
                    t0, t1 = seg.last_span
                    frames = cap.frames_between(t0 - a.buffer_preroll, t1 + 0.1)
                    print(f'sign ended: processing all {len(frames)} camera frames '
                          f'(the live loop saw {len(clip)})', flush=True)
                    started = time.time()
                    job = worker.submit(lambda fr=frames: predictor.predict(landmarks_for_frames(a, fr)))
                    pending = (f'buffered ({len(frames)} frames)', job, started, len(frames))
                else:
                    pending = (f'frame {n} ({len(clip)} frames)', worker.submit(predictor.predict, clip))
            if pending is not None and pending[1].done():
                tag, fut = pending[0], pending[1]
                if len(pending) > 2:
                    tag += f', {time.time() - pending[2]:.1f} s to process'
                pending = None
                state['result'] = fut.result()
                print_result(tag, state['result'])
                word = gate.check(state['result'], time.time())
                if word:
                    print(f'  -> speaking "{word}"', flush=True)
                    speaker.say(word)
                elif state['result']:
                    print(f'  -> not spoken (top guess below {100 * a.threshold:.0f}% or repeated)', flush=True)

            if a.status_every and time.time() - t_status >= a.status_every:
                print(f'status: hands up in {seen_hands}/{seen_n} frames, recording={seg.recording}, '
                      f'waiting_for_hands_to_leave={seg.need_drop}, dropped_clips={seg.discarded}, '
                      f'fps={state["fps"]:.1f}, busy={busy}', flush=True)
                t_status, seen_n, seen_hands = time.time(), 0, 0
            # average the frame interval, then invert: averaging 1/dt lets one 2 ms gap read as 50+ fps
            now = time.time(); avg_dt = 0.9 * avg_dt + 0.1 * (now - t_prev); t_prev = now
            state['fps'] = 1.0 / max(avg_dt, 1e-6)
            show = preview is not None and preview.watched and n % 2 == 0   # every 2nd frame is plenty for a preview
            if show or rec:
                if res is None:
                    pass                                       # buffered sign being processed: no live landmarks
                elif hasattr(holistic, 'draw'):
                    holistic.draw(frame)
                else:
                    draw, hol = mp.solutions.drawing_utils, mp.solutions.holistic
                    for lms, conn in ((res.pose_landmarks, hol.POSE_CONNECTIONS), (res.left_hand_landmarks, hol.HAND_CONNECTIONS),
                                      (res.right_hand_landmarks, hol.HAND_CONNECTIONS)):
                        if lms is not None:
                            draw.draw_landmarks(frame, lms, conn)
                frame = draw_overlay(cv2, frame, state)
                if busy and pending is not None:               # still working (it may have finished this pass)
                    cv2.putText(frame, f'PROCESSING {busy_frames} frames...', (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2, cv2.LINE_AA)
                if rec:
                    rec.add(frame, t_cap)
                if show:
                    # small frames: the board-to-laptop link (a phone hotspot) carried as little as 19 KB/s
                    small = cv2.resize(frame, (a.preview_width, a.preview_width * frame.shape[0] // frame.shape[1]),
                                       interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    if ok:
                        preview.publish(buf.tobytes(), t_cap)
    except KeyboardInterrupt:
        print('\nstopped.')
    finally:
        worker.shutdown(wait=True)
        if rec:
            rec.close()
        cap.release(); holistic.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='sign_model.tflite', help='TFLite sign model from convert_sign_model.py')
    ap.add_argument('--model-threads', type=int, default=2, help='threads for the sign model (landmarks keep running meanwhile)')
    ap.add_argument('--weights', help='use the Keras .h5 model instead of --model (needs TensorFlow)')
    ap.add_argument('--signmap', default='sign_to_prediction_index_map.json')
    ap.add_argument('--max-len', type=int, default=128, help='MUST equal the CFG_MAX_LEN used in training')
    ap.add_argument('--dim', type=int, default=192, help='MUST equal the training dim (default 192)')
    ap.add_argument('--camera', default=ARDUCAM, help='camera device path, or a video file to run the same pipeline on')
    ap.add_argument('--width', type=int, default=640); ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--flip', action='store_true', help='mirror the image left-right before MediaPipe')
    ap.add_argument('--landmarker', default='custom', choices=['custom', 'split', 'holistic'])
    ap.add_argument('--hand-model', default='full', choices=['full', 'lite'], help='custom: hand landmark model (Holistic uses full)')
    ap.add_argument('--complexity', type=int, default=0, choices=[0, 1, 2], help='Holistic model_complexity; 0 is fastest')
    ap.add_argument('--hand-complexity', type=int, default=1, choices=[0, 1], help='split: Hands model; 1 = the full model Holistic uses')
    ap.add_argument('--face-every', type=int, default=2, help='split/custom: refresh the face mesh every N frames while signing')
    ap.add_argument('--palm-every', type=int, default=6, help='custom: palm detection every N frames while one hand is tracked')
    ap.add_argument('--threshold', type=float, default=0.40, help='only speak when the top guess is at least this confident')
    ap.add_argument('--cooldown', type=float, default=1.0, help='seconds before the same word may be spoken again')
    ap.add_argument('--audio-dir', default=AUDIO_DIR); ap.add_argument('--audio-device', default=AUDIO_DEVICE)
    ap.add_argument('--volume', type=int, default=100, help='speaker PCM %% set before each word (-1 = leave as is)')
    ap.add_argument('--port', type=int, default=5000); ap.add_argument('--no-preview', action='store_true')
    ap.add_argument('--preview-width', type=int, default=400, help='width of preview frames sent to the browser')
    ap.add_argument('--record', metavar='OUT.mp4', help='also write the annotated video on the board')
    ap.add_argument('--record-fps', type=float, default=15.0)
    ap.add_argument('--sign-end', type=float, default=0.4, help='seconds without hands that end a sign')
    ap.add_argument('--max-sign', type=float, default=4.0, help='hands in view longer than this (s) = not a sign; dropped')
    ap.add_argument('--rest-line', type=float, default=0.8,
                    help='hands whose centre is below this fraction of the frame height count as down (resting)')
    ap.add_argument('--status-every', type=float, default=5.0, help='seconds between status lines in the log (0 = off)')
    ap.add_argument('--buffered', action='store_true', help='run landmarks on every camera frame of a finished sign')
    ap.add_argument('--buffer-preroll', type=float, default=0.3, help='--buffered: extra seconds before the detected start')
    ap.add_argument('--test-speaker', metavar='WORD', help='play one word through the speaker and exit')
    ap.add_argument('--benchmark', action='store_true', help='measure speed and memory, then exit')
    ap.add_argument('--bench-frames', type=int, default=100)
    a = ap.parse_args()

    speaker = Speaker(a.audio_dir, a.audio_device, None if a.volume < 0 else a.volume)
    if a.test_speaker:
        if speaker.missing([a.test_speaker]):
            sys.exit(f'no audio file {speaker.path(a.test_speaker)}')
        sys.exit(speaker.say(a.test_speaker).wait())

    if a.benchmark:
        return benchmark(a, make_predictor(a))

    missing = speaker.missing(json.load(open(a.signmap)))
    if missing:
        print(f'WARNING: {len(missing)} signs have no audio file in {a.audio_dir} and will be silent: {", ".join(missing)}')
    predictor = make_predictor(a)
    print(f'model loaded: {len(predictor.names)} classes, max_len={a.max_len}, dim={a.dim}, '
          f'threshold={100 * a.threshold:.0f}%, landmarker={a.landmarker}, '
          f'model={a.weights or a.model}', flush=True)
    run(a, predictor, speaker, None if a.no_preview else Preview(a.port))


if __name__ == '__main__':
    main()
