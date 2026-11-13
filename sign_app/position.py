"""Tells the signer, on the Grove LCD, whether they are placed well for the camera.

States (first match wins):
  MANY   two or more faces in the picture
  NONE   no body and no face found
  EDGE   someone is partly in view but no body pose was found, or they are off-centre / their face is cut off
  CLOSE  shoulders wider than `near` (fraction of the picture width)
  FAR    shoulders narrower than `far`
  OK     none of the above

`classify()` is plain arithmetic so it can be tested without a camera. `PositionWatcher` wraps the landmarker that
sign_speaker uses: it counts faces with MediaPipe's face detector every few frames, classifies each Holistic result,
and sends a state to the LCD once it has held for `hold` seconds. The LCD is driven by the lcd_status sketch through
the router's monitor port (127.0.0.1:7500).
"""
import socket, time

import numpy as np

L_SHOULDER, R_SHOULDER, NOSE = 11, 12, 0

#            line 1 (16 chars)   line 2              backlight r, g, b
MESSAGES = {
    'OK':    ('IN POSITION',     'Ready to sign',    (0, 90, 0)),
    'NONE':  ('NO ONE IN VIEW',  'Step in front',    (0, 0, 90)),
    'EDGE':  ('NOT IN POSITION', 'Center yourself',  (90, 60, 0)),
    'CLOSE': ('TOO CLOSE',       'Move back',        (110, 0, 0)),
    'FAR':   ('TOO FAR',         'Move closer',      (110, 0, 0)),
    'MANY':  ('TOO MANY PEOPLE', 'One signer only',  (110, 0, 0)),
}


def classify(pose, n_faces, near=0.55, far=0.18):
    """pose: None or array (33, 3) of x, y, visibility in picture fractions. Returns (state, detail text)."""
    if n_faces >= 2:
        return 'MANY', f'{n_faces} faces'
    if pose is None:
        return ('NONE', 'nothing found') if n_faces == 0 else ('EDGE', 'face found but no body pose')
    ls, rs, nose = pose[L_SHOULDER], pose[R_SHOULDER], pose[NOSE]
    width = abs(float(ls[0] - rs[0]))
    mid = float(ls[0] + rs[0]) / 2
    detail = f'shoulders {width:.2f} of width, centre x {mid:.2f}, nose y {float(nose[1]):.2f}'
    if width > near:
        return 'CLOSE', detail
    if width < far:
        return 'FAR', detail
    if not 0.25 <= mid <= 0.75 or not 0.05 <= float(nose[1]) <= 0.70:
        return 'EDGE', detail
    return 'OK', detail


class LCD:
    """Sends two lines and a colour to the lcd_status sketch. Never raises: with no display the program still runs."""

    def __init__(self, addr=('127.0.0.1', 7500)):
        self.addr, self.sock, self.last_try = addr, None, 0.0

    def _connect(self):
        if self.sock is None and time.time() - self.last_try > 3:
            self.last_try = time.time()
            try:
                self.sock = socket.create_connection(self.addr, timeout=1)
                self.sock.settimeout(0.5)
            except OSError:
                self.sock = None
        return self.sock is not None

    def show(self, line1, line2, rgb):
        if not self._connect():
            return False
        try:
            self.sock.sendall(f'T{line1[:16]}\t{line2[:16]}\t{rgb[0]} {rgb[1]} {rgb[2]}\n'.encode('ascii', 'replace'))
            return True
        except OSError:
            self.sock = None
            return False

    def close(self):
        if self.sock is not None:
            self.show('Sign program', 'stopped', (40, 40, 40))
            self.sock.close()


class PositionWatcher:
    def __init__(self, inner, mp, near=0.55, far=0.18, hold=0.7, face_every=4, lcd=None):
        self.inner, self.near, self.far, self.hold, self.face_every = inner, near, far, hold, face_every
        self.faces = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
        self.lcd = lcd if lcd is not None else LCD()
        self.n, self.n_faces = 0, 0
        self.shown, self.candidate, self.since, self.last_send = None, None, 0.0, 0.0
        if hasattr(inner, 'draw'):               # sign_speaker checks for this attribute
            self.draw = inner.draw

    def process(self, rgb):
        res = self.inner.process(rgb)
        if self.n % self.face_every == 0:
            det = self.faces.process(rgb).detections
            self.n_faces = len(det) if det else 0
        self.n += 1
        pose = None
        if res.pose_landmarks is not None:
            pose = np.array([(p.x, p.y, p.visibility) for p in res.pose_landmarks.landmark], np.float32)
        state, detail = classify(pose, self.n_faces, self.near, self.far)
        now = time.time()
        if state != self.candidate:
            self.candidate, self.since = state, now
        changed = state != self.shown and now - self.since >= self.hold
        if changed or now - self.last_send > 10:  # resend now and then: covers a display plugged in late
            if changed:
                print(f'position: {MESSAGES[state][0]} ({detail})', flush=True)
            self.shown, self.last_send = state if changed else self.shown, now
            if self.shown is not None:
                self.lcd.show(*MESSAGES[self.shown])
        return res

    def close(self):
        self.lcd.close()
        self.faces.close()
        self.inner.close()
