"""MediaPipe Holistic on several CPU cores.

One Holistic graph uses about 1.3 of the UNO Q's four cores and takes ~225 ms per frame (~4.5 fps). This runs N
Holistic instances, each in its own process, and deals camera frames to them in turn, so about N times as many frames
get landmarks. Each instance tracks its own every-Nth-frame stream, which is the frame spacing a single instance had
before, so tracking behaves as it did.

`process(rgb)` hands the frame to the next worker and returns the result that worker has just finished, i.e. the
landmarks of the frame given N calls earlier (about 0.2-0.3 s old). The first N calls return an empty result.
`draw(frame)` replaces the picture with the frame those landmarks belong to before drawing, so the preview stays in step.
"""
import multiprocessing as mp_
from collections import deque
from types import SimpleNamespace

import numpy as np

PARTS = ('face_landmarks', 'left_hand_landmarks', 'pose_landmarks', 'right_hand_landmarks')
EMPTY = SimpleNamespace(**{p: None for p in PARTS})


def _worker(conn, complexity):
    import sys
    # mediapipe imports TensorFlow when it is installed, though Holistic never uses it: ~300 MB and ~19 s per worker.
    # A None entry makes `import tensorflow` fail, and mediapipe treats it as an optional dependency.
    sys.modules['tensorflow'] = None
    import mediapipe as mp
    hol = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=complexity,
                                         min_detection_confidence=0.5, min_tracking_confidence=0.5)
    conn.send('ready')
    try:
        while True:
            msg = conn.recv()                     # EOFError when the parent is gone: exit instead of lingering
            if msg is None:
                break
            shape, buf = msg
            res = hol.process(np.frombuffer(buf, np.uint8).reshape(shape))
            conn.send([None if getattr(res, p) is None else getattr(res, p).SerializeToString() for p in PARTS])
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        hol.close()


class ParallelHolistic:
    def __init__(self, workers=3, complexity=1):
        import mediapipe as mp
        from mediapipe.framework.formats import landmark_pb2
        self.mp, self._pb, self.last = mp, landmark_pb2.NormalizedLandmarkList, EMPTY
        ctx = mp_.get_context('spawn')            # fresh interpreters: no TensorFlow or camera state copied in
        self.conns, self.procs = [], []
        for _ in range(workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, complexity), daemon=True)
            p.start(); child.close()
            self.conns.append(parent); self.procs.append(p)
        for c in self.conns:
            if c.recv() != 'ready':
                raise RuntimeError('Holistic worker failed to start')
        self.n, self.k = workers, 0
        self.busy = [False] * workers
        self.frames = deque()                     # frames whose landmarks are still being computed, oldest first
        self.shown = None                         # frame that the last returned landmarks belong to

    def _parse(self, parts):
        out = {}
        for name, blob in zip(PARTS, parts):
            if blob is None:
                out[name] = None
            else:
                m = self._pb(); m.ParseFromString(blob); out[name] = m
        return SimpleNamespace(**out)

    def process(self, rgb):
        i = self.k % self.n
        res = EMPTY
        if self.busy[i]:                          # this worker's previous frame is the oldest one outstanding
            res = self._parse(self.conns[i].recv())
            self.shown = self.frames.popleft()
        self.conns[i].send((rgb.shape, rgb.tobytes()))
        self.busy[i] = True
        self.frames.append(rgb)
        self.k += 1
        self.last = res
        return res

    def draw(self, frame):
        """Put the landmarks' own frame into `frame` (BGR, in place), then draw them like sign_speaker does."""
        if self.shown is None:
            return
        res, mp = self.last, self.mp
        frame[:] = self.shown[..., ::-1]
        draw, hol = mp.solutions.drawing_utils, mp.solutions.holistic
        for lms, conn in ((res.pose_landmarks, hol.POSE_CONNECTIONS), (res.left_hand_landmarks, hol.HAND_CONNECTIONS),
                          (res.right_hand_landmarks, hol.HAND_CONNECTIONS)):
            if lms is not None:
                draw.draw_landmarks(frame, lms, conn)

    def close(self):
        for c in self.conns:
            try:
                c.send(None)
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
