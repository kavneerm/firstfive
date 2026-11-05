#!/usr/bin/env python3
"""
Webcam demo for the ISLR isolated-sign model (250 everyday signs, Kaggle "asl-signs" vocabulary).

  webcam --> MediaPipe Holistic (543 landmarks/frame, same layout as the training data)
         --> record one sign --> the notebook's Preprocess --> model --> top-5 guesses

Run (from this folder, after installing the packages -- see README steps in the chat):
    python demo_webcam.py --weights islr250-fold0-fold0-best.h5 --signmap sign_to_prediction_index_map.json --max-len 128

Keys in the video window:
    SPACE  start / stop recording one sign (manual mode, default)
    A      toggle AUTO mode (records while a hand is visible, predicts when hands drop out)
    F      flip the camera image left-right before MediaPipe (try both! see notes)
    C      clear the last result        Q / ESC  quit

Other modes:
    --list-signs         print the 250 signs the model knows
    --dry-run            load the model and run it on random data (checks your install + weights, no camera needed)
    --video clip.mp4     predict a recorded clip (whole file = one sign), no window; add --auto to split into signs
    --benchmark          measure MediaPipe speed, model latency and memory (no window) -- run this on the UNO Q first
    --headless           camera + auto mode with NO window; prints each prediction (and appends it to --out-file)
"""
import argparse, json, os, sys, time
from collections import deque
import numpy as np

N_LM = 543
LHAND = slice(468, 489)
RHAND = slice(522, 543)


# ----------------------------------------------------------------------------- landmarks
def pack_landmarks(res):
    """MediaPipe Holistic result -> float32 (543, 3), NaN where nothing was detected.
    Order = face 0-467, left hand 468-488, pose 489-521, right hand 522-542 (same as the training data)."""
    out = np.full((N_LM, 3), np.nan, np.float32)
    for lms, off, n in ((res.face_landmarks, 0, 468), (res.left_hand_landmarks, 468, 21),
                        (res.pose_landmarks, 489, 33), (res.right_hand_landmarks, 522, 21)):
        if lms is not None:
            pts = lms.landmark[:n]
            out[off:off + len(pts)] = [(p.x, p.y, p.z) for p in pts]
    return out


def hand_visible(lm):
    return not (np.isnan(lm[LHAND]).all() and np.isnan(lm[RHAND]).all())


class AutoSegmenter:
    """Cuts a stream of per-frame landmarks into sign-sized clips using hand visibility:
    starts after `start_frames` consecutive frames with a hand, ends after `end_frames` frames without one."""

    def __init__(self, start_frames=3, end_frames=8, min_len=6, max_frames=300, preroll=4):
        self.start_frames, self.end_frames, self.min_len, self.max_frames = start_frames, end_frames, min_len, max_frames
        self.recent = deque(maxlen=preroll + start_frames)
        self.clip, self.active, self.hand_run, self.miss = [], False, 0, 0

    @property
    def recording(self):
        return self.active

    def update(self, lm):
        """Feed one frame. Returns a finished clip as an array (T,543,3), or None."""
        seen = hand_visible(lm)
        if not self.active:
            self.recent.append(lm)
            self.hand_run = self.hand_run + 1 if seen else 0
            if self.hand_run >= self.start_frames:
                self.active, self.clip, self.miss = True, list(self.recent), 0
            return None
        self.clip.append(lm)
        self.miss = 0 if seen else self.miss + 1
        if self.miss >= self.end_frames or len(self.clip) >= self.max_frames:
            clip = self.clip[:len(self.clip) - max(0, self.miss - 2)]          # drop most of the trailing no-hand frames
            self.clip, self.active, self.hand_run, self.miss = [], False, 0, 0
            self.recent.clear()
            return np.stack(clip) if len(clip) >= self.min_len else None
        return None


# ----------------------------------------------------------------------------- model
class Predictor:
    def __init__(self, weights, signmap, max_len, dim):
        import islr_model as M                       # imported here so --list-signs works without TensorFlow
        self.M, self.max_len = M, max_len
        sign_to_idx = json.load(open(signmap))
        self.names = {int(i): n for n, i in sign_to_idx.items()}
        self.model = M.load_model(weights, max_len=max_len, dim=dim, num_classes=len(self.names))
        # graph mode: eager calls take ~670 ms per sign on the UNO Q, the compiled function ~105 ms (same outputs)
        import tensorflow as tf
        self._infer = tf.function(lambda t: self.model(t, training=False),
                                  input_signature=[tf.TensorSpec((1, max_len, M.CHANNELS), tf.float32)])

    def predict(self, clip, topk=5):
        x = self.M.prepare_input(clip, self.max_len)
        if x is None:
            return None
        logits = self._infer(x).numpy()[0]
        p = np.exp(logits - logits.max()); p /= p.sum()
        order = np.argsort(-p)[:topk]
        return [(self.names.get(int(i), f'class_{i}'), float(p[i])) for i in order]


def print_result(tag, result):
    if result is None:
        print(f'{tag}: no usable frames (no face/hands found)')
    else:
        print(f'{tag}: ' + ' | '.join(f'{n} {100 * p:.0f}%' for n, p in result))


# ----------------------------------------------------------------------------- mediapipe / camera
def make_holistic(complexity):
    try:
        import mediapipe as mp
    except ImportError:
        sys.exit('mediapipe is not installed.  pip install "mediapipe==0.10.14"')
    if not hasattr(mp, 'solutions'):
        sys.exit(f'mediapipe {mp.__version__} no longer has the Holistic API this demo needs.  pip install "mediapipe==0.10.14"')
    return mp, mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=complexity,
                                              min_detection_confidence=0.5, min_tracking_confidence=0.5)


def open_camera(index):
    import cv2
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f'Could not open camera {index}. Try --camera 1, and close other apps that use the webcam.')
    return cap


def draw_overlay(cv2, img, state):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 34), (30, 30, 30), -1)
    mode = 'AUTO' if state['auto'] else 'MANUAL'
    status = 'REC' if state['recording'] else ('ready' if state['auto'] else 'SPACE = record')
    col = (0, 0, 255) if state['recording'] else (200, 200, 200)
    cv2.putText(img, f'{mode} | {status} | flip={"on" if state["flip"] else "off"} | {state["fps"]:.0f} fps | frames {state["nframes"]}',
                (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    res = state['result']
    if res:
        cv2.rectangle(img, (0, h - 150), (w, h), (30, 30, 30), -1)
        for i, (name, p) in enumerate(res[:5]):
            y = h - 125 + i * 26
            bar = int(p * (w - 220))
            cv2.rectangle(img, (170, y - 14), (170 + bar, y + 4), (0, 200, 0) if i == 0 else (120, 120, 120), -1)
            cv2.putText(img, f'{name[:16]:16s}', (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 if i == 0 else 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f'{100 * p:.0f}%', (178 + bar, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def run_webcam(a, predictor):
    import cv2
    mp, holistic = make_holistic(a.complexity)
    draw = mp.solutions.drawing_utils
    hol = mp.solutions.holistic
    cap = open_camera(a.camera)
    if a.width: cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    if a.height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    seg = AutoSegmenter()
    manual = []                       # frames of the current manual recording
    state = dict(auto=a.auto, flip=a.flip, recording=False, result=None, fps=0.0, nframes=0)
    last, t_prev = None, time.time()
    print('Window open. SPACE = record a sign, again = predict. Q to quit.')
    while True:
        ok, frame = cap.read()
        if not ok:
            print('Camera returned no frame; stopping.'); break
        if state['flip']:
            frame = cv2.flip(frame, 1)
        res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        lm = pack_landmarks(res)

        if state['auto']:
            clip = seg.update(lm)
            state['recording'], state['nframes'] = seg.recording, len(seg.clip)
            if clip is not None:
                state['result'] = predictor.predict(clip); print_result(f'auto ({len(clip)} frames)', state['result'])
        elif state['recording']:
            manual.append(lm); state['nframes'] = len(manual)

        for lms, conn in ((res.pose_landmarks, hol.POSE_CONNECTIONS), (res.left_hand_landmarks, hol.HAND_CONNECTIONS),
                          (res.right_hand_landmarks, hol.HAND_CONNECTIONS)):
            if lms is not None:
                draw.draw_landmarks(frame, lms, conn)
        now = time.time(); state['fps'] = 0.9 * state['fps'] + 0.1 / max(now - t_prev, 1e-6); t_prev = now
        cv2.imshow('ISLR sign demo', draw_overlay(cv2, frame, state))

        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k == ord(' ') and not state['auto']:
            if not state['recording']:
                manual, state['recording'], state['result'] = [], True, None
            else:
                state['recording'] = False
                if len(manual) >= 3:
                    state['result'] = predictor.predict(np.stack(manual)); print_result(f'manual ({len(manual)} frames)', state['result'])
                manual, state['nframes'] = [], 0
        elif k == ord('a'):
            state['auto'] = not state['auto']; state['recording'] = False; manual = []; seg = AutoSegmenter()
        elif k == ord('f'):
            state['flip'] = not state['flip']; print('flip', 'on' if state['flip'] else 'off')
        elif k == ord('c'):
            state['result'] = None
    cap.release(); cv2.destroyAllWindows(); holistic.close()


def run_video(a, predictor):
    import cv2
    _, holistic = make_holistic(a.complexity)
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f'Could not open {a.video}')
    seg, frames, n = AutoSegmenter(), [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if a.flip:
            frame = cv2.flip(frame, 1)
        lm = pack_landmarks(holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        n += 1
        if a.auto:
            clip = seg.update(lm)
            if clip is not None:
                print_result(f'frames ending at {n} ({len(clip)} long)', predictor.predict(clip))
        else:
            frames.append(lm)
    cap.release(); holistic.close()
    if not a.auto:
        arr = np.stack(frames) if frames else np.zeros((0, N_LM, 3), np.float32)
        have = int(np.sum([hand_visible(f) for f in frames])) if frames else 0
        print(f'{n} frames read, hands visible in {have}')
        print_result('whole video', predictor.predict(arr) if len(arr) else None)


def _mem_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0      # Linux: KB -> MB
    except Exception:
        return float('nan')


def run_benchmark(a, predictor):
    """Headless speed/memory check. Works with a camera or --video; --weights is optional (adds model latency)."""
    import cv2
    _, holistic = make_holistic(a.complexity)
    src = cv2.VideoCapture(a.video) if a.video else open_camera(a.camera)
    if a.width: src.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    if a.height: src.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    t_read = t_mp = 0.0; n = 0; shape = None
    print(f'benchmarking {a.bench_frames} frames (complexity={a.complexity}) ... keep your hands in view')
    while n < a.bench_frames:
        t0 = time.time(); ok, frame = src.read(); t1 = time.time()
        if not ok:
            break
        shape = frame.shape
        if a.flip: frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); t2 = time.time()
        pack_landmarks(holistic.process(rgb)); t3 = time.time()
        if n >= 3:                                   # skip warm-up frames
            t_read += t1 - t0; t_mp += t3 - t2
        n += 1
    src.release(); holistic.close()
    m = max(n - 3, 1)
    read_ms, mp_ms = 1000 * t_read / m, 1000 * t_mp / m
    print(f'\nframe size          : {shape}')
    print(f'camera read         : {read_ms:6.1f} ms/frame')
    print(f'MediaPipe Holistic  : {mp_ms:6.1f} ms/frame')
    print(f'=> achievable rate  : {1000 / max(read_ms + mp_ms, 1e-6):6.1f} frames/second (landmarks only)')
    if predictor is not None:
        fake = np.random.RandomState(0).rand(30, N_LM, 3).astype(np.float32)
        predictor.predict(fake)                      # warm-up
        t = time.time()
        for _ in range(5): predictor.predict(fake)
        print(f'model + preprocess  : {1000 * (time.time() - t) / 5:6.1f} ms per sign (30-frame clip)')
    print(f'peak memory (RSS)   : {_mem_mb():6.0f} MB')
    print('\nHow to read this: a webcam normally delivers ~30 frames/second. If the achievable rate is far below that, '
          'frames get skipped, so each sign reaches the model with fewer frames than it had when it was recorded -- '
          'it looks sped-up, which can cost accuracy. Try --complexity 0, a smaller --width/--height, or the TFLite route.')

def run_headless(a, predictor):
    """Camera + auto mode, no window. Prints every prediction; optionally appends JSON lines to --out-file."""
    import cv2
    _, holistic = make_holistic(a.complexity)
    cap = open_camera(a.camera)
    if a.width: cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    if a.height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    seg, n = AutoSegmenter(), 0
    print('Headless auto mode running. Sign with your hands in view; Ctrl+C to stop.', flush=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('Camera returned no frame; stopping.'); break
            if a.flip: frame = cv2.flip(frame, 1)
            n += 1
            clip = seg.update(pack_landmarks(holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
            if clip is None:
                continue
            result = predictor.predict(clip)
            print_result(f'frame {n} ({len(clip)} frames)', result)
            if result and a.out_file:
                with open(a.out_file, 'a') as fh:
                    fh.write(json.dumps({'time': time.time(), 'sign': result[0][0], 'confidence': round(result[0][1], 3),
                                         'top5': [[nm, round(p, 3)] for nm, p in result]}) + '\n')
    except KeyboardInterrupt:
        print('\nstopped.')
    finally:
        cap.release(); holistic.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', help='the *-best.h5 (or -last.h5) file from training')
    ap.add_argument('--signmap', default='sign_to_prediction_index_map.json')
    ap.add_argument('--max-len', type=int, default=128, help='MUST equal the CFG_MAX_LEN used in training')
    ap.add_argument('--dim', type=int, default=192, help='MUST equal the training dim (default 192)')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--width', type=int, default=0); ap.add_argument('--height', type=int, default=0)
    ap.add_argument('--flip', action='store_true', help='start with the image flipped left-right (toggle with F)')
    ap.add_argument('--auto', action='store_true', help='start in auto mode (toggle with A)')
    ap.add_argument('--complexity', type=int, default=1, choices=[0, 1, 2], help='MediaPipe model_complexity; 0 is faster')
    ap.add_argument('--video', help='predict a recorded video file instead of using the camera')
    ap.add_argument('--dry-run', action='store_true', help='load model, run it on random input, exit')
    ap.add_argument('--benchmark', action='store_true', help='measure speed and memory, no window (--weights optional)')
    ap.add_argument('--bench-frames', type=int, default=100)
    ap.add_argument('--headless', action='store_true', help='camera + auto mode with no window; prints predictions')
    ap.add_argument('--out-file', help='with --headless: append each prediction as a JSON line to this file')
    ap.add_argument('--list-signs', action='store_true')
    a = ap.parse_args()

    if a.list_signs:
        names = sorted(json.load(open(a.signmap)), key=str.lower)
        print(f'{len(names)} signs:'); print(', '.join(names)); return
    if a.benchmark and not a.weights:
        return run_benchmark(a, None)
    if not a.weights:
        sys.exit('--weights is required (path to the .h5 file)')
    predictor = Predictor(a.weights, a.signmap, a.max_len, a.dim)
    print(f'model loaded: {len(predictor.names)} classes, max_len={a.max_len}, dim={a.dim}')
    if a.dry_run:
        rng = np.random.RandomState(0)
        fake = rng.rand(30, N_LM, 3).astype(np.float32)
        print_result('random input (answers are meaningless; this only checks the pipeline)', predictor.predict(fake, topk=3))
        return
    if a.benchmark:
        return run_benchmark(a, predictor)
    if a.headless:
        return run_headless(a, predictor)
    run_video(a, predictor) if a.video else run_webcam(a, predictor)


if __name__ == '__main__':
    main()
