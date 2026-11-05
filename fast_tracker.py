"""
Custom hand + face landmark tracker for the UNO Q, built directly on MediaPipe's own .tflite models with
tflite-runtime (no MediaPipe, no TensorFlow). Replaces Holistic, whose fixed graph re-runs a detector almost every
frame and uses about one core.

Per frame:
  - every tracked hand / the face is cropped from the ROI predicted by last frame's landmarks and run through its
    landmark model (each interpreter uses all cores; measured faster on the UNO Q than running them side by side)
  - the palm / face *detectors* run only while a hand slot or the face is empty, and only every few frames: every
    2nd frame with no hand tracked (to catch a new sign fast), every `palm_every` frames while one hand is already
    tracked (then it is only looking for a second hand; measured as the biggest per-frame cost)
  - the face mesh is refreshed every `face_every` frames while signing (lips/eyes move less than hands)
  - after ~1 s without hands it idles: palm detection every 3rd frame, face refreshed every 3rd frame

ROI geometry follows MediaPipe's graphs (detection -> rect with rotation from two keypoints, rect transformation,
landmarks -> next rect). Those subgraph options are compiled into MediaPipe's binary rather than shipped as config
files, so the constants below come from MediaPipe's source and are checked against Holistic with compare_landmarkers.py.

`process(rgb)` returns an object with Holistic's attributes (face_landmarks, left/right_hand_landmarks,
pose_landmarks=None), so demo_webcam.pack_landmarks and the rest of the pipeline are unchanged.
"""
import math, os, time
from collections import defaultdict
from types import SimpleNamespace
import numpy as np
import cv2

from sign_infer_lite import make_interpreter

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
# same convention question as fast_landmarks: which Holistic slot a hand-model "Right"/"Left" belongs to
from fast_landmarks import HANDS_LABEL_TO_SLOT

HAND_SUBSET = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]      # landmarks MediaPipe uses to place the next hand ROI


# ----------------------------------------------------------------------------- geometry
def norm_angle(a):
    return a - 2 * math.pi * math.floor((a + math.pi) / (2 * math.pi))


def rotation_from(p0, p1, target):
    """DetectionsToRects / HandLandmarksToRect rotation: angle that turns p0->p1 to `target` (pixel coords)."""
    return norm_angle(target - math.atan2(-(p1[1] - p0[1]), p1[0] - p0[0]))


def transform_rect(cx, cy, w, h, rot, scale, shift_y=0.0):
    """RectTransformation with square_long=True, in pixels: shift along the rotated y axis, square, then scale."""
    cx -= h * shift_y * math.sin(rot)
    cy += h * shift_y * math.cos(rot)
    side = max(w, h) * scale
    return cx, cy, side, side, rot


def crop(rgb, rect, size):
    """Rotated rect (cx, cy, w, h, rot) in pixels -> (size, size, 3) uint8 crop and the 2x3 affine used."""
    cx, cy, w, h, rot = rect
    c, s = math.cos(rot), math.sin(rot)
    def corner(u, v):
        return [cx + u * w * c - v * h * s, cy + u * w * s + v * h * c]
    src = np.float32([corner(-.5, -.5), corner(.5, -.5), corner(-.5, .5)])
    dst = np.float32([[0, 0], [size, 0], [0, size]])
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(rgb, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), M


def project(pts, rect, size):
    """Landmarks in crop pixels (N, 3) -> image pixels (N, 3); z scaled like MediaPipe (by ROI width)."""
    cx, cy, w, h, rot = rect
    c, s = math.cos(rot), math.sin(rot)
    u, v = pts[:, 0] / size - 0.5, pts[:, 1] / size - 0.5
    out = np.empty_like(pts)
    out[:, 0] = cx + u * w * c - v * h * s
    out[:, 1] = cy + u * w * s + v * h * c
    out[:, 2] = pts[:, 2] / size * w
    return out


def hand_rect_from_landmarks(px):
    """HandLandmarksToRect + RectTransformation(scale 2.0, shift_y -0.1) from (21, 3) image-pixel landmarks."""
    sub = px[HAND_SUBSET, :2]
    p1 = ((sub[4] + sub[8]) / 2 + sub[6]) / 2                # (index MCP + ring MCP)/2, averaged with middle MCP
    rot = rotation_from(sub[0], p1, math.pi / 2)
    c, s = math.cos(rot), math.sin(rot)
    rx = sub[:, 0] * c + sub[:, 1] * s                       # project onto the rotated axes
    ry = -sub[:, 0] * s + sub[:, 1] * c
    mx, my = (rx.min() + rx.max()) / 2, (ry.min() + ry.max()) / 2
    cx, cy = mx * c - my * s, mx * s + my * c
    return transform_rect(cx, cy, rx.max() - rx.min(), ry.max() - ry.min(), rot, 2.0, -0.1)


def face_rect_from_landmarks(px):
    """LandmarksToDetection (axis-aligned box of all 468) + rotation from eye corners 33 -> 263, scale 1.5."""
    x0, y0 = px[:, :2].min(0); x1, y1 = px[:, :2].max(0)
    rot = rotation_from(px[33, :2], px[263, :2], 0.0)
    return transform_rect((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, rot, 1.5)


def iou_center_close(a, b):
    """Cheap association test (stand-in for AssociationNormRect's IoU > 0.5): centers within half a side."""
    return math.hypot(a[0] - b[0], a[1] - b[1]) < 0.5 * max(a[2], b[2])


# ----------------------------------------------------------------------------- detectors
def ssd_anchors(size, strides=(8, 16, 16, 16)):
    """SsdAnchors with fixed_anchor_size and aspect ratio 1 (+ interpolated scale): 2 anchors per cell per layer;
    layers sharing a stride are merged. Gives 2016 anchors at 192 (palm) and 896 at 128 (face)."""
    out, i = [], 0
    while i < len(strides):
        j = i
        while j < len(strides) and strides[j] == strides[i]:
            j += 1
        n = 2 * (j - i)
        g = int(math.ceil(size / strides[i]))
        for y in range(g):
            for x in range(g):
                out += [((x + 0.5) / g, (y + 0.5) / g)] * n
        i = j
    return np.array(out, np.float32)


class SSDDetector:
    """Palm or face detector: letterbox -> model -> decode boxes + keypoints -> weighted NMS -> image pixels."""

    def __init__(self, path, n_keypoints, value_range, threads, min_score=0.5, nms_iou=0.3):
        self.it = make_interpreter(path, threads)
        d = self.it.get_input_details()[0]
        self.inp, self.size = d['index'], int(d['shape'][1])
        outs = sorted(self.it.get_output_details(), key=lambda o: -o['shape'][-1])   # regressors first
        self.reg, self.cls = outs[0]['index'], outs[1]['index']
        self.anchors = ssd_anchors(self.size)
        assert len(self.anchors) == outs[0]['shape'][1], (len(self.anchors), outs[0]['shape'])
        self.nk, self.lo, self.hi, self.min_score, self.nms_iou = n_keypoints, value_range[0], value_range[1], min_score, nms_iou

    def __call__(self, rgb):
        H, W = rgb.shape[:2]
        S = self.size
        scale = S / max(W, H)
        nw, nh = round(W * scale), round(H * scale)
        px, py = (S - nw) // 2, (S - nh) // 2
        lb = np.zeros((S, S, 3), np.uint8)
        lb[py:py + nh, px:px + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        x = lb.astype(np.float32) * ((self.hi - self.lo) / 255.0) + self.lo
        self.it.set_tensor(self.inp, x[None]); self.it.invoke()
        raw = self.it.get_tensor(self.reg)[0]
        score = 1 / (1 + np.exp(-np.clip(self.it.get_tensor(self.cls)[0, :, 0], -100, 100)))
        keep = np.where(score >= self.min_score)[0]
        if len(keep) == 0:
            return []
        a, r = self.anchors[keep], raw[keep] / S
        boxes = np.stack([r[:, 0] + a[:, 0], r[:, 1] + a[:, 1], r[:, 2], r[:, 3]], 1)
        kps = r[:, 4:4 + 2 * self.nk].reshape(-1, self.nk, 2) + a[:, None, :]
        dets = []
        for b, k, s in self._weighted_nms(boxes, kps, score[keep]):
            # undo the letterbox: letterboxed-normalised -> image pixels
            to_px = lambda xs, ys: ((xs * S - px) / scale, (ys * S - py) / scale)
            cx, cy = to_px(b[0], b[1])
            kx, ky = to_px(k[:, 0], k[:, 1])
            dets.append(dict(cx=cx, cy=cy, w=b[2] * S / scale, h=b[3] * S / scale, kp=np.stack([kx, ky], 1), score=s))
        return dets

    def _weighted_nms(self, boxes, kps, scores):
        x0, y0 = boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2
        x1, y1 = boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2
        area = boxes[:, 2] * boxes[:, 3]
        left = np.ones(len(scores), bool)
        out = []
        while left.any():
            i = int(np.argmax(np.where(left, scores, -1)))
            iw = np.clip(np.minimum(x1[i], x1) - np.maximum(x0[i], x0), 0, None)
            ih = np.clip(np.minimum(y1[i], y1) - np.maximum(y0[i], y0), 0, None)
            inter = iw * ih
            grp = left & (inter / (area[i] + area - inter + 1e-9) > self.nms_iou)
            grp[i] = True
            w = scores[grp][:, None]
            out.append(((boxes[grp] * w).sum(0) / w.sum(), (kps[grp] * w[:, :, None]).sum(0) / w.sum(), float(scores[i])))
            left &= ~grp
        return out


# ----------------------------------------------------------------------------- landmark models
class LandmarkModel:
    def __init__(self, path, value_range, threads):
        self.it = make_interpreter(path, threads)
        d = self.it.get_input_details()[0]
        self.inp, self.size = d['index'], int(d['shape'][1])
        self.outs = {o['name']: o['index'] for o in self.it.get_output_details()}
        self.lo, self.hi = value_range

    def run(self, rgb, rect):
        img, _ = crop(rgb, rect, self.size)
        x = img.astype(np.float32) * ((self.hi - self.lo) / 255.0) + self.lo
        self.it.set_tensor(self.inp, x[None]); self.it.invoke()
        return {n: self.it.get_tensor(i).reshape(-1) for n, i in self.outs.items()}


def _sigmoid(v):
    return float(1 / (1 + math.exp(-max(-100.0, min(100.0, float(v))))))


class _Landmarks:
    """Minimal NormalizedLandmarkList look-alike: `.landmark[i].x/.y/.z`, built only when someone reads it."""

    def __init__(self, norm):
        self.arr = norm

    @property
    def landmark(self):
        return [SimpleNamespace(x=float(a), y=float(b), z=float(c)) for a, b, c in self.arr]


class FastTracker:
    def __init__(self, hand_model='full', threads=4, max_hands=2, presence=0.5, idle_after=1.0, fps_hint=10.0,
                 value_ranges=None, face_every=2, palm_every=6):
        vr = dict(palm=(0.0, 1.0), hand=(0.0, 1.0), face_det=(-1.0, 1.0), face=(0.0, 1.0))
        vr.update(value_ranges or {})
        self.palm = SSDDetector(os.path.join(MODELS, 'palm_detection_full.tflite'), 7, vr['palm'], threads)
        hand_path = os.path.join(MODELS, f'hand_landmark_{hand_model}.tflite')
        self.hand = LandmarkModel(hand_path, vr['hand'], threads)
        self.face_det = SSDDetector(os.path.join(MODELS, 'face_detection_short_range.tflite'), 6, vr['face_det'], threads)
        self.face = LandmarkModel(os.path.join(MODELS, 'face_landmark.tflite'), vr['face'], threads)
        hn = sorted(self.hand.outs)                # Identity: landmarks, _1: presence, _2: handedness, _3: world
        self.h_lm, self.h_pres, self.h_hand = hn[0], hn[1], hn[2]
        fn = sorted(self.face.outs, key=lambda n: -self.face.it.get_tensor(self.face.outs[n]).size)
        self.f_lm, self.f_pres = fn[0], fn[1]
        self.max_hands, self.presence = max_hands, presence
        self.idle_frames = max(1, int(idle_after * fps_hint))
        self.hands = []          # [(rect, label, label_score)] carried to the next frame
        self.face_rect = None
        self.n = self.no_hand_run = 0
        self.last = None         # for draw(): dict(face=px, hands=[px...])
        self.face_every, self.palm_every = max(1, face_every), max(1, palm_every)
        self.timing = defaultdict(lambda: [0.0, 0])   # stage -> [total seconds, runs], read by the benchmark

    def _timed(self, stage, fn, *args):
        t = time.time()
        out = fn(*args)
        self.timing[stage][0] += time.time() - t; self.timing[stage][1] += 1
        return out

    @property
    def idle(self):
        return self.no_hand_run >= self.idle_frames

    def _hand_from_rect(self, rgb, rect):
        o = self.hand.run(rgb, rect)
        pres = float(o[self.h_pres][0])
        if not 0.0 <= pres <= 1.0:
            pres = _sigmoid(pres)
        if pres < self.presence:
            return None
        h = float(o[self.h_hand][0])
        if not 0.0 <= h <= 1.0:
            h = _sigmoid(h)
        px = project(o[self.h_lm].reshape(21, 3).astype(np.float32), rect, self.hand.size)
        return px, ('Right' if h > 0.5 else 'Left'), max(h, 1 - h)

    def process(self, rgb):
        H, W = rgb.shape[:2]
        self.n += 1
        # ---- hands: track existing ROIs, then (re)detect only if a slot is empty
        found = []
        for rect, _, _ in self.hands:
            r = self._timed('hand landmarks', self._hand_from_rect, rgb, rect)
            if r is not None:
                found.append(r)
        every = 3 if self.idle else (self.palm_every if found else 2)
        if len(found) < self.max_hands and self.n % every == 0:
            tracked = [hand_rect_from_landmarks(px) for px, _, _ in found]
            for d in sorted(self._timed('palm detect', self.palm, rgb), key=lambda d: -d['score']):
                if len(found) >= self.max_hands:
                    break
                rect = transform_rect(d['cx'], d['cy'], d['w'], d['h'],
                                      rotation_from(d['kp'][0], d['kp'][2], math.pi / 2), 2.6, -0.5)
                if any(iou_center_close(rect, t) for t in tracked):
                    continue
                r = self._timed('hand landmarks', self._hand_from_rect, rgb, rect)
                if r is not None:
                    found.append(r); tracked.append(hand_rect_from_landmarks(r[0]))
        self.hands = [(hand_rect_from_landmarks(px), lab, sc) for px, lab, sc in found]
        self.no_hand_run = 0 if found else self.no_hand_run + 1

        # ---- face: refresh every `face_every` frames while signing, every 3rd frame while idle
        face_px = self.last['face'] if self.last else None
        if self.n % (3 if self.idle else self.face_every) == 0 or (face_px is None and not self.idle):
            face_px = None
            if self.face_rect is None:
                dets = self._timed('face detect', self.face_det, rgb)
                if dets:
                    d = max(dets, key=lambda d: d['score'])
                    self.face_rect = transform_rect(d['cx'], d['cy'], d['w'], d['h'],
                                                    rotation_from(d['kp'][0], d['kp'][1], 0.0), 1.5)
            if self.face_rect is not None:
                o = self._timed('face landmarks', self.face.run, rgb, self.face_rect)
                if _sigmoid(o[self.f_pres][0]) >= self.presence:
                    face_px = project(o[self.f_lm].reshape(468, 3).astype(np.float32), self.face_rect, self.face.size)
                    self.face_rect = face_rect_from_landmarks(face_px)
                else:
                    self.face_rect = None

        # ---- Holistic-shaped result (normalised coordinates)
        out = SimpleNamespace(face_landmarks=None, pose_landmarks=None, left_hand_landmarks=None, right_hand_landmarks=None)
        scale = np.float32([W, H, W])
        if face_px is not None:
            out.face_landmarks = _Landmarks(face_px / scale)
        for px, lab, sc in sorted(found, key=lambda f: -f[2]):
            slot = HANDS_LABEL_TO_SLOT[lab]
            if getattr(out, slot) is not None:
                slot = 'right_hand_landmarks' if slot == 'left_hand_landmarks' else 'left_hand_landmarks'
            if getattr(out, slot) is None:
                setattr(out, slot, _Landmarks(px / scale))
        self.last = dict(face=face_px, hands=[px for px, _, _ in found])
        return out

    def draw(self, bgr):
        """Preview overlay without MediaPipe's drawing utils: hand skeletons and a sparse face outline."""
        if not self.last:
            return bgr
        for px in self.last['hands']:
            for a, b in HAND_EDGES:
                cv2.line(bgr, tuple(int(v) for v in px[a, :2]), tuple(int(v) for v in px[b, :2]), (0, 255, 0), 2)
            for p in px:
                cv2.circle(bgr, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
        if self.last['face'] is not None:
            for p in self.last['face'][::6]:
                cv2.circle(bgr, (int(p[0]), int(p[1])), 1, (255, 255, 255), -1)
        return bgr

    def close(self):
        pass


HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
