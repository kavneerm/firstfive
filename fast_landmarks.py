"""
Faster stand-in for MediaPipe Holistic on the UNO Q.

Holistic runs pose -> face crop -> face mesh -> hand crops -> hand models one after another, so a frame uses about
one core of four. The sign model never reads the pose landmarks (POSE is left out of POINT_LANDMARKS), so this runs
only the Hands and Face Mesh solutions (the same landmark models Holistic uses, shipped inside the mediapipe wheel),
on two threads at once, and optionally refreshes the face only every `face_every` frames.

`process(rgb)` returns an object with the same attributes as a Holistic result, so pack_landmarks() and the drawing
code work unchanged.
"""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

# Hands and Holistic name the two hands by different conventions; this maps a Hands label to the Holistic slot
# the training data used. Checked frame-by-frame against Holistic with compare_landmarkers.py.
HANDS_LABEL_TO_SLOT = {'Right': 'left_hand_landmarks', 'Left': 'right_hand_landmarks'}


class SplitLandmarker:
    def __init__(self, mp, hand_complexity=1, face_every=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        sol = mp.solutions
        self.hands = sol.hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=hand_complexity,
                                     min_detection_confidence=min_detection_confidence,
                                     min_tracking_confidence=min_tracking_confidence)
        # refine_landmarks=False gives the 468-point mesh that Holistic (and the training data) use
        self.face = sol.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False,
                                           min_detection_confidence=min_detection_confidence,
                                           min_tracking_confidence=min_tracking_confidence)
        self.face_every, self.n, self.last_face = max(1, face_every), 0, None
        self.pool = ThreadPoolExecutor(max_workers=1)

    def _face(self, rgb):
        res = self.face.process(rgb)
        return res.multi_face_landmarks[0] if res.multi_face_landmarks else None

    def process(self, rgb):
        rgb.flags.writeable = False                       # lets MediaPipe use the buffer without copying it
        refresh_face = self.n % self.face_every == 0
        self.n += 1
        face_job = self.pool.submit(self._face, rgb) if refresh_face else None
        hands = self.hands.process(rgb)                   # runs while the face mesh works on the other thread
        if face_job is not None:
            self.last_face = face_job.result()

        out = SimpleNamespace(face_landmarks=self.last_face, pose_landmarks=None,
                              left_hand_landmarks=None, right_hand_landmarks=None)
        found = list(zip(hands.multi_hand_landmarks or [], hands.multi_handedness or []))
        found.sort(key=lambda h: -h[1].classification[0].score)    # most confident first wins a contested slot
        for lms, handed in found:
            slot = HANDS_LABEL_TO_SLOT.get(handed.classification[0].label)
            if getattr(out, slot) is not None:            # both hands got the same label: give this one the other slot
                slot = 'right_hand_landmarks' if slot == 'left_hand_landmarks' else 'left_hand_landmarks'
            if getattr(out, slot) is None:
                setattr(out, slot, lms)
        return out

    def close(self):
        self.pool.shutdown(wait=True)
        self.hands.close(); self.face.close()


def make_landmarker(kind, complexity=0, hand_complexity=1, face_every=1, hand_model='full', palm_every=6):
    """'custom' (fast_tracker.FastTracker, no MediaPipe needed), 'holistic' (the original pipeline) or 'split'
    (this module). Returns (mediapipe module or None, landmarker)."""
    if kind == 'custom':
        from fast_tracker import FastTracker
        return None, FastTracker(hand_model=hand_model, face_every=face_every, palm_every=palm_every)
    from demo_webcam import make_holistic
    if kind == 'holistic':
        return make_holistic(complexity)
    import mediapipe as mp
    return mp, SplitLandmarker(mp, hand_complexity=hand_complexity, face_every=face_every)
