#!/usr/bin/env python3
"""
Accuracy check on recorded clips. Each file in --clips is named <expected sign>.mp4 (or <sign>_2.mp4, ...) and
holds exactly one sign. Prints PASS/FAIL per clip and a top-1 / top-5 total; exits 1 if any clip fails top-1.

    python check_clips.py --weights aslModelENGS21-fold0-best.h5 --clips clips/ [--flip]
"""
import argparse, glob, os, sys
import numpy as np

from demo_webcam import Predictor, pack_landmarks
from fast_landmarks import make_landmarker


def clip_landmarks(cv2, holistic, path, flip):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if flip:
            frame = cv2.flip(frame, 1)
        frames.append(pack_landmarks(holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    cap.release()
    return np.stack(frames) if frames else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--signmap', default='sign_to_prediction_index_map.json')
    ap.add_argument('--max-len', type=int, default=128); ap.add_argument('--dim', type=int, default=192)
    ap.add_argument('--clips', required=True, help='folder of <sign>.mp4 files')
    ap.add_argument('--flip', action='store_true'); ap.add_argument('--complexity', type=int, default=0)
    ap.add_argument('--landmarker', default='custom', choices=['custom', 'split', 'holistic'])
    ap.add_argument('--hand-model', default='full', choices=['full', 'lite'])
    ap.add_argument('--hand-complexity', type=int, default=1); ap.add_argument('--face-every', type=int, default=2)
    a = ap.parse_args()

    import cv2
    predictor = Predictor(a.weights, a.signmap, a.max_len, a.dim)
    known = set(predictor.names.values())
    paths = sorted(glob.glob(os.path.join(a.clips, '*.mp4')))
    if not paths:
        sys.exit(f'no .mp4 files in {a.clips}')
    top1 = top5 = total = 0
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        expected = stem if stem in known else stem.rsplit('_', 1)[0]    # "hello_2" -> "hello"; "TV" stays "TV"
        if expected not in known:
            print(f'SKIP  {os.path.basename(path)}: "{expected}" is not one of the model\'s signs'); continue
        _, holistic = make_landmarker(a.landmarker, a.complexity, a.hand_complexity, a.face_every, a.hand_model)  # fresh per clip
        lms = clip_landmarks(cv2, holistic, path, a.flip)
        holistic.close()
        result = predictor.predict(lms) if lms is not None else None
        total += 1
        if not result:
            print(f'FAIL  {expected:12s} -> no usable frames'); continue
        names = [n for n, _ in result]
        top1 += names[0] == expected; top5 += expected in names
        print(f'{"PASS" if names[0] == expected else "FAIL"}  {expected:12s} -> {names[0]} {100 * result[0][1]:.0f}%'
              f'   (top5: {", ".join(names)})')
    print(f'\ntop-1 {top1}/{total}   top-5 {top5}/{total}   (flip={"on" if a.flip else "off"}, landmarker={a.landmarker})')
    sys.exit(0 if total and top1 == total else 1)


if __name__ == '__main__':
    main()
