#!/usr/bin/env python3
"""
Check a faster landmarker (--landmarker custom = fast_tracker.py, or split = fast_landmarks.py) against Holistic on the same frames, per clip:
  - ms/frame for each, and the speed-up
  - hand-slot mapping: mean xy distance to Holistic's hand in the SAME slot vs the OPPOSITE slot
    (same must be much smaller, otherwise HANDS_LABEL_TO_SLOT is backwards)
  - mean xy distance per part (normalised image units; 0.01 = 6 px at 640 wide) and detection agreement
  - with --weights: the sign model's top-5 on each landmark stream

    python compare_landmarkers.py --clips clips --weights aslModelENGS21-fold0-best.h5
"""
import argparse, glob, os, time
import numpy as np

from demo_webcam import pack_landmarks
from fast_landmarks import make_landmarker

PARTS = {'face': slice(0, 468), 'left': slice(468, 489), 'right': slice(522, 543)}


def run(cv2, lm, frames):
    out, t = [], time.time()
    for f in frames:
        out.append(pack_landmarks(lm.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))))
    return np.stack(out), 1000 * (time.time() - t) / len(frames)


def xy_dist(a, b):
    """Mean xy distance over frames where both have the part, and how many such frames there were."""
    both = ~np.isnan(a[:, :, 0]).all(1) & ~np.isnan(b[:, :, 0]).all(1)
    if not both.any():
        return float('nan'), 0
    return float(np.nanmean(np.linalg.norm(a[both, :, :2] - b[both, :, :2], axis=-1))), int(both.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clips', required=True); ap.add_argument('--weights')
    ap.add_argument('--signmap', default='sign_to_prediction_index_map.json')
    ap.add_argument('--max-len', type=int, default=128); ap.add_argument('--dim', type=int, default=192)
    ap.add_argument('--complexity', type=int, default=0); ap.add_argument('--hand-complexity', type=int, default=1)
    ap.add_argument('--face-every', type=int, default=1)
    ap.add_argument('--landmarker', default='custom', choices=['custom', 'split'])
    ap.add_argument('--hand-model', default='full', choices=['full', 'lite'])
    a = ap.parse_args()

    import cv2
    predictor = None
    if a.weights:
        from demo_webcam import Predictor
        predictor = Predictor(a.weights, a.signmap, a.max_len, a.dim)
    t_h = t_s = 0.0; nframes = 0
    for path in sorted(glob.glob(os.path.join(a.clips, '*.mp4'))):
        cap, frames = cv2.VideoCapture(path), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            continue
        # fresh trackers per clip so both start from detection, like an independent recording
        _, hol = make_landmarker('holistic', a.complexity); H, ms_h = run(cv2, hol, frames); hol.close()
        _, spl = make_landmarker(a.landmarker, a.complexity, a.hand_complexity, a.face_every, a.hand_model)
        S, ms_s = run(cv2, spl, frames); spl.close()
        t_h += ms_h * len(frames); t_s += ms_s * len(frames); nframes += len(frames)

        print(f'\n== {os.path.basename(path)}: {len(frames)} frames   holistic {ms_h:.0f} ms/frame   '
              f'{a.landmarker} {ms_s:.0f} ms/frame   speed-up x{ms_h / ms_s:.2f}')
        for part, sl in PARTS.items():
            d, n = xy_dist(H[:, sl], S[:, sl])
            have_h, have_s = int((~np.isnan(H[:, sl, 0]).all(1)).sum()), int((~np.isnan(S[:, sl, 0]).all(1)).sum())
            print(f'   {part:5s}  detected holistic {have_h:3d} / {a.landmarker} {have_s:3d} frames   '
                  f'same-slot dist {d:.4f} over {n} frames')
        same = [xy_dist(H[:, PARTS[p]], S[:, PARTS[p]])[0] for p in ('left', 'right')]
        swap = [xy_dist(H[:, PARTS['left']], S[:, PARTS['right']])[0], xy_dist(H[:, PARTS['right']], S[:, PARTS['left']])[0]]
        print(f'   hand mapping: same-slot {np.nanmean(same) if not np.isnan(same).all() else float("nan"):.4f}   '
              f'swapped {np.nanmean(swap) if not np.isnan(swap).all() else float("nan"):.4f}   '
              f'(same should be far smaller)')
        if predictor:
            for name, L in (('holistic', H), (a.landmarker, S)):
                r = predictor.predict(L)
                print(f'   model on {name:8s}: ' + (' | '.join(f'{n} {100 * p:.0f}%' for n, p in r) if r else 'no usable frames'))
    if nframes:
        print(f'\nOVERALL {nframes} frames: holistic {t_h / nframes:.0f} ms/frame ({1000 * nframes / t_h:.1f} fps)   '
              f'{a.landmarker} {t_s / nframes:.0f} ms/frame ({1000 * nframes / t_s:.1f} fps)   speed-up x{t_h / t_s:.2f}')


if __name__ == '__main__':
    main()
