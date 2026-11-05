#!/usr/bin/env python3
"""
One-time conversion of the Keras sign model to TFLite (built-in ops only, so tflite-runtime can run it), followed by
the equivalence checks. Run in the full TF environment (~/demo_env):

    python convert_sign_model.py --weights aslModelENGS21-fold0-best.h5 --out sign_model.tflite [--clips clips]

Checks (exit 1 if any fails):
  V1  Keras vs TFLite logits: max abs diff < 1e-3 and identical top-5, on random clips (+ fixture clips)
  V2  numpy prepare_input vs TF prepare_input: max abs diff < 1e-5, on the same clips
"""
import argparse, glob, os, sys
import numpy as np


def fixture_landmarks(clips_dir):
    """(name, (T,543,3)) per clip, landmarks from Holistic so the checks see realistic NaN patterns."""
    import cv2
    from demo_webcam import make_holistic, pack_landmarks
    out = []
    for path in sorted(glob.glob(os.path.join(clips_dir, '*.mp4'))):
        _, hol = make_holistic(0)
        cap, frames = cv2.VideoCapture(path), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(pack_landmarks(hol.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))))
        cap.release(); hol.close()
        if frames:
            out.append((os.path.basename(path), np.stack(frames)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', required=True); ap.add_argument('--out', default='sign_model.tflite')
    ap.add_argument('--signmap', default='sign_to_prediction_index_map.json')
    ap.add_argument('--max-len', type=int, default=128); ap.add_argument('--dim', type=int, default=192)
    ap.add_argument('--clips', help='folder of fixture .mp4 clips to include in the checks')
    a = ap.parse_args()

    import tensorflow as tf
    import islr_model as M
    import sign_infer_lite as L
    from demo_webcam import Predictor

    assert L.POINT_LANDMARKS == M.POINT_LANDMARKS and L.CHANNELS == M.CHANNELS, 'landmark lists out of sync'
    keras = Predictor(a.weights, a.signmap, a.max_len, a.dim)
    fn = tf.function(lambda t: keras.model(t, training=False),
                     input_signature=[tf.TensorSpec((1, a.max_len, M.CHANNELS), tf.float32)])
    conv = tf.lite.TFLiteConverter.from_concrete_functions([fn.get_concrete_function()], keras.model)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]       # no Flex ops: tflite-runtime can't run them
    blob = conv.convert()
    open(a.out, 'wb').write(blob)
    print(f'wrote {a.out} ({len(blob) / 1e6:.1f} MB)')

    lite = L.LitePredictor(a.out, a.signmap, a.max_len)
    rng = np.random.RandomState(0)
    clips = [(f'random{T}', rng.rand(T, 543, 3).astype(np.float32)) for T in (1, 2, 3, 30, 200)]
    holey = rng.rand(40, 543, 3).astype(np.float32); holey[:, 468:489] = np.nan; holey[::3, 522:543] = np.nan
    clips.append(('random40-missing-hands', holey))
    if a.clips:
        clips += fixture_landmarks(a.clips)

    ok = True
    for name, c in clips:
        x_tf, x_np = M.prepare_input(c, a.max_len), L.prepare_input(c, a.max_len)
        d_pre = float(np.abs(x_tf - x_np).max())
        lk, ll = keras._infer(x_tf).numpy()[0], lite.logits(x_tf)
        d_log = float(np.abs(lk - ll).max())
        same5 = list(np.argsort(-lk)[:5]) == list(np.argsort(-ll)[:5])
        good = d_pre < 1e-5 and d_log < 1e-3 and same5
        ok &= good
        print(f'{"PASS" if good else "FAIL"}  {name:24s} V2 preprocess diff {d_pre:.2e}   V1 logit diff {d_log:.2e}   '
              f'top-5 {"same" if same5 else "DIFFERENT"}')
    print('ALL PASS' if ok else 'SOME FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
