"""Parity checks (exit 1 on failure):
  P1  common.preprocess on the stored 118-point subset == islr_model.Preprocess on the full 543-point frame
  P2  model.get_model() defaults load baseline.h5 and give the same logits as islr_model.load_model
  P3  augment() runs on edge-case lengths and returns finite features after preprocess
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tensorflow as tf
import islr_model as M
import common as C
import model

ok = True
rng = np.random.RandomState(0)
clips = []
for T in (1, 2, 3, 23, 200):
    c = rng.rand(T, 543, 3).astype(np.float32)
    c[:, 468:489][rng.rand(T) < 0.5] = np.nan
    clips.append(c)

pre = M.Preprocess(max_len=128)
for c in clips:
    a = pre(tf.constant(c))[0].numpy()
    b = C.preprocess(tf.constant(c[:, C.POINTS]), 128).numpy()
    d = float(np.abs(a - b).max())
    good = a.shape == b.shape and d < 1e-5
    ok &= good
    print(f'{"PASS" if good else "FAIL"} P1 T={len(c):3d} shape {b.shape} max diff {d:.1e}')

w = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'baseline.h5')
ref = M.load_model(w, 128)
mine = model.get_model(max_len=128)
mine.load_weights(w)
for c in clips:
    x = M.prepare_input(c, 128)
    d = float(np.abs(ref(x, training=False).numpy() - mine(x, training=False).numpy()).max())
    good = d < 1e-5
    ok &= good
    print(f'{"PASS" if good else "FAIL"} P2 T={len(c):3d} logit diff {d:.1e}')
print(f'     baseline params: {mine.count_params():,}')

for T in (1, 2, 5, 300):
    x = tf.constant(rng.rand(T, C.NP, 3).astype(np.float32))
    for _ in range(20):
        f = C.preprocess(C.augment(x), 128).numpy()
        if not (np.isfinite(f).all() and 1 <= len(f) <= 128):
            ok = False
            print(f'FAIL P3 T={T} got shape {f.shape}, finite={np.isfinite(f).all()}')
            break
    else:
        print(f'PASS P3 T={T}')
print('ALL PASS' if ok else 'SOME FAILED')
sys.exit(0 if ok else 1)
