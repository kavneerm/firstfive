#!/usr/bin/env python3
"""Find the signers the baseline never trained on, and make them our validation set.

The baseline was trained on ~80% of the 21 participants. On its own training signers it scores much higher than on
held-out ones, so per-participant top-1 should split into two groups. We take the low group (2-8 participants,
at the largest gap) as validation, so baseline and new models are compared on signers neither trained on.

    python find_heldout.py --data data/islr --weights baseline.h5 --out data/islr/split.json
Exit 1 without writing if there is no clean gap (gap < 3 points or group means < 5 points apart).
"""
import argparse, json, sys
import numpy as np
import tensorflow as tf
import common as C
import model as Mdl

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--weights', default='baseline.h5')
ap.add_argument('--out', required=True)
a = ap.parse_args()

D = C.Data(a.data)
m = Mdl.get_model(max_len=C.MAX_LEN, num_classes=len(D.signmap))
m.load_weights(a.weights)
pred = m.predict(D.dataset(D.meta, 512, False), verbose=0).argmax(-1)   # predict runs in graph mode; an eager per-batch loop leaks GPU memory
D.meta['ok'] = pred == D.meta.label.values
acc = D.meta.groupby('participant_id').ok.agg(['mean', 'size']).sort_values('mean')
print(acc.to_string())

v = acc['mean'].values
gaps = {k: v[k] - v[k - 1] for k in range(2, 9)}
k = max(gaps, key=gaps.get)
low, high = acc.iloc[:k], acc.iloc[k:]
lo_mean = float((low['mean'] * low['size']).sum() / low['size'].sum())
hi_mean = float((high['mean'] * high['size']).sum() / high['size'].sum())
print(f'\nlargest gap after the {k} lowest: {gaps[k]:.3f}; held-out mean {lo_mean:.4f} vs trained-on mean {hi_mean:.4f}')
if gaps[k] < 0.03 or hi_mean - lo_mean < 0.05:
    sys.exit('NO CLEAN GAP: fall back to retraining the baseline architecture on our own split')
split = dict(val_participants=[int(p) for p in low.index], method='baseline per-participant accuracy gap',
             baseline_heldout_top1_keras=lo_mean, baseline_trained_top1_keras=hi_mean, gap=float(gaps[k]),
             n_val=int(low['size'].sum()))
json.dump(split, open(a.out, 'w'), indent=1)
print('wrote', a.out, json.dumps(split))
