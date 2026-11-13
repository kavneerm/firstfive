"""Build a small zip in the Kaggle asl-signs layout (train.csv, sign map, train_landmark_files/<pid>/<sid>.parquet)
for smoke tests while the real download isn't available. Each sign gets a distinct hand trajectory so a model can learn it.

    python tests/make_fake_kaggle.py out.zip [--signs 10] [--participants 6] [--per 8]
"""
import argparse, io, json, sys, zipfile
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--signs', type=int, default=10)
ap.add_argument('--participants', type=int, default=6); ap.add_argument('--per', type=int, default=8)
a = ap.parse_args()

real_map = json.load(open('sign_to_prediction_index_map.json'))
rng = np.random.RandomState(0)
types = ['face'] * 468 + ['left_hand'] * 21 + ['pose'] * 33 + ['right_hand'] * 21
lidx = list(range(468)) + list(range(21)) + list(range(33)) + list(range(21))
rows, sid = [], 1000
with zipfile.ZipFile(a.out, 'w') as z:
    for p in range(a.participants):
        pid = 10000 + p
        for s in range(a.signs):
            sign = list(real_map)[s]
            for k in range(a.per):
                T = rng.randint(8, 160)
                t = np.linspace(0, 1, T)[:, None]
                base = rng.rand(1, 543, 3) * 0.2 + 0.4
                coords = np.repeat(base, T, 0)
                coords[:, 522:543, 0] += 0.2 * np.sin(2 * np.pi * (s + 1) * t / 3)
                coords[:, 522:543, 1] += 0.2 * np.cos(2 * np.pi * (s % 3 + 1) * t)
                coords += rng.randn(*coords.shape) * 0.01
                coords[:, 468:489] = np.nan
                if k == 0 and p == 0 and s == 0:
                    coords[:] = np.nan          # one sequence with no usable frames
                df = pd.DataFrame({'frame': np.repeat(np.arange(T), 543), 'type': types * T,
                                   'landmark_index': lidx * T,
                                   'x': coords[..., 0].ravel(), 'y': coords[..., 1].ravel(), 'z': coords[..., 2].ravel()})
                b = io.BytesIO(); df.to_parquet(b)
                path = f'train_landmark_files/{pid}/{sid}.parquet'
                z.writestr(path, b.getvalue())
                rows.append((path, pid, sid, sign)); sid += 1
    z.writestr('train.csv', pd.DataFrame(rows, columns=['path', 'participant_id', 'sequence_id', 'sign']).to_csv(index=False))
    z.writestr('sign_to_prediction_index_map.json', json.dumps(real_map))
print(f'wrote {a.out}: {len(rows)} sequences')
