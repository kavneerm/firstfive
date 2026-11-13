#!/usr/bin/env python3
"""Convert the Kaggle asl-signs download into one array for fast training.

    python prepare_kaggle.py --zip asl-signs.zip --out data/islr [--workers 16] [--limit N]

Writes to --out:
  landmarks.npy   float32 (total_frames, 118, 3): only the landmarks the model uses (common.POINTS), NaN kept;
                  frames where all 118 are NaN are dropped (prepare_input does the same at inference)
  meta.csv        sequence_id, participant_id, sign, label, start, length (row range in landmarks.npy)
  sign_to_prediction_index_map.json   copied from the competition
Prints frame-length percentiles and how many sequences were skipped (no usable frames).
"""
import argparse, io, json, os, sys, time, zipfile
from multiprocessing import Pool
import numpy as np
import pandas as pd

ROWS = 543
_Z = {}


def _points():
    import islr_model as M          # no TF op is run; only the landmark lists are read
    return np.array(M.POINT_LANDMARKS)


def _init(zpath):
    _Z['z'] = zipfile.ZipFile(zpath)
    _Z['pts'] = _points()


def load_one(path):
    raw = _Z['z'].read(path)
    df = pd.read_parquet(io.BytesIO(raw), columns=['x', 'y', 'z'])
    if len(df) % ROWS:
        return path, None, f'{len(df)} rows is not a multiple of {ROWS}'
    a = df.values.astype(np.float32).reshape(-1, ROWS, 3)[:, _Z['pts']]
    a = a[~np.all(np.isnan(a), axis=(1, 2))]
    if len(a) == 0:
        return path, None, 'no usable frames'
    return path, a, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zip', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--limit', type=int, help='only the first N sequences (smoke test)')
    a = ap.parse_args()

    z = zipfile.ZipFile(a.zip)
    train = pd.read_csv(io.BytesIO(z.read('train.csv')))
    signmap = json.loads(z.read('sign_to_prediction_index_map.json'))
    if a.limit:
        train = train.sample(n=min(a.limit, len(train)), random_state=0).reset_index(drop=True)
    print(f'train.csv: {len(train)} sequences, {train.participant_id.nunique()} participants, {train.sign.nunique()} signs', flush=True)
    os.makedirs(a.out, exist_ok=True)

    t0, arrays, keep, skipped = time.time(), {}, [], []
    with Pool(a.workers, initializer=_init, initargs=(a.zip,)) as pool:
        for i, (path, arr, err) in enumerate(pool.imap_unordered(load_one, train.path.tolist(), chunksize=64)):
            if err:
                skipped.append((path, err))
            else:
                arrays[path] = arr
            if (i + 1) % 5000 == 0:
                print(f'  {i + 1}/{len(train)}  {time.time() - t0:.0f}s', flush=True)

    starts, lens, parts, n = [], [], [], 0
    for _, r in train.iterrows():
        arr = arrays.get(r.path)
        if arr is None:
            continue
        keep.append(r)
        starts.append(n); lens.append(len(arr)); parts.append(arr); n += len(arr)
    meta = pd.DataFrame(keep).reset_index(drop=True)
    meta['label'] = meta.sign.map(signmap)
    meta['start'], meta['length'] = starts, lens
    np.save(os.path.join(a.out, 'landmarks.npy'), np.concatenate(parts))
    meta[['sequence_id', 'participant_id', 'sign', 'label', 'start', 'length']].to_csv(os.path.join(a.out, 'meta.csv'), index=False)
    json.dump(signmap, open(os.path.join(a.out, 'sign_to_prediction_index_map.json'), 'w'))

    L = np.array(lens)
    print(f'wrote {len(meta)} sequences, {n} frames in {time.time() - t0:.0f}s; skipped {len(skipped)}')
    for p, e in skipped[:10]:
        print(f'  skipped {p}: {e}')
    print('frames/sequence min/25/50/75/95/99/max:', np.percentile(L, [0, 25, 50, 75, 95, 99, 100]).astype(int).tolist())
    print(f'longer than 128: {(L > 128).mean():.1%}')
    if meta.label.isna().any():
        sys.exit('ERROR: some signs are missing from the sign map')


if __name__ == '__main__':
    main()
