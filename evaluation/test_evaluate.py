#!/usr/bin/env python3
"""Checks for evaluation/evaluate.py. Plain asserts, prints PASS/FAIL per check and ALL PASS / exit 1 at the end.

    python evaluation/test_evaluate.py                                   # metric math only (numpy)
    python evaluation/test_evaluate.py --real <kaggle sample dir>        # + model run vs raw Kaggle parquet (needs pandas)

The --real folder holds Kaggle's train.csv, the train_landmark_files/ it names, and out/ made from them by
model_search/prepare_kaggle.py. The check feeds each raw 543-point clip straight to sign_infer_lite.LitePredictor
and requires evaluate.run_model (which rebuilds clips from the 118-point landmarks.npy) to give the same logits.
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate as E

FAILED = []


def check(name, cond, detail=''):
    print(('PASS  ' if cond else 'FAIL  ') + name + (f'  ({detail})' if detail else ''))
    if not cond:
        FAILED.append(name)


def synthetic(n=3000, c=12, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, c, n)
    lg = rng.normal(size=(n, c)).astype(np.float32)
    lg[np.arange(n), y] += rng.uniform(0, 3, n)          # mostly right, sometimes not
    lg[5] = np.nan                                         # one clip with no landmarks
    return dict(logits=lg, label=y, length=rng.integers(1, 300, n), participant=rng.choice([11, 22, 33], n),
                sequence_id=np.arange(n), hand_share=rng.choice([0., .25, .5, .7, .95, 1.], n).astype(np.float32),
                names=np.array([f'w{i:02d}' for i in range(c)]), model=np.array('x'), runtime=np.array('x'),
                seconds=np.array(0.))


def test_metrics():
    # Cole's report: 13149 / 18528 correct, 95% CI 70.31-71.62%. Same interval method?
    lo, hi = E.wilson(13149, 18528)
    check('Wilson CI reproduces the CI in Cole\'s report', (round(100 * lo, 2), round(100 * hi, 2)) == (70.31, 71.62),
          f'{100 * lo:.2f}-{100 * hi:.2f}')

    P = synthetic()
    R = E.compute(P)
    S, F = R['summary'], R['fig_stats']
    ok = ~np.isnan(P['logits']).any(-1)
    lg, y = P['logits'][ok], P['label'][ok]
    N = len(y)
    check('unusable clip counted and left out', S['unusable_sequences'] == 1 and S['sequences'] == N)
    for k in (1, 3, 5, 10):
        brute = np.mean([y[i] in np.argsort(-lg[i])[:k] for i in range(N)])
        check(f'top-{k} matches brute force', abs(R['topk'][k] - brute) < 1e-12, f'{R["topk"][k]:.4f} vs {brute:.4f}')
    pred = lg.argmax(-1)
    check('correct count', S['correct'] == int((pred == y).sum()))
    for r in R['per']:
        m = y == r['index']
        if abs(r['top1'] - (pred[m] == y[m]).mean()) > 1e-12 or r['n'] != m.sum():
            check(f'per-word accuracy for {r["sign"]}', False); break
        wrong = pred[m][pred[m] != r['index']]
        if len(wrong):
            vals, cnt = np.unique(wrong, return_counts=True)
            if cnt.max() != r['times_confused'] or P['names'][vals[cnt.argmax()]] != r['most_confused_with'] and \
                    (cnt == cnt.max()).sum() == 1:
                check(f'most-confused for {r["sign"]}', False, f'{r}'); break
    else:
        check('per-word accuracy and most-confused word match brute force', True)
    check('macro accuracy = mean of word accuracies',
          abs(S['macro_accuracy'] - np.mean([(pred[y == c] == c).mean() for c in np.unique(y)])) < 1e-12)
    check('confusion matrix rows sum to clips per word', all(R['cm'][c].sum() == (y == c).sum() for c in range(12)))
    pairs = sum(c for c, _, _ in R['conf_pairs'])
    check('mix-up counts add up to all wrong answers', pairs == int((pred != y).sum()))
    for key in ('clip_length', 'hand_detection', 'per_signer'):
        tot = sum(g['n'] for g in F[key])
        check(f'{key} groups cover every clip exactly once', tot == N, f'{tot} vs {N}')
    conf = E.softmax(lg.astype(np.float64)).max(-1)
    for b in S['by_confidence']:
        m = conf >= b['threshold']
        check(f'confidence >= {b["threshold"]}: share and accuracy',
              abs(b['share_of_predictions'] - m.mean()) < 1e-12 and
              abs(b['accuracy_when_that_confident'] - (pred[m] == y[m]).mean()) < 1e-12)
    check('answer-when-sure curve ends at overall accuracy', abs(R['cum_acc'][-1] - S['top1_accuracy']) < 1e-12)
    ece = sum((np.minimum((conf * 10).astype(int), 9) == b).mean() *
              abs((pred == y)[np.minimum((conf * 10).astype(int), 9) == b].mean() -
                  conf[np.minimum((conf * 10).astype(int), 9) == b].mean())
              for b in range(10) if (np.minimum((conf * 10).astype(int), 9) == b).any())
    check('calibration error', abs(R['ece'] - ece) < 1e-12)
    text = E.report_text(R, P, 'synthetic')
    check('report text has every section', all(s in text for s in ('OVERALL ACCURACY', 'WEAKEST 15', 'STRONGEST 10',
                                                                     'MOST COMMON MIX-UPS', 'BY SIGNER')))


def test_real(real):
    import pandas as pd
    import sign_infer_lite as L
    os.environ['SIGN_NO_XNNPACK'] = '1'
    model = os.path.join(HERE, '..', 'sign_app', 'sign_model_int8w.tflite')
    out = os.path.join(real, 'out')
    P = E.run_model(model, out, '', None, 1)
    meta = pd.read_csv(os.path.join(out, 'meta.csv'))
    train = pd.read_csv(os.path.join(real, 'train.csv')).set_index('sequence_id')
    pred = L.LitePredictor(model, os.path.join(out, 'sign_to_prediction_index_map.json'))
    found = {f: os.path.join(d, f) for d, _, fs in os.walk(real) for f in fs if f.endswith('.parquet')}
    diffs, same_top = [], 0
    for k, sid in enumerate(meta.sequence_id):
        df = pd.read_parquet(found[os.path.basename(train.loc[sid, 'path'])], columns=['x', 'y', 'z'])
        raw = df.values.astype(np.float32).reshape(-1, 543, 3)
        ref = pred.logits(L.prepare_input(raw, pred.max_len))
        diffs.append(np.abs(ref - P['logits'][k]).max())
        same_top += int(ref.argmax() == P['logits'][k].argmax())
    check(f'{len(diffs)} real clips: logits match the board code on raw Kaggle files', max(diffs) < 1e-5,
          f'max diff {max(diffs):.2e}')
    check('same top answer on every real clip', same_top == len(diffs), f'{same_top}/{len(diffs)}')
    check('labels come through from meta.csv', (P['label'] == meta.label.values).all())
    R = E.compute(P)
    print(f'      (real sample: top-1 {100 * R["summary"]["top1_accuracy"]:.1f}% on {R["summary"]["sequences"]} clips)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--real')
    a = ap.parse_args()
    test_metrics()
    if a.real:
        test_real(a.real)
    print('ALL PASS' if not FAILED else f'{len(FAILED)} FAILED: {FAILED}')
    sys.exit(1 if FAILED else 0)
