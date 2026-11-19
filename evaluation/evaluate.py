#!/usr/bin/env python3
"""Accuracy report for a sign model: the same numbers, CSVs and charts as Cole's fold-0 report, for any .tflite model.

    python evaluation/evaluate.py --model sign_app/sign_model_int8w.tflite --data <dir> \
        --participants 29302,49445,27610,62590 --out evaluation/results/int8w
    python evaluation/evaluate.py --from-predictions evaluation/results/int8w/predictions.npz --out evaluation/results/int8w

--data is a folder made by model_search/prepare_kaggle.py (landmarks.npy, meta.csv, sign_to_prediction_index_map.json).
Each clip runs through sign_app/sign_infer_lite.py, one at a time with XNNPACK off: the exact path the UNO Q runs.
Fold 0 of the original training (signers 29302, 49445, 27610, 62590) is the set no version of this model trained on.
"""
import argparse, csv, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'sign_app'))

CONF_THRESHOLDS = (0.5, 0.7, 0.9)
LENGTH_BINS = [(0, 10, '≤10'), (11, 20, '11-20'), (21, 40, '21-40'), (41, 80, '41-80'), (81, 160, '81-160'),
               (161, 10 ** 9, '>160')]
HAND_BINS = [(-1, .25, '≤25%'), (.25, .50, '25-50%'), (.50, .75, '50-75%'), (.75, .95, '75-95%'), (.95, 1., '>95%')]  # (a, b]


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def softmax(lg):
    e = np.exp(lg - lg.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------- inference

def run_model(model, data, participants, limit, threads):
    os.environ.setdefault('SIGN_NO_XNNPACK', '1')
    import sign_infer_lite as L
    lm = np.load(os.path.join(data, 'landmarks.npy'), mmap_mode='r')
    rows = list(csv.DictReader(open(os.path.join(data, 'meta.csv'))))
    if participants:
        keep = {int(p) for p in participants.split(',')}
        rows = [r for r in rows if int(r['participant_id']) in keep]
    if limit:
        rows = rows[:limit]
    if lm.shape[1] != len(L.POINT_LANDMARKS):
        raise SystemExit(f'landmarks.npy has {lm.shape[1]} points per frame, expected {len(L.POINT_LANDMARKS)}')
    hand_cols = [L.POINT_LANDMARKS.index(i) for i in L.LHAND + L.RHAND]
    pred = L.LitePredictor(model, os.path.join(data, 'sign_to_prediction_index_map.json'), num_threads=threads)
    n_cls = pred.it.get_output_details()[0]['shape'][-1]

    logits = np.full((len(rows), n_cls), np.nan, np.float32)
    hand_share = np.zeros(len(rows), np.float32)
    t0 = time.time()
    for k, r in enumerate(rows):
        s, n = int(r['start']), int(r['length'])
        clip118 = np.asarray(lm[s:s + n], np.float32)
        clip = np.full((n, 543, 3), np.nan, np.float32)
        clip[:, L.POINT_LANDMARKS] = clip118
        hand_share[k] = (~np.all(np.isnan(clip118[:, hand_cols]), axis=(1, 2))).mean() if n else 0.
        x = L.prepare_input(clip, pred.max_len)
        if x is not None:
            logits[k] = pred.logits(x)
        if (k + 1) % 2000 == 0:
            print(f'  {k + 1}/{len(rows)} clips, {time.time() - t0:.0f}s', flush=True)
    names = [n for n, _ in sorted(json.load(open(os.path.join(data, 'sign_to_prediction_index_map.json'))).items(),
                                  key=lambda t: t[1])]
    return dict(logits=logits, label=np.array([int(r['label']) for r in rows]),
                length=np.array([int(r['length']) for r in rows]),
                participant=np.array([int(r['participant_id']) for r in rows]),
                sequence_id=np.array([int(r['sequence_id']) for r in rows]),
                hand_share=hand_share, names=np.array(names), model=np.array(os.path.basename(model)),
                runtime=np.array('tflite, XNNPACK off' if os.environ.get('SIGN_NO_XNNPACK') == '1' else 'tflite, XNNPACK on'),
                seconds=np.array(time.time() - t0))


# ---------------------------------------------------------------- metrics

def compute(P):
    lg, y, names = P['logits'], P['label'], [str(n) for n in P['names']]
    ok = ~np.isnan(lg).any(-1)
    unusable = int((~ok).sum())
    lg, y = lg[ok], y[ok]
    length, part, hand = P['length'][ok], P['participant'][ok], P['hand_share'][ok]
    N, C = lg.shape
    prob = softmax(lg.astype(np.float64))
    order = np.argsort(-lg, -1)
    rank = np.argmax(order == y[:, None], -1)               # 0 = right first guess
    top1 = order[:, 0]
    conf = prob[np.arange(N), top1]
    correct = top1 == y

    def acc_ci(mask):
        k, n = int(correct[mask].sum()), int(mask.sum())
        lo, hi = wilson(k, n)
        return dict(n=n, acc=k / n if n else float('nan'), lo=lo, hi=hi)

    topk = {k: float((rank < k).mean()) for k in range(1, 11)}

    per = []
    cm = np.zeros((C, C), np.int64)
    np.add.at(cm, (y, top1), 1)
    for c in range(C):
        m = y == c
        n = int(m.sum())
        if n == 0:
            continue
        k = int(correct[m].sum())
        lo, hi = wilson(k, n)
        row = cm[c].copy(); row[c] = 0
        j = int(row.argmax())
        per.append(dict(sign=names[c], index=c, n=n, top1=k / n, top1_ci_low=lo, top1_ci_high=hi,
                        top5=float((rank[m] < 5).mean()), mean_prob_of_true=float(prob[m, c].mean()),
                        most_confused_with=names[j] if row[j] else '', times_confused=int(row[j])))
    per.sort(key=lambda r: (r['top1'], r['sign']))
    word_acc = np.array([r['top1'] for r in per])

    off = cm.copy(); np.fill_diagonal(off, 0)
    pairs = np.argwhere(off > 0)
    conf_pairs = sorted(((int(off[a, b]), names[a], names[b]) for a, b in pairs), key=lambda t: (-t[0], t[1], t[2]))

    # calibration: 10 equal-width confidence bins, error = weighted |accuracy - confidence|
    bins = np.minimum((conf * 10).astype(int), 9)
    calib, ece = [], 0.
    for b in range(10):
        m = bins == b
        if m.any():
            calib.append(dict(conf=float(conf[m].mean()), acc=float(correct[m].mean()), n=int(m.sum())))
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())

    srt = np.argsort(-conf, kind='stable')
    cum_acc = np.cumsum(correct[srt]) / np.arange(1, N + 1)
    coverage = np.arange(1, N + 1) / N
    by_conf = []
    for t in CONF_THRESHOLDS:
        m = conf >= t
        by_conf.append(dict(threshold=t, share_of_predictions=float(m.mean()),
                            accuracy_when_that_confident=float(correct[m].mean()) if m.any() else float('nan')))
    most_conf = {f'{int(c * 100)}pct': float(cum_acc[max(0, int(round(c * N)) - 1)]) for c in (0.5, 0.8, 0.9)}

    clip_len = [dict(label=lab, **acc_ci((length >= a) & (length <= b))) for a, b, lab in LENGTH_BINS]
    hand_det = [dict(label=lab, **acc_ci((hand > a) & (hand <= b))) for a, b, lab in HAND_BINS]
    signers = [dict(label=str(p), **acc_ci(part == p)) for p in dict.fromkeys(part.tolist())]

    k1 = int(correct.sum())
    lo1, hi1 = wilson(k1, N)
    lo5, hi5 = wilson(int((rank < 5).sum()), N)
    summary = dict(sequences=N, classes_in_test=len(per), classes_total=C, unusable_sequences=unusable,
                   top1_accuracy=topk[1], top1_ci=[lo1, hi1], correct=k1, top5_accuracy=topk[5], top5_ci=[lo5, hi5],
                   macro_accuracy=float(word_acc.mean()), median_word_accuracy=float(np.median(word_acc)),
                   words_at_100pct=int((word_acc == 1).sum()), words_at_90pct_or_more=int((word_acc >= .9).sum()),
                   words_below_50pct=int((word_acc < .5).sum()), min_word_accuracy=float(word_acc.min()),
                   max_word_accuracy=float(word_acc.max()), by_confidence=by_conf)
    fig_stats = dict(sequences=N, top1=topk[1], top3=topk[3], top5=topk[5], top10=topk[10], calibration_error=float(ece),
                     accuracy_when_answering_most_confident=most_conf, clip_length=clip_len, hand_detection=hand_det,
                     per_signer=signers)
    return dict(summary=summary, fig_stats=fig_stats, per=per, conf_pairs=conf_pairs, cm=cm, names=names, topk=topk,
                calib=calib, ece=float(ece), coverage=coverage, cum_acc=cum_acc, conf=conf, correct=correct)


# ---------------------------------------------------------------- report

def report_text(R, P, title):
    S, F = R['summary'], R['fig_stats']
    pc = lambda v: f'{100 * v:.2f}%'
    L = [f'model         : {P["model"]}  ({P["runtime"]}, one clip at a time, sign_app/sign_infer_lite.py)',
         f'tested on     : {title}',
         f'signers       : {", ".join(s["label"] for s in F["per_signer"])}', '']
    bar = '=' * 64
    L += [bar, 'OVERALL ACCURACY', bar,
          f'sequences tested        : {S["sequences"]}  ({S["classes_in_test"]} of {S["classes_total"]} words present)',
          f'top-1 accuracy          : {pc(S["top1_accuracy"])}   (95% CI {100 * S["top1_ci"][0]:.2f}-{100 * S["top1_ci"][1]:.2f}%)   [{S["correct"]} correct]',
          f'top-5 accuracy          : {pc(S["top5_accuracy"])}   (95% CI {100 * S["top5_ci"][0]:.2f}-{100 * S["top5_ci"][1]:.2f}%)',
          f'macro (per-word) average: {pc(S["macro_accuracy"])}   <- every word counts equally',
          f'median word accuracy    : {pc(S["median_word_accuracy"])}',
          f'chance level            : {pc(1 / S["classes_total"])}', '',
          f'words at 100%           : {S["words_at_100pct"]}',
          f'words at >= 90%         : {S["words_at_90pct_or_more"]}',
          f'words below 50%         : {S["words_below_50pct"]}',
          f'worst / best word       : {100 * S["min_word_accuracy"]:.1f}% / {100 * S["max_word_accuracy"]:.1f}%',
          f'unusable clips          : {S["unusable_sequences"]}  (no landmarks at all; left out)', '',
          'When the model is confident, how often is it right?']
    for b in S['by_confidence']:
        L.append(f'  confidence >= {b["threshold"]}:  {100 * b["share_of_predictions"]:.1f}% of predictions, '
                 f'{100 * b["accuracy_when_that_confident"]:.1f}% correct')
    L += ['', bar, 'WEAKEST 15 WORDS', bar, f'{"word":<20}{"n":>7}{"top-1":>8}   most often mistaken for']
    for r in R['per'][:15]:
        L.append(f'{r["sign"]:<20}{r["n"]:>7}{100 * r["top1"]:>7.1f}%   {r["most_confused_with"]} ({r["times_confused"]}x)')
    L += ['', bar, 'STRONGEST 10 WORDS', bar]
    for r in R['per'][::-1][:10]:
        L.append(f'{r["sign"]:<20}{r["n"]:>7}{100 * r["top1"]:>7.1f}%')
    L += ['', bar, 'MOST COMMON MIX-UPS (true word -> what the model said)', bar]
    for c, a, b in R['conf_pairs'][:15]:
        L.append(f'  {c:>3}x  {a} -> {b}')
    L += ['', bar, 'BY SIGNER', bar]
    for s in F['per_signer']:
        L.append(f'  {s["label"]:<8} n={s["n"]:<6} top-1 {100 * s["acc"]:.2f}%  (95% CI {100 * s["lo"]:.1f}-{100 * s["hi"]:.1f}%)')
    L += ['', "Note: per-word results rest on few examples per word, so a single word's accuracy is rough; see the CI",
          "columns in per_sign_accuracy.csv. The overall number is much more precise than any single word's."]
    return '\n'.join(L) + '\n'


def write_files(R, P, out, title):
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, 'report.txt'), 'w').write(report_text(R, P, title))
    json.dump(R['summary'], open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    json.dump(R['fig_stats'], open(os.path.join(out, 'figure_stats.json'), 'w'), indent=1)
    cols = ['sign', 'index', 'n', 'top1', 'top1_ci_low', 'top1_ci_high', 'top5', 'mean_prob_of_true',
            'most_confused_with', 'times_confused']
    with open(os.path.join(out, 'per_sign_accuracy.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(R['per'])
    with open(os.path.join(out, 'top_confusions.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['true', 'predicted', 'count'])
        w.writerows([(a, b, c) for c, a, b in R['conf_pairs'][:200]])


# ---------------------------------------------------------------- charts

BLUE, RED, GREY = '#3a7dd8', '#d9534f', '#888888'


def charts(R, out, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    S, F = R['summary'], R['fig_stats']
    overall = 100 * S['top1_accuracy']

    def topk(ax):
        ks = list(range(1, 11)); v = [100 * R['topk'][k] for k in ks]
        ax.plot(ks, v, 'o-', color=BLUE)
        for k in (1, 3, 5):
            ax.annotate(f'{v[k - 1]:.1f}%', (k, v[k - 1]), textcoords='offset points', xytext=(0, -16), ha='center', fontsize=8)
        ax.axhline(100 / S['classes_total'], color=GREY, ls=':', lw=1)
        ax.text(10.3, 1.5, 'chance for top-1', ha='right', fontsize=7, color=GREY)
        ax.set(ylim=(0, 102), xticks=ks, title='Top-K accuracy', ylabel='% of signs',
               xlabel="the right word is among the model's top K guesses")

    def hist(ax):
        ax.hist(100 * np.array([r['top1'] for r in R['per']]), bins=np.arange(0, 105, 5), color=BLUE, edgecolor='white')
        ax.axvline(overall, color=RED, ls='--', lw=1, label=f'overall {overall:.1f}%')
        ax.legend(fontsize=8, loc='upper left')
        ax.set(title='How accurate is each word?', xlabel='accuracy of a single word (%)', ylabel='number of words')

    def calib(ax):
        ax.plot([0, 100], [0, 100], '--', color=GREY, label='perfectly calibrated')
        ax.plot([100 * c['conf'] for c in R['calib']], [100 * c['acc'] for c in R['calib']], 'o-', color=BLUE, label='the model')
        ax.legend(fontsize=8, loc='upper left')
        ax.set(xlim=(0, 100), ylim=(0, 100), title=f'Can you trust its confidence?  (calibration error {100 * R["ece"]:.1f} points)',
               xlabel='how confident the model says it is (%)', ylabel='how often it is actually right (%)')

    def sure(ax):
        cov, acc = 100 * R['coverage'], 100 * R['cum_acc']
        ax.plot(cov, acc, color=BLUE)
        for b in S['by_confidence']:
            x = 100 * b['share_of_predictions']; yv = 100 * b['accuracy_when_that_confident']
            ax.plot(x, yv, 'o', color=RED, ms=4)
            ax.annotate(f'conf ≥ {b["threshold"]}', (x, yv), textcoords='offset points', xytext=(5, 6), fontsize=8)
        ax.set(xlim=(0, 100), ylim=(min(acc.min(), overall) - 5, 101),
               title='Answer only when sure: accuracy vs. how much it answers',
               xlabel='% of signs the model answers (most confident first)', ylabel='accuracy on those (%)')

    def bars(ax, groups, ttl, xl):
        x = np.arange(len(groups))
        acc = [100 * g['acc'] if g['n'] else 0 for g in groups]
        err = [[max(0, 100 * (g['acc'] - g['lo'])) if g['n'] else 0 for g in groups],
               [max(0, 100 * (g['hi'] - g['acc'])) if g['n'] else 0 for g in groups]]
        ax.bar(x, acc, color=BLUE, width=.7, yerr=err, capsize=3)
        for i, g in enumerate(groups):
            ax.text(i, 2, f'n={g["n"]}', ha='center', color='white', fontsize=7)
        ax.axhline(overall, color=RED, ls='--', lw=1, label=f'overall {overall:.1f}%')
        ax.legend(fontsize=8, loc='upper center', frameon=False)
        ax.set(xticks=x, ylim=(0, 120), yticks=range(0, 101, 20), title=ttl, xlabel=xl, ylabel='top-1 accuracy (%)')
        ax.set_xticklabels([g['label'] for g in groups], fontsize=8)

    def weak_strong(fig):
        per = R['per']; strong = per[::-1][:15][::-1]
        for ax, rows, col, ttl in ((fig.add_subplot(121), per[:15][::-1], RED, 'Weakest 15 words'),
                                   (fig.add_subplot(122), strong, BLUE, 'Strongest 15 words')):
            v = [100 * r['top1'] for r in rows]
            lo = [100 * (r['top1'] - r['top1_ci_low']) for r in rows]; hi = [100 * (r['top1_ci_high'] - r['top1']) for r in rows]
            ax.barh(range(len(rows)), v, color=col, xerr=[lo, hi], capsize=2)
            lab = [f'{r["sign"]}  → {r["most_confused_with"]}' if col == RED and r['most_confused_with'] else r['sign'] for r in rows]
            ax.set_yticks(range(len(rows))); ax.set_yticklabels(lab, fontsize=8)
            ax.axvline(overall, color=GREY, ls='--', lw=1)
            ax.set(xlim=(0, 100), title=ttl, xlabel='top-1 accuracy (%), with 95% CI')

    def confusion(ax, idx, ttl, labels):
        cm = R['cm'][np.ix_(idx, idx)].astype(float)
        cm = cm / np.maximum(R['cm'][idx].sum(1, keepdims=True), 1)
        im = ax.imshow(100 * cm, cmap='Blues', vmin=0, vmax=100 if labels else 30, interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=.046, label='% of that word\'s clips')
        if labels:
            nm = [R['names'][i] for i in idx]
            ax.set_xticks(range(len(idx))); ax.set_xticklabels(nm, rotation=90, fontsize=7)
            ax.set_yticks(range(len(idx))); ax.set_yticklabels(nm, fontsize=7)
        ax.set(title=ttl, xlabel='what the model said', ylabel='true word')

    def single(name, fn, size=(9, 6.2)):
        fig, ax = plt.subplots(figsize=size, dpi=100)
        fn(ax); fig.tight_layout(); fig.savefig(os.path.join(out, name)); plt.close(fig)

    single('2_topk_accuracy.png', topk)
    single('3_word_accuracy_histogram.png', hist)
    fig = plt.figure(figsize=(13, 6.5), dpi=100); weak_strong(fig); fig.tight_layout()
    fig.savefig(os.path.join(out, '4_weakest_and_strongest_words.png')); plt.close(fig)
    single('5_confusion_matrix_all_words.png',
           lambda ax: confusion(ax, list(range(len(R['names']))), 'Confusion matrix, all words (alphabetical index)', False),
           size=(10, 9))
    words = []
    for _, a, b in R['conf_pairs']:
        for w in (a, b):
            if w not in words:
                words.append(w)
        if len(words) >= 24:
            break
    idx = [R['names'].index(w) for w in words[:24]]
    single('6_confusion_matrix_zoom.png', lambda ax: confusion(ax, idx, 'Confusion matrix, most-confused words', True),
           size=(10, 9))

    def mixups(ax):
        top = R['conf_pairs'][:20][::-1]
        ax.barh(range(len(top)), [c for c, _, _ in top], color=BLUE)
        ax.set_yticks(range(len(top))); ax.set_yticklabels([f'{a} → {b}' for _, a, b in top], fontsize=8)
        ax.set(title='Most common mix-ups', xlabel='number of clips (true word → what the model said)')
    single('7_top_mixups.png', mixups, size=(9, 7))
    single('8_confidence_calibration.png', calib)
    single('9_answer_only_when_sure.png', sure)
    single('10_accuracy_by_clip_length.png',
           lambda ax: bars(ax, F['clip_length'], 'Does clip length matter?', 'length of the signed clip (video frames)'))
    single('11_accuracy_by_hand_detection.png',
           lambda ax: bars(ax, F['hand_detection'], 'Does hand detection matter?', 'share of frames in which a hand was detected'))
    single('12_accuracy_by_signer.png',
           lambda ax: bars(ax, F['per_signer'], 'Accuracy for each signer', 'signer (participant id)'))

    def per_sign(ax):
        v = 100 * np.array([r['top1'] for r in R['per']])
        ax.bar(range(len(v)), v, width=1., color=[RED if a < 50 else BLUE for a in v])
        ax.axhline(overall, color=GREY, ls='--', lw=1, label=f'overall {overall:.1f}%')
        ax.legend(fontsize=8, loc='upper left')
        ax.set(xlim=(-1, len(v)), ylim=(0, 100), title='Top-1 accuracy of every word, worst to best (red: below 50%)',
               xlabel='words, sorted by accuracy', ylabel='top-1 accuracy (%)')
    single('per_sign_accuracy.png', per_sign, size=(13, 5))

    fig, axs = plt.subplots(2, 3, figsize=(20, 11), dpi=100)
    fig.suptitle(f'Sign model tested on {S["sequences"]:,} signs across {S["classes_in_test"]} words: '
                 f'top-1 {overall:.1f}%, top-5 {100 * S["top5_accuracy"]:.1f}%\n{title}', fontsize=15)
    topk(axs[0, 0]); hist(axs[0, 1]); calib(axs[0, 2]); sure(axs[1, 0])
    bars(axs[1, 1], F['clip_length'], 'Does clip length matter?', 'length of the signed clip (video frames)')
    bars(axs[1, 2], F['per_signer'], 'Accuracy for each signer', 'signer (participant id)')
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(os.path.join(out, 'dashboard.png')); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model'); ap.add_argument('--data')
    ap.add_argument('--participants', default='29302,49445,27610,62590', help="comma list; '' = every clip in --data")
    ap.add_argument('--from-predictions', help='redraw from a saved predictions.npz instead of running the model '
                                               '(--participants then picks a subset of its clips)')
    ap.add_argument('--out', required=True); ap.add_argument('--limit', type=int)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--title', default='fold 0: the 4 signers no version of this model trained on')
    a = ap.parse_args()
    if a.from_predictions:
        P = dict(np.load(a.from_predictions, allow_pickle=False))
        if a.participants:
            keep = np.isin(P['participant'], [int(p) for p in a.participants.split(',')])
            P.update({k: v[keep] for k, v in P.items() if v.ndim and len(v) == len(keep) and k != 'names'})
    else:
        if not (a.model and a.data):
            ap.error('--model and --data are required unless --from-predictions is given')
        P = run_model(a.model, a.data, a.participants, a.limit, a.threads)
        os.makedirs(a.out, exist_ok=True)
        np.savez_compressed(os.path.join(a.out, 'predictions.npz'), **P)
        print(f'inference: {len(P["label"])} clips in {float(P["seconds"]):.0f}s')
    R = compute(P)
    write_files(R, P, a.out, a.title)
    charts(R, a.out, a.title)
    S = R['summary']
    print(f'top-1 {100 * S["top1_accuracy"]:.2f}%  top-5 {100 * S["top5_accuracy"]:.2f}%  on {S["sequences"]} clips '
          f'({S["unusable_sequences"]} unusable) -> {a.out}')


if __name__ == '__main__':
    main()
