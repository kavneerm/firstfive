#!/usr/bin/env python3
"""The fixed evaluation harness. Every candidate (and the baseline) is scored by this file only.

    python eval_candidate.py runs/<id> --data data/islr --split data/islr/split.json [--results results.jsonl]

For each export of runs/<id>/best.h5 (float32 TFLite; int8w = 8-bit weights via dynamic-range quantization; int8w-plain =
the same file run without the XNNPACK delegate, which changes both memory and the int8 arithmetic, so it is scored separately):
  val top-1 / top-5 of the TFLite model on every validation sequence (batch 1, what the board runs),
  .tflite size, and runtime memory from mem_probe.py in a fresh process.
Writes runs/<id>/eval.json and appends one line per export to --results.
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import tensorflow as tf
import common as C
import model as Mdl

HERE = os.path.dirname(os.path.abspath(__file__))


def export(student, quant):
    fn = tf.function(lambda t: student(t, training=False),
                     input_signature=[tf.TensorSpec((1, C.MAX_LEN, C.CHANNELS), tf.float32)])
    conv = tf.lite.TFLiteConverter.from_concrete_functions([fn.get_concrete_function()], student)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if quant == 'int8w':
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    return conv.convert()


def tflite_scores(path, ds, threads, plain=False):
    kw = dict(experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES) if plain else {}
    it = tf.lite.Interpreter(model_path=path, num_threads=threads, **kw)
    it.allocate_tensors()
    i, o = it.get_input_details()[0]['index'], it.get_output_details()[0]['index']
    top1 = top5 = n = 0
    logits_all = []
    for xb, yb in ds:
        xb, yb = xb.numpy(), np.argmax(yb.numpy(), -1)
        for x, y in zip(xb, yb):
            it.set_tensor(i, x[None])
            it.invoke()
            lg = it.get_tensor(o)[0]
            r = np.argsort(-lg)[:5]
            top1 += int(r[0] == y); top5 += int(y in r); n += 1
            logits_all.append(lg)
    return top1 / n, top5 / n, n, np.stack(logits_all)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run'); ap.add_argument('--data', required=True); ap.add_argument('--split', required=True)
    ap.add_argument('--results'); ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--quants', default='float32,int8w,int8w-plain')
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.run, 'cfg.json')))
    D = C.Data(a.data)
    _, va = D.split(a.split)
    student = Mdl.get_model(max_len=C.MAX_LEN, num_classes=len(D.signmap), **cfg.get('arch', {}))
    student.load_weights(os.path.join(a.run, 'best.h5'))
    ds = D.dataset(va, 256, False)

    keras_logits = student.predict(ds, verbose=0)
    rows, t0 = [], time.time()
    for q in a.quants.split(','):
        fmt, plain = q.replace('-plain', ''), q.endswith('-plain')
        path = os.path.join(a.run, f'model_{fmt}.tflite')
        open(path, 'wb').write(export(student, fmt))
        top1, top5, n, lg = tflite_scores(path, ds, a.threads, plain)
        mem = json.loads(subprocess.run([sys.executable, os.path.join(HERE, 'mem_probe.py'), path] + (['plain'] if plain else []),
                                        capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
        agree = float((lg.argmax(-1) == keras_logits.argmax(-1)).mean())
        row = dict(id=cfg.get('id', os.path.basename(a.run)), export=q, arch={**Mdl.BASELINE, **cfg.get('arch', {})},
                   distill=bool(cfg.get('distill')), epochs=cfg.get('epochs'), params=int(student.count_params()),
                   val_top1=round(top1, 5), val_top5=round(top5, 5), n_val=n, keras_agreement=round(agree, 5),
                   file_bytes=mem['file_bytes'], mem_delta_kb=mem['delta_kb'], mem_runtime=mem['runtime'],
                   time=time.strftime('%Y-%m-%d %H:%M:%S'))
        rows.append(row)
        print('EVAL', json.dumps(row), flush=True)
    ej = os.path.join(a.run, 'eval.json')
    prev = [r for r in (json.load(open(ej)) if os.path.exists(ej) else []) if r['export'] not in {r['export'] for r in rows}]
    json.dump(prev + rows, open(ej, 'w'), indent=1)
    if a.results:
        with open(a.results, 'a') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
    print(f'eval done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
