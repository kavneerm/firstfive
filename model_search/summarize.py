#!/usr/bin/env python3
"""Baseline vs every candidate, with the pass criterion.

    python summarize.py --results results.jsonl [--out SUMMARY.md] [--tol 0.01] [--mem-frac 0.5]

PASS = val top-1 >= baseline float32 top-1 - tol AND runtime memory <= mem-frac x baseline float32 memory.
The winner is the passing export with the least runtime memory.
"""
import argparse, json, sys

ap = argparse.ArgumentParser()
ap.add_argument('--results', default='results.jsonl'); ap.add_argument('--out')
ap.add_argument('--tol', type=float, default=0.01); ap.add_argument('--mem-frac', type=float, default=0.5)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.results) if l.strip()]
base = [r for r in rows if r['id'] == 'baseline' and r['export'] == 'float32']
if not base:
    sys.exit('no baseline float32 row in results')
b = base[-1]
lines = [f"Baseline (float32): top-1 {b['val_top1']:.4f}, top-5 {b['val_top5']:.4f}, {b['file_bytes'] / 1e6:.2f} MB file, "
         f"{b['mem_delta_kb'] / 1024:.1f} MB runtime memory, {b['params']:,} params, n_val {b['n_val']}",
         f"PASS = top-1 >= {b['val_top1'] - a.tol:.4f} and memory <= {a.mem_frac * b['mem_delta_kb'] / 1024:.1f} MB", '',
         '| id | export | epochs | distill | dim/stages/convs/k | params | top-1 | Δ top-1 | file MB | mem MB | mem vs base | pass |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
passing = []
for r in sorted(rows, key=lambda r: (r['mem_delta_kb'], -r['val_top1'])):
    ar = r['arch']
    ok = not (r['id'] == 'baseline' and r['export'] == 'float32') and r['val_top1'] >= b['val_top1'] - a.tol and r['mem_delta_kb'] <= a.mem_frac * b['mem_delta_kb']
    if ok:
        passing.append(r)
    lines.append(f"| {r['id']} | {r['export']} | {r['epochs']} | {'y' if r['distill'] else ''} | "
                 f"{ar['dim']}/{ar['stages']}/{ar['convs']}/{ar['ksize']} | {r['params']:,} | {r['val_top1']:.4f} | "
                 f"{(r['val_top1'] - b['val_top1']) * 100:+.2f} | {r['file_bytes'] / 1e6:.2f} | {r['mem_delta_kb'] / 1024:.1f} | "
                 f"{r['mem_delta_kb'] / b['mem_delta_kb']:.0%} | {'PASS' if ok else ''} |")
lines.append('')
if passing:
    w = min(passing, key=lambda r: (r['mem_delta_kb'], -r['val_top1']))
    lines.append(f"WINNER: {w['id']} ({w['export']}): top-1 {w['val_top1']:.4f} ({(w['val_top1'] - b['val_top1']) * 100:+.2f} pts), "
                 f"memory {w['mem_delta_kb'] / 1024:.1f} MB = {w['mem_delta_kb'] / b['mem_delta_kb']:.0%} of baseline")
else:
    lines.append('NO CANDIDATE PASSES')
text = '\n'.join(lines)
print(text)
if a.out:
    open(a.out, 'w').write(text + '\n')
