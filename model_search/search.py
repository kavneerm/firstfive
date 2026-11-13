#!/usr/bin/env python3
"""Overnight driver: trains and scores candidates one at a time until the deadline; wakes Fable 5.2 between trials
to read results and edit the queue. Works without Fable (falls back to fallback_queue.jsonl).

    python search.py --data data/islr --split data/islr/split.json --deadline "2026-11-12 10:30"

Files: queue.jsonl (Fable edits), fallback_queue.jsonl, results.jsonl (harness output), failures.jsonl,
status.json, notes.md (Fable's notes), runs/<id>/, fable_logs/. Protected files are hashed at start; if anything
other than queue.jsonl / notes.md changes, it is restored from .protected/ and Fable is switched off.
"""
import argparse, datetime as dt, hashlib, json, os, shutil, subprocess, sys, time

PROTECTED = ['common.py', 'model.py', 'islr_model.py', 'train.py', 'eval_candidate.py', 'mem_probe.py', 'search.py',
             'summarize.py', 'FABLE.md', 'fallback_queue.jsonl', 'baseline.h5']
PY = sys.executable


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    open('search.log', 'a').write(line + '\n')


def jl(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({'_bad': line})
    return out


def append(path, obj):
    open(path, 'a').write(json.dumps(obj) + '\n')


class Driver:
    def __init__(self, a):
        self.a = a
        self.deadline = dt.datetime.strptime(a.deadline, '%Y-%m-%d %H:%M').timestamp()
        self.fable_on = not a.no_fable
        self.last_fable = 0.
        os.makedirs('.protected', exist_ok=True); os.makedirs('runs', exist_ok=True); os.makedirs('fable_logs', exist_ok=True)
        self.hashes = {}
        for p in PROTECTED + [a.split]:
            shutil.copy2(p, os.path.join('.protected', p.replace('/', '__')))
            self.hashes[p] = sha(p)
        for p in ('queue.jsonl', 'notes.md', 'results.jsonl', 'failures.jsonl'):
            open(p, 'a').close()

    def check_tamper(self):
        bad = [p for p, h in self.hashes.items() if not os.path.exists(p) or sha(p) != h]
        for p in bad:
            shutil.copy2(os.path.join('.protected', p.replace('/', '__')), p)
        if bad:
            log(f'PROTECTED FILES CHANGED {bad}: restored; Fable switched off for the rest of the night')
            self.fable_on = False

    def done_ids(self):
        ids = {r.get('id') for r in jl('results.jsonl')} | {r.get('id') for r in jl('failures.jsonl')}
        return ids | {r.get('id') for r in jl('skipped.jsonl')}

    def sec_per_epoch(self, cfg):
        """Median measured sec/epoch for the same dim, else the slowest seen, else 240 s."""
        seen = {}
        for d in os.listdir('runs'):
            s = os.path.join('runs', d, 'summary.json')
            if os.path.exists(s):
                j = json.load(open(s)); c = json.load(open(os.path.join('runs', d, 'cfg.json')))
                seen.setdefault(c['arch'].get('dim', 192), []).append(j['sec_per_epoch'] * j['n_train'] / max(1, c.get('max_train') or j['n_train']))
        dim = cfg.get('arch', {}).get('dim', 192)
        if dim in seen:
            return sorted(seen[dim])[len(seen[dim]) // 2]
        return max([max(v) for v in seen.values()], default=240.)

    def next_cfg(self):
        done = self.done_ids()
        for src in ('queue.jsonl', 'fallback_queue.jsonl'):
            for c in jl(src):
                if '_bad' in c:
                    continue
                if c.get('id') and c['id'] not in done and not os.path.exists(os.path.join('runs', c['id'], 'summary.json')):
                    return src, c
        return None, None

    def validate(self, cfg):
        r = subprocess.run([PY, '-c', 'import sys,json,train; train.load_cfg(sys.argv[1])', json.dumps(cfg)],
                           capture_output=True, text=True)
        return None if r.returncode == 0 else (r.stderr.strip().splitlines() or ['invalid'])[-1]

    def run_trial(self, src, cfg):
        cid = cfg['id']
        err = self.validate(cfg)
        if err:
            append('failures.jsonl', dict(id=cid, stage='validate', error=err, cfg=cfg)); log(f'{cid}: invalid config: {err}')
            return
        est = self.sec_per_epoch(cfg) * cfg.get('epochs', 10) * 1.1 + 600
        left = self.deadline - time.time()
        if est > left:
            append('skipped.jsonl', dict(id=cid, reason=f'needs ~{est / 60:.0f} min, {left / 60:.0f} min left'))
            log(f'{cid}: skipped, needs ~{est / 60:.0f} min but {left / 60:.0f} min left')
            return
        out = os.path.join('runs', cid)
        os.makedirs(out, exist_ok=True)
        self.status(dict(state='training', id=cid, source=src, est_min=round(est / 60), started=time.strftime('%H:%M')))
        log(f'{cid} ({src}): training, est {est / 60:.0f} min: {json.dumps(cfg)}')
        with open(os.path.join(out, 'train.log'), 'w') as f:
            r = subprocess.run([PY, 'train.py', '--cfg', json.dumps(cfg), '--data', self.a.data, '--split', self.a.split,
                                '--out', out], stdout=f, stderr=subprocess.STDOUT, timeout=max(600, left))
        if r.returncode:
            tail = open(os.path.join(out, 'train.log')).read()[-1500:]
            append('failures.jsonl', dict(id=cid, stage='train', returncode=r.returncode, log_tail=tail)); log(f'{cid}: training FAILED')
            return
        self.status(dict(state='evaluating', id=cid))
        with open(os.path.join(out, 'eval.log'), 'w') as f:
            r = subprocess.run([PY, 'eval_candidate.py', out, '--data', self.a.data, '--split', self.a.split,
                                '--results', 'results.jsonl'], stdout=f, stderr=subprocess.STDOUT)
        if r.returncode:
            tail = open(os.path.join(out, 'eval.log')).read()[-1500:]
            append('failures.jsonl', dict(id=cid, stage='eval', returncode=r.returncode, log_tail=tail)); log(f'{cid}: eval FAILED')
            return
        for row in json.load(open(os.path.join(out, 'eval.json'))):
            log(f"{cid} {row['export']}: top1 {row['val_top1']:.4f}  {row['file_bytes'] / 1e6:.2f} MB  mem {row['mem_delta_kb'] / 1024:.1f} MB")

    def wake_fable(self, why):
        if not self.fable_on:
            return
        self.check_tamper()
        left = (self.deadline - time.time()) / 60
        prompt = (open('FABLE.md').read() + f'\n\n## Right now\nTime {time.strftime("%H:%M")}, {left:.0f} minutes until the '
                  f'deadline. Woken because: {why}. Read results.jsonl, failures.jsonl, skipped.jsonl, search.log, '
                  f'notes.md and runs/*/summary.json, then update queue.jsonl and notes.md.')
        cmd = ['claude', '-p', prompt, '--model', self.a.model, '--max-turns', '40',
               '--allowedTools', 'Read', 'Grep', 'Glob', 'Edit(queue.jsonl)', 'Write(queue.jsonl)',
               'Edit(notes.md)', 'Write(notes.md)']
        logf = os.path.join('fable_logs', time.strftime('%H%M%S') + '.log')
        log(f'waking Fable ({why})')
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            open(logf, 'w').write(r.stdout + '\n--- stderr ---\n' + r.stderr)
            log(f'Fable finished rc={r.returncode} (log {logf})')
        except subprocess.TimeoutExpired:
            log('Fable timed out after 20 min')
        self.last_fable = time.time()
        self.check_tamper()

    def status(self, d):
        d.update(deadline=dt.datetime.fromtimestamp(self.deadline).strftime('%H:%M'), fable_on=self.fable_on,
                 results=len(jl('results.jsonl')))
        json.dump(d, open('status.json', 'w'), indent=1)

    def loop(self):
        log(f'driver start; deadline {self.a.deadline}; fable {"on" if self.fable_on else "off"}')
        self.wake_fable('start of the night')
        while time.time() < self.deadline - 900:
            self.check_tamper()
            src, cfg = self.next_cfg()
            if cfg is None:
                if self.fable_on and time.time() - self.last_fable > 300:
                    self.wake_fable('the queue is empty')
                    continue
                log('nothing left to run'); break
            self.run_trial(src, cfg)
            if time.time() - self.last_fable > self.a.fable_every:
                self.wake_fable(f'trial {cfg["id"]} finished')
        self.status(dict(state='finished'))
        subprocess.run([PY, 'summarize.py', '--results', 'results.jsonl', '--out', 'SUMMARY.md'])
        log('driver done; see SUMMARY.md')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True); ap.add_argument('--split', required=True)
    ap.add_argument('--deadline', required=True, help='local time "YYYY-MM-DD HH:MM"; no trial starts that cannot finish')
    ap.add_argument('--model', default='claude-fable-5-2'); ap.add_argument('--fable-every', type=int, default=0,
                                                                          help='min seconds between Fable wakes (0 = after every trial)')
    ap.add_argument('--no-fable', action='store_true')
    Driver(ap.parse_args()).loop()
