#!/usr/bin/env python3
"""Compare sign models on the UNO Q: runtime memory, speed, and whether they give the same answers.

    cd ~/sign_demo && ~/sign_env/bin/python board_check.py sign_model.tflite sign_model_int8w.tflite

For each model and runtime (XNNPACK on = default, off = SIGN_NO_XNNPACK=1) it starts a fresh process and prints:
memory the model adds (peak RSS minus RSS after importing tflite-runtime), ms per inference (2 threads, as
sign_speaker uses), and the top-1 sign for 20 fixed random inputs. Agreement with the first model is shown at the end.
"""
import json, os, subprocess, sys

CHILD = r'''
import json, os, sys, time
import numpy as np
from tflite_runtime.interpreter import Interpreter, OpResolverType
def kb(key):
    for line in open('/proc/self/status'):
        if line.startswith(key):
            return int(line.split()[1])
rng = np.random.RandomState(0)
xs = []
for _ in range(20):
    x = rng.randn(1, 128, 708).astype(np.float32); x[:, rng.randint(10, 128):] = -100.; xs.append(x)
before = kb('VmRSS:')
kw = dict(experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES) if sys.argv[2] == 'off' else {}
it = Interpreter(model_path=sys.argv[1], num_threads=2, **kw); it.allocate_tensors()
i, o = it.get_input_details()[0]['index'], it.get_output_details()[0]['index']
top = []
t0 = time.perf_counter()
for x in xs:
    it.set_tensor(i, x); it.invoke(); top.append(int(it.get_tensor(o)[0].argmax()))
ms = (time.perf_counter() - t0) / len(xs) * 1000
print(json.dumps(dict(mem_kb=kb('VmHWM:') - before, ms=ms, top=top)))
'''

ref = None
for path in sys.argv[1:]:
    for mode in ('on', 'off'):
        r = subprocess.run([sys.executable, '-c', CHILD, path, mode], capture_output=True, text=True)
        if r.returncode:
            print(f'{path} XNNPACK {mode}: FAILED\n{r.stderr[-800:]}')
            continue
        j = json.loads(r.stdout.strip().splitlines()[-1])
        ref = ref or j['top']
        agree = sum(a == b for a, b in zip(ref, j['top']))
        print(f'{os.path.basename(path):28s} XNNPACK {mode:3s}  file {os.path.getsize(path) / 1e6:5.2f} MB  '
              f'memory +{j["mem_kb"] / 1024:5.1f} MB  {j["ms"]:7.1f} ms/inference  same top-1 as first: {agree}/20')
