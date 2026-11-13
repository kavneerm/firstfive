#!/usr/bin/env python3
"""Memory one sign model needs at runtime, measured in a fresh process the way the board runs it (batch 1, 4 threads).

    python mem_probe.py model.tflite [plain]   -> prints one JSON line
"plain" turns off the default XNNPACK delegate (it keeps its own repacked copy of the weights).

delta_kb = peak RSS after loading + 20 inferences - RSS after importing the runtime (so the runtime itself is excluded).
Uses tflite_runtime when installed (as on the UNO Q), otherwise tf.lite.
"""
import json, os, resource, sys
import numpy as np


def rss_kb():
    try:
        for line in open('/proc/self/status'):
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    except FileNotFoundError:
        pass
    return None


def peak_kb():
    try:
        for line in open('/proc/self/status'):
            if line.startswith('VmHWM:'):
                return int(line.split()[1])
    except FileNotFoundError:
        pass
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r // 1024 if sys.platform == 'darwin' else r


try:
    from tflite_runtime.interpreter import Interpreter, OpResolverType
    runtime = 'tflite_runtime'
except ImportError:
    import tensorflow as tf
    Interpreter, OpResolverType, runtime = tf.lite.Interpreter, tf.lite.experimental.OpResolverType, 'tf.lite'

path = sys.argv[1]
before = rss_kb() or peak_kb()
plain = len(sys.argv) > 2 and sys.argv[2] == 'plain'
kw = dict(experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES) if plain else {}
it = Interpreter(model_path=path, num_threads=4, **kw)
it.allocate_tensors()
inp, out = it.get_input_details()[0], it.get_output_details()[0]
x = np.random.RandomState(0).randn(*inp['shape']).astype(inp['dtype'])
x[:, 60:] = -100.
for _ in range(20):
    it.set_tensor(inp['index'], x)
    it.invoke()
    it.get_tensor(out['index'])
peak = peak_kb()
print(json.dumps(dict(runtime=runtime + ('-plain' if plain else ''), rss_before_kb=before, rss_peak_kb=peak, delta_kb=peak - before,
                      file_bytes=os.path.getsize(path))))
