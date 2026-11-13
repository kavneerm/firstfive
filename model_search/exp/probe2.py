"""Like mem_probe.py, plus a runtime switch: python probe2.py model.tflite [xnnpack|plain]
Prints JSON: delta_kb (peak RSS after load + 20 runs minus RSS before load) and ms per inference (4 threads)."""
import json, os, resource, sys, time
import numpy as np


def peak_kb():
    try:
        for line in open('/proc/self/status'):
            if line.startswith('VmHWM:'):
                return int(line.split()[1])
    except FileNotFoundError:
        pass
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r // 1024 if sys.platform == 'darwin' else r


def rss_kb():
    try:
        for line in open('/proc/self/status'):
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    except FileNotFoundError:
        return None


try:
    from tflite_runtime.interpreter import Interpreter, OpResolverType
except ImportError:
    import tensorflow as tf
    Interpreter, OpResolverType = tf.lite.Interpreter, tf.lite.experimental.OpResolverType

path, mode = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'xnnpack')
x = np.random.RandomState(0).randn(1, 128, 708).astype(np.float32)
x[:, 60:] = -100.
before = rss_kb() or peak_kb()
kw = {} if mode == 'xnnpack' else dict(experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
it = Interpreter(model_path=path, num_threads=4, **kw)
it.allocate_tensors()
inp, out = it.get_input_details()[0], it.get_output_details()[0]
xi = x.astype(inp['dtype'])
for _ in range(3):
    it.set_tensor(inp['index'], xi); it.invoke()
t0 = time.perf_counter()
for _ in range(20):
    it.set_tensor(inp['index'], xi); it.invoke(); it.get_tensor(out['index'])
ms = (time.perf_counter() - t0) / 20 * 1000
print(json.dumps(dict(mode=mode, delta_kb=peak_kb() - before, ms=round(ms, 2), file_bytes=os.path.getsize(path))))
