"""Memory/speed of runtime variants for already-trained candidates (no retraining):
exports: float32, int8w (8-bit weights), int8full (8-bit weights + activations, float in/out, 200-clip calibration)
runtimes: xnnpack (default) vs plain (no default delegates).
Also: top-1 agreement of each export with the Keras model on the 200 calibration clips (a quick quality signal only;
real accuracy needs the val set on the box).

    python exp/runtime_variants.py exp/<run> [...]
"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tensorflow as tf
import common as C
import model as Mdl

HERE = os.path.dirname(os.path.abspath(__file__))
calib = np.load(os.path.join(HERE, 'calib200.npy'))


def export_full_int8(student):
    fn = tf.function(lambda t: student(t, training=False),
                     input_signature=[tf.TensorSpec((1, C.MAX_LEN, C.CHANNELS), tf.float32)])
    conv = tf.lite.TFLiteConverter.from_concrete_functions([fn.get_concrete_function()], student)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([calib[i:i + 1]] for i in range(len(calib)))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    return conv.convert()


def agree(path, ref):
    it = tf.lite.Interpreter(model_path=path, num_threads=4)
    it.allocate_tensors()
    i, o = it.get_input_details()[0]['index'], it.get_output_details()[0]['index']
    top = []
    for x in calib:
        it.set_tensor(i, x[None]); it.invoke(); top.append(it.get_tensor(o)[0].argmax())
    return float((np.array(top) == ref).mean())


for run in sys.argv[1:]:
    cfg = json.load(open(os.path.join(run, 'cfg.json')))
    m = Mdl.get_model(max_len=C.MAX_LEN, num_classes=250, **cfg.get('arch', {}))
    m.load_weights(os.path.join(run, 'best.h5'))
    ref = m.predict(calib, verbose=0).argmax(-1)
    full = os.path.join(run, 'model_int8full.tflite')
    try:
        open(full, 'wb').write(export_full_int8(m))
    except Exception as e:
        print(f'{run}: int8full export FAILED: {type(e).__name__}: {str(e)[:300]}')
    for q in ('float32', 'int8w', 'int8full'):
        p = os.path.join(run, f'model_{q}.tflite')
        if not os.path.exists(p):
            continue
        ag = agree(p, ref)
        for mode in ('xnnpack', 'plain'):
            r = json.loads(subprocess.run([sys.executable, os.path.join(HERE, 'probe2.py'), p, mode],
                                          capture_output=True, text=True).stdout.strip().splitlines()[-1])
            print(f"{os.path.basename(run):10s} {q:9s} {mode:8s} file {r['file_bytes'] / 1e6:5.2f} MB  "
                  f"mem {r['delta_kb'] / 1024:5.1f} MB  {r['ms']:6.1f} ms  agree {ag:.3f}", flush=True)
