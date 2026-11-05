"""
TensorFlow-free sign inference: a numpy port of islr_model.Preprocess / prepare_input plus a TFLite predictor with
the same interface as demo_webcam.Predictor. Needs only numpy and tflite-runtime (falls back to tf.lite when
tflite-runtime is not installed, so the comparison checks can run in the full TF environment).
"""
import json
import numpy as np

# same landmark selection as islr_model.py (copied so this module never imports TensorFlow)
NOSE = [1, 2, 98, 327]
LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
       78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE
CHANNELS = 6 * len(POINT_LANDMARKS)
PAD = -100.


def _nan_mean(x, axis):
    """tf_nan_mean: sum of non-NaN / count of non-NaN, NaN when there are none (0/0), kept dims."""
    ok = ~np.isnan(x)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(ok, x, 0).sum(axis=axis, keepdims=True) / ok.sum(axis=axis, keepdims=True)


def preprocess(coords, max_len):
    """numpy version of islr_model.Preprocess for one clip (T, 543, 3) -> (T', CHANNELS), T' = min(T, max_len)."""
    x = np.asarray(coords, np.float32)[None]
    mean = _nan_mean(x[:, :, [17]], axis=(1, 2))
    mean = np.where(np.isnan(mean), np.float32(0.5), mean)
    x = x[:, :, POINT_LANDMARKS]
    with np.errstate(invalid='ignore', divide='ignore'):
        std = np.sqrt(_nan_mean((x - mean) ** 2, axis=(1, 2)))
        x = (x - mean) / std
    x = x[:, :max_len, :, :2]
    n, p = x.shape[1], x.shape[2]
    dx = np.zeros_like(x); dx2 = np.zeros_like(x)
    if n > 1:
        dx[:, :-1] = x[:, 1:] - x[:, :-1]
    if n > 2:
        dx2[:, :-2] = x[:, 2:] - x[:, :-2]
    out = np.concatenate([a.reshape(1, n, 2 * p) for a in (x, dx, dx2)], axis=-1)
    return np.where(np.isnan(out), np.float32(0), out).astype(np.float32)[0]


def prepare_input(coords, max_len):
    """Same contract as islr_model.prepare_input: drop frames with no selected landmarks, preprocess, pad to max_len
    with PAD. Returns (1, max_len, CHANNELS) float32 or None."""
    coords = np.asarray(coords, np.float32)
    keep = ~np.all(np.isnan(coords[:, POINT_LANDMARKS]), axis=(1, 2))
    coords = coords[keep]
    if len(coords) == 0:
        return None
    feats = preprocess(coords, max_len)
    out = np.full((max_len, CHANNELS), PAD, np.float32)
    out[:len(feats)] = feats
    return out[None]


def make_interpreter(path, num_threads=1):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=path, num_threads=num_threads)
    it.allocate_tensors()
    return it


class LitePredictor:
    """Drop-in for demo_webcam.Predictor running sign_model.tflite."""

    def __init__(self, model_path, signmap, max_len=128, num_threads=1):
        self.max_len = max_len
        self.names = {int(i): n for n, i in json.load(open(signmap)).items()}
        self.it = make_interpreter(model_path, num_threads)
        self.inp = self.it.get_input_details()[0]['index']
        self.out = self.it.get_output_details()[0]['index']

    def logits(self, x):
        self.it.set_tensor(self.inp, x)
        self.it.invoke()
        return self.it.get_tensor(self.out)[0]

    def predict(self, clip, topk=5):
        x = prepare_input(clip, self.max_len)
        if x is None:
            return None
        logits = self.logits(x)
        p = np.exp(logits - logits.max()); p /= p.sum()
        order = np.argsort(-p)[:topk]
        return [(self.names.get(int(i), f'class_{i}'), float(p[i])) for i in order]
