"""Shared data code: the 118-landmark subset we store, preprocessing that matches islr_model.Preprocess exactly,
training augmentation, and tf.data pipelines."""
import json, os
import numpy as np
import pandas as pd
import tensorflow as tf
import islr_model as M

MAX_LEN = 128
PAD = M.PAD
POINTS = M.POINT_LANDMARKS            # 543-layout indices we keep, in model order
NP = len(POINTS)                      # 118
CHANNELS = M.CHANNELS                 # 708
MEAN_IDX = POINTS.index(17)           # Preprocess centres on landmark 17 (a lip point)


def _flip_perm():
    """Index permutation over the 118 stored points that swaps left/right parts (used with x -> 1 - x)."""
    pairs = list(zip(M.LHAND, M.RHAND)) + list(zip(M.LLIP, M.RLIP)) + list(zip(M.LEYE, M.REYE)) + \
        list(zip(M.LNOSE, M.RNOSE))
    perm = list(range(NP))
    for a, b in pairs:
        if a in POINTS and b in POINTS:
            ia, ib = POINTS.index(a), POINTS.index(b)
            perm[ia], perm[ib] = ib, ia
    return perm


FLIP_PERM = _flip_perm()


def drop_empty_frames(x):
    """Same as prepare_input: drop frames where every kept landmark is NaN."""
    keep = tf.logical_not(tf.reduce_all(tf.math.is_nan(x), axis=[1, 2]))
    return tf.boolean_mask(x, keep)


def preprocess(x, max_len=MAX_LEN):
    """(T, 118, 3) -> (min(T, max_len), 708). Mirrors islr_model.Preprocess on the 543-point input."""
    x = x[None]
    mean = M.tf_nan_mean(x[:, :, MEAN_IDX:MEAN_IDX + 1], axis=[1, 2], keepdims=True)
    mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
    std = M.tf_nan_std(x, center=mean, axis=[1, 2], keepdims=True)
    x = (x - mean) / std
    x = x[:, :max_len]
    length = tf.shape(x)[1]
    x = x[..., :2]
    dx = tf.cond(length > 1, lambda: tf.pad(x[:, 1:] - x[:, :-1], [[0, 0], [0, 1], [0, 0], [0, 0]]), lambda: tf.zeros_like(x))
    dx2 = tf.cond(length > 2, lambda: tf.pad(x[:, 2:] - x[:, :-2], [[0, 0], [0, 2], [0, 0], [0, 0]]), lambda: tf.zeros_like(x))
    x = tf.concat([tf.reshape(t, (-1, length, 2 * NP)) for t in (x, dx, dx2)], axis=-1)
    x = tf.where(tf.math.is_nan(x), tf.constant(0., x.dtype), x)
    return x[0]


def augment(x, max_len=MAX_LEN):
    """x: (T, 118, 3) raw MediaPipe coords with NaN. Random speed, crop, mirror, affine, time mask."""
    T = tf.shape(x)[0]
    # speed: resample to T * U(0.8, 1.2) frames
    if tf.random.uniform(()) < 0.8:
        newT = tf.maximum(1, tf.cast(tf.round(tf.cast(T, tf.float32) * tf.random.uniform((), 0.8, 1.2)), tf.int32))
        flat = tf.reshape(x, (T, 1, NP * 3))
        x = tf.reshape(tf.image.resize(flat, (newT, 1)), (newT, NP, 3))
        T = newT
    # random temporal crop when longer than max_len (eval keeps the first max_len frames)
    if T > max_len:
        s = tf.random.uniform((), 0, T - max_len + 1, dtype=tf.int32)
        x = x[s:s + max_len]
        T = tf.constant(max_len)
    # mirror
    if tf.random.uniform(()) < 0.5:
        x = tf.gather(x, FLIP_PERM, axis=1)
        x = tf.concat([1. - x[..., :1], x[..., 1:]], axis=-1)
    # affine on xy around the image centre
    if tf.random.uniform(()) < 0.75:
        xy, z = x[..., :2] - 0.5, x[..., 2:]
        sc = tf.random.uniform((), 0.8, 1.2)
        sh = tf.random.uniform((2,), -0.15, 0.15)
        th = tf.random.uniform((), -np.pi / 6, np.pi / 6)
        c, s_ = tf.cos(th), tf.sin(th)
        A = tf.stack([[c, -s_], [s_, c]]) @ tf.stack([[1., sh[0]], [sh[1], 1.]]) * sc
        xy = tf.einsum('tpi,ji->tpj', xy, A) + 0.5 + tf.random.uniform((1, 1, 2), -0.1, 0.1)
        x = tf.concat([xy, z * sc], axis=-1)
    # time mask: blank a contiguous block of up to 40% of frames
    if tf.random.uniform(()) < 0.25 and T > 4:
        w = tf.random.uniform((), 1, tf.maximum(2, tf.cast(tf.cast(T, tf.float32) * 0.4, tf.int32)), dtype=tf.int32)
        s = tf.random.uniform((), 0, T - w + 1, dtype=tf.int32)
        m = tf.logical_and(tf.range(T) >= s, tf.range(T) < s + w)[:, None, None]
        x = tf.where(m, tf.constant(np.nan, x.dtype), x)
    return x


class Data:
    """Loads prepare_kaggle.py output: landmarks.npy (frames, 118, 3) float32, meta.csv (start, length, label, participant)."""

    def __init__(self, root):
        self.root = root
        self.meta = pd.read_csv(os.path.join(root, 'meta.csv'))
        self.lm = np.load(os.path.join(root, 'landmarks.npy'), mmap_mode='r')
        self.signmap = json.load(open(os.path.join(root, 'sign_to_prediction_index_map.json')))

    def split(self, split_file):
        val_p = set(json.load(open(split_file))['val_participants'])
        is_val = self.meta.participant_id.isin(val_p).values
        return self.meta[~is_val], self.meta[is_val]

    def ragged(self, rows):
        starts, lens = rows.start.values, rows.length.values
        idx = np.concatenate([np.arange(s, s + n) for s, n in zip(starts, lens)])
        vals = np.asarray(self.lm[idx], np.float32)
        return tf.RaggedTensor.from_row_lengths(vals, lens)

    def dataset(self, rows, batch, train, max_len=MAX_LEN, shuffle_seed=None):
        with tf.device('/CPU:0'):        # the landmark array is ~5 GB; on the GPU it leaves no room for the model
            rt = self.ragged(rows)
            y = tf.one_hot(rows.label.values, len(self.signmap))
        ds = tf.data.Dataset.from_tensor_slices((rt, y))
        if train:
            ds = ds.shuffle(len(rows), seed=shuffle_seed, reshuffle_each_iteration=True)

        def f(x, y):
            x = drop_empty_frames(x)
            if train:
                x = augment(x, max_len)
            return preprocess(x, max_len), y

        ds = ds.map(f, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not train)
        ds = ds.padded_batch(batch, padded_shapes=([max_len, CHANNELS], [len(self.signmap)]),
                             padding_values=(tf.constant(PAD, tf.float32), 0.), drop_remainder=train)
        return ds.prefetch(tf.data.AUTOTUNE)
