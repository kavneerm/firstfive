"""Configurable version of islr_model.get_model. The defaults rebuild the baseline exactly (same layers in the same
order, so the baseline .h5 loads into it)."""
import tensorflow as tf
import islr_model as M

BASELINE = dict(dim=192, stages=2, convs=3, ksize=17, expand=2, heads=4, conv_expand=2, top_mult=2)

LIMITS = dict(dim=(32, 256), stages=(1, 3), convs=(1, 4), ksize=(3, 31), expand=(1, 4), heads=(1, 8),
              conv_expand=(1, 4), top_mult=(1, 2))


def check_arch(a):
    for k, (lo, hi) in LIMITS.items():
        v = a.get(k, BASELINE[k])
        if not isinstance(v, int) or not lo <= v <= hi:
            raise ValueError(f'{k}={v!r} outside [{lo}, {hi}]')
    if a.get('dim', 192) % a.get('heads', 4):
        raise ValueError('dim must be divisible by heads')
    extra = set(a) - set(BASELINE)
    if extra:
        raise ValueError(f'unknown arch keys {sorted(extra)}')


def get_model(max_len=128, num_classes=250, dropout_step=0, **arch):
    a = {**BASELINE, **arch}
    check_arch(a)
    dim = a['dim']
    inp = tf.keras.Input((max_len, M.CHANNELS))
    x = tf.keras.layers.Masking(mask_value=M.PAD, input_shape=(max_len, M.CHANNELS))(inp)
    x = tf.keras.layers.Dense(dim, use_bias=False, name='stem_conv')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95, name='stem_bn')(x)
    for _ in range(a['stages']):
        for _ in range(a['convs']):
            x = M.Conv1DBlock(dim, a['ksize'], drop_rate=0.2, expand_ratio=a['conv_expand'])(x)
        x = M.TransformerBlock(dim, num_heads=a['heads'], expand=a['expand'])(x)
    x = tf.keras.layers.Dense(dim * a['top_mult'], activation=None, name='top_conv')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = M.LateDropout(0.8, start_step=dropout_step)(x)
    x = tf.keras.layers.Dense(num_classes, name='classifier')(x)
    return tf.keras.Model(inp, x)
