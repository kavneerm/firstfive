#!/usr/bin/env python3
"""Train one candidate. Config is a JSON object (file path or inline), e.g.
    {"id": "d96-s2c3", "arch": {"dim": 96}, "epochs": 10, "distill": {"alpha": 0.5, "T": 2.0}}

    python train.py --cfg cfg.json --data data/islr --split data/islr/split.json --out runs/<id>

Writes runs/<id>/: cfg.json, best.h5 (best val top-1), last.h5, log.csv, summary.json.
Keys (defaults): arch {} (model.BASELINE overrides), epochs 10, batch 64, lr 5e-4, wd 0.05, warmup 0.1,
label_smoothing 0.1, distill null | {alpha, T, teacher (baseline.h5)}, seed 42, max_train (all).
"""
import argparse, json, math, os, sys, time
import numpy as np
import tensorflow as tf
import common as C
import model as Mdl

DEFAULTS = dict(arch={}, epochs=10, batch=64, lr=5e-4, wd=0.05, warmup=0.1, label_smoothing=0.1,
                distill=None, seed=42, max_train=None)


def load_cfg(s):
    cfg = json.load(open(s)) if os.path.exists(s) else json.loads(s)
    bad = set(cfg) - set(DEFAULTS) - {'id', 'note'}
    if bad:
        raise ValueError(f'unknown config keys {sorted(bad)}')
    cfg = {**DEFAULTS, **cfg}
    Mdl.check_arch({**Mdl.BASELINE, **cfg['arch']})
    if not 1 <= cfg['epochs'] <= 120:
        raise ValueError('epochs must be 1..120')
    return cfg


class OneCycle(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, total, warmup):
        self.lr, self.total, self.warm = lr, float(total), max(1., float(total) * warmup)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        up = self.lr * step / self.warm
        prog = tf.clip_by_value((step - self.warm) / tf.maximum(1., self.total - self.warm), 0., 1.)
        down = 1e-6 + 0.5 * (self.lr - 1e-6) * (1 + tf.cos(math.pi * prog))
        return tf.where(step < self.warm, up, down)


class KD(tf.keras.Model):
    """Student trained on smoothed CE plus KL to a frozen teacher's softened logits."""

    def __init__(self, student, teacher, alpha, T, smoothing):
        super().__init__()
        self.student, self.teacher, self.alpha, self.T = student, teacher, alpha, T
        self.ce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=smoothing)
        self.kl = tf.keras.losses.KLDivergence()
        self.acc = tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')
        self.loss_m = tf.keras.metrics.Mean(name='loss')

    @property
    def metrics(self):
        return [self.loss_m, self.acc]

    def call(self, x, training=False):
        return self.student(x, training=training)

    def train_step(self, data):
        x, y = data
        t = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            s = self.student(x, training=True)
            kd = self.kl(tf.nn.softmax(t / self.T), tf.nn.softmax(s / self.T)) * self.T ** 2
            loss = (1 - self.alpha) * self.ce(y, s) + self.alpha * kd
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.student.trainable_variables),
                                           self.student.trainable_variables))
        self.loss_m.update_state(loss); self.acc.update_state(y, s)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        s = self.student(x, training=False)
        self.loss_m.update_state(self.ce(y, s)); self.acc.update_state(y, s)
        return {m.name: m.result() for m in self.metrics}


class SaveBest(tf.keras.callbacks.Callback):
    def __init__(self, student, out):
        super().__init__()
        self.student, self.out, self.best, self.best_epoch, self.times = student, out, -1., -1, []

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.t0)
        acc = logs['val_categorical_accuracy']
        if acc > self.best:
            self.best, self.best_epoch = acc, epoch + 1
            self.student.save_weights(os.path.join(self.out, 'best.h5'))
        self.student.save_weights(os.path.join(self.out, 'last.h5'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--split', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    os.makedirs(a.out, exist_ok=True)
    json.dump(cfg, open(os.path.join(a.out, 'cfg.json'), 'w'), indent=1)
    tf.keras.utils.set_random_seed(cfg['seed'])
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    print('GPUs:', tf.config.list_physical_devices('GPU'), flush=True)

    D = C.Data(a.data)
    tr, va = D.split(a.split)
    if cfg['max_train']:
        tr = tr.sample(n=min(cfg['max_train'], len(tr)), random_state=cfg['seed'])
    ncls = len(D.signmap)
    print(f'train {len(tr)}  val {len(va)}  classes {ncls}  cfg {json.dumps(cfg)}', flush=True)
    ds_tr = D.dataset(tr, cfg['batch'], True, shuffle_seed=cfg['seed'])
    ds_va = D.dataset(va, 256, False)
    steps = len(tr) // cfg['batch']
    total = steps * cfg['epochs']

    student = Mdl.get_model(max_len=C.MAX_LEN, num_classes=ncls, dropout_step=int(total * 0.25), **cfg['arch'])
    opt = tf.keras.optimizers.AdamW(learning_rate=OneCycle(cfg['lr'], total, cfg['warmup']), weight_decay=cfg['wd'])
    if cfg['distill']:
        d = cfg['distill']
        teacher = Mdl.get_model(max_len=C.MAX_LEN, num_classes=ncls)
        teacher.load_weights(d.get('teacher', 'baseline.h5'))
        teacher.trainable = False
        net = KD(student, teacher, d.get('alpha', 0.5), d.get('T', 2.0), cfg['label_smoothing'])
        net.compile(optimizer=opt)
    else:
        net = student
        net.compile(optimizer=opt, metrics=['categorical_accuracy'],
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=cfg['label_smoothing']))
    print(f'params {student.count_params():,}  steps/epoch {steps}', flush=True)

    sb = SaveBest(student, a.out)
    net.fit(ds_tr, epochs=cfg['epochs'], validation_data=ds_va, verbose=2,
            callbacks=[sb, tf.keras.callbacks.CSVLogger(os.path.join(a.out, 'log.csv'))])
    summ = dict(id=cfg.get('id'), params=int(student.count_params()), best_val_top1_keras=float(sb.best),
                best_epoch=sb.best_epoch, epochs=cfg['epochs'], sec_per_epoch=float(np.median(sb.times)),
                n_train=len(tr), n_val=len(va))
    json.dump(summ, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    print('SUMMARY', json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
