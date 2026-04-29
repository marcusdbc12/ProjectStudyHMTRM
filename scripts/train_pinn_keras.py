#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


def parse_args():
    p = argparse.ArgumentParser(description='Train Keras PINN-style inverse model')
    p.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=1e-3)
    p.add_argument('--physics-weight', type=float, default=0.15)
    p.add_argument('--save-name', type=str, default='best_pinn.keras')
    return p.parse_args()


def load_sample(root: Path, sid: str):
    target = pd.read_csv(root/'input'/f'{sid}_target.csv', header=None).values.astype('float32')
    refocus = pd.read_csv(root/'input'/f'{sid}_refocus.csv', header=None).values.astype('float32')
    traces = pd.read_csv(root/'input'/f'{sid}_traces.csv').values.astype('float32')
    n = target.shape[0]
    traces_img = np.zeros((n, n), dtype='float32')
    flat = traces.mean(axis=0)
    traces_img.flat[:min(traces_img.size, flat.size)] = flat[:min(traces_img.size, flat.size)]
    x = np.stack([traces_img, refocus], axis=-1)
    return x, target[...,None]


def load_split(root: Path, split_name: str):
    ids = pd.read_csv(root/'artifacts'/split_name)['sample_id'].tolist()
    xs, ys = zip(*(load_sample(root, sid) for sid in ids))
    return np.stack(xs), np.stack(ys)


def laplacian_loss(y_pred: tf.Tensor) -> tf.Tensor:
    k = tf.constant([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]], dtype=tf.float32)
    k = tf.reshape(k, [3,3,1,1])
    lap = tf.nn.conv2d(y_pred, k, strides=1, padding='SAME')
    return tf.reduce_mean(tf.square(lap))


class PinnModel(keras.Model):
    def __init__(self, physics_weight: float):
        super().__init__()
        self.physics_weight = physics_weight
        self.net = keras.Sequential([
            keras.layers.Input(shape=(96,96,2)),
            keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            keras.layers.UpSampling2D(),
            keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            keras.layers.UpSampling2D(),
            keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid'),
        ])
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.bce_tracker = keras.metrics.Mean(name='bce')
        self.physics_tracker = keras.metrics.Mean(name='physics')

    @property
    def metrics(self):
        return [self.loss_tracker, self.bce_tracker, self.physics_tracker]

    def call(self, inputs, training=False):
        return self.net(inputs, training=training)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            pred = self(x, training=True)
            bce = tf.reduce_mean(keras.losses.binary_crossentropy(y, pred))
            phys = laplacian_loss(pred)
            loss = bce + self.physics_weight * phys
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.bce_tracker.update_state(bce)
        self.physics_tracker.update_state(phys)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        pred = self(x, training=False)
        bce = tf.reduce_mean(keras.losses.binary_crossentropy(y, pred))
        phys = laplacian_loss(pred)
        loss = bce + self.physics_weight * phys
        self.loss_tracker.update_state(loss)
        self.bce_tracker.update_state(bce)
        self.physics_tracker.update_state(phys)
        return {m.name: m.result() for m in self.metrics}


def main():
    args = parse_args()
    root = args.project_root
    log_path = root/'artifacts'/'logs'/'train_pinn.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler()])
    x_train, y_train = load_split(root, 'train_split.csv')
    x_val, y_val = load_split(root, 'validation_split.csv')
    model = PinnModel(args.physics_weight)
    model.compile(optimizer=keras.optimizers.Adam(args.learning_rate))
    hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs, batch_size=args.batch_size, verbose=2)
    model.net.save(root/'artifacts'/args.save_name)
    (root/'artifacts'/'training_config_pinn.json').write_text(json.dumps(vars(args), indent=2, default=str))
    pd.DataFrame(hist.history).to_csv(root/'artifacts'/'training_history_pinn.csv', index=False)
    for key in ['loss', 'val_loss', 'bce', 'val_bce', 'physics', 'val_physics']:
        if key in hist.history:
            plt.figure(); plt.plot(hist.history[key]); plt.title(key); plt.tight_layout(); plt.savefig(root/'artifacts'/f'{key}.png', dpi=140); plt.close()
    logging.info('Saved PINN model to %s', root/'artifacts'/args.save_name)

if __name__ == '__main__':
    main()
