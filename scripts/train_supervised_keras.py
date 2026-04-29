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
    p = argparse.ArgumentParser(description='Train supervised Keras inverse model')
    p.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=1e-3)
    p.add_argument('--channels', choices=['traces', 'traces_refocus'], default='traces_refocus')
    p.add_argument('--save-name', type=str, default='best_supervised.keras')
    return p.parse_args()


def load_one(root: Path, sample_id: str, channels: str):
    traces = pd.read_csv(root/'input'/f'{sample_id}_traces.csv').values.astype('float32')
    target = pd.read_csv(root/'input'/f'{sample_id}_target.csv', header=None).values.astype('float32')
    n = target.shape[0]
    if traces.shape[1] < n*n:
        tiled = np.zeros((n*n, 1), dtype='float32')
        flat = traces.mean(axis=0)
        tiled[:len(flat),0] = flat[:min(len(flat), n*n)]
        traces_img = tiled.reshape(n, n, 1)
    else:
        traces_img = traces.mean(axis=0)[:n*n].reshape(n, n, 1)
    if channels == 'traces_refocus':
        refocus = pd.read_csv(root/'input'/f'{sample_id}_refocus.csv', header=None).values.astype('float32')[...,None]
        x = np.concatenate([traces_img, refocus], axis=-1)
    else:
        x = traces_img
    return x, target[...,None]


def load_split(root: Path, split_file: str, channels: str):
    df = pd.read_csv(root/'artifacts'/split_file)
    xs, ys = [], []
    for sid in df['sample_id']:
        x, y = load_one(root, sid, channels)
        xs.append(x); ys.append(y)
    return np.stack(xs), np.stack(ys)


def build_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = keras.layers.UpSampling2D()(x)
    x = keras.layers.Conv2D(16, 3, padding='same', activation='relu')(x)
    outputs = keras.layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x)
    return keras.Model(inputs, outputs)


def main():
    args = parse_args()
    root = args.project_root
    log_path = root/'artifacts'/'logs'/'train_supervised.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler()])
    x_train, y_train = load_split(root, 'train_split.csv', args.channels)
    x_val, y_val = load_split(root, 'validation_split.csv', args.channels)
    model = build_model(x_train.shape[1:])
    model.compile(optimizer=keras.optimizers.Adam(args.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    callbacks = [keras.callbacks.ModelCheckpoint(root/'artifacts'/args.save_name, save_best_only=True, monitor='val_loss'), keras.callbacks.CSVLogger(root/'artifacts'/'training_history_supervised.csv')]
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=2)
    (root/'artifacts'/'training_config_supervised.json').write_text(json.dumps(vars(args), indent=2, default=str))
    plt.figure(); plt.plot(history.history['loss'], label='train'); plt.plot(history.history['val_loss'], label='val'); plt.legend(); plt.title('Supervised loss'); plt.tight_layout(); plt.savefig(root/'artifacts'/'loss_supervised.png', dpi=140); plt.close()
    plt.figure(); plt.plot(history.history['accuracy'], label='train'); plt.plot(history.history['val_accuracy'], label='val'); plt.legend(); plt.title('Supervised accuracy'); plt.tight_layout(); plt.savefig(root/'artifacts'/'accuracy_supervised.png', dpi=140); plt.close()
    logging.info('Saved model to %s', root/'artifacts'/args.save_name)

if __name__ == '__main__':
    main()
