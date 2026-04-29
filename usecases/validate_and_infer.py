#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def load_input(root: Path, sample_id: str):
    target = pd.read_csv(root/'input'/f'{sample_id}_target.csv', header=None).values.astype('float32')
    refocus = pd.read_csv(root/'input'/f'{sample_id}_refocus.csv', header=None).values.astype('float32')
    traces = pd.read_csv(root/'input'/f'{sample_id}_traces.csv').values.astype('float32')
    n = target.shape[0]
    traces_img = np.zeros((n, n), dtype='float32')
    flat = traces.mean(axis=0)
    traces_img.flat[:min(flat.size, traces_img.size)] = flat[:min(flat.size, traces_img.size)]
    x = np.stack([traces_img, refocus], axis=-1)
    return x, target


def main():
    p = argparse.ArgumentParser(description='Run inference on one sample')
    p.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--sample-id', type=str, default='sample_0000')
    p.add_argument('--model-path', type=Path, default=None)
    args = p.parse_args()
    root = args.project_root
    model_path = args.model_path or (root/'artifacts'/'best_supervised.keras')
    model = tf.keras.models.load_model(model_path, compile=False)
    x, target = load_input(root, args.sample_id)
    pred = model.predict(x[None,...], verbose=0)[0,...,0]
    plt.figure(figsize=(10,3.6))
    plt.subplot(1,3,1); plt.imshow(x[...,1], cmap='inferno'); plt.title('Refocus')
    plt.subplot(1,3,2); plt.imshow(target, cmap='magma'); plt.title('Target')
    plt.subplot(1,3,3); plt.imshow(pred, cmap='viridis'); plt.title('Prediction')
    plt.tight_layout();
    out = root/'artifacts'/f'{args.sample_id}_prediction.png'
    plt.savefig(out, dpi=140)
    print(f'Saved prediction figure to {out}')

if __name__ == '__main__':
    main()
