#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser(description='Split metadata into train/validation/test')
    p.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--train-ratio', type=float, default=0.70)
    p.add_argument('--validation-ratio', type=float, default=0.15)
    p.add_argument('--test-ratio', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--copy-test-files', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    root = args.project_root
    log_path = root/'artifacts'/'logs'/'split.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler()])
    total = args.train_ratio + args.validation_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError('Ratios must sum to 1.0')
    meta = pd.read_csv(root/'input'/'metadata.csv')
    train_df, temp_df = train_test_split(meta, test_size=(1.0-args.train_ratio), random_state=args.seed, shuffle=True)
    val_fraction_of_temp = args.validation_ratio / (args.validation_ratio + args.test_ratio)
    validation_df, test_df = train_test_split(temp_df, test_size=(1.0-val_fraction_of_temp), random_state=args.seed, shuffle=True)
    out = root/'artifacts'
    train_df.to_csv(out/'train_split.csv', index=False)
    validation_df.to_csv(out/'validation_split.csv', index=False)
    test_df.to_csv(out/'test_split.csv', index=False)
    logging.info('Split complete: train=%d validation=%d test=%d', len(train_df), len(validation_df), len(test_df))
    if args.copy_test_files:
        test_dir, val_dir = root/'test', root/'validation'
        for d in [test_dir, val_dir]:
            d.mkdir(exist_ok=True)
            for f in d.glob('*'): 
                if f.is_file(): f.unlink()
        for df, folder in [(test_df, test_dir), (validation_df, val_dir)]:
            for sid in df['sample_id']:
                for suffix in ['traces','target','sound_speed','density','attenuation','refocus']:
                    src = root/'input'/f'{sid}_{suffix}.csv'
                    if src.exists(): shutil.copy2(src, folder/src.name)
        logging.info('Validation and test folders refreshed')

if __name__ == '__main__':
    main()
