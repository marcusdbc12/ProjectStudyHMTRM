#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_csv(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).to_numpy(dtype=np.float64)


def safe_log10(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return np.log10(np.maximum(np.abs(x), eps))


def row_normalize(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    denom = np.max(np.abs(x), axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def depth_compensate(x: np.ndarray, start_row: int = 3, eps: float = 1e-30) -> np.ndarray:
    out = x.copy()
    profile = np.mean(np.abs(out), axis=1, keepdims=True)
    profile = np.maximum(profile, eps)
    out[start_row:, :] = out[start_row:, :] / profile[start_row:, :]
    return out


def stats_dict(x: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "nonzero_frac": float(np.count_nonzero(x) / x.size),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Debug TRM CSV outputs.")
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--sample-id", type=str, default="sample_0000")
    p.add_argument("--surface-mask-depth", type=int, default=3,
                   help="Rows near the top to ignore in some diagnostics.")
    args = p.parse_args()

    root = args.project_root
    input_dir = root / "input"
    out_dir = root / "artifacts" / "debug_trm"
    out_dir.mkdir(parents=True, exist_ok=True)

    sid = args.sample_id

    refocus = load_csv(input_dir / f"{sid}_refocus.csv")
    trm_energy = load_csv(input_dir / f"{sid}_trm_energy.csv")
    target = load_csv(input_dir / f"{sid}_target.csv")
    sound_speed = load_csv(input_dir / f"{sid}_sound_speed.csv")
    traces = pd.read_csv(input_dir / f"{sid}_traces.csv")

    # Diagnostics
    refocus_log = safe_log10(refocus)
    energy_log = safe_log10(trm_energy)

    refocus_rownorm = row_normalize(refocus)
    energy_rownorm = row_normalize(trm_energy)

    refocus_depthcomp = depth_compensate(refocus, start_row=args.surface_mask_depth)
    energy_depthcomp = depth_compensate(trm_energy, start_row=args.surface_mask_depth)

    # Ignore masked top rows for some stats
    refocus_below = refocus[args.surface_mask_depth:, :]
    energy_below = trm_energy[args.surface_mask_depth:, :]

    stats = {
        "sample_id": sid,
        "refocus_stats": stats_dict(refocus),
        "trm_energy_stats": stats_dict(trm_energy),
        "refocus_below_surface_stats": stats_dict(refocus_below),
        "trm_energy_below_surface_stats": stats_dict(energy_below),
        "target_sum": float(np.sum(target)),
        "trace_columns": list(traces.columns),
        "trace_shape": list(traces.shape),
    }

    # Save stats text
    stats_path = out_dir / f"{sid}_stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

    # Multi-panel debug figure
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    axes[0, 0].imshow(sound_speed, aspect="auto", cmap="viridis")
    axes[0, 0].set_title("Sound speed")

    axes[0, 1].imshow(target, aspect="auto", cmap="magma")
    axes[0, 1].set_title("Target")

    axes[0, 2].imshow(refocus, aspect="auto", cmap="inferno")
    axes[0, 2].set_title("TRM max raw")

    axes[1, 0].imshow(refocus_log, aspect="auto", cmap="inferno")
    axes[1, 0].set_title("TRM max log10")

    axes[1, 1].imshow(refocus_rownorm, aspect="auto", cmap="RdBu_r")
    axes[1, 1].set_title("TRM max row-normalized")

    axes[1, 2].imshow(refocus_depthcomp, aspect="auto", cmap="RdBu_r")
    axes[1, 2].set_title("TRM max depth-compensated")

    axes[2, 0].imshow(energy_log, aspect="auto", cmap="inferno")
    axes[2, 0].set_title("TRM energy log10")

    axes[2, 1].imshow(energy_rownorm, aspect="auto", cmap="RdBu_r")
    axes[2, 1].set_title("TRM energy row-normalized")

    axes[2, 2].imshow(energy_depthcomp, aspect="auto", cmap="RdBu_r")
    axes[2, 2].set_title("TRM energy depth-compensated")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_dir / f"{sid}_debug_panels.png", dpi=160)
    plt.close(fig)

    # Trace figure
    plt.figure(figsize=(10, 4))
    first_col = traces.columns[0]
    plt.plot(traces[first_col].to_numpy())
    plt.title(f"{sid} first trace: {first_col}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{sid}_first_trace.png", dpi=160)
    plt.close()

    # Surface-vs-depth profile figure
    plt.figure(figsize=(10, 4))
    plt.plot(np.mean(np.abs(refocus), axis=1), label="mean |TRM max| by row")
    plt.plot(np.mean(np.abs(trm_energy), axis=1), label="mean |TRM energy| by row")
    plt.axvline(args.surface_mask_depth, linestyle="--")
    plt.legend()
    plt.title(f"{sid} depth profiles")
    plt.tight_layout()
    plt.savefig(out_dir / f"{sid}_depth_profiles.png", dpi=160)
    plt.close()

    print(f"Saved debug outputs to: {out_dir}")
    print(f"Stats file: {stats_path}")
    print("Important note:")
    print("- With only refocus.csv and trm_energy.csv, we can debug contrast and depth bias.")
    print("- We cannot compute late-time-only TRM from CSVs alone.")
    print("- For that, the generator would need to save the full trm_field_np time history or selected time slices.")


if __name__ == "__main__":
    main()
