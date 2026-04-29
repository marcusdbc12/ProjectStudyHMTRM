#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_traces_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path)
    cols = list(df.columns)
    arr = df.to_numpy(dtype=np.float64)
    return arr, cols


def parse_receiver_indices(columns: list[str]) -> list[int]:
    rx = []
    for c in columns:
        try:
            rx.append(int(c.split("_rx_")[1]))
        except Exception:
            rx.append(len(rx))
    return rx


def estimate_arrival_index(trace: np.ndarray) -> int:
    env = np.abs(trace)
    if np.max(env) <= 0:
        return 0
    threshold = 0.15 * np.max(env)
    idx = np.argmax(env >= threshold)
    return int(idx)


def main() -> None:
    p = argparse.ArgumentParser(description="Create professor-friendly plots for the trace stack.")
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--sample-id", type=str, default="sample_0000")
    p.add_argument("--t-end", type=float, default=1.2e-7)
    p.add_argument("--source-center-frac", type=float, default=0.28)
    p.add_argument("--old-gate-frac", type=float, default=0.10)
    p.add_argument("--frequency", type=float, default=25e6)
    p.add_argument("--margin-time-ns", type=float, default=10.0)
    args = p.parse_args()

    root = args.project_root
    input_dir = root / "input"
    out_dir = root / "artifacts" / "trace_explanation"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_id = args.sample_id
    traces, columns = load_traces_csv(input_dir / f"{sample_id}_traces.csv")
    receiver_indices = parse_receiver_indices(columns)

    nt, nr = traces.shape
    t = np.linspace(0.0, args.t_end, nt)
    t_ns = t * 1e9

    amp_max = np.max(np.abs(traces)) if np.max(np.abs(traces)) > 0 else 1.0
    offset = 1.4 * amp_max

    plt.figure(figsize=(11, 8))
    for r in range(nr):
        plt.plot(t_ns, traces[:, r] + r * offset, linewidth=1.2)
    plt.xlabel("time (ns)")
    plt.ylabel("receiver index with vertical offset")
    plt.title(f"{sample_id}: 16 receiver traces with offsets")
    plt.tight_layout()
    plt.savefig(out_dir / f"{sample_id}_offset_traces.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.imshow(
        traces.T,
        aspect="auto",
        cmap="RdBu_r",
        extent=[t_ns[0], t_ns[-1], receiver_indices[0], receiver_indices[-1]],
        origin="lower",
    )
    plt.xlabel("time (ns)")
    plt.ylabel("receiver index")
    plt.title(f"{sample_id}: trace stack with physical time axis")
    plt.tight_layout()
    plt.savefig(out_dir / f"{sample_id}_trace_stack_physical_axes.png", dpi=180)
    plt.close()

    arrival_idx = np.array([estimate_arrival_index(traces[:, r]) for r in range(nr)], dtype=int)
    arrival_ns = t_ns[arrival_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(receiver_indices, arrival_ns, marker="o")
    plt.xlabel("receiver index")
    plt.ylabel("estimated first strong arrival time (ns)")
    plt.title(f"{sample_id}: arrival time vs receiver index")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{sample_id}_arrival_times.png", dpi=180)
    plt.close()

    selected = [0, max(0, nr//2 - 1), min(nr-1, nr//2), nr-1]
    plt.figure(figsize=(10, 6))
    for r in selected:
        plt.plot(t_ns, traces[:, r], label=f"rx {r}")
    plt.xlabel("time (ns)")
    plt.ylabel("amplitude (simulation units)")
    plt.title(f"{sample_id}: selected traces for explanation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{sample_id}_selected_traces.png", dpi=180)
    plt.close()

    source_center_time = args.source_center_frac * args.t_end
    source_width_buggy = max(args.old_gate_frac * args.t_end, 3.0 / args.frequency)
    margin_time = args.margin_time_ns * 1e-9

    t_direct_est = np.maximum(arrival_idx / max(nt - 1, 1) * args.t_end - source_center_time, 0.0)
    old_gate_time = source_center_time + t_direct_est + source_width_buggy + margin_time
    new_gate_time = source_center_time + t_direct_est + margin_time

    plt.figure(figsize=(10, 6))
    plt.plot(receiver_indices, old_gate_time * 1e9, marker="o", label="old buggy gate time")
    plt.plot(receiver_indices, new_gate_time * 1e9, marker="s", label="fixed gate time")
    plt.axhline(args.t_end * 1e9, linestyle="--", label="t_end")
    plt.xlabel("receiver index")
    plt.ylabel("gate time (ns)")
    plt.title(f"{sample_id}: old vs fixed gate time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{sample_id}_gate_comparison.png", dpi=180)
    plt.close()

    md = f"""# Trace explanation for {sample_id}

## Demonstration values vs simulation values
The hand-sketch coordinate values given before were **demonstration values only**.
They were made to help reproduce the V-shape by hand.
They were **not** applied in the Python simulation code.

The real plotted data come from:
- `{sample_id}_traces.csv`

## Gate-window bug explanation
Originally, the code used

- `t_center = 0.28 * t_end`
- `source_width = max(0.10 * t_end, 3/frequency)`
- `gate_time = t_center + t_direct + source_width + margin`

With
- `t_end = {args.t_end:.3e} s`
- `frequency = {args.frequency:.3e} Hz`

the term
- `3/f = {3.0/args.frequency:.3e} s`

can become too large relative to the full time window.

That makes `gate_time` exceed the recording duration, so nearly the whole trace is set to zero before time reversal.

The fixed gate is

- `gate_time = t_center + t_direct + margin`

which removes the direct arrival more conservatively and keeps useful later data.

## Files produced in this folder
- offset traces of all receivers
- trace stack with time in ns
- estimated arrival time vs receiver index
- selected receiver traces
- comparison of old and fixed gate times
"""
    (out_dir / f"{sample_id}_trace_explanation.md").write_text(md, encoding="utf-8")

    print(f"Saved plots and explanation to: {out_dir}")


if __name__ == "__main__":
    main()
