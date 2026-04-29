#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_step_pulse_signal(
    t_end: float,
    dt: float,
    source_mag: float = 1.3e-5,
    start_frac: float = 0.28,
    pulse_width_ns: float = 5.0,
):
    """
    Build a perfect rectangular step pulse of finite duration.

    s(t) = (source_mag / dt) * [H(t - t0) - H(t - (t0 + tau))]
    """
    t = np.arange(0.0, t_end, dt, dtype=np.float64)

    t0 = start_frac * t_end
    tau = pulse_width_ns * 1e-9
    amplitude = source_mag / dt

    signal = amplitude * (((t >= t0) & (t < t0 + tau)).astype(np.float64))

    meta = {
        "mode": "rectangular_step_pulse",
        "t_end_s": t_end,
        "dt_s": dt,
        "source_mag": source_mag,
        "amplitude_over_dt": amplitude,
        "start_frac": start_frac,
        "start_time_s": t0,
        "pulse_width_ns": pulse_width_ns,
        "pulse_width_s": tau,
        "end_time_s": t0 + tau,
        "n_samples": len(t),
    }
    return t, signal, meta


def main() -> None:
    p = argparse.ArgumentParser(description="Plot only a finite-duration rectangular step pulse.")
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--t-end", type=float, default=1.2e-7)
    p.add_argument("--dt", type=float, default=1.186e-10)
    p.add_argument("--source-mag", type=float, default=1.3e-5)
    p.add_argument("--start-frac", type=float, default=0.28)
    p.add_argument("--pulse-width-ns", type=float, default=5.0)
    args = p.parse_args()

    out_dir = args.project_root / "artifacts" / "source_behavior_step"
    out_dir.mkdir(parents=True, exist_ok=True)

    t, signal, meta = build_step_pulse_signal(
        t_end=args.t_end,
        dt=args.dt,
        source_mag=args.source_mag,
        start_frac=args.start_frac,
        pulse_width_ns=args.pulse_width_ns,
    )

    t_ns = t * 1e9
    start_ns = meta["start_time_s"] * 1e9
    end_ns = meta["end_time_s"] * 1e9

    plt.figure(figsize=(10, 4))
    plt.plot(t_ns, signal, label="rectangular step pulse")
    plt.axvline(start_ns, linestyle="--", label="pulse start")
    plt.axvline(end_ns, linestyle="--", label="pulse end")
    plt.xlabel("time (ns)")
    plt.ylabel("amplitude (simulation units)")
    plt.title("Rectangular step-pulse source waveform")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "source_step_waveform.png", dpi=180)
    plt.close()

    info_lines = [
        "# Rectangular step-pulse metadata",
        "mode: rectangular_step_pulse",
        f"t_end_s: {meta['t_end_s']}",
        f"dt_s: {meta['dt_s']}",
        f"source_mag: {meta['source_mag']}",
        f"amplitude_over_dt: {meta['amplitude_over_dt']}",
        f"start_frac: {meta['start_frac']}",
        f"start_time_s: {meta['start_time_s']}",
        f"pulse_width_ns: {meta['pulse_width_ns']}",
        f"pulse_width_s: {meta['pulse_width_s']}",
        f"end_time_s: {meta['end_time_s']}",
        f"n_samples: {meta['n_samples']}",
        "",
        "Interpretation:",
        "- This is a finite-duration step pulse, not an infinite Heaviside step.",
        "- The pulse turns on sharply at t0 and turns off sharply at t0 + tau.",
        "- It is useful as a test case, but less physically smooth than a Gaussian pulse.",
    ]
    (out_dir / "source_step_waveform_info.txt").write_text("\n".join(info_lines), encoding="utf-8")

    print(out_dir / "source_step_waveform.png")
    print(out_dir / "source_step_waveform_info.txt")


if __name__ == "__main__":
    main()
