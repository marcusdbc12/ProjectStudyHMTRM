#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_source_signal(
    t_end: float,
    dt: float,
    source_mag: float = 1.3e-5,
    center_frac: float = 0.28,
    pulse_fwhm_ns: float = 5.0,
):
    """
    Build the current source waveform as a PURE GAUSSIAN pulse.

    Mathematical form:
        s(t) = (source_mag / dt) * exp( -(t - t0)^2 / (2*sigma^2) )

    where:
        t0    = center_frac * t_end
        sigma = FWHM / (2*sqrt(2*ln 2))
    """
    t = np.arange(0.0, t_end, dt, dtype=np.float64)
    center = center_frac * t_end

    # Convert FWHM to sigma
    fwhm_s = pulse_fwhm_ns * 1e-9
    sigma = fwhm_s / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    amplitude = source_mag / dt
    signal = amplitude * np.exp(-0.5 * ((t - center) / sigma) ** 2)

    meta = {
        "mode": "pure_gaussian",
        "t_end_s": t_end,
        "dt_s": dt,
        "source_mag": source_mag,
        "amplitude_over_dt": amplitude,
        "center_frac": center_frac,
        "center_s": center,
        "pulse_fwhm_ns": pulse_fwhm_ns,
        "pulse_fwhm_s": fwhm_s,
        "sigma_s": sigma,
        "sigma_ns": sigma * 1e9,
        "2sigma_ns": 2.0 * sigma * 1e9,
        "4sigma_ns": 4.0 * sigma * 1e9,
        "6sigma_ns": 6.0 * sigma * 1e9,
        "n_samples": len(t),
    }
    return t, signal, meta


def main() -> None:
    p = argparse.ArgumentParser(description="Plot only the pure-Gaussian source waveform.")
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--t-end", type=float, default=1.2e-7)
    p.add_argument("--dt", type=float, default=1.186e-10)
    p.add_argument("--source-mag", type=float, default=1.3e-5)
    p.add_argument("--center-frac", type=float, default=0.28)
    p.add_argument("--pulse-fwhm-ns", type=float, default=5.0)
    args = p.parse_args()

    out_dir = args.project_root / "artifacts" / "source_behavior"
    out_dir.mkdir(parents=True, exist_ok=True)

    t, signal, meta = build_source_signal(
        t_end=args.t_end,
        dt=args.dt,
        source_mag=args.source_mag,
        center_frac=args.center_frac,
        pulse_fwhm_ns=args.pulse_fwhm_ns,
    )

    t_ns = t * 1e9
    center_ns = meta["center_s"] * 1e9
    fwhm = meta["pulse_fwhm_ns"]

    plt.figure(figsize=(10, 4))
    plt.plot(t_ns, signal, label="pure Gaussian pulse")
    plt.axvline(center_ns, linestyle="--", label="pulse center")
    plt.axvline(center_ns - fwhm / 2.0, linestyle=":", alpha=0.9, label="FWHM limits")
    plt.axvline(center_ns + fwhm / 2.0, linestyle=":", alpha=0.9)
    plt.xlabel("time (ns)")
    plt.ylabel("amplitude (simulation units)")
    plt.title("Pure Gaussian source waveform")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "source_waveform.png", dpi=180)
    plt.close()

    info_lines = [
        "# Source waveform metadata",
        "mode: pure_gaussian",
        f"t_end_s: {meta['t_end_s']}",
        f"dt_s: {meta['dt_s']}",
        f"source_mag: {meta['source_mag']}",
        f"amplitude_over_dt: {meta['amplitude_over_dt']}",
        f"center_frac: {meta['center_frac']}",
        f"center_s: {meta['center_s']}",
        f"pulse_fwhm_ns: {meta['pulse_fwhm_ns']}",
        f"pulse_fwhm_s: {meta['pulse_fwhm_s']}",
        f"sigma_s: {meta['sigma_s']}",
        f"sigma_ns: {meta['sigma_ns']}",
        f"2sigma_ns: {meta['2sigma_ns']}",
        f"4sigma_ns: {meta['4sigma_ns']}",
        f"6sigma_ns: {meta['6sigma_ns']}",
        f"n_samples: {meta['n_samples']}",
        "",
        "Interpretation:",
        "- This script now plots a pure Gaussian pulse only.",
        "- The visible pulse can look wider than the FWHM because a Gaussian has no hard cutoff.",
        "- FWHM is measured at half of the peak amplitude, not at the full visible width.",
        "- For a 5 ns FWHM Gaussian, the visually noticeable width is often closer to ~8-10 ns.",
        "- This is expected behavior and does not mean the implementation is wrong.",
    ]
    (out_dir / "source_waveform_info.txt").write_text("\n".join(info_lines), encoding="utf-8")

    print(out_dir / "source_waveform.png")
    print(out_dir / "source_waveform_info.txt")


if __name__ == "__main__":
    main()
