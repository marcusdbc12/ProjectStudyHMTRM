#!/usr/bin/env python3
from __future__ import annotations

"""Generate synthetic j-Wave data for Pd(50 nm)/Si(300 um) interface defect imaging.

This version stores multiple TRM-derived representations for later ML experiments:
- raw TRM max
- raw TRM energy
- log10-compressed TRM max
- log10-compressed TRM energy
- row-normalized TRM max
- row-normalized TRM energy
"""

import os
#os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import jax
import optax
from jax.scipy.signal import convolve2d




try:
    import jax.numpy as jnp
    from jwave.acoustics import simulate_wave_propagation
    from jwave.geometry import Domain, Medium, Sensors, Sources, TimeAxis
    from jwave.signal_processing import apply_ramp, gaussian_window
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "j-Wave and its dependencies are required to run this generator. "
        "Install requirements.txt first."
    ) from exc


@dataclass
class SampleMeta:
    sample_id: str
    defect_type: str
    source_mode: str
    grid_x: int
    grid_y: int
    dx_um: float
    n_sources: int
    n_sensors: int
    time_steps: int
    #frequency_hz: float
    attenuation_np_per_m: float
    pd_thickness_nm: float
    si_thickness_um: float
    pd_speed_m_per_s: float
    si_speed_m_per_s: float
    source_x_px: int
    source_y_px: int
    defect_center_x_px: int
    defect_center_y_px: int
    defect_width_px: int
    defect_height_px: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic j-Wave data")
    p.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--samples', type=int, default=1000)
    p.add_argument('--grid-x', type=int, default=128)
    p.add_argument('--grid-y', type=int, default=96)
    p.add_argument('--grid', type=int, default=None)
    p.add_argument('--dx-um', type=float, default=5.0)
    p.add_argument('--dx', type=float, default=None)
    #p.add_argument('--frequency', type=float, default=25e6)
    p.add_argument('--pd-thickness-nm', type=float, default=50.0)
    p.add_argument('--si-thickness-um', type=float, default=300.0)
    p.add_argument('--pd-speed', type=float, default=3070.0)
    p.add_argument('--si-speed', type=float, default=8433.0)
    p.add_argument('--sources', type=int, default=1)
    p.add_argument('--sensors', type=int, default=16)
    p.add_argument('--t-end', type=float, default=1.2e-7)
    p.add_argument('--cfl', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--min-defect-frac', type=float, default=0.10)
    p.add_argument('--max-defect-frac', type=float, default=0.32)
    p.add_argument('--p-no-defect', type=float, default=0.20)
    p.add_argument('--p-crack', type=float, default=0.28)
    p.add_argument('--p-void', type=float, default=0.24)
    p.add_argument('--p-missing-pd', type=float, default=0.14)
    p.add_argument('--p-delamination', type=float, default=0.14)
    p.add_argument('--source-position', type=str, choices=['left', 'center', 'right', 'custom'], default='center')
    p.add_argument('--source-x-px', type=int, default=None)
    p.add_argument('--source-y-px', type=int, default=1)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--method', type=str, choices=['trm', 'fwi'], default='trm', help='Choose reconstruction method')
    p.add_argument('--fwi-epochs', type=int, default=15, help='Number of optimization steps for FWI')
    p.add_argument('--fwi-lr', type=float, default=5000.0, help='Learning rate for FWI velocity updates')
    return p.parse_args()


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler()],
    )


def clear_generated_files(input_dir: Path, output_dir: Path) -> None:
    for path in input_dir.glob('*.csv'):
        if path.is_file():
            path.unlink()
    for path in output_dir.glob('*.png'):
        if path.is_file():
            path.unlink()


def save_csv(array: np.ndarray, path: Path) -> None:
    pd.DataFrame(array).to_csv(path, header=False, index=False)


def safe_log10(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return np.log10(np.maximum(np.abs(x), eps)).astype(np.float32)


def row_normalize(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    denom = np.max(np.abs(x), axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return (x / denom).astype(np.float32)


def rect_mask(shape: tuple[int, int], cx: int, cy: int, w: int, h: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    x1, x2 = max(0, cx - w // 2), min(shape[1], cx + (w + 1) // 2)
    y1, y2 = max(0, cy - h // 2), min(shape[0], cy + (h + 1) // 2)
    mask[y1:y2, x1:x2] = 1.0
    return mask


def ellipse_mask(shape: tuple[int, int], cx: int, cy: int, rx: int, ry: int, angle_deg: float = 0.0) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    x = xx - cx
    y = yy - cy
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    xr = c * x + s * y
    yr = -s * x + c * y
    mask = ((xr / max(rx, 1)) ** 2 + (yr / max(ry, 1)) ** 2) <= 1.0
    return mask.astype(np.float32)


def horizontal_band_mask(shape: tuple[int, int], y_top: int, thickness: int, x1: int, x2: int) -> np.ndarray:
    ny, nx = shape
    mask = np.zeros((ny, nx), dtype=np.float32)
    y1 = max(0, y_top)
    y2 = min(ny, y_top + max(1, thickness))
    xa = max(0, min(x1, x2))
    xb = min(nx, max(x1, x2))
    mask[y1:y2, xa:xb] = 1.0
    return mask


def validate_probabilities(args: argparse.Namespace) -> np.ndarray:
    probs = np.array(
        [args.p_no_defect, args.p_crack, args.p_void, args.p_missing_pd, args.p_delamination],
        dtype=np.float64,
    )
    total = probs.sum()
    if total <= 0:
        raise ValueError("At least one class probability must be positive.")
    probs /= total
    return probs


def sample_defect(rng: np.random.Generator, nx: int, ny: int, min_frac: float, max_frac: float, class_probs: np.ndarray):
    classes = ['no_defect', 'crack', 'void', 'missing_pd', 'delamination']
    defect_type = rng.choice(classes, p=class_probs)

    cx, cy, w, h = nx // 2, ny // 2, 0, 0
    mask = np.zeros((ny, nx), dtype=np.float32)

    if defect_type == 'no_defect':
        return defect_type, mask, cx, cy, w, h

    if defect_type == 'crack':
        cx = int(rng.integers(max(8, int(0.20 * nx)), max(9, int(0.80 * nx))))
        cy = int(rng.integers(max(6, int(0.08 * ny)), max(7, int(0.45 * ny))))
        major = int(rng.integers(max(8, int(min_frac * nx)), max(10, int(max_frac * nx))))
        minor = int(rng.integers(1, max(2, int(0.03 * ny))))
        angle = float(rng.uniform(-70.0, 70.0))
        rx = max(major // 2, 2)
        ry = max(minor // 2, 1)
        mask = ellipse_mask((ny, nx), cx, cy, rx, ry, angle)
        w, h = 2 * rx, 2 * ry
        return defect_type, mask, cx, cy, w, h

    if defect_type == 'void':
        cx = int(rng.integers(max(8, int(0.20 * nx)), max(9, int(0.80 * nx))))
        cy = int(rng.integers(max(6, int(0.10 * ny)), max(7, int(0.55 * ny))))
        rx = int(rng.integers(max(3, int(0.04 * nx)), max(4, int(0.12 * nx))))
        ry = int(rng.integers(max(3, int(0.04 * ny)), max(4, int(0.12 * ny))))
        angle = float(rng.uniform(0.0, 180.0))
        mask = ellipse_mask((ny, nx), cx, cy, rx, ry, angle)
        w, h = 2 * rx, 2 * ry
        return defect_type, mask, cx, cy, w, h

    if defect_type == 'missing_pd':
        band_h = max(1, int(rng.integers(1, max(2, int(0.04 * ny)))))
        cx = int(rng.integers(max(8, int(0.15 * nx)), max(9, int(0.85 * nx))))
        w = int(rng.integers(max(8, int(min_frac * nx)), max(10, int(max_frac * nx))))
        cy = max(0, band_h // 2)
        mask = rect_mask((ny, nx), cx, cy, w, band_h)
        h = band_h
        return defect_type, mask, cx, cy, w, h

    if defect_type == 'delamination':
        thickness = int(rng.integers(1, max(2, int(0.03 * ny))))
        y_top = int(rng.integers(0, max(1, int(0.05 * ny))))
        x1 = int(rng.integers(0, max(2, int(0.45 * nx))))
        span = int(rng.integers(max(8, int(0.15 * nx)), max(10, int(0.40 * nx))))
        x2 = min(nx, x1 + span)
        mask = horizontal_band_mask((ny, nx), y_top, thickness, x1, x2)
        cx = (x1 + x2) // 2
        cy = y_top + thickness // 2
        w = x2 - x1
        h = thickness
        return defect_type, mask, cx, cy, w, h

    return defect_type, mask, cx, cy, w, h


def make_maps(nx: int, ny: int, defect_mask: np.ndarray, defect_type: str, attenuation: float,
              pd_speed: float, si_speed: float, pd_thickness_nm: float, dx_m: float):
    si_c, si_rho = float(si_speed), 2330.0
    pd_c, pd_rho = float(pd_speed), 12023.0

    sound_speed = np.full((ny, nx), si_c, dtype=np.float32)
    density = np.full((ny, nx), si_rho, dtype=np.float32)
    attenuation_map = np.full((ny, nx), attenuation, dtype=np.float32)

    effective_pd_thickness_m = pd_thickness_nm * 1e-9
    #interface_rows = max(1, int(round(effective_pd_thickness_m / dx_m)))
    interface_rows = int(round(effective_pd_thickness_m / dx_m))
    interface_rows = min(interface_rows, max(1, ny // 8))

    interface = np.zeros((ny, nx), dtype=np.float32)
    interface[:interface_rows, :] = 1.0
    sound_speed[interface == 1] = pd_c
    density[interface == 1] = pd_rho

    if defect_type == 'no_defect':
        return sound_speed, density, attenuation_map

    if defect_type == 'void':
        affected = defect_mask > 0
        sound_speed[affected] = si_c * 0.25
        density[affected] = si_rho * 0.10
        attenuation_map[affected] *= 2.2
    elif defect_type == 'crack':
        affected = defect_mask > 0
        sound_speed[affected] = si_c * 0.55
        density[affected] = si_rho * 0.45
        attenuation_map[affected] *= 1.6
    elif defect_type == 'missing_pd':
        affected = (defect_mask > 0) & (interface == 1)
        sound_speed[affected] = si_c
        density[affected] = si_rho
        attenuation_map[affected] *= 1.2
    elif defect_type == 'delamination':
        affected = defect_mask > 0
        sound_speed[affected] = si_c * 0.80
        density[affected] = si_rho * 0.65
        attenuation_map[affected] *= 1.8

    return sound_speed, density, attenuation_map


def plot_offset_traces_with_defect(residual_traces, baseline_traces, dt, sensor_x_px, sensor_y_px, src_x_px, src_y_px, cx, cy, dx_m, velocity_m_s, output_dir, sample_id):
    plt.figure(figsize=(10, 8))
    n_rx = residual_traces.shape[2]
    n_t = residual_traces.shape[0]
    t_ns = np.arange(n_t) * dt * 1e9

    # --- EXTRACT BOTH SIGNALS ---
    raw_residual = residual_traces[:, 0, :]
    raw_baseline = baseline_traces[:, 0, :]
    
    # Boost only the microscopic echoes so they match the visual scale of the main wave
    boost_factor = 0.5e6 
    boosted_residual = raw_residual * boost_factor

    # Calculate vertical spacing based on the massive baseline wave so nothing overlaps
    spacing = np.max(np.abs(raw_baseline)) * 1.5
    if spacing == 0:
        spacing = 1.0

    # Calculate ToF Physics
    dist_in = np.sqrt(((src_x_px - cx) * dx_m)**2 + ((src_y_px - cy) * dx_m)**2)
    pulse_center_t = (n_t - 1) * dt * 0.28 

    expected_times_ns = []
    offsets = []

    for r in range(n_rx):
        offset = r * spacing
        
        # 1. Plot the giant main wave (Baseline) in light gray
        plt.plot(t_ns, raw_baseline[:, r] + offset, color='lightgray', alpha=0.7)
        
        # 2. Plot the boosted defect echo in blue right on top
        plt.plot(t_ns, boosted_residual[:, r] + offset, color='steelblue', alpha=0.9)
        
        dist_out = np.sqrt(((sensor_x_px[r] - cx) * dx_m)**2 + ((sensor_y_px[r] - cy) * dx_m)**2)
        total_time_s = pulse_center_t + ((dist_in + dist_out) / velocity_m_s)
        expected_times_ns.append(total_time_s * 1e9)
        offsets.append(offset)

    # 3. Draw a vertical line exactly at the Laser Pulse Center
    pulse_center_ns = pulse_center_t * 1e9
    plt.axvline(x=pulse_center_ns, color='green', linestyle='-', linewidth=1.5, label=f'5ns Laser Pulse ({pulse_center_ns:.1f} ns)')

    # 4. OVERLAY THE RED THEORETICAL CURVE (Uncommented!)
    plt.plot(expected_times_ns, offsets, 'r--', linewidth=2, label='Theoretical Defect Arrival')
    plt.scatter(expected_times_ns, offsets, color='red', marker='x', s=50, zorder=5)

    plt.title(f'{sample_id} - Main Surface Wave vs. Boosted Defect Echo')
    plt.xlabel('Time (ns)')
    plt.ylabel('Receiver Array')
    plt.yticks(offsets, [f'rx{i}' for i in range(n_rx)])
    plt.xlim(0, n_t * dt * 1e9)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / f'{sample_id}_offset_traces_combined.png', dpi=160)
    plt.close()


def make_signal(time_axis):
    t = time_axis.to_array()

    source_mag = 1
    #source_mag = source_mag / time_axis.dt

    center = float(t.max()) * 0.28

    fwhm = 5.0e-9
    sigma = fwhm / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))

    s0 = source_mag * jnp.exp(-0.5 * ((t - center) / sigma) ** 2)
    return s0


def top_surface_line_positions(nx: int, count: int, y_index: int = 1) -> tuple[np.ndarray, np.ndarray]:
    margin_x = max(4, nx // 10)
    x = np.linspace(margin_x, nx - 1 - margin_x, count, dtype=int)
    y = np.full(count, y_index, dtype=int)
    return y, x


def choose_source_position(nx: int, ny: int, mode: str, source_x_px: int | None, source_y_px: int) -> tuple[np.ndarray, np.ndarray]:
    margin_x = max(4, nx // 10)
    if mode == 'left':
        sx = margin_x
    elif mode == 'right':
        sx = nx - 1 - margin_x
    elif mode == 'custom':
        sx = nx // 2 if source_x_px is None else int(np.clip(source_x_px, 0, nx - 1))
    else:
        sx = nx // 2
    sy = int(np.clip(source_y_px, 0, ny - 1))
    return np.array([sy], dtype=int), np.array([sx], dtype=int)


def normalize_recordings_array(recordings: np.ndarray, n_sensors: int) -> np.ndarray:
    rec_np = np.asarray(recordings)
    rec_np = np.squeeze(rec_np)
    if rec_np.ndim == 1:
        rec_np = rec_np[:, None]
    elif rec_np.ndim == 2:
        if rec_np.shape[0] == n_sensors and rec_np.shape[1] != n_sensors:
            rec_np = rec_np.T
    else:
        rec_np = rec_np.reshape(rec_np.shape[0], -1)
    return rec_np.astype(np.float32)


def normalize_field_array(field_like) -> np.ndarray:
    obj = field_like.params if hasattr(field_like, 'params') else field_like
    arr = np.asarray(obj)
    arr = np.squeeze(arr)
    if arr.ndim not in (2, 3):
        raise ValueError(f'Unexpected TR field shape after normalization: {arr.shape}')
    return arr.astype(np.float32)


def traces_to_frame(traces: np.ndarray) -> pd.DataFrame:
    time_steps, n_src, n_rx = traces.shape
    cols = {}
    for s in range(n_src):
        for r in range(n_rx):
            cols[f'src_{s:02d}_rx_{r:02d}'] = traces[:, s, r]
    return pd.DataFrame(cols)



def plot_sample(
    sample_id: str,
    output_dir: Path,
    sound_speed: np.ndarray,
    target: np.ndarray,
    traces: np.ndarray,
    trm_max: np.ndarray,
    trm_max_log: np.ndarray,
    trm_max_row_norm: np.ndarray,
    trm_energy_log: np.ndarray,
    dx_um: float,
    t_end: float,
) -> None:
    """
    Create a summary figure with physically meaningful axis labels.

    Why this version is better:
    - spatial maps are shown in micrometers instead of raw pixel indices
    - the trace stack is shown in nanoseconds on x-axis
    - receiver index is shown explicitly on y-axis for the trace stack
    - axes are no longer hidden
    """

    ny, nx = sound_speed.shape

    # Spatial extent in micrometers for all 2D material/TRM maps.
    # extent = [x_min, x_max, y_max, y_min] is used so that y=0 appears at the top,
    # matching the matrix-style interpretation already used in the project.
    extent_xy = [0, nx * dx_um, ny * dx_um, 0]

    # Time axis for traces in nanoseconds.
    nt = traces.shape[0]
    nr = traces.shape[2]
    t_ns = np.linspace(0.0, t_end * 1e9, nt)

    # Trace stack extent:
    # x-axis = time in ns
    # y-axis = receiver index
    # Using [nr-1, 0] puts rx_00 at the top, matching the previous visual style.
    extent_traces = [t_ns[0], t_ns[-1], nr - 1, 0]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # -------------------------------------------------------------------------
    # Row 1
    # -------------------------------------------------------------------------
    axes[0, 0].imshow(sound_speed, cmap='viridis', aspect='auto', extent=extent_xy)
    axes[0, 0].set_title('Sound speed')
    axes[0, 0].set_xlabel('x (um)')
    axes[0, 0].set_ylabel('y (um)')

    axes[0, 1].imshow(target, cmap='magma', aspect='auto', extent=extent_xy)
    axes[0, 1].set_title('Target mask')
    axes[0, 1].set_xlabel('x (um)')
    axes[0, 1].set_ylabel('y (um)')

    # axes[0, 2].imshow(
    #     #traces[:, 0, :].T,
    #     traces[:, 0, :],
    #     aspect='auto',
    #     cmap='RdBu_r',
    #     extent=extent_traces,
    # )
    # axes[0, 2].set_title('Trace stack (src 0)')
    # axes[0, 2].set_xlabel('time (ns)')
    # axes[0, 2].set_ylabel('receiver index')
    
    # Trace stack with receiver index on x-axis and time on y-axis
    
    extent_traces = [0, nr - 1, t_ns[-1], t_ns[0]]

    axes[0, 2].imshow(
        traces[:, 0, :],   # shape = (Nt, Nr) -> y=time, x=receiver
        aspect='auto',
        cmap='RdBu_r',
        extent=extent_traces,
    )
    axes[0, 2].set_title('Trace stack (src 0)')
    axes[0, 2].set_xlabel('receiver index')
    axes[0, 2].set_ylabel('time (ns)')

    # axes[0, 3].imshow(trm_max, cmap='inferno', aspect='auto', extent=extent_xy)
    # axes[0, 3].set_title('TRM max raw')
    # axes[0, 3].set_xlabel('x (um)')
    # axes[0, 3].set_ylabel('y (um)')

    # -------------------------------------------------------------------------
    # Row 2
    # -------------------------------------------------------------------------
    axes[1, 0].imshow(trm_max_log, cmap='inferno', aspect='auto', extent=extent_xy)
    axes[1, 0].set_title('TRM max log10')
    axes[1, 0].set_xlabel('x (um)')
    axes[1, 0].set_ylabel('y (um)')

    axes[1, 1].imshow(trm_max_row_norm, cmap='RdBu_r', aspect='auto', extent=extent_xy)
    axes[1, 1].set_title('TRM max row-normalized')
    axes[1, 1].set_xlabel('x (um)')
    axes[1, 1].set_ylabel('y (um)')

    axes[1, 2].imshow(trm_energy_log, cmap='inferno', aspect='auto', extent=extent_xy)
    axes[1, 2].set_title('TRM energy log10')
    axes[1, 2].set_xlabel('x (um)')
    axes[1, 2].set_ylabel('y (um)')

    # Combined RGB image:
    # R = row-normalized TRM max
    # G = log10 TRM max
    # B = log10 TRM energy
    # Each channel is scaled independently to [0,1] for visualization.
    # combo = np.stack([
    #     (trm_max_row_norm - np.min(trm_max_row_norm)) /
    #     max(np.max(trm_max_row_norm) - np.min(trm_max_row_norm), 1e-12),

    #     (trm_max_log - np.min(trm_max_log)) /
    #     max(np.max(trm_max_log) - np.min(trm_max_log), 1e-12),

    #     (trm_energy_log - np.min(trm_energy_log)) /
    #     max(np.max(trm_energy_log) - np.min(trm_energy_log), 1e-12),
    # ], axis=-1)

    # axes[1, 3].imshow(combo, aspect='auto', extent=extent_xy)
    # axes[1, 3].set_title('Combined RGB')
    # axes[1, 3].set_xlabel('x (um)')
    # axes[1, 3].set_ylabel('y (um)')

    # Optional: make ticks a bit smaller so the figure stays readable.
    for ax in axes.ravel():
        ax.tick_params(axis='both', labelsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / f'{sample_id}_summary.png', dpi=160)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Separate single-trace figure
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 3))
    trace = np.asarray(traces[:, 0, 0])
    peak = np.max(np.abs(trace))
    trace_mV = trace * 1e3
    # if peak > 0:
    #     trace_mV = trace / peak * 3.0   # target peak = 3 mV
    # else:
    #     trace_mV = trace.copy()

    #plt.plot(t_ns, trace_mV)
    plt.plot(t_ns, trace_mV)
    plt.title(f'{sample_id} src0/rx0')
    plt.xlabel('Time (ns)')
    plt.ylabel('Amplitude (mV)')
    plt.tight_layout()
    plt.savefig(output_dir / f'{sample_id}_trace.png', dpi=160)
    plt.close()


######################################
#Function for velocity and spacing
######################################
def analyze_velocity_and_spacing(traces, dt, sensor_x_px, sensor_y_px, src_x_px, src_y_px, dx_m, output_dir, sample_id):
    # 1. Extract peak times
    peak_indices = np.argmax(np.abs(traces[:, 0, :]), axis=0)
    t_peaks = peak_indices * dt
    
    # 2. Calculate true propagation distance from source to each sensor
    dx_array = (sensor_x_px - src_x_px) * dx_m
    dy_array = (sensor_y_px - src_y_px) * dx_m
    distances = np.sqrt(dx_array**2 + dy_array**2)
    
    # 3. Linear Regression: distance = velocity * time + offset
    slope, intercept = np.polyfit(t_peaks, distances, 1)
    velocity_m_s = slope
    
    # 4. Plotting
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(t_peaks * 1e9, distances * 1e6, 'bo', label='Measured Peak Arrivals')
    
    t_line = np.linspace(np.min(t_peaks), np.max(t_peaks), 100)
    d_line = slope * t_line + intercept
    plt.plot(t_line * 1e9, d_line * 1e6, 'r-', label=f'Linear Fit: v $\\approx$ {velocity_m_s:.1f} m/s')
    
    plt.title(f'{sample_id} - Surface Wave Velocity Fit')
    plt.xlabel('Arrival Time (ns)')
    plt.ylabel('Absolute Distance from Source ($\\mu$m)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'{sample_id}_velocity_fit.png', dpi=160)
    plt.close()
    
    # 5. Generate CSV for Delta x between adjacent receivers
    sensor_positions_m = sensor_x_px * dx_m
    delta_x_array = np.diff(sensor_positions_m)
    
    csv_data = []
    for i in range(len(delta_x_array)):
        csv_data.append({
            'Receiver_Pair': f'rx{i}_rx{i+1}',
            'Receiver_A_Pos_um': sensor_positions_m[i] * 1e6,
            'Receiver_B_Pos_um': sensor_positions_m[i+1] * 1e6,
            'Delta_x_m': delta_x_array[i],
            'Delta_x_um': delta_x_array[i] * 1e6,
            'Calculated_Velocity_m_s': velocity_m_s
        })
        
    df_spacing = pd.DataFrame(csv_data)
    df_spacing.to_csv(output_dir / f'{sample_id}_receiver_spacing.csv', index=False)
    
    return velocity_m_s

##############################################################


#########
#Plotting for offset
########

def plot_three_step_subtraction(traces_arr, baseline_traces_arr, residual_traces, dt, dx_m, output_dir, sample_id):
    """
    Plots a 3-panel comparison for  documentation:
    Left: Raw signal with defect
    Center: Perfect baseline (no defect)
    Right: Isolated residual echo (boosted)
    """


    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    
    n_rx = traces_arr.shape[2]
    n_t = traces_arr.shape[0]
    t_ns = np.arange(n_t) * dt * 1e9

    # Extract the arrays
    raw = traces_arr[:, 0, :]
    base = baseline_traces_arr[:, 0, :]
    resid = residual_traces[:, 0, :]

    # Calculate uniform spacing based on the massive surface wave
    spacing = np.max(np.abs(base)) * 1.5
    if spacing == 0: spacing = 1.0

    # Boost the residual so it is visible in the third plot
    boost_factor = 1e6
    boosted_resid = resid * boost_factor

    titles = [
        "1. Raw Simulation (With Defect)", 
        "2. Perfect Baseline (No Defect)", 
        f"3. Residual Echo (Boosted {boost_factor:.0e}x)"
    ]
    data_sources = [raw, base, boosted_resid]

    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontweight='bold')
        ax.set_xlabel("Time (ns)")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlim(0, n_t * dt * 1e9)
        
        if i == 0:
            ax.set_ylabel("Receiver Array")
            offsets = [r * spacing for r in range(n_rx)]
            ax.set_yticks(offsets)
            ax.set_yticklabels([f'rx{r}' for r in range(n_rx)])

        # Plot all 16 receivers for this specific dataset
        for r in range(n_rx):
            offset = r * spacing
            ax.plot(t_ns, data_sources[i][:, r] + offset, color='steelblue', alpha=0.85)

    plt.suptitle(f"{sample_id} - Baseline Subtraction Pipeline", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / f'{sample_id}_subtraction_pipeline.png', dpi=160)
    plt.close()

################
#FWI
###############

def make_gaussian_kernel(sigma=1.5, size=5):
    """Creates a 2D Gaussian kernel for spatial smoothing."""
    x = jnp.arange(-size // 2 + 1., size // 2 + 1.)
    x, y = jnp.meshgrid(x, x)
    kernel = jnp.exp(-(x**2 + y**2) / (2. * sigma**2))
    return kernel / jnp.sum(kernel)

def plot_fwi_results(true_c, reconstructed_c, loss_history, dx_m, output_dir, sample_id):
    """Plots the True Speed, Reconstructed Speed, and Error Convergence."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    ny, nx = true_c.shape
    extent = [0, nx * dx_m * 1e6, ny * dx_m * 1e6, 0]

    # Plot 1: True Target
    im0 = axes[0].imshow(true_c, cmap='viridis', aspect='auto', extent=extent)
    axes[0].set_title("True Sound Speed (Target)")
    plt.colorbar(im0, ax=axes[0], label='Velocity (m/s)')

    # Plot 2: FWI Result
    im1 = axes[1].imshow(reconstructed_c, cmap='viridis', aspect='auto', extent=extent)
    axes[1].set_title("FWI Reconstructed Speed")
    plt.colorbar(im1, ax=axes[1], label='Velocity (m/s)')

    # Plot 3: Convergence Curve
    axes[2].plot(range(1, len(loss_history) + 1), loss_history, 'r.-', linewidth=2)
    axes[2].set_title("FWI Convergence (MSE Loss)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Error")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / f'{sample_id}_fwi_results.png', dpi=160)
    plt.close()

# 
def run_fwi_optimization(true_recordings, medium_guess, time_axis, src, sensors, epochs, lr, sample_id):
    """Robust FWI using Optax Adam optimizer and spatial gradient smoothing."""
    import logging
    logging.info(f"Starting Robust FWI for {sample_id} over {epochs} epochs...")
    
    blur_kernel = make_gaussian_kernel(sigma=2.0, size=5)

    # Helper function to strip jaxdf physics wrappers
    def extract_array(obj):
        try:
            return jnp.squeeze(obj.on_grid)
        except AttributeError:
            try:
                return jnp.squeeze(obj.params)
            except AttributeError:
                return jnp.squeeze(jnp.asarray(obj))

    # Strip ALL parameters so Medium doesn't complain about mixed types
    # c_start = extract_array(medium_guess.sound_speed)
    c0 = extract_array(medium_guess.sound_speed)

    rho_arr = extract_array(medium_guess.density)
    att_arr = extract_array(medium_guess.attenuation)
    true_recordings = jnp.asarray(true_recordings)
    true_recordings = jnp.squeeze(true_recordings)
    n_sensors = len(sensors.positions[0])

    if true_recordings.ndim == 1:
        true_recordings = true_recordings[:, None]
    elif true_recordings.ndim == 2:
        if true_recordings.shape[0] == n_sensors and true_recordings.shape[1] != n_sensors:
            true_recordings = true_recordings.T
    else:
        true_recordings = true_recordings.reshape(true_recordings.shape[0], -1)

    delta_c = jnp.zeros_like(c0)

    loss_history = []
    grad_history = []
    update_history = []

    c_min = 1500.0
    c_max = 9500.0

    
    @jax.jit
    def loss_fn(delta_c):
        c_array = jnp.clip(c0 + delta_c, a_min=c_min, a_max=c_max)
        m_temp = Medium(
            domain=medium_guess.domain,
            sound_speed=c_array,
            density=rho_arr,
            attenuation=att_arr
        )

        sim_recordings = simulate_wave_propagation(
            m_temp,
            time_axis,
            sources=src,
            sensors=sensors
        )

        # force simulated recordings to match target shape: (time, sensors)
        sim_recordings = jnp.asarray(sim_recordings)
        sim_recordings = jnp.squeeze(sim_recordings)

        # if squeeze leaves a 1D array, recover 2D
        if sim_recordings.ndim == 1:
            sim_recordings = sim_recordings[:, None]
        elif sim_recordings.ndim == 2:
            if sim_recordings.shape[0] == n_sensors and sim_recordings.shape[1] != n_sensors:
                sim_recordings = sim_recordings.T
        else:
            sim_recordings = sim_recordings.reshape(sim_recordings.shape[0], -1)

        error = sim_recordings - true_recordings

        t = jnp.arange(error.shape[0]) * time_axis.dt
        t_ns = t * 1e9

        weight = jnp.where((t_ns >= 105.0) & (t_ns <= 150.0), 1.0, 0.1)
        weight = weight[:, None]

        weighted_error = error * weight

        reg = 1e-8 * jnp.mean(jnp.square(delta_c))

        return jnp.mean(jnp.square(weighted_error)) + reg

    loss_and_grad_fn = jax.value_and_grad(loss_fn)
    
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(delta_c)
    
    loss_history = []

    for epoch in range(epochs):
        loss_val, raw_grad = loss_and_grad_fn(delta_c)
        
        # 1. Squeeze out any dummy dimensions
        grad_2d = jnp.squeeze(raw_grad)
        
        # 2. Apply the 2D spatial smoothing
        smoothed_2d = convolve2d(grad_2d, blur_kernel, mode='same')
        
        # 3. Reshape it exactly back to what JAX expects
        smoothed_grad = jnp.reshape(smoothed_2d, raw_grad.shape)
        
        # updates, opt_state = optimizer.update(smoothed_grad, opt_state)
        # delta_c = optax.apply_updates(delta_c, updates)        
        
        # Invertimos el signo porque queremos minimizar el error
        updates, opt_state = optimizer.update(smoothed_grad, opt_state, delta_c)
        delta_c = optax.apply_updates(delta_c, updates)

        # Evita que delta_c se quede microscópico respecto a velocidades de miles m/s
        delta_c = jnp.clip(delta_c, -3000.0, 3000.0)


        # Constrain the physics bounds (e.g., Silicon sound speed limits)
        #c_current = jnp.clip(c_current, a_min=10.0, a_max=9500.0)

        loss_history.append(float(loss_val))
        grad_history.append(float(jnp.linalg.norm(raw_grad)))
        update_history.append(float(jnp.linalg.norm(delta_c)))

        logging.info(
            "  FWI Epoch %02d/%d | Loss: %.12e | GradNorm: %.12e | SmoothGradNorm: %.12e | UpdateNorm: %.12e",
            epoch + 1,
            epochs,
            float(loss_val),
            float(jnp.linalg.norm(raw_grad)),
            float(jnp.linalg.norm(smoothed_grad)),
            float(jnp.linalg.norm(delta_c)),
        )

    c_final = jnp.clip(c0 + delta_c, a_min=c_min, a_max=c_max)
    return np.array(c_final), loss_history, grad_history, update_history
    
def main() -> None:
    args = parse_args()
    if args.grid is not None:
        args.grid_x = args.grid
        args.grid_y = args.grid

    dx_m = args.dx if args.dx is not None else args.dx_um * 1e-6
    class_probs = validate_probabilities(args)

    root = args.project_root
    input_dir = root / 'input'
    output_dir = root / 'output'
    art_dir = root / 'artifacts'

    for d in [input_dir, output_dir, art_dir, art_dir / 'logs', root / 'validation', root / 'test', root / 'usecases']:
        d.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        clear_generated_files(input_dir, output_dir)

    configure_logging(art_dir / 'logs' / 'generate.log')
    logging.info('Starting synthetic data generation')
    logging.info('Arguments: %s', vars(args))

    rng = np.random.default_rng(args.seed)

    nx, ny = args.grid_x, args.grid_y
    domain = Domain((ny, nx), (dx_m, dx_m))

    args.sources = 1
    source_y, source_x = choose_source_position(nx, ny, args.source_position, args.source_x_px, args.source_y_px)
    sensor_y, sensor_x = top_surface_line_positions(nx, args.sensors, y_index=1)

    logging.info('Using top-surface source=(%d,%d) and %d top-line sensors', int(source_x[0]), int(source_y[0]), args.sensors)

    time_axis = TimeAxis.from_medium(
        Medium(domain=domain, sound_speed=float(args.si_speed), density=2330.0),
        cfl=args.cfl,
        t_end=args.t_end,
    )
    signal = make_signal(time_axis)

    rows = []

    for i in range(args.samples):
        sample_id = f'sample_{i:04d}'
        logging.info('Generating %s', sample_id)

        defect_type, target, cx, cy, w, h = sample_defect(rng, nx, ny, args.min_defect_frac, args.max_defect_frac, class_probs)
        attenuation = float(rng.uniform(6.0, 12.0))

        sound_speed, density, attenuation_map = make_maps(
            nx, ny, target, defect_type, attenuation, args.pd_speed, args.si_speed, args.pd_thickness_nm, dx_m
        )

        medium = Medium(
            domain=domain,
            sound_speed=jnp.asarray(sound_speed),
            density=jnp.asarray(density),
            attenuation=jnp.asarray(attenuation_map),
        )

        sensors = Sensors(positions=(sensor_y, sensor_x))
        src = Sources(
            positions=(jnp.asarray([source_y[0]]), jnp.asarray([source_x[0]])),
            signals=jnp.asarray(signal[None, :]),
            dt=time_axis.dt,
            domain=domain,
        )
        #####
        #old one 
        ####
        # recordings = simulate_wave_propagation(medium, time_axis, sources=src, sensors=sensors)
        # rec_np = normalize_recordings_array(recordings, args.sensors)
        # traces_arr = rec_np[:, None, :]
        #######
        #####
        #new one
        #####

        # 1. SIMULACIÓN DE LÍNEA BASE (MEDIO PERFECTO)
        sound_speed_base, density_base, attenuation_map_base = make_maps(
            nx, ny, np.zeros_like(target), 'no_defect', attenuation, args.pd_speed, args.si_speed, args.pd_thickness_nm, dx_m
        )
        medium_base = Medium(
            domain=domain,
            sound_speed=jnp.asarray(sound_speed_base),
            density=jnp.asarray(density_base),
            attenuation=jnp.asarray(attenuation_map_base),
        )
        recordings_base = simulate_wave_propagation(medium_base, time_axis, sources=src, sensors=sensors)
        rec_np_base = normalize_recordings_array(recordings_base, args.sensors)
        baseline_traces_arr = rec_np_base[:, None, :]

        baseline_field = simulate_wave_propagation(
            medium_base,
            time_axis,
            sources=src,
        )

        baseline_field_np = normalize_field_array(baseline_field)

        # 2. SIMULACIÓN REAL (CON DEFECTO)
        recordings = simulate_wave_propagation(medium, time_axis, sources=src, sensors=sensors)
        rec_np = normalize_recordings_array(recordings, args.sensors)
        traces_arr = rec_np[:, None, :]

        # 3. RESTA PARA EXTRAER EL ECO PURO
        residual_traces = traces_arr - baseline_traces_arr


        # gated_traces = rec_np.copy()
        # dt = float(time_axis.dt)
        # margin_time = 10e-9
        # source_center_time = args.t_end * 0.28

        # for r in range(args.sensors):
        #     dx_r = abs(int(sensor_x[r]) - int(source_x[0])) * dx_m
        #     dy_r = abs(int(sensor_y[r]) - int(source_y[0])) * dx_m
        #     d_r = float(np.sqrt(dx_r * dx_r + dy_r * dy_r))
        #     t_direct = d_r / float(args.si_speed)
        #     gate_time = source_center_time + t_direct + margin_time
        #     gate_idx_r = min(gated_traces.shape[0], max(0, int(np.floor(gate_time / dt))))
        #     gated_traces[:gate_idx_r, r] = 0.0

        # reversed_traces = gated_traces[::-1, :]

        save_csv(target, input_dir / f'{sample_id}_target.csv')
        save_csv(sound_speed, input_dir / f'{sample_id}_sound_speed.csv')
        save_csv(density, input_dir / f'{sample_id}_density.csv')
        save_csv(attenuation_map, input_dir / f'{sample_id}_attenuation.csv')

        if args.method == 'trm':
            


            logging.info("Running old_v8-compatible TRM block")

            # old_v8 behavior: use the full recorded signal with defect, not residual
            gated_traces = rec_np.copy()

            dt = float(time_axis.dt)
            margin_time = 10e-9
            source_center_time = args.t_end * 0.28

            for r in range(args.sensors):
                dx_r = abs(int(sensor_x[r]) - int(source_x[0])) * dx_m
                dy_r = abs(int(sensor_y[r]) - int(source_y[0])) * dx_m
                d_r = float(np.sqrt(dx_r * dx_r + dy_r * dy_r))

                t_direct = d_r / float(args.si_speed)
                gate_time = source_center_time + t_direct + margin_time
                gate_idx_r = min(
                    gated_traces.shape[0],
                    max(0, int(np.floor(gate_time / dt)))
                )

                gated_traces[:gate_idx_r, r] = 0.0

            reversed_traces = gated_traces[::-1, :]
            signals_clean = jnp.asarray(reversed_traces.T)

            src_trm = Sources(
                positions=(jnp.asarray(sensor_y), jnp.asarray(sensor_x)),
                signals=signals_clean,
                dt=time_axis.dt,
                domain=medium.domain,
            )

            trm_field = simulate_wave_propagation(
                medium,
                time_axis,
                sources=src_trm,
            )

            trm_field_np = normalize_field_array(trm_field)

            if trm_field_np.ndim == 3:
                trm_max = np.max(np.abs(trm_field_np), axis=0).astype(np.float32)
                trm_energy = np.sum(np.square(np.abs(trm_field_np)), axis=0).astype(np.float32)
            else:
                trm_max = np.abs(trm_field_np).astype(np.float32)
                trm_energy = np.square(np.abs(trm_field_np)).astype(np.float32)

            surface_mask_depth = int(sensor_y[0]) + 2
            trm_max[:surface_mask_depth, :] = 0.0
            trm_energy[:surface_mask_depth, :] = 0.0

            logging.info("old_v8-compatible trm_field_np shape: %s", trm_field_np.shape)
            logging.info("trm_max stats: min=%g max=%g mean=%g", np.min(trm_max), np.max(trm_max), np.mean(trm_max))
            logging.info("trm_energy stats: min=%g max=%g mean=%g", np.min(trm_energy), np.max(trm_energy), np.mean(trm_energy))
            logging.info("target sum: %g", np.sum(target))



            trm_max_log = safe_log10(trm_max)
            trm_energy_log = safe_log10(trm_energy)
            trm_max_row_norm = row_normalize(trm_max)
            trm_energy_row_norm = row_normalize(trm_energy)

            logging.info('trm_field_np shape: %s', trm_field_np.shape)
            logging.info('trm_max stats: min=%g max=%g mean=%g', np.min(trm_max), np.max(trm_max), np.mean(trm_max))
            logging.info('trm_energy stats: min=%g max=%g mean=%g', np.min(trm_energy), np.max(trm_energy), np.mean(trm_energy))
            logging.info('target sum: %g', np.sum(target))



            save_csv(trm_max, input_dir / f'{sample_id}_refocus.csv')
            save_csv(trm_energy, input_dir / f'{sample_id}_trm_energy.csv')
            save_csv(trm_max_log, input_dir / f'{sample_id}_refocus_log.csv')
            save_csv(trm_energy_log, input_dir / f'{sample_id}_trm_energy_log.csv')
            save_csv(trm_max_row_norm, input_dir / f'{sample_id}_refocus_row_norm.csv')
            save_csv(trm_energy_row_norm, input_dir / f'{sample_id}_trm_energy_row_norm.csv')
            
            plot_sample(
                sample_id,
                output_dir,
                sound_speed,
                target,
                traces_arr,
                trm_max,
                trm_max_log,
                trm_max_row_norm,
                trm_energy_log,
                dx_um=float(dx_m * 1e6),
                t_end=args.t_end,
            )
            
        elif args.method == 'fwi':
            # ==========================================
            # FWI PIPELINE (The new code)
            # ==========================================
            if defect_type != 'no_defect':
                fwi_reconstructed_c, loss_curve, grad_history, update_history = run_fwi_optimization(
                    true_recordings=jnp.asarray(rec_np), 
                    medium_guess=medium_base, 
                    time_axis=time_axis, 
                    src=src, 
                    sensors=sensors, 
                    epochs=args.fwi_epochs, 
                    lr=args.fwi_lr, 
                    sample_id=sample_id
                )
                
                # Save the numerical array
                np.savetxt(output_dir / f'{sample_id}_fwi_c_map.csv', fwi_reconstructed_c, delimiter=',')
                
                # Generate the 3-panel FWI graph
                plot_fwi_results(
                    true_c=sound_speed, 
                    reconstructed_c=fwi_reconstructed_c, 
                    loss_history=loss_curve, 
                    dx_m=dx_m, 
                    output_dir=output_dir, 
                    sample_id=sample_id
                )
            else:
                logging.info(f"Sample {sample_id} is no_defect. Skipping FWI.")
                
        # 3. Rest of your loop continues here...

        # src_trm = Sources(
        #     positions=(jnp.asarray(sensor_y), jnp.asarray(sensor_x)),
        #     signals=jnp.asarray(reversed_traces.T),
        #     dt=time_axis.dt,
        #     domain=domain,
        # )

        # trm_field = simulate_wave_propagation(medium, time_axis, sources=src_trm)
        # trm_field_np = normalize_field_array(trm_field)

        # if trm_field_np.ndim == 3:
        #     trm_max = np.max(np.abs(trm_field_np), axis=0).astype(np.float32)
        #     trm_energy = np.sum(np.square(np.abs(trm_field_np)), axis=0).astype(np.float32)
        # else:
        #     trm_max = np.abs(trm_field_np).astype(np.float32)
        #     trm_energy = np.square(np.abs(trm_field_np)).astype(np.float32)

        # surface_mask_depth = int(sensor_y[0]) + 2
        # trm_max[:surface_mask_depth, :] = 0.0
        # trm_energy[:surface_mask_depth, :] = 0.0

        # trm_max_log = safe_log10(trm_max)
        # trm_energy_log = safe_log10(trm_energy)
        # trm_max_row_norm = row_normalize(trm_max)
        # trm_energy_row_norm = row_normalize(trm_energy)

        # logging.info('trm_field_np shape: %s', trm_field_np.shape)
        # logging.info('trm_max stats: min=%g max=%g mean=%g', np.min(trm_max), np.max(trm_max), np.mean(trm_max))
        # logging.info('trm_energy stats: min=%g max=%g mean=%g', np.min(trm_energy), np.max(trm_energy), np.mean(trm_energy))
        # logging.info('target sum: %g', np.sum(target))

         # --- SAVE ALL THREE TRACE DATASETS ---
        # 1. The simulation with the defect (kept original name so it doesn't break older scripts)
        traces_to_frame(traces_arr).to_csv(input_dir / f'{sample_id}_traces.csv', index=False)
        
        # 2. The perfect baseline simulation (no defect)
        traces_to_frame(baseline_traces_arr).to_csv(input_dir / f'{sample_id}_traces_baseline.csv', index=False)
        
        # 3. The isolated pure defect echoes
        traces_to_frame(residual_traces).to_csv(input_dir / f'{sample_id}_traces_residual.csv', index=False)
        # -------------------------------------







        # --- NEW INTEGRATION ---
        # 1. Calculate Velocity
        calculated_velocity = analyze_velocity_and_spacing(
            traces=traces_arr,
            dt=float(time_axis.dt),
            sensor_x_px=sensor_x,
            sensor_y_px=sensor_y[0],
            src_x_px=source_x[0],
            src_y_px=source_y[0],
            dx_m=dx_m,
            output_dir=output_dir,
            sample_id=sample_id
        )
        logging.info(f'{sample_id} calculated surface velocity: {calculated_velocity:.2f} m/s')

        # 2. Plot Waterfall with Defect Delay Overlay
        # if defect_type != 'no_defect':
        #     plot_offset_traces_with_defect(
        #         #traces=traces_arr,
        #         residual_traces=residual_traces,
        #         baseline_traces=baseline_traces_arr, 
        #         dt=float(time_axis.dt),
        #         sensor_x_px=sensor_x,
        #         sensor_y_px=sensor_y,
        #         src_x_px=source_x[0],
        #         src_y_px=source_y[0],
        #         cx=cx,
        #         cy=cy,
        #         dx_m=dx_m,
        #         velocity_m_s=calculated_velocity,
        #         output_dir=output_dir,
        #         sample_id=sample_id
        #     )
                # 2. Plot Waterfall with Defect Delay Overlay
        if defect_type != 'no_defect':
            
            # Graph A: The overlaid traces with the red theoretical curve
            plot_offset_traces_with_defect(
                residual_traces=residual_traces,
                baseline_traces=baseline_traces_arr, 
                dt=float(time_axis.dt),
                sensor_x_px=sensor_x,
                sensor_y_px=sensor_y,
                src_x_px=source_x[0],
                src_y_px=source_y[0],
                cx=cx,
                #cy=cy,
                cy=cy - (h / 2),
                dx_m=dx_m,
                #velocity_m_s=calculated_velocity,
                velocity_m_s=args.si_speed,
                output_dir=output_dir,
                sample_id=sample_id
            )

            # Graph B: The 3-panel A/B/C baseline subtraction comparison
            plot_three_step_subtraction(
                traces_arr=traces_arr,
                baseline_traces_arr=baseline_traces_arr,
                residual_traces=residual_traces,
                dt=float(time_axis.dt),
                dx_m=dx_m,
                output_dir=output_dir,
                sample_id=sample_id
            )
        # -----------------------
        # -----------------------







        rows.append(
            asdict(
                SampleMeta(
                    sample_id=sample_id,
                    defect_type=defect_type,
                    source_mode=args.source_position,
                    grid_x=nx,
                    grid_y=ny,
                    dx_um=float(dx_m * 1e6),
                    n_sources=args.sources,
                    n_sensors=args.sensors,
                    time_steps=traces_arr.shape[0],
                    #frequency_hz=args.frequency,
                    attenuation_np_per_m=attenuation,
                    pd_thickness_nm=args.pd_thickness_nm,
                    si_thickness_um=args.si_thickness_um,
                    pd_speed_m_per_s=args.pd_speed,
                    si_speed_m_per_s=args.si_speed,
                    source_x_px=int(source_x[0]),
                    source_y_px=int(source_y[0]),
                    defect_center_x_px=cx,
                    defect_center_y_px=cy,
                    defect_width_px=w,
                    defect_height_px=h,
                )
            )
        )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(input_dir / 'metadata.csv', index=False)

    manifest = {
        'project': 'jwave_keras_gui_workflow',
        'samples': len(rows),
        'grid_x': nx,
        'grid_y': ny,
        'dx_um': float(dx_m * 1e6),
        #'frequency_hz': args.frequency,
        'pd_thickness_nm': args.pd_thickness_nm,
        'si_thickness_um': args.si_thickness_um,
        'pd_speed_m_per_s': args.pd_speed,
        'si_speed_m_per_s': args.si_speed,
        'source_position': args.source_position,
        'source_x_px': int(source_x[0]),
        'source_y_px': int(source_y[0]),
        'class_probabilities': {
            'no_defect': float(class_probs[0]),
            'crack': float(class_probs[1]),
            'void': float(class_probs[2]),
            'missing_pd': float(class_probs[3]),
            'delamination': float(class_probs[4]),
        },
        'notes': (
            'Pd/Si interface-sensitive synthetic data using one top-surface pulsed-laser source, '
            'a top-surface receiver line, trace gating, and numerical TRM. '
            'Saved outputs include raw, log, and row-normalized TRM representations for ML experiments.'
        ),
    }

    (art_dir / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2))
    logging.info('Finished generation of %d samples', len(rows))


if __name__ == '__main__':
    main()
