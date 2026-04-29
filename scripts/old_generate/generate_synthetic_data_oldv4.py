#!/usr/bin/env python3
from __future__ import annotations

"""Generate synthetic j-Wave data for Pd(50 nm)/Si(300 um) interface defect imaging."""

import os
import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Force CPU while debugging.
# Remove or change this after configuring Metal/JAX GPU successfully on Apple Silicon.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

try:
    import jax.numpy as jnp
    from jwave.acoustics import simulate_wave_propagation
    from jwave.geometry import Domain, Medium, Sensors, Sources, TimeAxis, points_on_circle
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
    frequency_hz: float
    attenuation_np_per_m: float
    pd_thickness_nm: float
    si_thickness_um: float
    pd_speed_m_per_s: float
    si_speed_m_per_s: float
    defect_center_x_px: int
    defect_center_y_px: int
    defect_width_px: int
    defect_height_px: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic j-Wave data")
    p.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--samples', type=int, default=1000)

    p.add_argument('--grid-x', type=int, default=128, help='Grid width in pixels (x direction).')
    p.add_argument('--grid-y', type=int, default=96, help='Grid height in pixels (y direction).')
    p.add_argument('--grid', type=int, default=None, help='Legacy square-grid shortcut; overrides grid-x/grid-y if set.')

    p.add_argument(
        '--dx-um',
        type=float,
        default=5.0,
        help='Grid spacing in micrometers. Physical size = number of pixels × dx.'
    )
    p.add_argument('--dx', type=float, default=None, help='Legacy grid spacing in meters.')

    p.add_argument('--frequency', type=float, default=25e6, help='Excitation frequency in Hz.')
    p.add_argument(
        '--pd-thickness-nm',
        type=float,
        default=50.0,
        help='Physical palladium thickness in nm. Used as metadata and to estimate an effective near-surface layer.'
    )
    p.add_argument(
        '--si-thickness-um',
        type=float,
        default=300.0,
        help='Physical silicon thickness in um. Used as metadata for the modeled substrate.'
    )
    p.add_argument(
        '--pd-speed',
        type=float,
        default=3070.0,
        help='Reference palladium wave speed in m/s. Typical value used here: ~3070 m/s.'
    )
    p.add_argument(
        '--si-speed',
        type=float,
        default=8433.0,
        help='Reference silicon wave speed in m/s. Typical longitudinal value used here: ~8433 m/s.'
    )

    p.add_argument('--sources', type=int, default=16)
    p.add_argument('--sensors', type=int, default=16)
    p.add_argument('--t-end', type=float, default=1.2e-7)
    p.add_argument('--cfl', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=7)

    p.add_argument('--min-defect-frac', type=float, default=0.10)
    p.add_argument('--max-defect-frac', type=float, default=0.32)

    # Class probabilities
    p.add_argument('--p-no-defect', type=float, default=0.20)
    p.add_argument('--p-crack', type=float, default=0.28)
    p.add_argument('--p-void', type=float, default=0.24)
    p.add_argument('--p-missing-pd', type=float, default=0.14)
    p.add_argument('--p-delamination', type=float, default=0.14)

    p.add_argument('--source-position', type=str, choices=['left', 'center', 'right', 'custom'], default='center',
                   help='Pulsed-laser source position along the top surface.')
    p.add_argument('--source-x-px', type=int, default=None,
                   help='Exact lateral source position in pixels when --source-position=custom.')
    p.add_argument('--source-y-px', type=int, default=1,
                   help='Exact vertical source position in pixels.')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler()],
    )



def clear_generated_files(input_dir: Path, output_dir: Path) -> None:
    """Remove previously generated CSV and PNG artifacts from input/ and output/."""
    for path in input_dir.glob('*.csv'):
        if path.is_file():
            path.unlink()
    for path in output_dir.glob('*.png'):
        if path.is_file():
            path.unlink()

def save_csv(array: np.ndarray, path: Path) -> None:
    pd.DataFrame(array).to_csv(path, header=False, index=False)


def rect_mask(shape: tuple[int, int], cx: int, cy: int, w: int, h: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    x1, x2 = max(0, cx - w // 2), min(shape[1], cx + (w + 1) // 2)
    y1, y2 = max(0, cy - h // 2), min(shape[0], cy + (h + 1) // 2)
    mask[y1:y2, x1:x2] = 1.0
    return mask


def ellipse_mask(
    shape: tuple[int, int],
    cx: int,
    cy: int,
    rx: int,
    ry: int,
    angle_deg: float = 0.0,
) -> np.ndarray:
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


def horizontal_band_mask(
    shape: tuple[int, int],
    y_top: int,
    thickness: int,
    x1: int,
    x2: int,
) -> np.ndarray:
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
        [
            args.p_no_defect,
            args.p_crack,
            args.p_void,
            args.p_missing_pd,
            args.p_delamination,
        ],
        dtype=np.float64,
    )
    total = probs.sum()
    if total <= 0:
        raise ValueError("At least one class probability must be positive.")
    probs /= total
    return probs


def sample_defect(
    rng: np.random.Generator,
    nx: int,
    ny: int,
    min_frac: float,
    max_frac: float,
    class_probs: np.ndarray,
):
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
        cx = int(rng.integers(max(8, int(0.20 * nx)), max(9, int(0.80 * nx))))
        cy = int(rng.integers(0, max(1, int(0.04 * ny))))

        w = int(rng.integers(max(6, int(0.10 * nx)), max(8, int(0.28 * nx))))
        h = int(rng.integers(1, max(2, int(0.05 * ny))))
        mask = rect_mask((ny, nx), cx, cy, w, h)

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


def make_maps(
    nx: int,
    ny: int,
    defect_mask: np.ndarray,
    defect_type: str,
    attenuation: float,
    pd_speed: float,
    si_speed: float,
    pd_thickness_nm: float,
    dx_m: float,
):
    si_c, si_rho = float(si_speed), 2330.0
    pd_c, pd_rho = float(pd_speed), 12023.0

    sound_speed = np.full((ny, nx), si_c, dtype=np.float32)
    density = np.full((ny, nx), si_rho, dtype=np.float32)
    attenuation_map = np.full((ny, nx), attenuation, dtype=np.float32)

    effective_pd_thickness_m = pd_thickness_nm * 1e-9
    interface_rows = max(1, int(round(effective_pd_thickness_m / dx_m)))
    interface_rows = min(interface_rows, max(1, ny // 8))

    interface = np.zeros((ny, nx), dtype=np.float32)
    interface[:interface_rows, :] = 1.0

    # Healthy layered medium
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


def make_signal(time_axis: TimeAxis, frequency: float) -> jnp.ndarray:
    t = time_axis.to_array()
    s0 = 8.0e2 / time_axis.dt * jnp.sin(2 * jnp.pi * frequency * t)
    center = float(t.max()) * 0.28
    width = max(float(t.max()) * 0.10, 3.0 / frequency)
    return gaussian_window(apply_ramp(s0, time_axis.dt, frequency), t, center, width)


def traces_to_frame(traces: np.ndarray) -> pd.DataFrame:
    time_steps, n_src, n_rx = traces.shape
    cols = {}
    for s in range(n_src):
        for r in range(n_rx):
            cols[f'src_{s:02d}_rx_{r:02d}'] = traces[:, s, r]
    return pd.DataFrame(cols)


def build_refocus_image(refocus: np.ndarray, ny: int, nx: int) -> np.ndarray:
    refocus = np.asarray(refocus).reshape(-1)
    refocus_img = np.zeros((ny, nx), dtype=np.float32)

    rows_to_copy = min(ny, refocus.shape[0])
    if rows_to_copy > 0:
        repeated = np.repeat(refocus[:rows_to_copy, None], nx, axis=1)
        refocus_img[:rows_to_copy, :] = repeated.astype(np.float32)

    return refocus_img


def plot_sample(
    sample_id: str,
    output_dir: Path,
    sound_speed: np.ndarray,
    target: np.ndarray,
    traces: np.ndarray,
    refocus: np.ndarray,
):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(sound_speed, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Sound speed')

    axes[0, 1].imshow(target, cmap='magma', aspect='auto')
    axes[0, 1].set_title('Target mask')

    axes[1, 0].imshow(traces[:, 0, :].T, aspect='auto', cmap='RdBu_r')
    axes[1, 0].set_title('Trace stack (src 0)')

    axes[1, 1].imshow(refocus, cmap='inferno', aspect='auto')
    axes[1, 1].set_title('Refocus field')

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_dir / f'{sample_id}_summary.png', dpi=140)
    plt.close(fig)

    plt.figure(figsize=(8, 3))
    plt.plot(traces[:, 0, 0])
    plt.title(f'{sample_id} src0/rx0')
    plt.tight_layout()
    plt.savefig(output_dir / f'{sample_id}_trace.png', dpi=140)
    plt.close()


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

    for d in [
        input_dir,
        output_dir,
        art_dir,
        art_dir / 'logs',
        root / 'validation',
        root / 'test',
        root / 'usecases',
    ]:
        d.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        clear_generated_files(input_dir, output_dir)

    configure_logging(art_dir / 'logs' / 'generate.log')
    logging.info('Starting synthetic data generation')
    logging.info('Arguments: %s', vars(args))

    rng = np.random.default_rng(args.seed)

    nx, ny = args.grid_x, args.grid_y
    domain = Domain((ny, nx), (dx_m, dx_m))

    radius = int(0.36 * min(nx, ny))
    center = (ny // 2, nx // 2)

    source_y, source_x = points_on_circle(args.sources, radius, center)
    sensor_y, sensor_x = points_on_circle(
        args.sensors,
        radius,
        center,
        angle=np.pi / args.sensors,
    )

    time_axis = TimeAxis.from_medium(
        Medium(
            domain=domain,
            sound_speed=float(args.si_speed),
            density=2330.0,
        ),
        cfl=args.cfl,
        t_end=args.t_end,
    )

    signal = make_signal(time_axis, args.frequency)
    rows = []

    for i in range(args.samples):
        sample_id = f'sample_{i:04d}'
        logging.info('Generating %s', sample_id)

        defect_type, target, cx, cy, w, h = sample_defect(
            rng,
            nx,
            ny,
            args.min_defect_frac,
            args.max_defect_frac,
            class_probs,
        )

        attenuation = float(rng.uniform(6.0, 12.0))

        sound_speed, density, attenuation_map = make_maps(
            nx,
            ny,
            target,
            defect_type,
            attenuation,
            args.pd_speed,
            args.si_speed,
            args.pd_thickness_nm,
            dx_m,
        )

        medium = Medium(
            domain=domain,
            sound_speed=jnp.asarray(sound_speed),
            density=jnp.asarray(density),
            attenuation=jnp.asarray(attenuation_map),
        )

        traces = []
        sensors = Sensors(positions=(sensor_y, sensor_x))

        for sx, sy in zip(source_x, source_y):
            src = Sources(
                positions=(np.array([sy]), np.array([sx])),
                signals=signal[None, :],
                dt=time_axis.dt,
                domain=domain,
            )

            recordings = simulate_wave_propagation(
                medium,
                time_axis,
                sources=src,
                sensors=sensors,
            )

            rec_np = np.asarray(recordings)
            rec_np = np.squeeze(rec_np)

            if rec_np.ndim == 1:
                rec_np = rec_np[:, None]
            elif rec_np.ndim == 2:
                if rec_np.shape[0] == args.sensors and rec_np.shape[1] != args.sensors:
                    rec_np = rec_np.T
            else:
                rec_np = rec_np.reshape(rec_np.shape[0], -1)

            traces.append(rec_np.astype(np.float32))

        traces_arr = np.stack(traces, axis=1).astype(np.float32)

        refocus = np.mean(np.abs(traces_arr), axis=(1, 2))
        refocus_img = build_refocus_image(refocus, ny, nx)

        traces_to_frame(traces_arr).to_csv(input_dir / f'{sample_id}_traces.csv', index=False)
        save_csv(target, input_dir / f'{sample_id}_target.csv')
        save_csv(sound_speed, input_dir / f'{sample_id}_sound_speed.csv')
        save_csv(density, input_dir / f'{sample_id}_density.csv')
        save_csv(attenuation_map, input_dir / f'{sample_id}_attenuation.csv')
        save_csv(refocus_img, input_dir / f'{sample_id}_refocus.csv')

        plot_sample(sample_id, output_dir, sound_speed, target, traces_arr, refocus_img)

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
                    frequency_hz=args.frequency,
                    attenuation_np_per_m=attenuation,
                    pd_thickness_nm=args.pd_thickness_nm,
                    si_thickness_um=args.si_thickness_um,
                    pd_speed_m_per_s=args.pd_speed,
                    si_speed_m_per_s=args.si_speed,
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
        'frequency_hz': args.frequency,
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
            'Pd/Si interface-sensitive synthetic data. '
            'dx is specified in micrometers; physical size = grid pixels × dx. '
            'Pd thickness may be sub-grid, so it is represented as an effective near-surface layer.'
        ),
    }

    (art_dir / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2))
    logging.info('Finished generation of %d samples', len(rows))


if __name__ == '__main__':
    main()