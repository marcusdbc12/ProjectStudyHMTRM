#!/usr/bin/env python3
from __future__ import annotations

"""
Generate synthetic j-Wave data for:
- Pd(50 nm)/Si(300 um), or
- pure Si

with a numerical time-reversal mirror (TRM):
1) forward simulation from one source
2) record traces on a top-surface receiver line
3) reverse traces in time
4) backpropagate from the same receiver positions
5) save TRM max map and TRM energy map

Notes:
- For compatibility with existing downstream scripts, *_refocus.csv stores the TRM max map.
- *_trm_energy.csv stores the TRM energy-like map.
- The script forces JAX to CPU for stability on Apple Metal / FFT issues.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    structure_mode: str
    defect_type: str
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
    source_x_px: int
    source_y_px: int
    defect_center_x_px: int
    defect_center_y_px: int
    defect_width_px: int
    defect_height_px: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic j-Wave data with numerical TRM")
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--grid-x", type=int, default=128)
    p.add_argument("--grid-y", type=int, default=96)
    p.add_argument("--grid", type=int, default=None,
                   help="Legacy square-grid shortcut; overrides grid-x/grid-y if set.")
    p.add_argument("--dx-um", type=float, default=5.0, help="Grid spacing in micrometers.")
    p.add_argument("--dx", type=float, default=None, help="Legacy grid spacing in meters.")
    p.add_argument("--frequency", type=float, default=25e6)
    p.add_argument("--pd-thickness-nm", type=float, default=50.0,
                   help="Physical palladium thickness in nm.")
    p.add_argument("--si-thickness-um", type=float, default=300.0,
                   help="Physical silicon thickness in um.")
    p.add_argument("--pd-speed", type=float, default=3070.0,
                   help="Reference palladium wave speed in m/s.")
    p.add_argument("--si-speed", type=float, default=8433.0,
                   help="Reference silicon wave speed in m/s.")
    p.add_argument("--structure-mode", type=str, choices=["pd_si", "pure_si"], default="pd_si")
    p.add_argument("--sources", type=int, default=1,
                   help="Physical setup uses one source; kept for metadata compatibility.")
    p.add_argument("--sensors", type=int, default=16)
    p.add_argument("--t-end", type=float, default=1.2e-7)
    p.add_argument("--cfl", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-defect-frac", type=float, default=0.10)
    p.add_argument("--max-defect-frac", type=float, default=0.32)
    p.add_argument("--p-no-defect", type=float, default=0.20)
    p.add_argument("--p-crack", type=float, default=0.28)
    p.add_argument("--p-void", type=float, default=0.24)
    p.add_argument("--p-missing-pd", type=float, default=0.14)
    p.add_argument("--p-delamination", type=float, default=0.14)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )


def clear_generated_files(input_dir: Path, output_dir: Path) -> None:
    """
    Remove previously generated CSV and PNG artifacts from input/ and output/.
    Keeps the folders themselves.
    """
    input_patterns = ["*.csv"]
    output_patterns = ["*.png"]

    for pattern in input_patterns:
        for path in input_dir.glob(pattern):
            if path.is_file():
                path.unlink()

    for pattern in output_patterns:
        for path in output_dir.glob(pattern):
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


def ellipse_mask(shape: tuple[int, int], cx: int, cy: int, a: int, b: int, theta: float) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float32)
    x = xx - float(cx)
    y = yy - float(cy)

    ct, st = np.cos(theta), np.sin(theta)
    xp = x * ct + y * st
    yp = -x * st + y * ct

    a = max(1, a)
    b = max(1, b)
    mask = ((xp ** 2) / (a ** 2) + (yp ** 2) / (b ** 2) <= 1.0).astype(np.float32)
    return mask


def choose_defect_type(args: argparse.Namespace, rng: np.random.Generator) -> str:
    if args.structure_mode == "pure_si":
        labels = ["no_defect", "crack", "void"]
        probs = np.array([args.p_no_defect, args.p_crack, args.p_void], dtype=np.float64)
    else:
        labels = ["no_defect", "crack", "void", "missing_pd", "delamination"]
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

    probs = np.maximum(probs, 0.0)
    if probs.sum() <= 0:
        probs = np.ones_like(probs)
    probs = probs / probs.sum()
    return str(rng.choice(labels, p=probs))


def sample_defect(
    rng: np.random.Generator,
    nx: int,
    ny: int,
    min_frac: float,
    max_frac: float,
    structure_mode: str,
    defect_type: str,
):
    if defect_type == "no_defect":
        mask = np.zeros((ny, nx), dtype=np.float32)
        return defect_type, mask, nx // 2, ny // 2, 0, 0

    if defect_type == "crack":
        cx = int(rng.integers(max(6, int(0.20 * nx)), min(nx - 6, int(0.80 * nx))))
        cy = int(rng.integers(max(6, int(0.10 * ny)), min(ny - 6, int(0.60 * ny))))
        a = int(rng.integers(max(4, int(min_frac * nx)), max(5, int(max_frac * nx))))
        b = int(rng.integers(1, max(2, int(0.03 * ny))))
        theta = float(rng.uniform(-np.pi / 3, np.pi / 3))
        mask = ellipse_mask((ny, nx), cx, cy, a, b, theta)
        return defect_type, mask, cx, cy, 2 * a, 2 * b

    if defect_type == "void":
        cx = int(rng.integers(max(8, int(0.20 * nx)), min(nx - 8, int(0.80 * nx))))
        cy = int(rng.integers(max(8, int(0.10 * ny)), min(ny - 8, int(0.60 * ny))))
        a = int(rng.integers(max(3, int(0.03 * nx)), max(4, int(0.09 * nx))))
        b = int(rng.integers(max(3, int(0.03 * ny)), max(4, int(0.09 * ny))))
        theta = float(rng.uniform(0, np.pi))
        mask = ellipse_mask((ny, nx), cx, cy, a, b, theta)
        return defect_type, mask, cx, cy, 2 * a, 2 * b

    if defect_type == "missing_pd":
        cx = int(rng.integers(max(6, int(0.20 * nx)), min(nx - 6, int(0.80 * nx))))
        cy = 1
        w = int(rng.integers(max(4, int(min_frac * nx)), max(5, int(max_frac * nx))))
        h = int(rng.integers(1, max(2, int(0.04 * ny))))
        mask = rect_mask((ny, nx), cx, cy, w, h)
        return defect_type, mask, cx, cy, w, h

    if defect_type == "delamination":
        cx = int(rng.integers(max(8, int(0.20 * nx)), min(nx - 8, int(0.80 * nx))))
        cy = 1
        w = int(rng.integers(max(6, int(min_frac * nx)), max(7, int(max_frac * nx))))
        h = max(1, int(0.02 * ny))
        mask = rect_mask((ny, nx), cx, cy, w, h)
        return defect_type, mask, cx, cy, w, h

    raise ValueError(f"Unsupported defect type: {defect_type}")


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
    structure_mode: str,
):
    si_c, si_rho = float(si_speed), 2330.0
    pd_c, pd_rho = float(pd_speed), 12023.0

    sound_speed = np.full((ny, nx), si_c, np.float32)
    density = np.full((ny, nx), si_rho, np.float32)
    attenuation_map = np.full((ny, nx), attenuation, np.float32)

    interface = np.zeros((ny, nx), dtype=np.float32)

    if structure_mode == "pd_si" and pd_thickness_nm > 0:
        effective_pd_thickness_m = pd_thickness_nm * 1e-9
        interface_rows = max(1, int(round(effective_pd_thickness_m / dx_m)))
        interface_rows = min(interface_rows, max(1, ny // 8))

        interface[:interface_rows, :] = 1.0
        sound_speed[interface == 1] = pd_c
        density[interface == 1] = pd_rho

    if defect_type == "no_defect":
        return sound_speed, density, attenuation_map

    if defect_type == "missing_pd":
        affected = (defect_mask > 0) & (interface == 1)
        sound_speed[affected] = si_c * 0.95
        density[affected] = si_rho * 0.80
        attenuation_map[affected] *= 1.4
        return sound_speed, density, attenuation_map

    if defect_type == "delamination":
        affected = defect_mask > 0
        sound_speed[affected] = si_c * 0.90
        density[affected] = si_rho * 0.70
        attenuation_map[affected] *= 1.5
        return sound_speed, density, attenuation_map

    if defect_type == "void":
        affected = defect_mask > 0
        sound_speed[affected] = si_c * 0.25
        density[affected] = si_rho * 0.10
        attenuation_map[affected] *= 2.2
        return sound_speed, density, attenuation_map

    if defect_type == "crack":
        affected = defect_mask > 0
        sound_speed[affected] = si_c * 0.55
        density[affected] = si_rho * 0.45
        attenuation_map[affected] *= 1.6
        return sound_speed, density, attenuation_map

    return sound_speed, density, attenuation_map


def make_signal(time_axis: TimeAxis, frequency: float) -> jnp.ndarray:
    t = time_axis.to_array()
    s0 = 8.0e2 / time_axis.dt * jnp.sin(2 * jnp.pi * frequency * t)
    center = float(t.max()) * 0.28
    width = max(float(t.max()) * 0.10, 3.0 / frequency)
    return gaussian_window(apply_ramp(s0, time_axis.dt, frequency), t, center, width)


def normalize_recordings(recordings, n_sensors: int) -> np.ndarray:
    arr = np.asarray(recordings)
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim == 2:
        if arr.shape[0] == n_sensors and arr.shape[1] != n_sensors:
            arr = arr.T
    else:
        arr = arr.reshape(arr.shape[0], -1)

    return arr.astype(np.float32)


def normalize_field_array(field_like) -> np.ndarray:
    obj = field_like

    if hasattr(obj, "params"):
        obj = obj.params

    arr = np.asarray(obj)
    arr = np.squeeze(arr)

    if arr.ndim not in (2, 3):
        raise ValueError(f"Unexpected TR field shape after normalization: {arr.shape}")

    return arr.astype(np.float32)


def traces_to_frame(traces: np.ndarray) -> pd.DataFrame:
    nt, n_src, n_rx = traces.shape
    cols = {}
    for s in range(n_src):
        for r in range(n_rx):
            cols[f"src_{s:02d}_rx_{r:02d}"] = traces[:, s, r]
    return pd.DataFrame(cols)


def plot_sample(
    sample_id: str,
    output_dir: Path,
    sound_speed: np.ndarray,
    target: np.ndarray,
    traces: np.ndarray,
    trm_max: np.ndarray,
):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(sound_speed, cmap="viridis", aspect="auto")
    axes[0, 0].set_title("Sound speed")

    axes[0, 1].imshow(target, cmap="magma", aspect="auto")
    axes[0, 1].set_title("Target mask")

    axes[1, 0].imshow(traces[:, 0, :].T, aspect="auto", cmap="RdBu_r")
    axes[1, 0].set_title("Forward traces (src 0)")

    axes[1, 1].imshow(trm_max, cmap="inferno", aspect="auto")
    axes[1, 1].set_title("TRM max map")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_dir / f"{sample_id}_summary.png", dpi=140)
    plt.close(fig)

    plt.figure(figsize=(8, 3))
    plt.plot(traces[:, 0, 0])
    plt.title(f"{sample_id} src0/rx0")
    plt.tight_layout()
    plt.savefig(output_dir / f"{sample_id}_trace.png", dpi=140)
    plt.close()


def main() -> None:
    args = parse_args()

    if args.grid is not None:
        args.grid_x = args.grid
        args.grid_y = args.grid

    args.sources = 1
    dx_m = args.dx if args.dx is not None else args.dx_um * 1e-6

    root = args.project_root
    input_dir = root / "input"
    output_dir = root / "output"
    art_dir = root / "artifacts"

    for d in [input_dir, output_dir, art_dir, art_dir / "logs", root / "validation", root / "test", root / "usecases"]:
        d.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        clear_generated_files(input_dir, output_dir)

    configure_logging(art_dir / "logs" / "generate.log")
    logging.info("Starting synthetic data generation")
    logging.info("Arguments: %s", vars(args))

    rng = np.random.default_rng(args.seed)

    nx, ny = args.grid_x, args.grid_y
    domain = Domain((ny, nx), (dx_m, dx_m))

    margin_x = max(4, nx // 10)
    sensor_x = np.linspace(margin_x, nx - 1 - margin_x, args.sensors, dtype=int)
    sensor_y = np.full(args.sensors, 1, dtype=int)

    source_x = np.array([nx // 2], dtype=int)
    source_y = np.array([1], dtype=int)

    time_axis = TimeAxis.from_medium(
        Medium(domain=domain, sound_speed=float(args.si_speed), density=2330.0),
        cfl=args.cfl,
        t_end=args.t_end,
    )
    signal = make_signal(time_axis, args.frequency)

    rows = []

    for i in range(args.samples):
        sample_id = f"sample_{i:04d}"
        logging.info("Generating %s", sample_id)

        defect_type = choose_defect_type(args, rng)

        defect_type, target, cx, cy, w, h = sample_defect(
            rng,
            nx,
            ny,
            args.min_defect_frac,
            args.max_defect_frac,
            args.structure_mode,
            defect_type,
        )

        attenuation = float(rng.uniform(6.0, 12.0))
        pd_thickness_nm_effective = 0.0 if args.structure_mode == "pure_si" else args.pd_thickness_nm

        sound_speed, density, attenuation_map = make_maps(
            nx=nx,
            ny=ny,
            defect_mask=target,
            defect_type=defect_type,
            attenuation=attenuation,
            pd_speed=args.pd_speed,
            si_speed=args.si_speed,
            pd_thickness_nm=pd_thickness_nm_effective,
            dx_m=dx_m,
            structure_mode=args.structure_mode,
        )

        medium = Medium(
            domain=domain,
            sound_speed=jnp.asarray(sound_speed),
            density=jnp.asarray(density),
            attenuation=jnp.asarray(attenuation_map),
        )

        sensors = Sensors(positions=(sensor_y, sensor_x))

        src_forward = Sources(
            positions=(jnp.asarray([source_y[0]]), jnp.asarray([source_x[0]])),
            signals=jnp.asarray(signal[None, :]),
            dt=time_axis.dt,
            domain=domain,
        )

        recordings = simulate_wave_propagation(
            medium,
            time_axis,
            sources=src_forward,
            sensors=sensors,
        )

        forward_traces = normalize_recordings(recordings, args.sensors)
        traces_arr = forward_traces[:, None, :]

        reversed_traces = forward_traces[::-1, :]

        src_trm = Sources(
            positions=(jnp.asarray(sensor_y), jnp.asarray(sensor_x)),
            signals=jnp.asarray(reversed_traces.T),
            dt=time_axis.dt,
            domain=domain,
        )

        trm_field = simulate_wave_propagation(
            medium,
            time_axis,
            sources=src_trm,
        )

        logging.info("TR field type: %s", type(trm_field))
        if hasattr(trm_field, "params"):
            try:
                logging.info("TR field params shape before squeeze: %s", np.asarray(trm_field.params).shape)
            except Exception as exc:
                logging.info("Could not inspect trm_field.params: %s", exc)
        else:
            try:
                logging.info("TR field raw shape before squeeze: %s", np.asarray(trm_field).shape)
            except Exception as exc:
                logging.info("Could not inspect trm_field: %s", exc)

        trm_field_np = normalize_field_array(trm_field)

        if trm_field_np.ndim == 3:
            # trm_max = np.max(np.abs(trm_field_np), axis=0).astype(np.float32)
            # trm_energy = np.sum(np.square(np.abs(trm_field_np)), axis=0).astype(np.float32)
            gate_idx = int(0.20 * trm_field_np.shape[0])  # start by skipping first 20%
            trm_field_gated = trm_field_np[gate_idx:, :, :]

            trm_max = np.max(np.abs(trm_field_gated), axis=0).astype(np.float32)
            trm_energy = np.sum(np.square(np.abs(trm_field_gated)), axis=0).astype(np.float32)
            
        elif trm_field_np.ndim == 2:
            trm_max = np.abs(trm_field_np).astype(np.float32)
            trm_energy = np.square(np.abs(trm_field_np)).astype(np.float32)
        else:
            raise ValueError(f"Unsupported normalized TR field shape: {trm_field_np.shape}")

        logging.info("trm_field_np shape: %s", trm_field_np.shape)
        logging.info("forward_traces shape: %s", forward_traces.shape)
        logging.info("trm_max stats: min=%g max=%g mean=%g", np.min(trm_max), np.max(trm_max), np.mean(trm_max))
        logging.info("trm_energy stats: min=%g max=%g mean=%g", np.min(trm_energy), np.max(trm_energy), np.mean(trm_energy))
        logging.info("target sum: %g", np.sum(target))

        traces_to_frame(traces_arr).to_csv(input_dir / f"{sample_id}_traces.csv", index=False)
        save_csv(target, input_dir / f"{sample_id}_target.csv")
        save_csv(sound_speed, input_dir / f"{sample_id}_sound_speed.csv")
        save_csv(density, input_dir / f"{sample_id}_density.csv")
        save_csv(attenuation_map, input_dir / f"{sample_id}_attenuation.csv")
        save_csv(trm_max, input_dir / f"{sample_id}_refocus.csv")
        save_csv(trm_energy, input_dir / f"{sample_id}_trm_energy.csv")

        plot_sample(sample_id, output_dir, sound_speed, target, traces_arr, trm_max)

        rows.append(
            asdict(
                SampleMeta(
                    sample_id=sample_id,
                    structure_mode=args.structure_mode,
                    defect_type=defect_type,
                    grid_x=nx,
                    grid_y=ny,
                    dx_um=float(dx_m * 1e6),
                    n_sources=1,
                    n_sensors=args.sensors,
                    time_steps=traces_arr.shape[0],
                    frequency_hz=args.frequency,
                    attenuation_np_per_m=attenuation,
                    pd_thickness_nm=pd_thickness_nm_effective,
                    si_thickness_um=args.si_thickness_um,
                    pd_speed_m_per_s=args.pd_speed,
                    si_speed_m_per_s=args.si_speed,
                    source_x_px=int(source_x[0]),
                    source_y_px=int(source_y[0]),
                    defect_center_x_px=int(cx),
                    defect_center_y_px=int(cy),
                    defect_width_px=int(w),
                    defect_height_px=int(h),
                )
            )
        )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(input_dir / "metadata.csv", index=False)

    manifest = {
        "project": "jwave_keras_gui_workflow",
        "samples": len(rows),
        "grid_x": nx,
        "grid_y": ny,
        "dx_um": float(dx_m * 1e6),
        "frequency_hz": args.frequency,
        "structure_mode": args.structure_mode,
        "pd_thickness_nm": 0.0 if args.structure_mode == "pure_si" else args.pd_thickness_nm,
        "si_thickness_um": args.si_thickness_um,
        "pd_speed_m_per_s": args.pd_speed,
        "si_speed_m_per_s": args.si_speed,
        "notes": (
            "One-source numerical TRM dataset. "
            "Top-surface receiver line. "
            "refocus.csv stores the TRM max map for compatibility."
        ),
    }
    (art_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    logging.info("Finished generation of %d samples", len(rows))


if __name__ == "__main__":
    main()
