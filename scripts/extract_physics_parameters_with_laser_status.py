#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def safe_literal_eval(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def extract_add_argument_defaults(script_path: Path) -> list[dict[str, Any]]:
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    rows = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_argument" and node.args:
                arg_name = safe_literal_eval(node.args[0])
                if not isinstance(arg_name, str):
                    continue
                entry = {
                    "argument": arg_name,
                    "default": None,
                    "type": None,
                    "help": None,
                    "choices": None,
                    "action": None,
                }
                for kw in node.keywords:
                    if kw.arg == "default":
                        entry["default"] = safe_literal_eval(kw.value)
                    elif kw.arg == "help":
                        entry["help"] = safe_literal_eval(kw.value)
                    elif kw.arg == "choices":
                        entry["choices"] = safe_literal_eval(kw.value)
                    elif kw.arg == "action":
                        entry["action"] = safe_literal_eval(kw.value)
                    elif kw.arg == "type":
                        if isinstance(kw.value, ast.Name):
                            entry["type"] = kw.value.id
                        elif isinstance(kw.value, ast.Attribute):
                            entry["type"] = kw.value.attr
                        else:
                            entry["type"] = type(kw.value).__name__
                rows.append(entry)
    rows.sort(key=lambda x: x["argument"])
    return rows


def get_default(args_df: pd.DataFrame, name: str):
    row = args_df.loc[args_df["argument"] == name]
    if len(row) == 0:
        return None
    return row.iloc[0]["default"]


def top_surface_sensor_positions(nx: int, sensors: int, y_index: int = 1):
    margin_x = max(4, nx // 10)
    x = np.linspace(margin_x, nx - 1 - margin_x, sensors, dtype=int)
    y = np.full(sensors, y_index, dtype=int)
    return y, x


def choose_source_position(nx: int, ny: int, mode: str, source_x_px: int | None, source_y_px: int):
    margin_x = max(4, nx // 10)
    if mode == "left":
        sx = margin_x
    elif mode == "right":
        sx = nx - 1 - margin_x
    elif mode == "custom":
        sx = nx // 2 if source_x_px is None else int(np.clip(source_x_px, 0, nx - 1))
    else:
        sx = nx // 2
    sy = int(np.clip(source_y_px, 0, ny - 1))
    return sy, sx


def build_physics_summary(args_df: pd.DataFrame) -> pd.DataFrame:
    dx_um = get_default(args_df, "--dx-um")
    t_end = get_default(args_df, "--t-end")
    cfl = get_default(args_df, "--cfl")
    si_speed = get_default(args_df, "--si-speed")
    pd_speed = get_default(args_df, "--pd-speed")
    grid_x = get_default(args_df, "--grid-x")
    grid_y = get_default(args_df, "--grid-y")
    frequency = get_default(args_df, "--frequency")
    sensors = get_default(args_df, "--sensors")
    sources = get_default(args_df, "--sources")
    pd_thickness_nm = get_default(args_df, "--pd-thickness-nm")
    si_thickness_um = get_default(args_df, "--si-thickness-um")

    dx_m = dx_um * 1e-6 if isinstance(dx_um, (int, float)) else None
    lx_um = grid_x * dx_um if isinstance(grid_x, (int, float)) and isinstance(dx_um, (int, float)) else None
    ly_um = grid_y * dx_um if isinstance(grid_y, (int, float)) and isinstance(dx_um, (int, float)) else None
    dt_s = cfl * dx_m / si_speed if all(isinstance(v, (int, float)) for v in [cfl, dx_m, si_speed]) else None
    time_steps_est = int(round(t_end / dt_s)) if all(isinstance(v, (int, float)) and v > 0 for v in [t_end, dt_s]) else None
    wavelength_si_um = (si_speed / frequency) * 1e6 if all(isinstance(v, (int, float)) and v > 0 for v in [si_speed, frequency]) else None
    wavelength_pd_um = (pd_speed / frequency) * 1e6 if all(isinstance(v, (int, float)) and v > 0 for v in [pd_speed, frequency]) else None
    ppw_si = wavelength_si_um / dx_um if all(isinstance(v, (int, float)) and v > 0 for v in [wavelength_si_um, dx_um]) else None
    ppw_pd = wavelength_pd_um / dx_um if all(isinstance(v, (int, float)) and v > 0 for v in [wavelength_pd_um, dx_um]) else None
    total_time_ns = t_end * 1e9 if isinstance(t_end, (int, float)) else None

    rows = [
        {"parameter": "grid_x", "value": grid_x, "units": "pixels", "meaning": "Number of lateral grid points"},
        {"parameter": "grid_y", "value": grid_y, "units": "pixels", "meaning": "Number of depth grid points"},
        {"parameter": "dx", "value": dx_um, "units": "um", "meaning": "Spatial step used by the solver"},
        {"parameter": "domain_width", "value": lx_um, "units": "um", "meaning": "Local modeled lateral size = grid_x * dx"},
        {"parameter": "domain_depth", "value": ly_um, "units": "um", "meaning": "Local modeled depth size = grid_y * dx"},
        {"parameter": "frequency", "value": frequency, "units": "Hz", "meaning": "Effective acoustic center frequency used by the current source model"},
        {"parameter": "si_speed", "value": si_speed, "units": "m/s", "meaning": "Silicon reference wave speed"},
        {"parameter": "pd_speed", "value": pd_speed, "units": "m/s", "meaning": "Palladium reference wave speed"},
        {"parameter": "pd_thickness", "value": pd_thickness_nm, "units": "nm", "meaning": "Physical Pd layer thickness"},
        {"parameter": "si_thickness", "value": si_thickness_um, "units": "um", "meaning": "Physical Si substrate thickness"},
        {"parameter": "cfl", "value": cfl, "units": "dimensionless", "meaning": "Courant number for stable time stepping"},
        {"parameter": "dt_estimated", "value": dt_s, "units": "s", "meaning": "Estimated time step from CFL * dx / c_max"},
        {"parameter": "dt_estimated_ns", "value": dt_s * 1e9 if isinstance(dt_s, (int, float)) else None, "units": "ns", "meaning": "Estimated time step in nanoseconds"},
        {"parameter": "t_end", "value": t_end, "units": "s", "meaning": "Total simulation time window per sample"},
        {"parameter": "t_end_ns", "value": total_time_ns, "units": "ns", "meaning": "Total simulation time window per sample in nanoseconds"},
        {"parameter": "time_steps_estimated", "value": time_steps_est, "units": "samples", "meaning": "Estimated number of time samples"},
        {"parameter": "wavelength_si", "value": wavelength_si_um, "units": "um", "meaning": "Silicon wavelength = c_si / f"},
        {"parameter": "wavelength_pd", "value": wavelength_pd_um, "units": "um", "meaning": "Pd wavelength = c_pd / f"},
        {"parameter": "ppw_si", "value": ppw_si, "units": "points/wavelength", "meaning": "Silicon wavelength resolution"},
        {"parameter": "ppw_pd", "value": ppw_pd, "units": "points/wavelength", "meaning": "Pd wavelength resolution"},
        {"parameter": "sources", "value": sources, "units": "count", "meaning": "Configured sources argument"},
        {"parameter": "sensors", "value": sensors, "units": "count", "meaning": "Number of top-line receivers"},
    ]
    return pd.DataFrame(rows)


def build_pulse_summary(args_df: pd.DataFrame) -> pd.DataFrame:
    frequency = get_default(args_df, "--frequency")
    t_end = get_default(args_df, "--t-end")
    source_center_frac = 0.28
    source_center_time = source_center_frac * t_end
    source_width_rule1 = 0.10 * t_end
    source_width_rule2 = 3.0 / frequency
    source_width_used = max(source_width_rule1, source_width_rule2)
    period_s = 1.0 / frequency
    margin_time = 10e-9

    rows = [
        {
            "parameter": "effective_acoustic_frequency",
            "value": frequency,
            "units": "Hz",
            "why_used": "Controls the oscillation rate of the simulated source waveform.",
            "do_we_use_it": "yes",
            "laser_side_status": "not a pure laser-only parameter in current code",
        },
        {
            "parameter": "period",
            "value": period_s,
            "units": "s",
            "why_used": "One acoustic cycle duration = 1/frequency.",
            "do_we_use_it": "derived from frequency",
            "laser_side_status": "derived quantity",
        },
        {
            "parameter": "source_center_fraction",
            "value": source_center_frac,
            "units": "fraction of t_end",
            "why_used": "Places the pulse away from t=0 so the waveform has quiet time before emission.",
            "do_we_use_it": "yes, implicit script constant",
            "laser_side_status": "effective waveform timing parameter",
        },
        {
            "parameter": "source_center_time",
            "value": source_center_time,
            "units": "s",
            "why_used": "Approximate center of the Gaussian-windowed pulse used by the source generator.",
            "do_we_use_it": "yes",
            "laser_side_status": "effective waveform timing parameter",
        },
        {
            "parameter": "window_rule_fraction",
            "value": source_width_rule1,
            "units": "s",
            "why_used": "One candidate width: 10% of the full simulation time window.",
            "do_we_use_it": "yes, in max(...) rule",
            "laser_side_status": "effective waveform width rule",
        },
        {
            "parameter": "window_rule_cycles",
            "value": source_width_rule2,
            "units": "s",
            "why_used": "Second candidate width: approximately three periods of the source frequency.",
            "do_we_use_it": "yes, in max(...) rule",
            "laser_side_status": "effective waveform width rule",
        },
        {
            "parameter": "window_width_used_in_source",
            "value": source_width_used,
            "units": "s",
            "why_used": "Actual width passed to the Gaussian source window = max(0.10*t_end, 3/f).",
            "do_we_use_it": "yes",
            "laser_side_status": "effective waveform width actually used",
        },
        {
            "parameter": "gate_margin_time",
            "value": margin_time,
            "units": "s",
            "why_used": "Extra safety delay added after estimated direct arrival before gating.",
            "do_we_use_it": "yes",
            "laser_side_status": "gate parameter, not laser physics",
        },
    ]
    return pd.DataFrame(rows)


def build_laser_side_status_summary(args_df: pd.DataFrame) -> pd.DataFrame:
    frequency = get_default(args_df, "--frequency")
    t_end = get_default(args_df, "--t-end")
    source_center_time = 0.28 * t_end
    source_width_used = max(0.10 * t_end, 3.0 / frequency)

    rows = [
        {
            "laser_side_quantity": "pulse_duration_tau_L",
            "available_in_code": "no, not explicitly",
            "current_proxy_or_closest_quantity": "window_width_used_in_source",
            "current_value": source_width_used,
            "units": "s",
            "comments": "The code does not store a true optical pulse duration. It only uses an effective waveform width for the simulated source pulse.",
        },
        {
            "laser_side_quantity": "repetition_rate",
            "available_in_code": "no",
            "current_proxy_or_closest_quantity": "none",
            "current_value": None,
            "units": "Hz",
            "comments": "The workflow simulates one excitation event per sample, not a repeated pulse train.",
        },
        {
            "laser_side_quantity": "optical_absorption_process",
            "available_in_code": "no, not explicitly",
            "current_proxy_or_closest_quantity": "none",
            "current_value": None,
            "units": "n/a",
            "comments": "The code does not include separate mu_a, fluence F, or Gruneisen parameter Gamma. It uses a direct effective acoustic source instead.",
        },
        {
            "laser_side_quantity": "effective_acoustic_source_frequency",
            "available_in_code": "yes",
            "current_proxy_or_closest_quantity": "--frequency",
            "current_value": frequency,
            "units": "Hz",
            "comments": "This is the main source-control parameter currently present in the code. It acts as an effective acoustic center frequency, not a full laser-side model.",
        },
        {
            "laser_side_quantity": "effective_pulse_center_time",
            "available_in_code": "yes",
            "current_proxy_or_closest_quantity": "0.28 * t_end",
            "current_value": source_center_time,
            "units": "s",
            "comments": "This controls where the simulated source pulse is centered inside the total time window.",
        },
    ]
    return pd.DataFrame(rows)


def build_probability_summary(args_df: pd.DataFrame) -> pd.DataFrame:
    prob_args = ["--p-no-defect", "--p-crack", "--p-void", "--p-missing-pd", "--p-delamination"]
    total = 0.0
    rows = []
    for name in prob_args:
        val = get_default(args_df, name)
        total += float(val) if val is not None else 0.0
    for name in prob_args:
        val = get_default(args_df, name)
        val = float(val) if val is not None else 0.0
        rows.append({
            "class_argument": name,
            "default_weight": val,
            "normalized_probability": val / total if total > 0 else None,
        })
    return pd.DataFrame(rows)


def build_sample_summary(project_root: Path) -> pd.DataFrame:
    metadata_path = project_root / "input" / "metadata.csv"
    manifest_path = project_root / "artifacts" / "dataset_manifest.json"
    rows = []

    if metadata_path.exists():
        meta = pd.read_csv(metadata_path)
        rows.append({"source": "metadata.csv", "field": "rows", "value": len(meta), "meaning": "Number of generated samples"})
        if "time_steps" in meta.columns:
            rows.append({"source": "metadata.csv", "field": "time_steps_first_sample", "value": meta["time_steps"].iloc[0], "meaning": "Actual time samples in first sample"})
        for col in ["source_x_px", "source_y_px", "grid_x", "grid_y", "frequency_hz", "defect_type"]:
            if col in meta.columns:
                rows.append({"source": "metadata.csv", "field": f"first_sample_{col}", "value": meta[col].iloc[0], "meaning": f"First-sample value of {col}"})

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ["samples", "grid_x", "grid_y", "dx_um", "frequency_hz", "source_position", "source_x_px", "source_y_px"]:
            if key in manifest:
                rows.append({"source": "dataset_manifest.json", "field": key, "value": manifest[key], "meaning": f"Manifest field {key}"})

    return pd.DataFrame(rows)


def build_gate_summary(args_df: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    grid_x = get_default(args_df, "--grid-x")
    grid_y = get_default(args_df, "--grid-y")
    dx_um = get_default(args_df, "--dx-um")
    si_speed = get_default(args_df, "--si-speed")
    sensors = get_default(args_df, "--sensors")
    t_end = get_default(args_df, "--t-end")
    cfl = get_default(args_df, "--cfl")
    source_position = get_default(args_df, "--source-position") or "center"
    source_x_px = get_default(args_df, "--source-x-px")
    source_y_px = get_default(args_df, "--source-y-px")
    if source_y_px is None:
        source_y_px = 1

    metadata_path = project_root / "input" / "metadata.csv"
    if metadata_path.exists():
        meta = pd.read_csv(metadata_path)
        first = meta.iloc[0]
        grid_x = int(first["grid_x"]) if "grid_x" in first else grid_x
        grid_y = int(first["grid_y"]) if "grid_y" in first else grid_y
        source_x_px = int(first["source_x_px"]) if "source_x_px" in first else source_x_px
        source_y_px = int(first["source_y_px"]) if "source_y_px" in first else source_y_px

    dx_m = dx_um * 1e-6
    dt_s = cfl * dx_m / si_speed
    source_center_time = 0.28 * t_end
    margin_time = 10e-9

    sy, sx = choose_source_position(int(grid_x), int(grid_y), source_position, source_x_px, int(source_y_px))
    sensor_y, sensor_x = top_surface_sensor_positions(int(grid_x), int(sensors), y_index=1)

    rows = []
    for r in range(int(sensors)):
        dx_r = abs(int(sensor_x[r]) - int(sx)) * dx_m
        dy_r = abs(int(sensor_y[r]) - int(sy)) * dx_m
        d_r = float(np.sqrt(dx_r * dx_r + dy_r * dy_r))
        t_direct = d_r / float(si_speed)
        gate_time = source_center_time + t_direct + margin_time
        gate_idx = max(0, int(np.floor(gate_time / dt_s)))
        rows.append({
            "receiver_index": r,
            "sensor_x_px": int(sensor_x[r]),
            "sensor_y_px": int(sensor_y[r]),
            "source_x_px": int(sx),
            "source_y_px": int(sy),
            "distance_m": d_r,
            "t_direct_s": t_direct,
            "t_direct_ns": t_direct * 1e9,
            "source_center_time_s": source_center_time,
            "source_center_time_ns": source_center_time * 1e9,
            "margin_time_s": margin_time,
            "margin_time_ns": margin_time * 1e9,
            "gate_time_s": gate_time,
            "gate_time_ns": gate_time * 1e9,
            "estimated_gate_index": gate_idx,
            "estimated_dt_ns": dt_s * 1e9,
            "estimated_total_time_ns": t_end * 1e9,
        })
    return pd.DataFrame(rows)


def build_markdown(args_df: pd.DataFrame, physics_df: pd.DataFrame, pulse_df: pd.DataFrame, laser_df: pd.DataFrame, probs_df: pd.DataFrame, gate_df: pd.DataFrame) -> str:
    return f"""# Physics, source, laser-side status, and gate extraction report

## What this script does
It scans `generate_synthetic_data.py`, extracts CLI defaults from `add_argument(...)`,
computes derived physical quantities, extracts pulsed-laser/source timing parameters,
adds a dedicated table describing which laser-side quantities are actually present
or missing, and computes the gate window per receiver using:

- `gate_time = source_center_time + t_direct + margin`

## Important interpretation
The current code uses an **effective acoustic source model**.
It does **not** yet contain a full separated model of:
- true optical pulse duration tau_L,
- repetition rate,
- optical absorption coefficient mu_a,
- fluence F,
- Gruneisen parameter Gamma.

## Main formulas
- `dt ≈ CFL * dx / c_max`
- `time_steps ≈ t_end / dt`
- `wavelength = c / f`
- `PPW = wavelength / dx`
- `source_center_time = 0.28 * t_end`
- `source_window_width = max(0.10 * t_end, 3/f)`
- `t_direct = distance / c_si`
- `gate_time = source_center_time + t_direct + margin`

## What to look at
- `physics_parameter_summary.csv` contains the full simulation time per sample
- `pulse_parameter_summary.csv` contains source timing choices and why they are used
- `laser_side_status_summary.csv` says what laser-side quantities exist or are missing
- `gate_window_summary.csv` contains the gate time for each receiver

## Counts
- extracted CLI arguments: {len(args_df)}
- derived physics quantities: {len(physics_df)}
- pulse/source quantities: {len(pulse_df)}
- laser-side status entries: {len(laser_df)}
- class-probability entries: {len(probs_df)}
- gate entries (receivers): {len(gate_df)}
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Extract all physics, pulsed-laser, numerical, and gate parameters from generate_synthetic_data.py")
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--script-path", type=Path, default=None)
    args = p.parse_args()

    project_root = args.project_root
    script_path = args.script_path or (project_root / "scripts" / "generate_synthetic_data.py")

    out_dir = project_root / "artifacts" / "physics_parameters"
    out_dir.mkdir(parents=True, exist_ok=True)

    args_df = pd.DataFrame(extract_add_argument_defaults(script_path))
    physics_df = build_physics_summary(args_df)
    pulse_df = build_pulse_summary(args_df)
    laser_df = build_laser_side_status_summary(args_df)
    probs_df = build_probability_summary(args_df)
    sample_df = build_sample_summary(project_root)
    gate_df = build_gate_summary(args_df, project_root)

    (out_dir / "cli_argument_defaults.csv").write_text(args_df.to_csv(index=False), encoding="utf-8")
    (out_dir / "physics_parameter_summary.csv").write_text(physics_df.to_csv(index=False), encoding="utf-8")
    (out_dir / "pulse_parameter_summary.csv").write_text(pulse_df.to_csv(index=False), encoding="utf-8")
    (out_dir / "laser_side_status_summary.csv").write_text(laser_df.to_csv(index=False), encoding="utf-8")
    (out_dir / "class_probability_summary.csv").write_text(probs_df.to_csv(index=False), encoding="utf-8")
    (out_dir / "dataset_sample_summary.csv").write_text(sample_df.to_csv(index=False), encoding="utf-8")
    (out_dir / "gate_window_summary.csv").write_text(gate_df.to_csv(index=False), encoding="utf-8")
    (out_dir / "physics_parameter_report.md").write_text(
        build_markdown(args_df, physics_df, pulse_df, laser_df, probs_df, gate_df),
        encoding="utf-8"
    )

    print(f"Saved: {out_dir / 'cli_argument_defaults.csv'}")
    print(f"Saved: {out_dir / 'physics_parameter_summary.csv'}")
    print(f"Saved: {out_dir / 'pulse_parameter_summary.csv'}")
    print(f"Saved: {out_dir / 'laser_side_status_summary.csv'}")
    print(f"Saved: {out_dir / 'class_probability_summary.csv'}")
    print(f"Saved: {out_dir / 'dataset_sample_summary.csv'}")
    print(f"Saved: {out_dir / 'gate_window_summary.csv'}")
    print(f"Saved: {out_dir / 'physics_parameter_report.md'}")


if __name__ == "__main__":
    main()
