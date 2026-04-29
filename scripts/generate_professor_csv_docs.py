#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_data_dictionary(sample_id: str) -> pd.DataFrame:
    rows = [
        {
            "file_pattern": f"{sample_id}_traces.csv",
            "representation_name": "Forward measured traces",
            "data_type": "time-series matrix",
            "row_index_meaning": "time index n",
            "column_index_meaning": "measurement channel name src_XX_rx_YY",
            "value_meaning": "recorded acoustic amplitude at receiver YY from source XX",
            "typical_shape": "Nt x Nr (current setup: Nt x 16)",
            "units": "simulation amplitude units",
            "for_ml": "yes",
            "description": "Main measured data before time reversal. Each column is one receiver trace.",
        },
        {
            "file_pattern": f"{sample_id}_target.csv",
            "representation_name": "Binary target mask",
            "data_type": "2D image/matrix",
            "row_index_meaning": "depth pixel y",
            "column_index_meaning": "lateral pixel x",
            "value_meaning": "0 = healthy region, 1 = defect region",
            "typical_shape": "Ny x Nx",
            "units": "dimensionless",
            "for_ml": "yes (label)",
            "description": "Ground-truth defect mask used as label for localization or segmentation.",
        },
        {
            "file_pattern": f"{sample_id}_sound_speed.csv",
            "representation_name": "Sound-speed map",
            "data_type": "2D image/matrix",
            "row_index_meaning": "depth pixel y",
            "column_index_meaning": "lateral pixel x",
            "value_meaning": "local acoustic wave speed c(x,y)",
            "typical_shape": "Ny x Nx",
            "units": "m/s",
            "for_ml": "optional",
            "description": "Physical medium map used by the simulator.",
        },
        {
            "file_pattern": f"{sample_id}_density.csv",
            "representation_name": "Density map",
            "data_type": "2D image/matrix",
            "row_index_meaning": "depth pixel y",
            "column_index_meaning": "lateral pixel x",
            "value_meaning": "local material density rho(x,y)",
            "typical_shape": "Ny x Nx",
            "units": "kg/m^3",
            "for_ml": "optional",
            "description": "Material density distribution used in the simulation.",
        },
        {
            "file_pattern": f"{sample_id}_attenuation.csv",
            "representation_name": "Attenuation map",
            "data_type": "2D image/matrix",
            "row_index_meaning": "depth pixel y",
            "column_index_meaning": "lateral pixel x",
            "value_meaning": "local attenuation coefficient",
            "typical_shape": "Ny x Nx",
            "units": "Np/m (synthetic model units)",
            "for_ml": "optional",
            "description": "Synthetic attenuation distribution used by the simulator.",
        },
        {
            "file_pattern": f"{sample_id}_refocus.csv",
            "representation_name": "TRM max map",
            "data_type": "2D image/matrix",
            "row_index_meaning": "depth pixel y",
            "column_index_meaning": "lateral pixel x",
            "value_meaning": "max_t |p_TR(x,y,t)|",
            "typical_shape": "Ny x Nx",
            "units": "simulation amplitude units",
            "for_ml": "yes",
            "description": "Raw numerical time-reversal maximum-amplitude map.",
        },
        {
            "file_pattern": f"{sample_id}_trm_energy.csv",
            "representation_name": "TRM energy map",
            "data_type": "2D image/matrix",
            "row_index_meaning": "depth pixel y",
            "column_index_meaning": "lateral pixel x",
            "value_meaning": "sum_t |p_TR(x,y,t)|^2",
            "typical_shape": "Ny x Nx",
            "units": "simulation energy-like units",
            "for_ml": "yes",
            "description": "Integrated TRM energy representation.",
        },
        {
            "file_pattern": f"{sample_id}_refocus_log.csv",
            "representation_name": "TRM max log10 map",
            "data_type": "2D image/matrix",
            "row_index_meaning": "depth pixel y",
            "column_index_meaning": "lateral pixel x",
            "value_meaning": "log10(max(|TRM max|, eps))",
            "typical_shape": "Ny x Nx",
            "units": "log10(amplitude)",
            "for_ml": "yes",
            "description": "Log-compressed TRM maximum map for dynamic-range reduction.",
        },
        {
            "file_pattern": f"{sample_id}_trm_energy_log.csv",
            "representation_name": "TRM energy log10 map",
            "data_type": "2D image/matrix",
            "row_index_meaning": "depth pixel y",
            "column_index_meaning": "lateral pixel x",
            "value_meaning": "log10(max(TRM energy, eps))",
            "typical_shape": "Ny x Nx",
            "units": "log10(energy)",
            "for_ml": "yes",
            "description": "Log-compressed TRM energy map.",
        },
        {
            "file_pattern": f"{sample_id}_refocus_row_norm.csv",
            "representation_name": "TRM max row-normalized map",
            "data_type": "2D image/matrix",
            "row_index_meaning": "depth pixel y",
            "column_index_meaning": "lateral pixel x",
            "value_meaning": "TRM max divided by the maximum of each row",
            "typical_shape": "Ny x Nx",
            "units": "dimensionless",
            "for_ml": "yes",
            "description": "Contrast-enhanced TRM map. Useful to reveal lateral defect structure.",
        },
        {
            "file_pattern": f"{sample_id}_trm_energy_row_norm.csv",
            "representation_name": "TRM energy row-normalized map",
            "data_type": "2D image/matrix",
            "row_index_meaning": "depth pixel y",
            "column_index_meaning": "lateral pixel x",
            "value_meaning": "TRM energy divided by the maximum of each row",
            "typical_shape": "Ny x Nx",
            "units": "dimensionless",
            "for_ml": "yes",
            "description": "Row-normalized TRM energy representation for ML experiments.",
        },
        {
            "file_pattern": "metadata.csv",
            "representation_name": "Sample metadata table",
            "data_type": "tabular CSV",
            "row_index_meaning": "one row per sample",
            "column_index_meaning": "metadata field name",
            "value_meaning": "sample-level parameter or label",
            "typical_shape": "Nsamples x Nfields",
            "units": "mixed",
            "for_ml": "supporting metadata",
            "description": "Global information per sample: defect type, source position, grid, frequency, and geometry.",
        },
    ]
    return pd.DataFrame(rows)


def build_trace_columns_description(n_receivers: int = 16) -> pd.DataFrame:
    rows = []
    for rx in range(n_receivers):
        rows.append(
            {
                "column_name": f"src_00_rx_{rx:02d}",
                "meaning": f"Trace recorded at receiver {rx}",
                "source_index": 0,
                "receiver_index": rx,
                "description": f"Time-domain acoustic signal measured at receiver {rx} for the only physical pulsed-laser source.",
            }
        )
    return pd.DataFrame(rows)


def build_markdown_explanation(sample_id: str) -> str:
    return f"""# Data explanation for {sample_id}

## Coordinate convention
- Rows in 2D CSV files correspond to **depth** pixel `y`
- Columns in 2D CSV files correspond to **lateral** pixel `x`

## Main groups of files
1. **Physical medium**
   - `{sample_id}_sound_speed.csv`
   - `{sample_id}_density.csv`
   - `{sample_id}_attenuation.csv`

2. **Ground truth**
   - `{sample_id}_target.csv`

3. **Measured data**
   - `{sample_id}_traces.csv`

4. **Numerical time-reversal outputs**
   - `{sample_id}_refocus.csv`
   - `{sample_id}_trm_energy.csv`
   - `{sample_id}_refocus_log.csv`
   - `{sample_id}_trm_energy_log.csv`
   - `{sample_id}_refocus_row_norm.csv`
   - `{sample_id}_trm_energy_row_norm.csv`

## How to explain the key TRM files
- **TRM max raw**: stores the strongest back-propagated response at each pixel
- **TRM energy**: stores the time-integrated TRM energy at each pixel
- **log10 versions**: compress dynamic range to reveal weak structures
- **row-normalized versions**: normalize each depth row independently to enhance lateral defect visibility

## Recommended explanation to professor
The raw TRM maps preserve the direct physical amplitude information, while the log and row-normalized maps are derived representations created to improve interpretability and to test different neural-network input strategies.
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Create documentation tables for generated CSV files.")
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--sample-id", type=str, default="sample_0000")
    p.add_argument("--receivers", type=int, default=16)
    args = p.parse_args()

    root = args.project_root
    out_dir = root / "artifacts" / "professor_docs"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dict = build_data_dictionary(args.sample_id)
    trace_dict = build_trace_columns_description(args.receivers)
    md_text = build_markdown_explanation(args.sample_id)

    data_dict_path = out_dir / "csv_data_dictionary.csv"
    trace_dict_path = out_dir / "traces_column_dictionary.csv"
    md_path = out_dir / "professor_explanation.md"

    data_dict.to_csv(data_dict_path, index=False)
    trace_dict.to_csv(trace_dict_path, index=False)
    md_path.write_text(md_text, encoding="utf-8")

    print(f"Saved: {data_dict_path}")
    print(f"Saved: {trace_dict_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
