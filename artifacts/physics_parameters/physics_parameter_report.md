# Physics, source, laser-side status, and gate extraction report

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
- extracted CLI arguments: 28
- derived physics quantities: 22
- pulse/source quantities: 8
- laser-side status entries: 5
- class-probability entries: 5
- gate entries (receivers): 16
