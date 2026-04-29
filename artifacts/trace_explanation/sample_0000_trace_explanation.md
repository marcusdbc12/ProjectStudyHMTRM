# Trace explanation for sample_0000

## Demonstration values vs simulation values
The hand-sketch coordinate values given before were **demonstration values only**.
They were made to help reproduce the V-shape by hand.
They were **not** applied in the Python simulation code.

The real plotted data come from:
- `sample_0000_traces.csv`

## Gate-window bug explanation
Originally, the code used

- `t_center = 0.28 * t_end`
- `source_width = max(0.10 * t_end, 3/frequency)`
- `gate_time = t_center + t_direct + source_width + margin`

With
- `t_end = 1.200e-07 s`
- `frequency = 2.500e+07 Hz`

the term
- `3/f = 1.200e-07 s`

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
