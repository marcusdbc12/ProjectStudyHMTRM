# Data explanation for sample_0000

## Coordinate convention
- Rows in 2D CSV files correspond to **depth** pixel `y`
- Columns in 2D CSV files correspond to **lateral** pixel `x`

## Main groups of files
1. **Physical medium**
   - `sample_0000_sound_speed.csv`
   - `sample_0000_density.csv`
   - `sample_0000_attenuation.csv`

2. **Ground truth**
   - `sample_0000_target.csv`

3. **Measured data**
   - `sample_0000_traces.csv`

4. **Numerical time-reversal outputs**
   - `sample_0000_refocus.csv`
   - `sample_0000_trm_energy.csv`
   - `sample_0000_refocus_log.csv`
   - `sample_0000_trm_energy_log.csv`
   - `sample_0000_refocus_row_norm.csv`
   - `sample_0000_trm_energy_row_norm.csv`

## How to explain the key TRM files
- **TRM max raw**: stores the strongest back-propagated response at each pixel
- **TRM energy**: stores the time-integrated TRM energy at each pixel
- **log10 versions**: compress dynamic range to reveal weak structures
- **row-normalized versions**: normalize each depth row independently to enhance lateral defect visibility

## Recommended explanation to professor
The raw TRM maps preserve the direct physical amplitude information, while the log and row-normalized maps are derived representations created to improve interpretability and to test different neural-network input strategies.
