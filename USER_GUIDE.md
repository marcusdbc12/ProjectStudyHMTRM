# User Guide: j-Wave + Keras GUI Workflow

This project implements a **Keras/TensorFlow workflow** for synthetic data generation, dataset splitting, supervised inverse training, PINN-style training, validation, inference, and a **graphic front end with a live log panel**.

## Folder structure

- `input/` — generated CSV files for traces, targets, sound speed, density, attenuation, and refocus maps.
- `output/` — generated images for quick inspection.
- `artifacts/` — saved models, training histories, charts, manifests, predictions, and logs.
- `artifacts/logs/` — process logs shown by the GUI log viewer.
- `validation/` — validation samples copied after splitting.
- `test/` — test samples copied after splitting.
- `scripts/` — command-line scripts with adjustable options.
- `usecases/` — validation and inference use cases.
- `app/` — Streamlit GUI.

## What the GUI can do

The GUI provides buttons and controls for:

1. **Generate synthetic data** with parameter controls.
2. **Visualize synthetic data** from the generated dataset.
3. **Split the dataset** into train, validation, and test sets.
4. **Train a supervised inverse model** with Keras.
5. **Train a PINN-style model** with Keras and an additional physics penalty.
6. **Inspect saved models**.
7. **Run use cases / validation inference** on a selected sample.
8. **Watch the process logs live** in a log panel.

## Install

```bash
pip install -r requirements.txt
```

## Run the GUI

From the project root:

```bash
streamlit run app/streamlit_app.py
```

## Command-line workflow

### 1. Generate synthetic data

```bash
python scripts/generate_synthetic_data.py --samples 24 --grid 96 --dx 5e-6 --sources 12 --sensors 12 --frequency 25e6
```

Useful options:
- `--samples`
- `--grid`
- `--dx`
- `--sources`
- `--sensors`
- `--frequency`
- `--seed`
- `--min-defect-frac`
- `--max-defect-frac`

### 2. Split the dataset

```bash
python scripts/split_dataset.py --train-ratio 0.7 --validation-ratio 0.15 --test-ratio 0.15 --copy-test-files
```

### 3. Train the supervised inverse model

```bash
python scripts/train_supervised_keras.py --epochs 20 --batch-size 4 --learning-rate 1e-3
```

### 4. Train the PINN-style model

```bash
python scripts/train_pinn_keras.py --epochs 20 --batch-size 4 --learning-rate 1e-3 --physics-weight 0.15
```

### 5. Run a use case / validation inference

```bash
python usecases/validate_and_infer.py --sample-id sample_0000 --model-path artifacts/best_supervised.keras
```

## Notes about the PINN option

The PINN script here is a **practical PINN-style implementation** rather than a full PDE-residual solver coupled end-to-end through j-Wave. It uses:

- supervised reconstruction loss,
- a spatial smoothness / Laplacian penalty as the physics term.

This keeps the workflow manageable while still moving toward a physics-informed approach.

## Typical workflow inside the GUI

1. Generate data.
2. Inspect sample images.
3. Split the data.
4. Train the supervised model first.
5. Compare it with the PINN-style model.
6. Use the inference panel to validate on a chosen sample.
7. Watch the log viewer for detailed process status.

## Important project-specific scope

This workflow is adapted to the earlier design discussion:

- **Sample:** Pd(50 nm) / Si(300 µm)
- **Frequency:** 25 MHz fixed
- **Detectable features:** lateral micro-defects, typically around **100–300 µm**
- **Not feasible:** direct imaging of the 50 nm Pd thickness itself

## Main output files

- `artifacts/best_supervised.keras`
- `artifacts/best_pinn.keras`
- `artifacts/training_history_supervised.csv`
- `artifacts/training_history_pinn.csv`
- `artifacts/loss_supervised.png`
- `artifacts/accuracy_supervised.png`
- `artifacts/*prediction.png`
- `artifacts/logs/*.log`

#Professor csv files

python generate_professor_csv_docs.py \
  --project-root /Users/mbarrionuevo/Downloads/jwave_keras_gui_workflow \
  --sample-id sample_0000 \
  --receivers 16

#Professor plots

  python scripts/explain_trace_stack.py \
  --project-root /Users/mbarrionuevo/Downloads/jwave_keras_gui_workflow \
  --sample-id sample_0000 \
  --t-end 1.2e-7 \
  --frequency 25000000


  