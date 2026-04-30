from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from task_runner import run_and_stream

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
LOGS = ROOT / 'artifacts' / 'logs'

st.set_page_config(page_title='j-Wave Keras Workflow', layout='wide')
st.title('j-Wave + Keras Workflow for Pd/Si Interface Defect Imaging')
st.caption('GUI for synthetic data generation, visualization, splitting, training, validation, and inference.')

left, right = st.columns([1.1, 1.0])
with left:
    st.header('1. Generate synthetic data')
    st.info('The previous defaults were placeholders for a quick demo. For real training, 500–2000 samples is a much more sensible starting range.')
    samples = st.number_input('Samples', min_value=10, max_value=20000, value=1000, step=100)
    col1, col2 = st.columns(2)
    grid_x = col1.number_input('Grid X (pixels)', min_value=32, max_value=512, value=128, step=8)
    grid_y = col2.number_input('Grid Y (pixels)', min_value=32, max_value=512, value=96, step=8)
    dx_um = st.number_input('dx (µm)', min_value=0.1, max_value=100.0, value=5.0, step=0.5, help='Grid spacing in micrometers. Physical width = Grid X × dx. Physical height = Grid Y × dx.')
    st.caption(f'Domain size: {grid_x * dx_um:.1f} µm × {grid_y * dx_um:.1f} µm. Comment on dx: dx converts pixels into physical length. Example: 128 × 5 µm = 640 µm. Choose dx small enough to sample the wavelength and defect geometry, but large enough to keep runtime manageable.')
    pd_thickness_nm = st.number_input('Pd thickness (nm)', min_value=1.0, max_value=5000.0, value=50.0, step=5.0, help='Physical palladium film thickness. In this workflow it is modeled as an effective near-surface/interface layer because 50 nm is far below the 25 MHz wavelength.')
    si_thickness_um = st.number_input('Si thickness (µm)', min_value=10.0, max_value=5000.0, value=300.0, step=10.0, help='Physical silicon substrate thickness.')
    pd_speed = st.number_input('Pd wave speed (m/s)', min_value=100.0, max_value=20000.0, value=3070.0, step=10.0, help='Typical bulk longitudinal/extensional reference value used here for palladium.')
    si_speed = st.number_input('Si wave speed (m/s)', min_value=100.0, max_value=20000.0, value=8433.0, step=10.0, help='Typical bulk longitudinal reference value used here for silicon.')
    st.caption('Default comments: Pd ≈ 3070 m/s, Si ≈ 8433 m/s. These are reference values for the synthetic generator and can be adjusted for your study.')
    sources = st.number_input('Sources', min_value=1, max_value=128, value=1, step=1)
    sensors = st.number_input('Sensors', min_value=4, max_value=128, value=16, step=1)
    freq_unit = st.selectbox('Frequency unit', ['MHz', 'GHz'], index=0)
    freq_value = st.number_input(f'Frequency ({freq_unit})', min_value=0.1, max_value=10_000.0, value=25.0 if freq_unit == 'MHz' else 0.025, step=1.0 if freq_unit == 'MHz' else 0.01)
    frequency_hz = freq_value * (1e6 if freq_unit == 'MHz' else 1e9)
    st.caption(f'Effective frequency: {frequency_hz:.3e} Hz')
    st.markdown('**Pulsed-laser source position**')
    src_col1, src_col2, src_col3 = st.columns(3)
    source_position = src_col1.selectbox('Source position', ['center', 'left', 'right', 'custom'], index=0, help='Top-surface pulsed-laser location.')
    source_x_px = src_col2.number_input('Custom source x (px)', min_value=0, max_value=int(grid_x - 1), value=int(grid_x // 2), step=1, disabled=(source_position != 'custom'))
    source_y_px = src_col3.number_input('Source y (px)', min_value=0, max_value=int(grid_y - 1), value=1, step=1)
    if st.button('Generate synthetic data'):
        cmd = [
            sys.executable, str(SCRIPTS / 'generate_synthetic_data.py'),
            '--project-root', str(ROOT),
            '--samples', str(samples),
            '--grid-x', str(grid_x),
            '--grid-y', str(grid_y),
            '--dx-um', str(dx_um),
            '--sources', str(sources),
            '--sensors', str(sensors),
            # FIX: script no soporta frequency
            # '--frequency', str(frequency_hz),
            '--pd-thickness-nm', str(pd_thickness_nm),
            '--si-thickness-um', str(si_thickness_um),
            '--pd-speed', str(pd_speed),
            '--si-speed', str(si_speed),
            '--source-position', str(source_position),
            '--source-y-px', str(int(source_y_px)),
        ]
        if source_position == 'custom':
            cmd.extend(['--source-x-px', str(int(source_x_px))])
        cmd.append('--overwrite')
        log_box = st.empty(); lines = []
        try:
            for line in run_and_stream(cmd, LOGS / 'generate.log'):
                lines.append(line)
                log_box.code('\n'.join(lines[-60:]), language='text')
            st.success('Synthetic data generated')
        except Exception as exc:
            log_box.code('\n'.join(lines[-60:]), language='text')
            st.error(str(exc))

    st.header('2. Visualize synthetic data')
    metadata_path = ROOT / 'input' / 'metadata.csv'
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        st.dataframe(metadata.head(20), use_container_width=True)
        sample_id = st.selectbox('Sample to visualize', metadata['sample_id'].tolist())
        c1, c2 = st.columns(2)
        for img_path, col in [(ROOT / 'output' / f'{sample_id}_summary.png', c1), (ROOT / 'output' / f'{sample_id}_trace.png', c2)]:
            if img_path.exists():
                col.image(Image.open(img_path), caption=img_path.name, use_container_width=True)
    else:
        st.info('Generate data first to populate visualizations.')

    st.header('3. Split dataset')
    train_ratio = st.slider('Train ratio', 0.1, 0.9, 0.7, 0.05)
    val_ratio = st.slider('Validation ratio', 0.05, 0.45, 0.15, 0.05)
    test_ratio = round(1.0 - train_ratio - val_ratio, 2)
    st.write(f'Test ratio: {test_ratio:.2f}')
    if st.button('Split data'):
        cmd = [
            sys.executable, str(SCRIPTS / 'split_dataset.py'),
            '--project-root', str(ROOT),
            '--train-ratio', str(train_ratio),
            '--validation-ratio', str(val_ratio),
            '--test-ratio', str(test_ratio),
            '--copy-test-files'
        ]
        log_box = st.empty(); lines = []
        try:
            for line in run_and_stream(cmd, LOGS / 'split.log'):
                lines.append(line)
                log_box.code('\n'.join(lines[-60:]), language='text')
            st.success('Dataset split completed')
        except Exception as exc:
            log_box.code('\n'.join(lines[-60:]), language='text')
            st.error(str(exc))

with right:
    st.header('4. Train models')
    epochs = st.number_input('Epochs', 1, 500, 25)
    batch_size = st.number_input('Batch size', 1, 256, 8)
    lr = st.number_input('Learning rate', value=1e-3, format='%.6f')
    physics_weight = st.number_input('PINN physics weight', value=0.15, format='%.4f')
    c1, c2 = st.columns(2)
    if c1.button('Train supervised inverse model'):
        cmd = [sys.executable, str(SCRIPTS / 'train_supervised_keras.py'), '--project-root', str(ROOT), '--epochs', str(epochs), '--batch-size', str(batch_size), '--learning-rate', str(lr)]
        lines = []; box = st.empty()
        try:
            for line in run_and_stream(cmd, LOGS / 'train_supervised.log'):
                lines.append(line)
                box.code('\n'.join(lines[-60:]), language='text')
            st.success('Supervised training completed')
        except Exception as exc:
            box.code('\n'.join(lines[-60:]), language='text')
            st.error(str(exc))
    if c2.button('Train PINN-style model'):
        cmd = [sys.executable, str(SCRIPTS / 'train_pinn_keras.py'), '--project-root', str(ROOT), '--epochs', str(epochs), '--batch-size', str(batch_size), '--learning-rate', str(lr), '--physics-weight', str(physics_weight)]
        lines = []; box = st.empty()
        try:
            for line in run_and_stream(cmd, LOGS / 'train_pinn.log'):
                lines.append(line)
                box.code('\n'.join(lines[-60:]), language='text')
            st.success('PINN training completed')
        except Exception as exc:
            box.code('\n'.join(lines[-60:]), language='text')
            st.error(str(exc))

    st.subheader('Training curves')
    for img_name in ['loss_supervised.png', 'accuracy_supervised.png', 'loss.png', 'val_loss.png', 'physics.png', 'val_physics.png']:
        path = ROOT / 'artifacts' / img_name
        if path.exists():
            st.image(str(path), caption=img_name, use_container_width=True)

    st.header('5. Save / inspect models')
    models = list((ROOT / 'artifacts').glob('*.keras'))
    st.write([m.name for m in models] or 'No .keras models yet')

    st.header('6. Validation and use cases')
    available = [file.name.replace('_target.csv', '') for file in (ROOT / 'input').glob('*_target.csv')]
    selected_sample = st.selectbox('Sample for use case', sorted(available) if available else ['sample_0000'])
    model_choice = st.selectbox('Model', [m.name for m in models] if models else ['best_supervised.keras'])
    if st.button('Run use case / validation inference'):
        cmd = [sys.executable, str(ROOT / 'usecases' / 'validate_and_infer.py'), '--project-root', str(ROOT), '--sample-id', selected_sample, '--model-path', str(ROOT / 'artifacts' / model_choice)]
        lines = []; box = st.empty()
        try:
            for line in run_and_stream(cmd, LOGS / 'usecase.log'):
                lines.append(line)
                box.code('\n'.join(lines[-60:]), language='text')
            st.success('Inference completed')
        except Exception as exc:
            box.code('\n'.join(lines[-60:]), language='text')
            st.error(str(exc))

    pred_path = ROOT / 'artifacts' / f'{selected_sample}_prediction.png'
    if pred_path.exists():
        st.image(str(pred_path), caption='Inference output', use_container_width=True)

st.header('Live log viewer')
log_files = sorted(LOGS.glob('*.log'))
selected_log = st.selectbox('Log file', [p.name for p in log_files] if log_files else ['No logs yet'])
if log_files:
    st.code((LOGS / selected_log).read_text(encoding='utf-8')[-16000:], language='text')
