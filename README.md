# Parkivo Part C Experiments

PyTorch-based experiment pipeline for UTS 42028 Assignment 3 Part C:
parking space occupancy detection using lecture-aligned CNNs trained from
scratch on PKLot.

## Project Layout

```text
data/
  raw/
  processed/
outputs/
  checkpoints/
  plots/
  reports/
src/
  config.py
  dataset.py
  evaluate.py
  models.py
  run_experiments.py
  train.py
  utils.py
```

## Environment Setup

Create and activate a virtual environment in VS Code or the terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

For this RTX 5080 system, install the official CUDA 12.8 PyTorch wheels first,
then install the remaining dependencies:

```bash
.venv/bin/python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
```

If you are using the project virtual environment, prefer:

```bash
source .venv/bin/activate
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Verify real CUDA usability, not just `torch.cuda.is_available()`:

```bash
.venv/bin/python - <<'PY'
import torch
x = torch.randn(16, 3, 128, 128, device="cuda")
conv = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1).to("cuda")
y = conv(x)
loss = y.square().mean()
loss.backward()
torch.cuda.synchronize()
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), tuple(y.shape))
PY
```

## Dataset Setup

The pipeline expects parking-space crops organized as binary classes:

```text
data/processed/pklot_binary/
  empty/
  occupied/
```

### Preferred path: prepare from Hugging Face PKLot

The code first tries to support PKLot from:

`https://huggingface.co/datasets/Voxel51/PKLot`

This path uses `fiftyone` to load the dataset and export parking-space crops
into `data/processed/pklot_binary/`.

Run:

```bash
.venv/bin/python src/run_experiments.py --prepare-only
```

If Hugging Face preparation succeeds, the processed dataset is built
automatically and the experiment command below can be run immediately.

### Fallback path: manual dataset setup

If the Hugging Face preparation path is unavailable or too heavy on your
machine, manually place PKLot parking-space crops into:

```text
data/processed/pklot_binary/empty/
data/processed/pklot_binary/occupied/
```

The pipeline does not fake data. Training only runs when a real dataset is
available.

## Run All Experiments

The current active comparison set is:

- LeNet-5 CNN
- AlexNet CNN
- ResNet-18 CNN

The final trained checkpoints already exist under `outputs/checkpoints/`.
Do not rerun training unless you intentionally want to replace those results.

```bash
.venv/bin/python src/run_experiments.py --batch-size 512 --epochs 15 --image-size 128 --num-workers -1
```

Useful options:

```bash
.venv/bin/python src/run_experiments.py --batch-size 512 --epochs 15 --image-size 128 --num-workers -1
.venv/bin/python src/run_experiments.py --auto-batch-size --batch-size-candidates 512 1024 1536 2048 2304 2560
.venv/bin/python src/run_experiments.py --max-samples-per-class 2500
.venv/bin/python src/run_experiments.py --prepare-only
```

## Reuse Trained Models

Use `load_trained_model` to load the saved lecture-aligned checkpoints without
retraining:

```python
from pathlib import Path
from src.models import load_trained_model

project_root = Path(".").resolve()
model = load_trained_model("resnet18", image_size=128, project_root=project_root)
```

Checkpoint mapping:

- `lenet5` -> `outputs/checkpoints/lenet_5_cnn.pt`
- `alexnet` -> `outputs/checkpoints/alexnet_cnn.pt`
- `resnet18` -> `outputs/checkpoints/resnet_18_cnn.pt`

GitHub publication note: all three active checkpoints are included in the
private repository. The 228 MB AlexNet checkpoint is tracked with Git LFS
because GitHub rejects regular blobs over 100 MB.

## Part D Streamlit GUI Prototype

The Part D GUI prototype is implemented in `app_streamlit.py` for the Parkivo
parking availability system. It is a local Streamlit dashboard for a university
assignment demo, not a production backend.

The current Part D demo is still-frame based:

```text
clean full-frame PKLot image
  -> official PKLot polygon ROI map
  -> entry-gate trigger
  -> frame capture from the monitored parking zone
  -> CNN classification on each cropped slot
  -> availability map
  -> driver display recommendation
```

The active GUI uses `assets/demo_frame_clean.jpg` as the default frame and
`assets/custom_slot_map.json` as the official PKLot/Voxel51 ROI map. The map
contains 68 monitored slots from the selected parking zone. The trained LeNet-5,
AlexNet, and ResNet-18 checkpoints perform image classification only on cropped
parking-slot images; the GUI does not convert the project into an object
detection system.

The GUI includes:

- Dashboard with recommendation, metrics, clean frame ROI overlay, and slot grid
- Detection Demo for full-frame upload/default-frame CNN detection
- read-only ROI Calibration summary for the official PKLot ROI map
- concise methodology notes explaining fixed-camera ROI image classification

The GUI does not require `data/processed/pklot_binary` to exist. Users can
upload a full parking-zone frame for ROI cropping/classification. The previous
video/time-lapse asset is no longer the core demo path.

Do not use external parking-system source code, copied GUI implementations,
copied model pipelines, or external project logic.

Use the Conda `ai` environment on this machine:

```powershell
C:\Users\pc\miniconda3\envs\ai\python.exe -m streamlit run app_streamlit.py --server.port 8501
```

If Streamlit is missing, install it first:

```powershell
C:\Users\pc\miniconda3\envs\ai\python.exe -m pip install streamlit
```

ROI calibration for the submitted demo is based on official PKLot polygon
annotations and is read-only in the main GUI to avoid overwriting the official
map.

## Outputs

After a successful run, the pipeline saves:

- model checkpoints in `outputs/checkpoints/`
- training plots in `outputs/plots/`
- results table in `outputs/reports/lecture_aligned_initial_results.csv`
- Part C markdown summary in `outputs/reports/lecture_aligned_part_c_ready.md`
- recommended summary in `outputs/reports/recommended_lecture_aligned_part_c_ready.md`

For GitHub, the committed outputs are the active lecture-aligned reports,
training histories, plots, and all three active checkpoints. The local virtual
environment, processed PKLot images, Python caches, temporary logs, legacy
report/plot archives, and legacy checkpoint files are excluded to keep the
repository practical for a standard GitHub push.

## Notes

- Framework: PyTorch
- GPU-first runtime: CUDA smoke test, AMP, channels-last, cuDNN benchmark
- Full-data training keeps all processed PKLot crops and uses simple positive-class weighting instead of downsampling
- Task type: image classification
- Classes: `occupied` vs `empty`
- Split: 70% train / 15% validation / 15% test
- Optimizer: Adam
- Loss: `BCEWithLogitsLoss`
- Models: LeNet-5 CNN, AlexNet CNN, and ResNet-18 CNN from scratch
- Early stopping is enabled by default for the active model set

If CUDA is visible but fails the built-in Conv2d probe, the pipeline falls back
to a non-CUDA device. On this machine the recommended target is the official
PyTorch `cu128` build.
