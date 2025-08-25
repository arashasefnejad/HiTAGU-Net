# HiTAGU-Net — Q1-Ready
Reference PyTorch implementation for **HiTAGU‑Net** (Hierarchical Temporal‑Aware Gated Units) for human action recognition on **Kinetics‑700** and **Something‑Something V2** under **single‑clip/single‑crop** and **multi‑clip/multi‑crop** protocols.

> **Pipeline:** ResNet‑50 (truncated) → Motion Difference Embedding (MDE) → Three‑State Temporal Unit (3STU) → Temporal Self‑Attention + Spatial Channel Attention → Classifier (+ Center/Temporal Loss, optional).

---

## Key Features
- Modular PyTorch codebase with clear components (backbone, MDE, 3STU, attention, classifier).
- Training/Evaluation scripts for single‑clip and multi‑clip protocols.
- Optional **CenterLoss** and **Temporal Consistency** loss via YAML switches.
- TensorBoard logging, seed control, AMP, gradient clipping.
- Reproducible environment: `environment.yml` and `requirements.txt`.
- CI workflow and smoke test for forward‑pass on CPU.
- Data prep utilities (ffmpeg frame extraction + CSV builder).

---

## Environment
You can use either Conda or pip:

### Conda (recommended)
```bash
conda env create -f environment.yml
conda activate hitagu
```

### Pip
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

**Reference setup** (for reviewers): Python 3.10, PyTorch 2.1, torchvision 0.16, CUDA 11.8, cuDNN (see `environment.yml`).

---

## Data Preparation
Input format is **RGB frames per video** stored as JPEGs: `000001.jpg, 000002.jpg, ...`

1) **Extract frames** with ffmpeg:
```bash
bash tools/frames_from_video.sh /path/to/input.mp4 /path/to/output_frames_dir
```

2) **Build CSV** files (Kinetics‑style directory tree `root/class/video_id/frames`):
```bash
python tools/make_csv.py --root /data/kinetics700_frames/train --out /data/kinetics700/train.csv
python tools/make_csv.py --root /data/kinetics700_frames/val   --out /data/kinetics700/val.csv
```

Each CSV row is:
```
relative/path/to/frames_dir,label_index,num_frames
```

3) **Edit config paths**:
Update `data.root`, `train_list`, and `val_list` in `configs/k700.yaml` or `configs/ssv2.yaml`.

---

## Training
Single‑clip/single‑crop training (default):
```bash
python scripts/train.py --config configs/k700.yaml
# or
python scripts/train.py --config configs/ssv2.yaml
```

**Optional losses** in YAML (`train` section):
```yaml
center_loss: true
lambda_center: 0.1
temporal_loss: true
lambda_temporal: 0.05
```

TensorBoard logs are written to `checkpoints/logs`:
```bash
tensorboard --logdir checkpoints/logs
```

---

## Evaluation
**Single‑clip/single‑crop** (fast):
```bash
python scripts/evaluate.py --config configs/k700.yaml --checkpoint checkpoints/best.pt
```

**Multi‑clip / Multi‑crop** (paper‑style, slower):
```bash
python scripts/evaluate_multiclip.py   --config configs/k700.yaml   --checkpoint checkpoints/best.pt   --video_folder /path/to/one_video_frames   --num_clips 10
```

---

## Project Structure
```
HiTAGU-Net/
├─ configs/                # YAML configs for datasets/experiments
├─ docs/                   # high‑res figures for the paper (add your PNG/SVG here)
├─ hitagu_net/
│  ├─ data/                # datasets & transforms
│  ├─ engine/              # training loop, losses, logging, utils
│  └─ models/              # backbone, MDE, 3STU, attention, classifier, wrapper
├─ scripts/                # train / eval / multi‑clip eval
├─ tests/                  # smoke test (forward pass on CPU)
├─ tools/                  # ffmpeg frame extraction + CSV builder
├─ .github/workflows/      # CI for CPU smoke test
├─ requirements.txt
├─ environment.yml
├─ CITATION.cff
└─ README.md
```

---

## Checkpoints
By default, the best model is saved to `checkpoints/best.pt`.  
Pretrained weights for K700/SSv2 are **not included** here (training is time‑consuming), but once you upload them to a public host, you can link them in this README for reviewers.

---

## Reproducibility Notes
- Seeds are set in `engine/utils.py`.
- AMP and gradient clipping are enabled by default.
- For paper‑level results, also report multi‑clip/multi‑crop evaluation.

---

## Citation
If you use this codebase, please cite the accompanying manuscript (update `CITATION.cff` with your details).

---

## Pretrained Checkpoints (K700 / SSv2)
To help reviewers run evaluation quickly, place your trained weights here:

```bash
# Option A: automatic (edit URLs in the script first)
bash tools/download_checkpoints.sh checkpoints/

# Option B: manual
# Put your files as:
#   checkpoints/k700_best.pt
#   checkpoints/ssv2_best.pt
```

Then evaluate:
```bash
python scripts/eval.py --config configs/k700.yaml --checkpoint checkpoints/k700_best.pt
python scripts/eval.py --config configs/ssv2.yaml --checkpoint checkpoints/ssv2_best.pt
```
