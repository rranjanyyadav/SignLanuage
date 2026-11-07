# Indian Sign Language Live Recognition & Evaluation Suite

A production-grade Indian Sign Language (ISL) alphabet recognition system supporting dual pipelines:
1. **Geometric Hand-Landmark Pipeline** (MediaPipe 21-point tracking + PyTorch MLP)
2. **Deep Image Classification Pipeline** (MediaPipe-guided Hand Cropping + 4-Block CNN & ResNet-18)
3. **Interactive Text Composer** (Real-time sign holding, debounced letter commitment, and word composition)

---

## Repository Structure

```text
signlang/
├── app/                              # Live camera inference & interactive applications
│   ├── live_image_inference.py       # Production ResNet/CNN live recognizer
│   ├── live_landmark_inference.py    # MediaPipe LandmarkMLP live recognizer
│   ├── text_composer.py              # Desktop sign-to-text composer application
│   ├── model_config.py               # Active-model registry & checkpoint resolution
│   └── find_cameras.py               # Camera backend and device index diagnostic
│
├── training/                         # Model training pipelines
│   ├── train_image_models.py         # Trains CNN & ResNet-18 on image datasets
│   └── train_landmark_model.py       # Trains LandmarkMLP on CSV landmark vectors
│
├── evaluation/                       # Audit, benchmarking, and visualization
│   ├── audit_alphabet.py             # Interactive A-Z audit harness with H & X safeguards
│   ├── evaluate_models.py            # Offline & live benchmark evaluation
│   └── plot_training_curves.py       # Loss & accuracy presentation curve generator
│
├── data_preparation/                 # Preprocessing & dataset generation
│   ├── prepare_cropped_dataset.py    # Hand bounding box, crop & mask pipeline
│   ├── capture_webcam_dataset.py     # Interactive webcam dataset capture utility
│   └── generate_skeleton_dataset.py  # Skeleton visualization dataset generator
│
├── docs/                             # Project documentation, guides, and proposals
│   └── Deep Learning - PBL.docx      # Project proposal & Phase I documentation draft
│
├── tests/                            # Automated regression test suite (52 tests)
│   ├── test_audit.py
│   ├── test_composer.py
│   ├── test_inference.py
│   ├── test_models.py
│   ├── test_plot_curves.py
│   ├── test_preprocessing.py
│   └── test_splitting.py
│
├── models/                           # Checkpoints (git-ignored, see Model Distribution)
│   ├── active_model.json             # Written by switch_model.py
│   ├── isl_resnet_large_finetuned.pth# ResNet-18 Large + webcam fine-tune (current active)
│   ├── isl_resnet_large.pth          # ResNet-18 on 49k masked crops (74.63% block test)
│   ├── isl_resnet_balanced.pth       # Experiment B, session/demographic balanced (77.59%)
│   ├── isl_resnet.pth                # Baseline ResNet-18 (legacy mask, 68.19%)
│   ├── isl_resnet_fine.pth           # Experimental ResNet-18 (fine mask, 67.65%)
│   ├── isl_webcam_resnet.pth         # ResNet-18 trained on webcam_dataset
│   ├── isl_cnn.pth                   # 4-block CNN
│   ├── isl_skeleton_cnn.pth          # CNN on skeleton renderings
│   ├── isl_alphabet_cnn_v2/v3/v4.pth # Earlier CNN iterations (kept for comparison)
│   └── isl_landmark_mlp.pth          # Geometric LandmarkMLP
│
├── manifests/                        # Fixed train/val/test file lists (Experiment B)
│
├── reports/                          # Submission evidence & evaluation metrics
│   ├── training_loss_curve.png       # Generated Phase I loss plot (300 DPI)
│   ├── training_accuracy_curve.png   # Generated Phase I accuracy plot (300 DPI)
│   ├── image_model_results.json      # 15-epoch training history
│   ├── phase3_resnet_fine_evaluation.json
│   ├── image_model_large*_results.json / image_model_balanced_results.json
│   ├── test_evaluation_*_detailed.json   # Per-class metrics & confusion pairs
│   ├── head_to_head_*                # Balanced-test model comparison
│   └── alphabet_audit/               # Audit records and contact sheets
│
├── switch_model.py                   # CLI to view / change the active image model
└── [Root Launchers]                  # Backward-compatible entrypoints for all scripts
```

---

## Architecture Overview

```text
Pipeline 1: Landmark Geometry (Recommended for real-time webcam)
  Webcam Frame
    -> MediaPipe HandLandmarker (detects up to 2 hands)
    -> 21 (x, y, z) landmarks per hand
    -> Wrist-relative translation & max-distance scale normalization
    -> Two-hand indicator feature (127-dimensional vector)
    -> LandmarkMLP (256 -> 128 -> 26)
    -> Temporal smoothing (deque majority vote) & confidence thresholding
    -> Final Class [A-Z] / Uncertain State

Pipeline 2: Deep Image Classification
  Webcam Frame / Dataset Image
    -> MediaPipe Hand Region Detection (Bounding Box + EMA jitter smoothing)
    -> Square crop with 25% padding (Single-hand or Dual-hand bounding)
    -> Resize to 224x224 & ImageNet normalization
    -> CNN (4-Block spatial architecture) / ResNet-18 Transfer Learning
    -> Temporal smoothing & confidence thresholding
    -> Final Class [A-Z] / Uncertain State
```

---

## Key Improvements & Problem Resolutions

- **Clean Modular Layout**: Organized into `app/`, `training/`, `evaluation/`, `data_preparation/`, `docs/`, `models/`, and `reports/`, with backward-compatible root launchers.
- **Preprocessing Alignment**: Resolved the mismatch where image models were trained on full-frame backgrounds while live inference cropped hands. Added `prepare_cropped_dataset.py` and auto-mode detection in checkpoints and inference.
- **Leakage-Safe Dataset Splitting**: Removed data leakage in `train_landmark_model.py` (which previously applied hand-swapping before shuffling). Implemented pre-augmentation splitting and session-based `block` holdouts to prevent video-frame leakage.
- **Improved CNN Architecture**: Replaced the 128-channel 1x1 pooling CNN with a 4-block architecture retaining 4x4 spatial resolution before classification, boosting representational capacity on complex finger signs (F, E, D, Q, T).
- **Temporal Smoothing & Confidence Handling**: Both landmark and image inference now feature configurable temporal smoothing buffers (`--history-size 7`), majority consensus checks, and an "Uncertain" state below confidence thresholds (`--min-confidence 0.60`).
- **Interactive Text Composer**: Added `app/text_composer.py` providing a live text composition window with stable letter confirmation, space/undo/clear buttons, and disarm cooldowns.
- **Automated Test Suite**: Comprehensive 52-test regression suite covering preprocessing, normalization invariance, bounding box clamping, split isolation, letter guidance safeguards, and model architectures.

---

## Installation & Setup

### Requirements
- Windows 10/11
- Python 3.10 or newer
- Webcam (built-in, USB, or virtual camera like Camo Studio)
- CUDA-compatible GPU (recommended for training, CPU supported for inference)

### Environment Setup
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Required Model Assets
- `hand_landmarker.task`: Download from Google MediaPipe:
  ```text
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
  ```
- `Indian Sign Language Gesture Landmarks.csv`: Place in repository root.

---

## Dataset Directory Guide

| Dataset Folder | Size / Files | Structure & Description | Recommended Role |
| :--- | :--- | :--- | :--- |
| `Indian Sign Language Gesture Landmarks.csv` | 50,859 rows | 128 columns (`target`, `uses_two_hands`, 126 coordinate values). | Training & evaluating `LandmarkMLP`. |
| `dataset_ISL` | 12,637 files | 26 folders (`a`-`z`), ~486 images each (100x100 full frames). | Training baseline image models. |
| `dataset_ISL_cropped` | Generated | 26 folders of hand crops generated via `prepare_cropped_dataset.py`. | Training aligned image models matching live inference. |
| `dataset_ISL_masked` | Generated | Production masked dataset with legacy silhouette dilation. | Production ResNet-18 training. |
| `dataset_ISL_large_masked` | Generated (~49k crops) | Masked hand crops built from `ISL Images` / `kaggle_upload_v4`. | Training `isl_resnet_large.pth`. |
| `dataset_skeleton` | Generated | MediaPipe skeleton renderings via `generate_skeleton_dataset.py`. | Training `isl_skeleton_cnn.pth`. |
| `dataset_ISL_fine` | Generated | Experimental masked dataset with fine anatomical silhouette. | Phase 3 fine-mask ResNet experiment. |
| `dataset - Gesture Speech` | 32,400 files | 27 folders (`a`-`z` plus `{`), 384x384 segmented hands on black background. | Cross-dataset transfer evaluation. |
| `ISL Images` / `kaggle_upload_v4` | >100,000 files | Hierarchical: `Kids`, `Teenagers`, `Adults` in `Full Sleeves` and `Half Sleeves`. | Evaluating demographic & signer independence. |
| `webcam_dataset` | User-captured | 26 class folders populated via `capture_webcam_dataset.py`. | Target-domain fine-tuning and evaluation. |

Generated datasets and `webcam_dataset/` are git-ignored; rebuild them with the `data_preparation/` scripts.

---

## Model Selection

Image-based live tools (`live_image_inference.py`, `text_composer.py`) load the **active model** unless you pass `--model` or `--checkpoint`. Resolution order:

1. `--checkpoint path\to\model.pth`
2. `--model <name>` (e.g. `large`, `balanced`, `resnet`, `cnn`)
3. `ISL_MODEL` environment variable
4. `models/active_model.json` (set by `switch_model.py`)
5. Default: `large` if present, otherwise `resnet`

If a configured model is unknown or its checkpoint is missing, a warning is printed and the next option is used.

```powershell
.\.venv\Scripts\python.exe .\switch_model.py            # show active model & available checkpoints
.\.venv\Scripts\python.exe .\switch_model.py finetuned  # activate isl_resnet_large_finetuned.pth
.\.venv\Scripts\python.exe .\switch_model.py resnet     # fall back to the baseline
```

| Key | Checkpoint | Test accuracy | Notes |
| :--- | :--- | :--- | :--- |
| `finetuned` | `isl_resnet_large_finetuned.pth` | 100% (random split) | Fine-tuned on own webcam photos; random split, so not comparable with block-split numbers. |
| `large` | `isl_resnet_large.pth` | 74.63% (block) | Trained on `dataset_ISL_large_masked`. |
| `balanced` | `isl_resnet_balanced.pth` | 77.59% (balanced manifest) | Experiment B, uses `manifests/`. |
| `resnet` | `isl_resnet.pth` | 68.19% | Original baseline. |
| `cnn` | `isl_cnn.pth` | – | 4-block CNN. |
| `skeleton_cnn` | `isl_skeleton_cnn.pth` | – | Skeleton-image CNN. |

---

## Usage Guide

*(Note: Commands can be run either via their package path like `.\app\live_image_inference.py` or via the root launcher `.\live_image_inference.py`.)*

### 1. Live Applications

**Run Real-Time ResNet Image Recognition:**
```powershell
.\.venv\Scripts\python.exe .\app\live_image_inference.py `
  --model resnet `
  --camera 0 `
  --crop-mode auto `
  --min-confidence 0.60 `
  --history-size 7
```

**Run Interactive Live Text Composer:**
```powershell
.\.venv\Scripts\python.exe .\app\text_composer.py `
  --camera 0 `
  --hold-duration 0.8 `
  --min-confidence 0.50
```

**Run Real-Time Landmark Recognition:**
```powershell
.\.venv\Scripts\python.exe .\app\live_landmark_inference.py `
  --camera 0 `
  --min-confidence 0.65 `
  --history-size 7
```

**Run Camera Diagnostic Utility:**
```powershell
.\.venv\Scripts\python.exe .\app\find_cameras.py
```

### 2. Dataset Preparation

**Preprocess Dataset to Align Hand Cropping:**
```powershell
.\.venv\Scripts\python.exe .\data_preparation\prepare_cropped_dataset.py `
  --input .\dataset_ISL `
  --output .\dataset_ISL_cropped `
  --padding 0.25 `
  --size 224
```

**Capture Custom Webcam Dataset:**
```powershell
.\.venv\Scripts\python.exe .\data_preparation\capture_webcam_dataset.py --camera 0 --mirror
```

### 3. Training Pipelines

**Train Landmark Model (Leakage-Safe Block Split):**
```powershell
.\.venv\Scripts\python.exe .\training\train_landmark_model.py `
  --split-strategy block `
  --epochs 40 `
  --batch-size 512
```

**Train Image Models (CNN & ResNet-18):**
```powershell
# Train both CNN and ResNet on cropped dataset:
.\.venv\Scripts\python.exe .\training\train_image_models.py `
  --model all `
  --dataset-dir .\dataset_ISL_cropped `
  --split-strategy block `
  --epochs 15
```

### 4. Evaluation & Diagnostics

**Run ISL Alphabet Webcam Audit & Diagnostic Mode:**
```powershell
.\.venv\Scripts\python.exe .\evaluation\audit_alphabet.py --checkpoint models\isl_resnet.pth --camera 0
```

**Evaluate Model Checkpoints Offline:**
```powershell
# Evaluate ResNet on held-out test split:
.\.venv\Scripts\python.exe .\evaluation\evaluate_models.py `
  --model resnet `
  --dataset-dir .\dataset_ISL_masked `
  --split test `
  --split-strategy block
```

**Generate Presentation Training Loss & Accuracy Curves:**
```powershell
.\.venv\Scripts\python.exe .\evaluation\plot_training_curves.py
```

---

## Running Automated Tests

Run the complete 52-test suite with `pytest` (included in `requirements.txt`):
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

---

## Model Distribution Strategy

Because model checkpoints (`.pth`) and the MediaPipe `.task` model are binary files excluded by `.gitignore`:
- **Version Control**: Lightweight metadata and evaluation metrics are committed under `reports/`.
- **Checkpoint Hosting**: Checkpoints can be distributed via GitHub Releases or HuggingFace Model Hub.
- **Local Structure**:
  ```text
  models/
    ├── isl_resnet_large_finetuned.pth  (~44.8 MB, current active model)
    ├── isl_resnet_large.pth            (~44.8 MB, default if no config)
    ├── isl_resnet_balanced.pth         (~44.8 MB)
    ├── isl_resnet.pth                  (~44.8 MB, baseline)
    ├── isl_resnet_fine.pth             (~44.8 MB)
    ├── isl_webcam_resnet.pth           (~44.8 MB)
    ├── isl_cnn.pth / isl_skeleton_cnn.pth (~5.8 MB each)
    ├── isl_alphabet_cnn_v2/v3/v4.pth   (~1.3 MB each, earlier CNN iterations)
    └── isl_landmark_mlp.pth            (~280 KB)
  hand_landmarker.task         (~7.8 MB)
  ```
