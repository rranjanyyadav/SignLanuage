# Indian Sign Language Live Recognition

Real-time Indian Sign Language alphabet recognition using MediaPipe hand landmarks and a PyTorch multilayer perceptron (MLP).

## How it works

```text
Webcam frame
  -> MediaPipe hand detection
  -> 21 landmarks per hand
  -> wrist-relative and scale normalization
  -> landmark MLP classifier
  -> seven-frame prediction smoothing
  -> displayed A-Z letter
```

The live recognizer uses hand geometry rather than the background or full camera image. This makes it faster and more robust to changes in lighting, clothing, and camera framing than the older full-frame CNN experiments.

## Requirements

- Windows
- Python 3.10 or newer
- A webcam, Camo Studio camera, or other camera supported by OpenCV
- PyTorch
- MediaPipe
- OpenCV

Create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision mediapipe opencv-python
```

The repository intentionally excludes the virtual environment, datasets, model checkpoints, and MediaPipe task file. These files must be provided locally.

## Required local files

Place these files in the project:

```text
hand_landmarker.task
models\isl_landmark_mlp.pth
Indian Sign Language Gesture Landmarks.csv
```

The CSV is used for training. The `.pth` checkpoint is used for live prediction. The `.task` file is the MediaPipe hand-landmark model.

## Train the landmark model

From the project root:

```powershell
.\.venv\Scripts\python.exe .\train_landmark_model.py
```

The trainer reads:

```text
Indian Sign Language Gesture Landmarks.csv
```

and writes:

```text
models\isl_landmark_mlp.pth
```

The dataset contains A-Z classes and left/right hand landmark coordinates. Training normalizes each hand relative to its wrist and its hand scale.

## Run live recognition

Default camera:

```powershell
.\.venv\Scripts\python.exe .\live_landmark_inference.py --camera 0
```

Use another camera index:

```powershell
.\.venv\Scripts\python.exe .\live_landmark_inference.py --camera 1
```

Mirror the preview and input when needed:

```powershell
.\.venv\Scripts\python.exe .\live_landmark_inference.py --camera 1 --mirror
```

Press `Q` or `Esc` to quit.

## Camo Studio

Start Camo Studio before running the recognizer. On the tested Windows setup:

- Camera `0` was the laptop webcam.
- Camera `1` was the Camo camera.

Use the diagnostic script to inspect available cameras:

```powershell
.\.venv\Scripts\python.exe .\find_cameras.py
```

Then run:

```powershell
.\.venv\Scripts\python.exe .\live_landmark_inference.py --camera 1
```

If the preview is reversed, add `--mirror`.

## Repository files

- `live_landmark_inference.py` - live MediaPipe plus MLP recognition.
- `train_landmark_model.py` - trains the landmark classifier.
- `find_cameras.py` - checks camera indexes and Windows capture backends.
- `hand_landmarker.task` - required locally but excluded from Git.
- `models\isl_landmark_mlp.pth` - required locally but excluded from Git.

## Limitations

The current system recognizes individual alphabet signs. It does not yet perform continuous sentence translation, word segmentation, spaces, delete gestures, or language modeling.

Validation accuracy from a random holdout is not the same as accuracy on a completely unseen signer. For a reliable signer-independent evaluation, hold out complete people or recording sessions rather than randomly splitting individual landmark rows.
