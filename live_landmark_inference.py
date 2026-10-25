import argparse
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "isl_landmark_mlp.pth"
LANDMARKER_PATH = PROJECT_DIR / "hand_landmarker.task"


class LandmarkMLP(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(127, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Dropout(0.20), nn.Linear(256, 128), nn.LayerNorm(128),
            nn.ReLU(), nn.Dropout(0.15), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def normalize_hand(hand):
    if hand is None:
        return [0.0] * 63
    wrist = hand[0]
    points = [[point.x - wrist.x, point.y - wrist.y, point.z - wrist.z] for point in hand]
    scale = max((point[0] ** 2 + point[1] ** 2) ** 0.5 for point in points)
    scale = max(scale, 1e-6)
    return [coordinate / scale for point in points for coordinate in point]


def make_features(result, swap_hands=False):
    left = right = None
    for hand, handedness in zip(result.hand_landmarks, result.handedness):
        side = handedness[0].category_name.lower()
        if side == "left":
            left = hand
        elif side == "right":
            right = hand
    if swap_hands:
        left, right = right, left
    values = normalize_hand(left) + normalize_hand(right)
    values.append(1.0 if left is not None and right is not None else 0.0)
    return torch.tensor(values, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mirror", action="store_true")
    args = parser.parse_args()

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run train_landmark_model.py first.")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = LandmarkMLP(len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}.")

    history = deque(maxlen=7)
    timestamp = 0
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Could not read a camera frame.")
                if args.mirror:
                    frame = cv2.flip(frame, 1)
                timestamp += 1
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                result = landmarker.detect_for_video(image, timestamp)
                if result.hand_landmarks:
                    with torch.inference_mode():
                        features = torch.stack([
                            make_features(result),
                            make_features(result, swap_hands=True),
                        ])
                        probabilities = torch.softmax(model(features), dim=1).mean(dim=0)
                    confidence, index = probabilities.max(dim=0)
                    history.append(int(index))
                    stable_index, votes = Counter(history).most_common(1)[0]
                    text = f"{classes[stable_index].upper()} ({float(confidence) * 100:.1f}%)"
                    color = (0, 220, 0) if float(confidence) >= 0.65 else (0, 165, 255)
                else:
                    history.clear()
                    text, color = "No hand detected", (0, 165, 255)
                cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
                cv2.putText(frame, "Q/ESC: quit", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("ISL Landmark Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
