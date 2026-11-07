import sys
import argparse
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
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
    """Convert MediaPipe hand landmark result to 127-d feature vector matching the training CSV.

    DATASET SPECIFICATION:
    In 'Indian Sign Language Gesture Landmarks.csv':
    - All 11,894 single-handed signs (C, I, L, O, U, V) ALWAYS store the active hand in 'left_hand' (slots 0..62)
      and leave 'right_hand' as all zeros (slots 63..125).
    - In two-handed signs, left_hand corresponds to the screen-left hand, and right_hand to screen-right.
    """
    if not result.hand_landmarks:
        return torch.zeros(127)

    num_hands = len(result.hand_landmarks)

    if num_hands == 1:
        # SINGLE-HAND SIGN:
        # Always place in the primary slot (0..62) to match 100% of single-hand signs in the dataset.
        primary = normalize_hand(result.hand_landmarks[0])
        secondary = [0.0] * 63
        if swap_hands:
            primary, secondary = secondary, primary
        return torch.tensor(primary + secondary + [0.0], dtype=torch.float32)
    else:
        # TWO-HAND SIGNS:
        # Sort hands spatially by screen-x (wrist position): left hand is screen-left, right is screen-right.
        h1 = result.hand_landmarks[0]
        h2 = result.hand_landmarks[1]
        if h1[0].x > h2[0].x:
            h1, h2 = h2, h1

        if swap_hands:
            h1, h2 = h2, h1

        left = normalize_hand(h1)
        right = normalize_hand(h2)
        return torch.tensor(left + right + [1.0], dtype=torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Run real-time ISL landmark recognition.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (e.g. 0 or 1).")
    parser.add_argument("--mirror", action="store_true", help="Mirror preview horizontally.")
    parser.add_argument("--min-confidence", type=float, default=0.50,
                        help="Confidence threshold below which prediction is marked uncertain (0.0 to 1.0).")
    parser.add_argument("--history-size", type=int, default=5,
                        help="Number of consecutive frames for majority vote temporal smoothing.")
    parser.add_argument("--min-consensus", type=float, default=0.40,
                        help="Minimum vote agreement fraction in history buffer to confirm prediction.")
    parser.add_argument("--verbose", action="store_true", help="Display top-3 candidate probabilities.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run train_landmark_model.py first.")
    if not LANDMARKER_PATH.is_file():
        raise FileNotFoundError(f"MediaPipe hand landmarker task file not found: {LANDMARKER_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = LandmarkMLP(len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}.")

    history = deque(maxlen=max(1, args.history_size))
    last_timestamp_ms = 0

    print("\n--- ISL Real-Time Landmark Recognizer Started ---")
    print("Controls: Press Q or ESC to quit.")
    print("Signs: C, I, L, O, U, V are single-handed. Other letters are two-handed.\n")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Could not read a camera frame.")
                if args.mirror:
                    frame = cv2.flip(frame, 1)

                # Ensure strictly increasing millisecond timestamps for MediaPipe VIDEO mode
                timestamp_ms = int(time.time() * 1000)
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                result = landmarker.detect_for_video(image, timestamp_ms)

                if result.hand_landmarks:
                    num_hands = len(result.hand_landmarks)

                    with torch.inference_mode():
                        feat_normal = make_features(result, swap_hands=False).unsqueeze(0)
                        feat_swapped = make_features(result, swap_hands=True).unsqueeze(0)

                        probs_normal = torch.softmax(model(feat_normal), dim=1)[0]
                        probs_swapped = torch.softmax(model(feat_swapped), dim=1)[0]

                        # Choose whichever orientation has higher confidence (DO NOT average them!)
                        if probs_normal.max() >= probs_swapped.max():
                            probabilities = probs_normal
                        else:
                            probabilities = probs_swapped

                    confidence, index = probabilities.max(dim=0)
                    conf_val = float(confidence)
                    history.append(int(index))

                    stable_index, votes = Counter(history).most_common(1)[0]
                    vote_fraction = votes / len(history)

                    top3_probs, top3_indices = torch.topk(probabilities, k=min(3, len(classes)))
                    hand_info = "2-Hand" if num_hands > 1 else "1-Hand"

                    if conf_val >= args.min_confidence and vote_fraction >= args.min_consensus:
                        text = f"{classes[stable_index].upper()} ({conf_val * 100:.1f}%) [{hand_info}]"
                        color = (0, 220, 0)  # Confident Green
                    else:
                        top_candidate = classes[stable_index].upper()
                        text = f"Uncertain: {top_candidate} ({conf_val * 100:.1f}%) [{hand_info}]"
                        color = (0, 165, 255)  # Uncertain Amber

                    # Draw hand skeletons directly on frame for visual feedback
                    h, w = frame.shape[:2]
                    for hand_landmarks in result.hand_landmarks:
                        pts = [(int(pt.x * w), int(pt.y * h)) for pt in hand_landmarks]
                        for pt in pts:
                            cv2.circle(frame, pt, 4, (0, 255, 255), -1)
                        connections = [
                            (0, 1), (1, 2), (2, 3), (3, 4),
                            (0, 5), (5, 6), (6, 7), (7, 8),
                            (5, 9), (9, 10), (10, 11), (11, 12),
                            (9, 13), (13, 14), (14, 15), (15, 16),
                            (13, 17), (17, 18), (18, 19), (19, 20),
                            (0, 17),
                        ]
                        for c1, c2 in connections:
                            cv2.line(frame, pts[c1], pts[c2], (0, 200, 0), 2)

                    if args.verbose:
                        debug_lines = [
                            f"1. {classes[top3_indices[0]].upper()}: {top3_probs[0]:.1%}",
                            f"2. {classes[top3_indices[1]].upper()}: {top3_probs[1]:.1%}",
                            f"3. {classes[top3_indices[2]].upper()}: {top3_probs[2]:.1%}",
                        ]
                        for idx, d_text in enumerate(debug_lines):
                            cv2.putText(
                                frame, d_text, (20, 85 + idx * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA,
                            )
                else:
                    history.clear()
                    text, color = "No hand detected", (0, 165, 255)

                cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
                cv2.putText(
                    frame, "Q/ESC: quit", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
                )
                cv2.imshow("ISL Landmark Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
