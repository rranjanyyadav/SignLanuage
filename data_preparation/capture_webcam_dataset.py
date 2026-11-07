import sys
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp

from data_preparation.prepare_cropped_dataset import compute_hand_box_and_mask


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
LANDMARKER_PATH = PROJECT_DIR / "hand_landmarker.task"
CLASSES = tuple("abcdefghijklmnopqrstuvwxyz")


def find_box_and_mask(result, width, height, padding):
    if not result.hand_landmarks:
        return None, None
    return compute_hand_box_and_mask(result.hand_landmarks, width, height, padding=padding, square=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture webcam hand crops or full frames with hands-free countdown timer."
    )
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "webcam_dataset")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--mask-background", action="store_true",
                        help="Mask background out to black using hand silhouette.")
    parser.add_argument("--countdown", type=float, default=3.0,
                        help="Seconds of countdown delay to position both hands before capturing.")
    parser.add_argument("--burst-count", type=int, default=20,
                        help="Number of frames to capture in burst mode.")
    parser.add_argument("--full-frame", action="store_true", help="Save full frame instead of hand crop.")
    parser.add_argument("--signer-id", type=str, default="", help="Optional signer identifier tag.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not LANDMARKER_PATH.is_file():
        raise FileNotFoundError(f"MediaPipe model not found: {LANDMARKER_PATH}")
    if args.padding < 0:
        raise ValueError("padding must be non-negative.")

    base_output = args.output
    if args.signer_id:
        base_output = base_output / args.signer_id

    for letter in CLASSES:
        (base_output / letter).mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}.")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )

    label_idx = 0
    label = CLASSES[label_idx]
    last_timestamp_ms = 0

    # Countdown & burst state
    is_counting_down = False
    countdown_start_time = 0.0
    burst_remaining = 0
    last_burst_time = 0.0

    print("\n--- ISL Webcam Dataset Capture with Hands-Free Timer ---")
    print(f"Countdown delay: {args.countdown:.1f}s (plenty of time to pose with both hands)")
    print("Controls: Press SPACE or B to trigger countdown, ENTER for next letter, Q/ESC to quit.\n")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Could not read a camera frame.")
                if args.mirror:
                    frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]
                timestamp_ms = int(time.time() * 1000)
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                result = landmarker.detect_for_video(image, timestamp_ms)
                box, crop_mask = (None, None) if args.full_frame else find_box_and_mask(result, width, height, args.padding)
                display = frame.copy()

                existing_count = len(list((base_output / label).glob("*.jpg")))
                now = time.time()

                # Draw detected hand box
                if box is not None:
                    left, top, right, bottom = box
                    cv2.rectangle(display, (left, top), (right, bottom), (0, 220, 0), 2)

                # State 1: In Countdown
                if is_counting_down:
                    elapsed = now - countdown_start_time
                    remaining = args.countdown - elapsed
                    if remaining > 0:
                        # Draw high-visibility countdown banner in center of screen
                        overlay = display.copy()
                        cv2.rectangle(overlay, (20, height // 2 - 70), (width - 20, height // 2 + 70), (0, 0, 0), -1)
                        display = cv2.addWeighted(overlay, 0.6, display, 0.4, 0)

                        timer_text = f"POSE FOR '{label.upper()}'! Starting in {remaining:.1f}s"
                        cv2.putText(
                            display, timer_text, (40, height // 2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 215, 255), 3, cv2.LINE_AA,
                        )
                    else:
                        is_counting_down = False
                        burst_remaining = args.burst_count
                        last_burst_time = now

                # State 2: Actively Burst Capturing
                elif burst_remaining > 0:
                    if now - last_burst_time >= 0.10:  # 10 fps capture rate
                        if args.full_frame:
                            crop_to_save = frame
                        elif box is not None:
                            left, top, right, bottom = box
                            crop_to_save = frame[top:bottom, left:right]
                            if args.mask_background and crop_mask is not None:
                                if crop_mask.shape != crop_to_save.shape[:2]:
                                    crop_mask = cv2.resize(crop_mask, (crop_to_save.shape[1], crop_to_save.shape[0]), interpolation=cv2.INTER_NEAREST)
                                crop_to_save = cv2.bitwise_and(crop_to_save, crop_to_save, mask=crop_mask)
                        else:
                            crop_to_save = frame

                        path = base_output / label / f"{label}_{existing_count + 1:05d}.jpg"
                        cv2.imwrite(str(path), crop_to_save)
                        burst_remaining -= 1
                        last_burst_time = now

                    # Draw active recording indicator
                    cv2.circle(display, (width - 40, 40), 16, (0, 0, 255), -1)
                    cv2.putText(
                        display, f"RECORDING '{label.upper()}': {burst_remaining} frames left",
                        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2, cv2.LINE_AA,
                    )

                # State 3: Ready / Idle
                else:
                    status = f"Target: [{label.upper()}] ({existing_count} saved) | SPACE/B: Start {int(args.countdown)}s Timer"
                    cv2.putText(
                        display, status, (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, (0, 220, 0), 2, cv2.LINE_AA,
                    )

                # Bottom control guide
                cv2.putText(
                    display, "SPACE/B: Timer | ENTER: Next letter | A-Z: Select | Q: Quit", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
                )

                cv2.imshow("Capture Webcam Dataset", display)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break
                elif ord("a") <= key <= ord("z"):
                    label = chr(key)
                    label_idx = CLASSES.index(label)
                    is_counting_down = False
                    burst_remaining = 0
                elif key in (32, ord("b")) and not is_counting_down and burst_remaining == 0:
                    # Trigger hands-free countdown
                    is_counting_down = True
                    countdown_start_time = time.time()
                elif key == 13:  # ENTER key advances to next letter
                    label_idx = (label_idx + 1) % len(CLASSES)
                    label = CLASSES[label_idx]
                    is_counting_down = True
                    countdown_start_time = time.time()

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
