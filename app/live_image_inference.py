import sys
import argparse
import math
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms

from training.train_image_models import IMAGE_SIZE, NUM_CLASSES, SmallCNN, LegacySmallCNN
from data_preparation.prepare_cropped_dataset import compute_hand_box_and_mask
from data_preparation.generate_skeleton_dataset import render_skeleton_image
try:
    from app.model_config import resolve_checkpoint, get_active_model_info
except ImportError:
    from model_config import resolve_checkpoint, get_active_model_info


ISL_SINGLE_HAND_CLASSES = frozenset({"c", "i", "l", "o", "u", "v"})
ISL_TWO_HAND_CLASSES = frozenset(set("abcdefghijklmnopqrstuvwxyz") - ISL_SINGLE_HAND_CLASSES)


def stabilize_two_hand_box(raw_box, num_hands, last_two_hand_box, hold_counter, max_hold=4):
    """Prevent bounding box from instantly snapping to a 1-hand crop when a hand is momentarily occluded."""
    if num_hands >= 2 and raw_box is not None:
        return raw_box, raw_box, max_hold
    if num_hands == 1 and hold_counter > 0 and last_two_hand_box is not None:
        # Retain the spatial extent of the 2-hand box during momentary occlusion
        return last_two_hand_box, last_two_hand_box, hold_counter - 1
    return raw_box, None, 0


def verify_hand_count_consistency(predicted_char, num_hands):
    """Verify whether detected hand count is consistent with verified ISL dataset grammar."""
    char = str(predicted_char).lower()
    if char == "h" and num_hands < 2:
        return False, "H requires 2 flat hands (1 detected)"
    if num_hands == 1 and char in ISL_TWO_HAND_CLASSES:
        return False, "Needs 2 hands, 1 detected"
    if num_hands >= 2 and char in ISL_SINGLE_HAND_CLASSES:
        return False, "Needs 1 hand, 2 detected"
    return True, ""


def _extract_landmark_pt(pt):
    """Extract (x, y, z) tuple from a landmark point supporting objects, dicts, or tuples."""
    if hasattr(pt, "x") and hasattr(pt, "y"):
        return float(pt.x), float(pt.y), float(getattr(pt, "z", 0.0))
    if isinstance(pt, dict):
        return float(pt["x"]), float(pt["y"]), float(pt.get("z", 0.0))
    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
        return float(pt[0]), float(pt[1]), float(pt[2]) if len(pt) > 2 else 0.0
    raise ValueError(f"Unrecognized landmark format: {pt}")


def _dist_3d(p1, p2):
    """Compute Euclidean 3D distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def compute_ring_to_middle_ratio(landmarks_list, min_middle_span=0.05):
    """Compute ring-to-middle fingertip extension ratio from MediaPipe landmarks.

    Measures extension of Landmark 16 (ring fingertip) relative to Landmark 12
    (middle fingertip) from Landmark 0 (wrist) on the active signing hand.

    Returns:
        tuple: (ratio, active_hand_idx, d_mid, d_ring) or (None, None, None, None)
    """
    if not landmarks_list:
        return None, None, None, None

    best_ratio = None
    best_hand_idx = None
    best_d_mid = -1.0
    best_d_ring = None

    for h_idx, hand in enumerate(landmarks_list):
        if hand is None or len(hand) < 21:
            continue
        try:
            w_pt = _extract_landmark_pt(hand[0])    # Wrist (0)
            m_pt = _extract_landmark_pt(hand[12])   # Middle fingertip (12)
            r_pt = _extract_landmark_pt(hand[16])   # Ring fingertip (16)
        except (KeyError, IndexError, ValueError, TypeError):
            continue

        if not all(math.isfinite(c) for pt in (w_pt, m_pt, r_pt) for c in pt):
            continue

        d_mid = _dist_3d(w_pt, m_pt)
        d_ring = _dist_3d(w_pt, r_pt)

        # In ISL N and M signs, the active signing hand extends its middle finger
        # across the non-dominant palm. We select the hand with greater middle finger extension.
        if d_mid > best_d_mid:
            best_d_mid = d_mid
            best_d_ring = d_ring
            best_hand_idx = h_idx

    # Require minimum middle finger span to ensure hand is extended and reliably measurable
    if best_hand_idx is None or best_d_mid < min_middle_span:
        return None, None, None, None

    ratio = best_d_ring / best_d_mid
    if not math.isfinite(ratio):
        return None, None, None, None

    return ratio, best_hand_idx, best_d_mid, best_d_ring


def correct_n_m_prediction(
    predicted_char,
    landmarks_list,
    ratio_threshold=0.72,
    min_middle_span=0.05,
):
    """Guarded decision-level correction between 'N' and 'M' using ring finger extension.

    Experimental Basis:
      In ISL, 'M' (target 12) extends 3 fingers (index, middle, ring) across the base palm,
      whereas 'N' (target 13) extends 2 fingers (index, middle) while curling the ring
      finger tightly into the palm.
      In held-out block-split testing (153 M samples, 154 N samples from separate sessions):
        - M ratio (d_ring / d_mid): mean = 0.9418, min = 0.8954, max = 0.9821
        - N ratio (d_ring / d_mid): mean = 0.5087, min = 0.4558, max = 0.5626
        - Clear separation margin: max(N)=0.5626 < min(M)=0.8954 (gap = 0.3328).
      A threshold of 0.72 is placed at the center of this margin to distinguish
      curled (< 0.72 -> 'N') vs extended (>= 0.72 -> 'M') ring fingers.

    Guards:
      1. Only applies when predicted_char is 'n' or 'm' (case-insensitive).
         All other 24 letter predictions pass through completely unchanged.
      2. If landmarks are None, empty, missing required points, or middle finger
         span is below min_middle_span, original prediction is retained without alteration.

    Args:
        predicted_char (str): ResNet candidate letter (e.g. 'n', 'm', 'N', 'M', or other letter).
        landmarks_list (list): MediaPipe hand landmarks from live frame (result.hand_landmarks).
        ratio_threshold (float): Threshold distinguishing curled vs extended ring finger (default: 0.72).
        min_middle_span (float): Minimum Euclidean distance from wrist to middle tip (default: 0.05).

    Returns:
        tuple: (final_char, info_dict)
            final_char (str): Corrected or original letter preserving input case.
            info_dict (dict): Diagnostic metadata including applied flag, ratio, and threshold.
    """
    if not predicted_char:
        return predicted_char, {"applied": False, "reason": "empty_prediction"}

    char_lower = str(predicted_char).lower()
    is_upper = str(predicted_char).isupper()

    # Guard 1: Only N and M are candidate classes for this landmark disambiguation rule
    if char_lower not in ("n", "m"):
        return predicted_char, {"applied": False, "reason": "not_n_or_m", "char": predicted_char}

    # Guard 2: Missing, invalid, or unmeasurable landmarks
    ratio, hand_idx, d_mid, d_ring = compute_ring_to_middle_ratio(
        landmarks_list, min_middle_span=min_middle_span
    )
    if ratio is None:
        return predicted_char, {
            "applied": False,
            "reason": "unreliable_or_missing_landmarks",
            "char": predicted_char,
        }

    # Apply threshold rule
    new_char_lower = "m" if ratio >= ratio_threshold else "n"
    final_char = new_char_lower.upper() if is_upper else new_char_lower

    return final_char, {
        "applied": True,
        "original_char": predicted_char,
        "corrected_char": final_char,
        "changed": (final_char != predicted_char),
        "ratio": ratio,
        "threshold": ratio_threshold,
        "active_hand_idx": hand_idx,
        "d_mid": d_mid,
        "d_ring": d_ring,
    }


def compute_hand_box(landmarks_list, width, height, padding=0.25, square=True):
    box, _ = compute_hand_box_and_mask(landmarks_list, width, height, padding=padding, square=square)
    return box


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
MODEL_DIR = PROJECT_DIR / "models"
LANDMARKER_PATH = PROJECT_DIR / "hand_landmarker.task"


def build_model(model_name, checkpoint):
    state_dict = checkpoint["model_state_dict"]
    is_resnet = (model_name == "resnet") or ("conv1.weight" in state_dict)
    if is_resnet:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    else:
        if "features.12.weight" in state_dict or checkpoint.get("architecture_version") == 2:
            model = SmallCNN(NUM_CLASSES)
        else:
            model = LegacySmallCNN(NUM_CLASSES)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def create_transform(checkpoint, force_grayscale=False):
    normalization = checkpoint.get("normalization", {})
    mean = normalization.get("mean", [0.485, 0.456, 0.406])
    std = normalization.get("std", [0.229, 0.224, 0.225])
    image_size = checkpoint.get("image_size", IMAGE_SIZE)
    use_grayscale = force_grayscale or checkpoint.get("grayscale", False)

    transform_list = [
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
    ]
    if use_grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=3))
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transforms.Compose(transform_list)


def smooth_box(current_box, prev_box, alpha=0.40):
    if prev_box is None:
        return current_box
    if current_box is None:
        return None
    return tuple(int(round(alpha * c + (1 - alpha) * p)) for c, p in zip(current_box, prev_box))


def parse_args():
    parser = argparse.ArgumentParser(description="Run real-time CNN or ResNet ISL webcam inference.")
    parser.add_argument("--model", choices=("large", "resnet_large", "balanced", "resnet_balanced", "resnet", "cnn", "skeleton_cnn"), default=None,
                        help="Model architecture checkpoint to load ('skeleton_cnn' is the hybrid skeleton model).")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Explicit checkpoint file path to load.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--mirror", action="store_true", default=False,
                        help="Mirror preview horizontally (DEFAULT: False). Note: ISL signs are asymmetric; mirroring inverts hand geometry and significantly degrades accuracy.")
    parser.add_argument("--padding", type=float, default=0.25, help="Padding fraction around hand box.")
    parser.add_argument("--crop-mode", choices=("auto", "hand", "full_frame"), default="auto",
                        help="'auto' follows checkpoint input_mode; 'hand' crops detected hand; 'full_frame' classifies full frame.")
    parser.add_argument("--full-frame", action="store_true",
                        help="Convenience alias for --crop-mode full_frame.")
    parser.add_argument("--min-confidence", type=float, default=0.50,
                        help="Confidence threshold below which prediction is marked uncertain.")
    parser.add_argument("--history-size", type=int, default=5,
                        help="Number of frames for temporal smoothing buffer.")
    parser.add_argument("--min-consensus", type=float, default=0.40,
                        help="Minimum majority vote agreement fraction required.")
    parser.add_argument("--grayscale", action="store_true",
                        help="Force grayscale preprocessing for web camera feed.")
    parser.add_argument("--mask-background", action="store_true", default=None,
                        help="Mask background to solid black using detected hand silhouette.")
    parser.add_argument("--verbose", action="store_true", help="Display top-3 candidate probabilities.")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Enable comprehensive diagnostic overlay (exact model input preview, top-3 probabilities, and smoothing stats).")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path, model_display_name = resolve_checkpoint(model_arg=args.model, checkpoint_arg=args.checkpoint)
    model_name = args.model
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {checkpoint_path}. Train the image model first via train_image_models.py."
        )
    if not LANDMARKER_PATH.is_file():
        raise FileNotFoundError(f"MediaPipe model not found: {LANDMARKER_PATH}")
    if args.padding < 0:
        raise ValueError("padding must be non-negative.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    state_dict = checkpoint["model_state_dict"]

    # Auto-detect model architecture if not explicitly specified
    if model_name is None:
        ckpt_type = checkpoint.get("model_type")
        if ckpt_type == "resnet" or "conv1.weight" in state_dict:
            model_name = "resnet"
        elif "skeleton" in str(checkpoint_path).lower():
            model_name = "skeleton_cnn"
        elif ckpt_type == "cnn" or "features.0.weight" in state_dict:
            model_name = "cnn"
        else:
            model_name = "skeleton_cnn"

    model = build_model(model_name, checkpoint)
    transform = create_transform(checkpoint, force_grayscale=args.grayscale)

    # Determine effective crop and mask mode
    checkpoint_input_mode = checkpoint.get("input_mode", "crop")
    checkpoint_is_gray = checkpoint.get("grayscale", False)
    checkpoint_mask_mode = checkpoint.get("mask_mode", "legacy")
    if checkpoint_mask_mode not in ("legacy", "fine"):
        checkpoint_mask_mode = "legacy"

    if args.full_frame or args.crop_mode == "full_frame":
        effective_mode = "full_frame"
        should_mask = False
    elif args.crop_mode == "hand":
        effective_mode = "hand"
        should_mask = args.mask_background if args.mask_background is not None else checkpoint_is_gray
    else:  # auto
        effective_mode = "full_frame" if checkpoint_input_mode == "full_frame" else "hand"
        should_mask = args.mask_background if args.mask_background is not None else checkpoint_is_gray

    print(f"--- ISL Image Recognition Started: {model_display_name} ({checkpoint_path.name}) ---")
    print(f"Mode: '{effective_mode}' | Mask Background: {should_mask} (mode: {checkpoint_mask_mode}) | Grayscale: {checkpoint_is_gray or args.grayscale}")

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
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    history = deque(maxlen=max(1, args.history_size))
    smooth_box_coords = None
    smoothed_probs = None
    prob_ema_alpha = 0.40
    last_timestamp_ms = 0
    last_two_hand_box = None
    two_hand_hold_counter = 0

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

                display = frame.copy()
                nm_info = {}
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                result = landmarker.detect_for_video(image, timestamp_ms)

                if model_name == "skeleton_cnn":
                    # HYBRID SKELETON CNN MODE
                    if result.hand_landmarks:
                        num_hands = len(result.hand_landmarks)
                        if num_hands == 1:
                            hands = [result.hand_landmarks[0]]
                        else:
                            h1, h2 = result.hand_landmarks[0], result.hand_landmarks[1]
                            if h1[0].x > h2[0].x:
                                h1, h2 = h2, h1
                            hands = [h1, h2]

                        skeleton_canvas = render_skeleton_image(hands, canvas_size=224)
                        inputs = transform(cv2.cvtColor(skeleton_canvas, cv2.COLOR_BGR2RGB)).unsqueeze(0)
                        with torch.inference_mode():
                            probs = torch.softmax(model(inputs), dim=1)[0]

                        if smoothed_probs is None:
                            smoothed_probs = probs
                        else:
                            smoothed_probs = prob_ema_alpha * probs + (1 - prob_ema_alpha) * smoothed_probs

                        confidence, index = smoothed_probs.max(dim=0)
                        history.append(int(index))
                        stable_index, votes = Counter(history).most_common(1)[0]
                        vote_frac = votes / len(history)

                        conf_val = float(confidence)
                        hand_desc = "2-Hand" if num_hands > 1 else "1-Hand"

                        if conf_val >= args.min_confidence and vote_frac >= args.min_consensus:
                            text = f"{classes[stable_index].upper()} ({conf_val * 100:.1f}%) [{hand_desc}]"
                            color = (0, 220, 0)
                        else:
                            text = f"Uncertain: {classes[stable_index].upper()} ({conf_val * 100:.1f}%) [{hand_desc}]"
                            color = (0, 165, 255)

                        # Picture-in-picture preview of rendered skeleton in top right corner
                        pip_h, pip_w = 120, 120
                        pip_crop = cv2.resize(skeleton_canvas, (pip_w, pip_h))
                        display[10:10+pip_h, width-10-pip_w:width-10] = pip_crop
                        cv2.rectangle(display, (width-10-pip_w, 10), (width-10, 10+pip_h), (0, 220, 255), 2)
                    else:
                        history.clear()
                        smoothed_probs = None
                        text, color = "No hand detected", (0, 165, 255)

                elif effective_mode == "full_frame":
                    crop = frame
                    inputs = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0)
                    with torch.inference_mode():
                        probs = torch.softmax(model(inputs), dim=1)[0]

                    if smoothed_probs is None:
                        smoothed_probs = probs
                    else:
                        smoothed_probs = prob_ema_alpha * probs + (1 - prob_ema_alpha) * smoothed_probs

                    confidence, index = smoothed_probs.max(dim=0)
                    history.append(int(index))
                    stable_index, votes = Counter(history).most_common(1)[0]
                    vote_frac = votes / len(history)

                    conf_val = float(confidence)
                    if conf_val >= args.min_confidence and vote_frac >= args.min_consensus:
                        text = f"{classes[stable_index].upper()} ({conf_val * 100:.1f}%)"
                        color = (0, 220, 0)
                    else:
                        text = f"Uncertain: {classes[stable_index].upper()} ({conf_val * 100:.1f}%)"
                        color = (0, 165, 255)

                else:  # Hand crop mode
                    raw_box = None
                    crop_mask = None
                    num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
                    if num_hands > 0:
                        raw_box, crop_mask = compute_hand_box_and_mask(
                            result.hand_landmarks, width, height, padding=args.padding, square=True,
                            mask_mode=checkpoint_mask_mode,
                        )
                        effective_box, last_two_hand_box, two_hand_hold_counter = stabilize_two_hand_box(
                            raw_box, num_hands, last_two_hand_box, two_hand_hold_counter, max_hold=4
                        )
                    else:
                        effective_box = None
                        last_two_hand_box = None
                        two_hand_hold_counter = 0

                    if effective_box is not None:
                        smooth_box_coords = smooth_box(effective_box, smooth_box_coords, alpha=0.45)
                        left, top, right, bottom = smooth_box_coords
                        crop = frame[top:bottom, left:right]

                        if crop.size > 0:
                            if should_mask and crop_mask is not None:
                                if crop_mask.shape != crop.shape[:2]:
                                    crop_mask = cv2.resize(crop_mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
                                crop = cv2.bitwise_and(crop, crop, mask=crop_mask)

                            inputs = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0)
                            with torch.inference_mode():
                                probs = torch.softmax(model(inputs), dim=1)[0]

                            if smoothed_probs is None:
                                smoothed_probs = probs
                            else:
                                smoothed_probs = prob_ema_alpha * probs + (1 - prob_ema_alpha) * smoothed_probs

                            confidence, index = smoothed_probs.max(dim=0)
                            history.append(int(index))
                            stable_index, votes = Counter(history).most_common(1)[0]
                            vote_frac = votes / len(history)

                            conf_val = float(confidence)
                            hand_desc = "2-Hand" if num_hands > 1 else "1-Hand"
                            predicted_char = classes[stable_index].lower()

                            # Guarded N-versus-M landmark disambiguation reusing existing landmarks
                            effective_char, nm_info = correct_n_m_prediction(
                                predicted_char,
                                result.hand_landmarks if result else None,
                            )
                            is_consistent, reason = verify_hand_count_consistency(effective_char, num_hands)
                            display_char = effective_char.upper()

                            if not is_consistent:
                                text = f"Uncertain: {display_char} ({reason})"
                                color = (0, 165, 255)
                            elif conf_val >= args.min_confidence and vote_frac >= args.min_consensus:
                                text = f"{display_char} ({conf_val * 100:.1f}%) [{hand_desc}]"
                                color = (0, 220, 0)
                            else:
                                text = f"Uncertain: {display_char} ({conf_val * 100:.1f}%) [{hand_desc}]"
                                color = (0, 165, 255)

                            cv2.rectangle(display, (left, top), (right, bottom), color, 2)

                            # Picture-in-picture preview of what the model sees in top right corner
                            pip_h = 140
                            pip_w = 140
                            if checkpoint_is_gray or args.grayscale:
                                preview_img = cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                            else:
                                preview_img = crop.copy()
                            pip_crop = cv2.resize(preview_img, (pip_w, pip_h))
                            display[28:28+pip_h, width-10-pip_w:width-10] = pip_crop
                            cv2.rectangle(display, (width-10-pip_w, 28), (width-10, 28+pip_h), (0, 220, 255), 2)
                            cv2.putText(
                                display, "INPUT TO RESNET (224x224)", (width-10-pip_w, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA,
                            )
                        else:
                            text, color = "Crop error", (0, 165, 255)
                    else:
                        history.clear()
                        smooth_box_coords = None
                        smoothed_probs = None
                        text, color = "No hand detected", (0, 165, 255)

                cv2.putText(
                    display, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, color, 2, cv2.LINE_AA,
                )
                cv2.putText(
                    display, f"Model: {checkpoint_path.name} | Mode: {effective_mode} | Mask: {should_mask} | Q/ESC: quit", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
                )

                if (args.verbose or args.diagnostic) and smoothed_probs is not None:
                    top3_p, top3_i = torch.topk(smoothed_probs, k=min(3, len(classes)))
                    # Draw semi-transparent background box for diagnostic readability
                    hud_overlay = display.copy()
                    cv2.rectangle(hud_overlay, (15, 65), (340, 215), (20, 20, 20), -1)
                    display = cv2.addWeighted(hud_overlay, 0.70, display, 0.30, 0)
                    cv2.rectangle(display, (15, 65), (340, 215), (0, 200, 255), 1)

                    cv2.putText(display, "DIAGNOSTIC TOP-3:", (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)
                    for rank in range(3):
                        cand_letter = classes[top3_i[rank]].upper()
                        cand_prob = float(top3_p[rank])
                        cv2.putText(
                            display,
                            f"#{rank+1}: {cand_letter}  ({cand_prob * 100:.1f}%)",
                            (25, 110 + rank * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
                        )
                    votes_info = f"Votes: {votes}/{len(history)}" if history else ""
                    cv2.putText(
                        display,
                        f"Smoothed: {classes[stable_index].upper()} ({votes_info})",
                        (25, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 0) if (conf_val >= args.min_confidence and is_consistent) else (0, 165, 255), 1, cv2.LINE_AA,
                    )
                    if not is_consistent:
                        cv2.putText(
                            display,
                            f"Check: {reason}",
                            (25, 205),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA,
                        )
                    elif nm_info.get("applied"):
                        cv2.putText(
                            display,
                            f"NM Ratio: {nm_info['ratio']:.2f} (th={nm_info['threshold']:.2f}) -> {display_char}",
                            (25, 205),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA,
                        )

                cv2.imshow("ISL Image Recognition", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
