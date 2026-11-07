import sys
import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from training.train_landmark_model import LandmarkMLP, row_to_features, CLASSES, CSV_PATH
from training.train_image_models import IMAGE_SIZE, NUM_CLASSES, SmallCNN, LegacySmallCNN, make_transforms, split_dataset_indices


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"
LANDMARKER_PATH = PROJECT_DIR / "hand_landmarker.task"


def load_image_model(model_name, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = MODEL_DIR / f"isl_{model_name}.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model_state_dict"]

    # Strict architecture verification: fail clearly if ambiguous or incompatible
    is_resnet_weights = "conv1.weight" in state_dict or "layer1.0.conv1.weight" in state_dict
    is_cnn_weights = "features.0.weight" in state_dict

    if is_resnet_weights and is_cnn_weights:
        raise ValueError(
            f"Ambiguous architecture in checkpoint '{checkpoint_path}': "
            "contains both ResNet and SmallCNN weight keys."
        )
    if not is_resnet_weights and not is_cnn_weights:
        raise ValueError(
            f"Unrecognized architecture in checkpoint '{checkpoint_path}': "
            "neither ResNet nor SmallCNN weight keys found."
        )

    detected_arch = "resnet" if is_resnet_weights else "cnn"

    # Enforce compatibility with requested model name
    if model_name is not None and model_name not in ("auto", "skeleton_cnn"):
        if model_name != detected_arch:
            raise ValueError(
                f"Architecture mismatch: requested model '{model_name}', but checkpoint '{checkpoint_path}' "
                f"contains '{detected_arch}' architecture."
            )

    if detected_arch == "resnet":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    else:
        if "features.12.weight" in state_dict or checkpoint.get("architecture_version") == 2:
            model = SmallCNN(NUM_CLASSES)
        else:
            model = LegacySmallCNN(NUM_CLASSES)

    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def compute_metrics(predictions, targets, class_names):
    num_classes = len(class_names)
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for p, t in zip(predictions, targets):
        matrix[t][p] += 1

    per_class = {}
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = matrix[c][c]
        fp = sum(matrix[r][c] for r in range(num_classes)) - tp
        fn = sum(matrix[c]) - tp
        total = sum(matrix[c])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        per_class[class_names[c]] = {
            "samples": total,
            "accuracy": tp / total if total > 0 else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    overall_accuracy = sum(matrix[i][i] for i in range(num_classes)) / len(predictions) if predictions else 0.0
    return {
        "accuracy": overall_accuracy,
        "macro_precision": sum(precisions) / num_classes,
        "macro_recall": sum(recalls) / num_classes,
        "macro_f1": sum(f1s) / num_classes,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def evaluate_image_dataset(model, checkpoint, dataset_dir, split="test", split_strategy="block", batch_size=32, device="cpu"):
    dataset_dir = Path(dataset_dir)
    # Reproduce training preprocessing: pass grayscale setting from checkpoint
    is_grayscale = checkpoint.get("grayscale", False)
    _, eval_transform = make_transforms(grayscale=is_grayscale)
    base_dataset = datasets.ImageFolder(dataset_dir)

    # Filter out extra classes like '{' in dataset - Gesture Speech if present
    valid_classes = [c for c in base_dataset.classes if c.lower() in CLASSES]
    if len(valid_classes) != NUM_CLASSES:
        print(f"Warning: Dataset has classes {base_dataset.classes}, filtering to standard 26.")

    class_to_idx = {c: idx for idx, c in enumerate(CLASSES)}

    if split == "all":
        eval_dataset = datasets.ImageFolder(dataset_dir, transform=eval_transform)
        indices = list(range(len(eval_dataset)))
    else:
        train_idx, val_idx, test_idx = split_dataset_indices(
            base_dataset.targets, validation_fraction=0.15, test_fraction=0.15,
            strategy=split_strategy, seed=42,
        )
        indices = test_idx if split == "test" else val_idx
        eval_dataset = datasets.ImageFolder(dataset_dir, transform=eval_transform)

    loader = DataLoader(
        Subset(eval_dataset, indices),
        batch_size=batch_size,
        shuffle=False,
    )

    model.to(device)
    predictions, targets = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu().tolist()
            predictions.extend(preds)
            targets.extend(labels.tolist())

    return compute_metrics(predictions, targets, CLASSES)


def run_live_camera_evaluation(model_type="landmark", checkpoint_path=None, camera_id=0, mirror=False,
                               seconds_per_letter=3.0, countdown_seconds=2.0, mask_background=None):
    """Interactive live-camera evaluation protocol measuring real webcam performance."""
    print("\n" + "=" * 60)
    print("      INTERACTIVE LIVE-CAMERA EVALUATION PROTOCOL")
    print("=" * 60)
    print(f"Testing model: {model_type.upper()}")
    print(f"You will be prompted to sign each letter from A to Z.")
    print(f"Countdown: {countdown_seconds:.1f}s | Recording: {seconds_per_letter:.1f}s per letter.\n")

    camera = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_id)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {camera_id}.")

    if not LANDMARKER_PATH.is_file():
        raise FileNotFoundError(f"MediaPipe model not found: {LANDMARKER_PATH}")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )

    if model_type == "landmark":
        ckpt_path = checkpoint_path or (MODEL_DIR / "isl_landmark_mlp.pth")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = LandmarkMLP(len(CLASSES))
        model.load_state_dict(ckpt["model_state_dict"])
        transform = None
        should_mask = False
    else:
        model, ckpt = load_image_model(model_type, checkpoint_path)
        is_gray = ckpt.get("grayscale", False)
        # Check whether training used silhouette masking:
        # Reproduce the training pipeline: models trained with grayscale/crop on masked dataset use masking
        if mask_background is not None:
            should_mask = mask_background
        else:
            should_mask = is_gray

        normalization = ckpt.get("normalization", {})
        mean = normalization.get("mean", [0.485, 0.456, 0.406])
        std = normalization.get("std", [0.229, 0.224, 0.225])
        image_size = ckpt.get("image_size", 224)

        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
        ]
        if is_gray:
            transform_list.append(transforms.Grayscale(num_output_channels=3))
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform = transforms.Compose(transform_list)

    model.eval()

    results_per_letter = {}
    last_timestamp_ms = 0

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            for target_idx, letter in enumerate(CLASSES):
                letter_upper = letter.upper()

                # Phase 1: Countdown to get ready
                countdown_start = time.time()
                while time.time() - countdown_start < countdown_seconds:
                    ok, frame = camera.read()
                    if not ok: break
                    if mirror: frame = cv2.flip(frame, 1)
                    rem = countdown_seconds - (time.time() - countdown_start)
                    cv2.putText(
                        frame, f"GET READY FOR '{letter_upper}': {rem:.1f}s",
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3, cv2.LINE_AA,
                    )
                    cv2.imshow("Live Evaluation", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        return None

                # Phase 2: Recording and evaluating
                rec_start = time.time()
                letter_preds = []
                while time.time() - rec_start < seconds_per_letter:
                    ok, frame = camera.read()
                    if not ok: break
                    if mirror: frame = cv2.flip(frame, 1)

                    # Monotonic millisecond timestamps for MediaPipe
                    timestamp_ms = int(time.time() * 1000)
                    if timestamp_ms <= last_timestamp_ms:
                        timestamp_ms = last_timestamp_ms + 1
                    last_timestamp_ms = timestamp_ms

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)

                    pred = None
                    if model_type == "landmark" and result.hand_landmarks:
                        from app.live_landmark_inference import make_features
                        feat = torch.stack([make_features(result), make_features(result, swap_hands=True)])
                        with torch.inference_mode():
                            pred = int(torch.softmax(model(feat), dim=1).mean(dim=0).argmax())
                    elif model_type in ("resnet", "cnn"):
                        from data_preparation.prepare_cropped_dataset import compute_hand_box_and_mask
                        h, w = frame.shape[:2]
                        if result.hand_landmarks:
                            box, crop_mask = compute_hand_box_and_mask(
                                result.hand_landmarks, w, h, padding=0.25, square=True
                            )
                        else:
                            box, crop_mask = None, None

                        if box is not None:
                            crop = frame[box[1]:box[3], box[0]:box[2]]
                            if crop.size > 0:
                                if should_mask and crop_mask is not None:
                                    if crop_mask.shape != crop.shape[:2]:
                                        crop_mask = cv2.resize(
                                            crop_mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST
                                        )
                                    crop = cv2.bitwise_and(crop, crop, mask=crop_mask)

                                inputs = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0)
                                with torch.inference_mode():
                                    pred = int(model(inputs).argmax(dim=1)[0])

                    if pred is not None:
                        letter_preds.append(pred)

                    rem = seconds_per_letter - (time.time() - rec_start)
                    status_text = f"SIGN '{letter_upper}' NOW! ({rem:.1f}s left)"
                    cv2.putText(frame, status_text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 0), 2, cv2.LINE_AA)
                    cv2.imshow("Live Evaluation", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        return None

                # Tally letter results
                if letter_preds:
                    majority_pred = Counter(letter_preds).most_common(1)[0][0]
                    correct = (majority_pred == target_idx)
                    acc = sum(1 for p in letter_preds if p == target_idx) / len(letter_preds)
                else:
                    majority_pred = -1
                    correct = False
                    acc = 0.0

                results_per_letter[letter] = {
                    "target": letter,
                    "target_idx": target_idx,
                    "predicted": CLASSES[majority_pred] if majority_pred >= 0 else "None",
                    "correct": correct,
                    "frame_accuracy": acc,
                    "frames_collected": len(letter_preds),
                }
                status_str = "CORRECT" if correct else f"MISCLASSIFIED as '{CLASSES[majority_pred].upper()}'"
                print(f"Letter {letter_upper}: {status_str} (frame agreement: {acc * 100:.1f}%)")

    finally:
        camera.release()
        cv2.destroyAllWindows()

    total_correct = sum(1 for v in results_per_letter.values() if v["correct"])
    overall_accuracy = total_correct / len(CLASSES)
    print("\n" + "=" * 60)
    print(f"LIVE CAMERA EVALUATION RESULT: {total_correct}/{len(CLASSES)} ({overall_accuracy * 100:.2f}%)")
    print("=" * 60)

    report = {
        "timestamp": datetime.now().isoformat(),
        "model_type": model_type,
        "mode": "live_webcam_evaluation",
        "overall_accuracy": overall_accuracy,
        "correct_letters": total_correct,
        "total_letters": len(CLASSES),
        "details": results_per_letter,
    }
    out_file = REPORT_DIR / f"live_webcam_eval_{model_type}_{int(time.time())}.json"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved live evaluation report: {out_file}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ISL models offline and on live webcam.")
    parser.add_argument("--model", choices=("landmark", "resnet", "cnn"), default="landmark")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="Path to image dataset (defaults to dataset_ISL_masked for crop/masked models, or dataset_ISL).")
    parser.add_argument("--split", choices=("test", "val", "all"), default="test")
    parser.add_argument("--split-strategy", choices=("block", "random"), default="block")
    parser.add_argument("--mask-background", action="store_true", default=None,
                        help="Explicitly enable/disable silhouette masking for live evaluation.")
    parser.add_argument("--live-eval", action="store_true",
                        help="Launch interactive live-camera evaluation protocol.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mirror", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.live_eval:
        run_live_camera_evaluation(
            model_type=args.model,
            checkpoint_path=args.checkpoint,
            camera_id=args.camera,
            mirror=args.mirror,
            mask_background=args.mask_background,
        )
        return

    if args.model == "landmark":
        ckpt_path = args.checkpoint or (MODEL_DIR / "isl_landmark_mlp.pth")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        print(f"--- Landmark Model Checkpoint Metrics ---")
        print(f"Path: {ckpt_path}")
        print(f"Split strategy: {ckpt.get('split_strategy', 'unknown')}")
        print(f"Validation accuracy: {ckpt.get('validation_accuracy', 0.0) * 100:.2f}%")
        print(f"Test accuracy:       {ckpt.get('test_accuracy', 0.0) * 100:.2f}%")
        if "test_macro_f1" in ckpt:
            print(f"Test macro F1:       {ckpt.get('test_macro_f1', 0.0) * 100:.2f}%")
        return

    # Image models evaluation
    model, checkpoint = load_image_model(args.model, args.checkpoint)

    # Establish dataset directory: match training input_mode
    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        if checkpoint.get("input_mode") == "crop" and (PROJECT_DIR / "dataset_ISL_masked").is_dir():
            dataset_dir = PROJECT_DIR / "dataset_ISL_masked"
        else:
            dataset_dir = PROJECT_DIR / "dataset_ISL"

    print(f"Evaluating {args.model.upper()} on {dataset_dir} (split={args.split}, strategy={args.split_strategy}, grayscale={checkpoint.get('grayscale', False)})...")
    metrics = evaluate_image_dataset(
        model, checkpoint, dataset_dir,
        split=args.split, split_strategy=args.split_strategy, device=device,
    )

    print("\n--- Evaluation Results ---")
    print(f"Accuracy:        {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision: {metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall:    {metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F1:        {metrics['macro_f1'] * 100:.2f}%")

    report_path = REPORT_DIR / f"eval_{args.model}_{dataset_dir.name}_{args.split_strategy}_{args.split}.json"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved evaluation report: {report_path}")


if __name__ == "__main__":
    main()

