import sys
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
CSV_PATH = PROJECT_DIR / "Indian Sign Language Gesture Landmarks.csv"
MODEL_PATH = PROJECT_DIR / "models" / "isl_landmark_mlp.pth"
REPORT_PATH = PROJECT_DIR / "reports" / "landmark_model_results.json"
CLASSES = list("abcdefghijklmnopqrstuvwxyz")
SEED = 42


def normalize_hand(values):
    """Normalize 21 (x, y, z) hand landmarks relative to wrist and scale."""
    points = torch.tensor(values, dtype=torch.float32).reshape(21, 3)
    # Check if hand is missing (marked by negative coordinates such as -1.0)
    if points[0, 0] < -0.5:
        return torch.zeros(63)

    wrist = points[0].clone()
    points = points - wrist
    scale = torch.linalg.vector_norm(points[:, :2], dim=1).max().clamp_min(1e-6)
    return (points / scale).reshape(-1)


def row_to_features(row, swap_hands=False):
    """Convert a CSV row dictionary to normalized 127-dimensional feature tensor."""
    features = []
    sides = ("right_hand", "left_hand") if swap_hands else ("left_hand", "right_hand")
    for side in sides:
        values = [
            float(row[f"{side}_{axis}_{index}"])
            for index in range(21)
            for axis in ("x", "y", "z")
        ]
        features.append(normalize_hand(values))
    features.append(torch.tensor([float(row["uses_two_hands"])], dtype=torch.float32))
    return torch.cat(features)


class LandmarkMLP(nn.Module):
    """Multilayer Perceptron for 127-dim ISL hand landmark features."""
    def __init__(self, num_classes=len(CLASSES), dropout1=0.20, dropout2=0.15):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(127, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def split_sample_indices(class_indices_map, val_fraction, test_fraction, strategy="block", seed=SEED):
    """Split sample indices into train, validation, and test splits without leakage.

    Args:
        class_indices_map: Dict mapping class_id -> list of sample indices.
        val_fraction: Fraction of data reserved for validation.
        test_fraction: Fraction of data reserved for testing.
        strategy: 'block' for contiguous session holdout, 'random' for stratified random split.
        seed: Random seed for reproducibility.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices, test_indices = [], [], []

    for target in sorted(class_indices_map.keys()):
        indices = list(class_indices_map[target])
        n = len(indices)
        val_count = max(1, round(n * val_fraction))
        test_count = max(1, round(n * test_fraction))
        train_count = n - val_count - test_count
        if train_count < 1:
            raise ValueError(f"Class {target} has too few samples ({n}) to split 3 ways.")

        if strategy == "random":
            perm = torch.randperm(n, generator=generator).tolist()
            shuffled = [indices[i] for i in perm]
            train_indices.extend(shuffled[:train_count])
            val_indices.extend(shuffled[train_count:train_count + val_count])
            test_indices.extend(shuffled[train_count + val_count:])
        elif strategy == "block":
            # Session/chunk-based holdout: hold out contiguous end blocks to prevent adjacent-frame leakage
            train_indices.extend(indices[:train_count])
            val_indices.extend(indices[train_count:train_count + val_count])
            test_indices.extend(indices[train_count + val_count:])
        else:
            raise ValueError(f"Unknown split strategy: {strategy}. Choose 'block' or 'random'.")

    return train_indices, val_indices, test_indices


def load_dataset(csv_path=CSV_PATH, val_fraction=0.15, test_fraction=0.15,
                 strategy="block", augment_train=True, seed=SEED):
    """Load landmarks dataset, perform leakage-safe split, and apply train-only augmentation.

    Returns:
        (train_x, train_y, val_x, val_y, test_x, test_y)
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    # Read raw samples grouped by class
    raw_samples = []
    class_indices_map = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        for idx, row in enumerate(csv.DictReader(file)):
            target = int(row["target"])
            if not 0 <= target < len(CLASSES):
                continue
            raw_samples.append((row, target))
            class_indices_map[target].append(idx)

    # Perform split on sample indices BEFORE any augmentation
    train_idx, val_idx, test_idx = split_sample_indices(
        class_indices_map, val_fraction, test_fraction, strategy=strategy, seed=seed
    )

    # Build train features with augmentation (hand swapping) applied strictly to train set
    train_features, train_labels = [], []
    for idx in train_idx:
        row, target = raw_samples[idx]
        train_features.append(row_to_features(row, swap_hands=False))
        train_labels.append(target)
        if augment_train:
            train_features.append(row_to_features(row, swap_hands=True))
            train_labels.append(target)

    # Validation and test receive genuine (unaugmented) features
    val_features, val_labels = [], []
    for idx in val_idx:
        row, target = raw_samples[idx]
        val_features.append(row_to_features(row, swap_hands=False))
        val_labels.append(target)

    test_features, test_labels = [], []
    for idx in test_idx:
        row, target = raw_samples[idx]
        test_features.append(row_to_features(row, swap_hands=False))
        test_labels.append(target)

    return (
        torch.stack(train_features),
        torch.tensor(train_labels, dtype=torch.long),
        torch.stack(val_features),
        torch.tensor(val_labels, dtype=torch.long),
        torch.stack(test_features),
        torch.tensor(test_labels, dtype=torch.long),
    )


def compute_metrics(predictions, targets, num_classes=len(CLASSES)):
    """Compute per-class accuracy, macro precision/recall/f1, and confusion matrix."""
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
        per_class[CLASSES[c]] = {
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train ISL Landmark MLP Classifier (Leakage-Safe).")
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH)
    parser.add_argument("--output-model", type=Path, default=MODEL_PATH)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--split-strategy", choices=("block", "random"), default="block",
                        help="'block' holds out contiguous recording chunks to prevent temporal leakage; 'random' performs pre-augmentation stratified shuffle.")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable train-set hand-swapping data augmentation.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading landmark dataset from: {args.csv_path}")
    print(f"Split strategy: {args.split_strategy} (val={args.val_fraction:.0%}, test={args.test_fraction:.0%})")
    print(f"Augment training with hand-swapping: {not args.no_augment}")

    train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(
        csv_path=args.csv_path,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        strategy=args.split_strategy,
        augment_train=not args.no_augment,
        seed=args.seed,
    )
    print(f"Train samples: {len(train_y)}, Validation samples: {len(val_y)}, Test samples: {len(test_y)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LandmarkMLP(len(CLASSES)).to(device)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(dim=1) == y_batch).sum().item()

        model.eval()
        with torch.inference_mode():
            val_logits = model(val_x.to(device))
            val_preds = val_logits.argmax(dim=1).cpu()
            val_accuracy = (val_preds == val_y).float().mean().item()

        if epoch % 5 == 0 or epoch == args.epochs:
            train_acc = train_correct / len(train_y)
            print(f"Epoch {epoch:02d}/{args.epochs}: train_acc={train_acc * 100:.2f}%, val_acc={val_accuracy * 100:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Final evaluation on held-out test split
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        test_logits = model(test_x.to(device))
        test_preds = test_logits.argmax(dim=1).cpu().tolist()

    test_metrics = compute_metrics(test_preds, test_y.tolist())
    print("\n--- Final Model Evaluation ---")
    print(f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}%")
    print(f"Test Accuracy:           {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Test Macro F1:           {test_metrics['macro_f1'] * 100:.2f}%")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "classes": CLASSES,
        "feature_size": 127,
        "normalization": "wrist-relative, max-xy-distance scale",
        "split_strategy": args.split_strategy,
        "validation_accuracy": best_val_accuracy,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
    }
    torch.save(checkpoint, args.output_model)
    print(f"Saved model checkpoint: {args.output_model}")

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "model": "LandmarkMLP",
        "checkpoint": str(args.output_model),
        "split_strategy": args.split_strategy,
        "best_validation_accuracy": best_val_accuracy,
        "test_metrics": test_metrics,
        "train_samples": len(train_y),
        "val_samples": len(val_y),
        "test_samples": len(test_y),
    }
    args.report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"Saved evaluation report: {args.report_path}")


if __name__ == "__main__":
    main()
