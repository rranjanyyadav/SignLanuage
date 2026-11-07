import sys
import argparse
import json
import random
import time
from pathlib import Path
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
DATASET_DIR = PROJECT_DIR / "dataset_ISL"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"
IMAGE_SIZE = 224
NUM_CLASSES = 26
SEED = 42

try:
    from evaluation.plot_training_curves import plot_loss_curve, plot_accuracy_curve
except ImportError:
    plot_loss_curve = None
    plot_accuracy_curve = None


class LegacySmallCNN(nn.Module):
    """Original 3-layer SmallCNN architecture preserved for backward compatibility."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs):
        return self.classifier(self.features(inputs).flatten(1))


class SmallCNN(nn.Module):
    """Improved 4-block CNN for sign language alphabet recognition.

    Retains spatial feature resolution (4x4) before classification, allowing the
    network to capture intricate finger configurations that 1x1 pooling lost.
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Retain 4x4 spatial feature grid
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


def build_model(name, use_pretrained=True):
    if name == "cnn":
        return SmallCNN()

    weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def make_transforms(grayscale=False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_list = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    eval_list = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]

    if grayscale:
        train_list.append(transforms.Grayscale(num_output_channels=3))
        eval_list.append(transforms.Grayscale(num_output_channels=3))

    train_list.extend([
        transforms.RandomRotation(12),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.10, 0.10),
            scale=(0.90, 1.10),
        ),
        transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.10 if not grayscale else 0.0),
        transforms.ToTensor(),
        normalize,
    ])
    eval_list.extend([
        transforms.ToTensor(),
        normalize,
    ])

    return transforms.Compose(train_list), transforms.Compose(eval_list)


def split_dataset_indices(targets, validation_fraction, test_fraction, strategy="block", seed=SEED):
    """Split dataset sample indices without temporal/frame leakage.

    Args:
        targets: List of integer class targets.
        validation_fraction: Fraction for validation.
        test_fraction: Fraction for test.
        strategy: 'block' for contiguous session holdout per class, 'random' for stratified shuffle.
        seed: Random seed.
    """
    generator = torch.Generator().manual_seed(seed)
    split_indices = [[], [], []]

    for target in sorted(set(targets)):
        class_indices = torch.tensor(
            [idx for idx, val in enumerate(targets) if val == target],
            dtype=torch.long,
        )
        total_class = len(class_indices)
        test_count = max(1, round(total_class * test_fraction))
        val_count = max(1, round(total_class * validation_fraction))
        train_count = total_class - test_count - val_count
        if train_count < 1:
            raise ValueError(
                f"Class {target} needs at least three images for train/validation/test splits."
            )

        if strategy == "random":
            perm = class_indices[torch.randperm(total_class, generator=generator)]
            split_indices[0].extend(perm[:train_count].tolist())
            split_indices[1].extend(perm[train_count:train_count + val_count].tolist())
            split_indices[2].extend(perm[train_count + val_count:].tolist())
        elif strategy == "block":
            # Contiguous holdout to keep continuous video frames together
            split_indices[0].extend(class_indices[:train_count].tolist())
            split_indices[1].extend(class_indices[train_count:train_count + val_count].tolist())
            split_indices[2].extend(class_indices[train_count + val_count:].tolist())
        else:
            raise ValueError(f"Unknown split strategy: {strategy}")

    return [torch.tensor(indices, dtype=torch.long) for indices in split_indices]


class ManifestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, manifest_path, transform=None):
        self.root_dir = Path(root_dir)
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        self.transform = transform
        self.classes = [chr(ord('a') + i) for i in range(NUM_CLASSES)]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.targets = [self.class_to_idx[Path(p).parts[0].lower()] for p in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.root_dir / self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


def make_loaders(batch_size, workers, dataset_dir=DATASET_DIR, split_strategy="block", grayscale=False, manifest_dir=None):
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Image dataset not found: {dataset_dir}")

    train_transform, evaluation_transform = make_transforms(grayscale=grayscale)
    loader_args = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if manifest_dir is not None and Path(manifest_dir).is_dir():
        m_dir = Path(manifest_dir)
        tr_m = m_dir / "train_manifest_balanced.json" if (m_dir / "train_manifest_balanced.json").exists() else m_dir / "train_manifest.json"
        va_m = m_dir / "val_manifest_balanced.json" if (m_dir / "val_manifest_balanced.json").exists() else m_dir / "val_manifest.json"
        te_m = m_dir / "test_manifest_balanced.json" if (m_dir / "test_manifest_balanced.json").exists() else m_dir / "test_manifest.json"

        train_ds = ManifestDataset(dataset_dir, tr_m, transform=train_transform)
        val_ds = ManifestDataset(dataset_dir, va_m, transform=evaluation_transform)
        test_ds = ManifestDataset(dataset_dir, te_m, transform=evaluation_transform)
        print(f"Loaded datasets from manifests in {m_dir}:")
        print(f"  Train: {len(train_ds):,} samples | Val: {len(val_ds):,} samples | Test: {len(test_ds):,} samples")
        return (
            DataLoader(train_ds, shuffle=True, **loader_args),
            DataLoader(val_ds, shuffle=False, **loader_args),
            DataLoader(test_ds, shuffle=False, **loader_args),
            train_ds.classes,
        )

    base_dataset = datasets.ImageFolder(dataset_dir)
    if len(base_dataset.classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} class folders, found {len(base_dataset.classes)}"
        )
    train_dataset = datasets.ImageFolder(dataset_dir, transform=train_transform)
    evaluation_dataset = datasets.ImageFolder(
        dataset_dir, transform=evaluation_transform
    )
    train_indices, validation_indices, test_indices = split_dataset_indices(
        base_dataset.targets, validation_fraction=0.15, test_fraction=0.15,
        strategy=split_strategy, seed=SEED,
    )
    return (
        DataLoader(Subset(train_dataset, train_indices), shuffle=True, **loader_args),
        DataLoader(Subset(evaluation_dataset, validation_indices), shuffle=False, **loader_args),
        DataLoader(Subset(evaluation_dataset, test_indices), shuffle=False, **loader_args),
        base_dataset.classes,
    )


def run_epoch(model, loader, criterion, device, optimizer=None, progress_desc=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    labels_seen = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        batches = loader
        if progress_desc is not None:
            batches = tqdm(
                loader,
                desc=progress_desc,
                unit="batch",
                leave=False,
                dynamic_ncols=True,
                mininterval=0.2,
            )
        for images, labels in batches:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            predictions.extend(predicted.cpu().tolist())
            labels_seen.extend(labels.cpu().tolist())
            if progress_desc is not None:
                batches.set_postfix(
                    loss=f"{loss.item():.4f}",
                    accuracy=f"{correct / total:.1%}",
                )
    return total_loss / total, correct / total, predictions, labels_seen


def compute_metrics(predictions, targets, num_classes=NUM_CLASSES):
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for p, t in zip(predictions, targets):
        matrix[t][p] += 1

    per_class = {}
    precisions, recalls, f1s = [], [], []
    classes = [chr(ord('a') + i) for i in range(num_classes)]
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
        per_class[classes[c]] = {
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


def train_model(name, args, loaders, classes, device):
    train_loader, validation_loader, test_loader, _ = loaders
    model = build_model(name, use_pretrained=not args.no_pretrained).to(device)
    if getattr(args, "base_checkpoint", None) and Path(args.base_checkpoint).is_file():
        base_path = Path(args.base_checkpoint)
        base_data = torch.load(base_path, map_location="cpu", weights_only=True)
        model.load_state_dict(base_data["model_state_dict"])
        print(f"Loaded starting weights from base checkpoint: {base_path}")
    if name == "resnet" and args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_val_f1 = -1.0
    best_state = None
    history = []
    total_train_sec = 0.0

    print(f"\n{'='*75}")
    print(f"  TRAINING {name.upper()} | Dataset: {len(train_loader.dataset):,} train, {len(validation_loader.dataset):,} val, {len(test_loader.dataset):,} test")
    print(f"  Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | Split: {args.split_strategy}")
    print(f"{'='*75}\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_accuracy, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            progress_desc=f"{name.upper()} Epoch {epoch:02d}/{args.epochs} [Train]",
        )
        validation_loss, validation_accuracy, val_preds, val_labels = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            progress_desc=f"{name.upper()} Epoch {epoch:02d}/{args.epochs} [Val]",
        )
        val_metrics = compute_metrics(val_preds, val_labels)
        val_macro_f1 = val_metrics["macro_f1"]
        epoch_sec = time.perf_counter() - epoch_start
        total_train_sec += epoch_sec

        avg_sec = total_train_sec / epoch
        remaining_sec = avg_sec * (args.epochs - epoch)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining_sec)) if remaining_sec >= 3600 else time.strftime("%M:%S", time.gmtime(remaining_sec))

        improved = validation_accuracy > best_accuracy
        if improved:
            best_accuracy = validation_accuracy
            best_val_f1 = val_macro_f1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            status_indicator = f" --> [Best Val Model Saved: Acc {best_accuracy * 100:.2f}%, F1 {best_val_f1 * 100:.2f}%]"
        else:
            status_indicator = ""

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {validation_loss:.4f} Acc: {validation_accuracy * 100:.2f}% F1: {val_macro_f1 * 100:.2f}% | "
            f"Time: {epoch_sec:.1f}s (ETA: {eta_str}){status_indicator}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "validation_macro_f1": val_macro_f1,
            "epoch_duration_seconds": round(epoch_sec, 2),
        })

    if best_state is None:
        raise RuntimeError(f"No checkpoint was produced for {name}.")

    print(f"\n{'-'*75}")
    print(f"Evaluating best validation model on held-out Test Set ({len(test_loader.dataset):,} images)...")
    model.load_state_dict(best_state)
    test_loss, test_accuracy, predictions, labels = run_epoch(
        model, test_loader, criterion, device, progress_desc="Evaluating Test Set"
    )
    test_metrics = compute_metrics(predictions, labels)
    print(f"Test Set Evaluation: Loss={test_loss:.4f}, Accuracy={test_accuracy * 100:.2f}%, Macro F1={test_metrics['macro_f1'] * 100:.2f}%")
    print(f"{'-'*75}\n")

    ckpt_name = args.checkpoint_name if args.checkpoint_name else f"isl_{name}.pth"
    checkpoint_path = MODEL_DIR / ckpt_name
    checkpoint = {
        "model_state_dict": best_state,
        "classes": classes,
        "image_size": IMAGE_SIZE,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "validation_accuracy": best_accuracy,
        "validation_macro_f1": best_val_f1,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_metrics["macro_f1"],
        "model_type": name,
        "split_strategy": args.split_strategy,
        "input_mode": args.input_mode,
        "grayscale": args.grayscale,
        "mask_mode": getattr(args, "mask_mode", "legacy"),
        "architecture_version": 2 if name == "cnn" else 1,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved best model checkpoint to: {checkpoint_path}")
    return {
        "model": name,
        "checkpoint": str(checkpoint_path),
        "split_strategy": args.split_strategy,
        "input_mode": args.input_mode,
        "best_validation_accuracy": best_accuracy,
        "best_validation_macro_f1": best_val_f1,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_metrics": test_metrics,
        "history": history,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN and ResNet-18 ISL classifiers.")
    parser.add_argument("--model", choices=("cnn", "resnet", "all"), default="all")
    parser.add_argument("--base-checkpoint", type=Path, default=None,
                        help="Path to existing checkpoint to initialize weights from for fine-tuning.")
    parser.add_argument("--checkpoint-name", type=str, default="",
                        help="Custom checkpoint filename under models/ (e.g. isl_resnet_large.pth).")
    parser.add_argument("--report-path", type=Path, default=None,
                        help="Custom report path under reports/ (e.g. reports/image_model_large_results.json).")
    parser.add_argument("--no-plot-curves", action="store_true",
                        help="Do not automatically save training and validation loss/accuracy curves.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--split-strategy", choices=("block", "random"), default="block",
                        help="'block' holds out contiguous recording chunks to prevent video-frame leakage.")
    parser.add_argument("--input-mode", choices=("auto", "crop", "full_frame"), default="auto",
                        help="Document input distribution: 'crop' for hand crops, 'full_frame' for full scene.")
    parser.add_argument("--grayscale", action="store_true", default=None,
                        help="Convert inputs to grayscale (auto-detected if dataset path contains 'masked' or 'gray').")
    parser.add_argument("--mask-mode", choices=("legacy", "fine", "none", "auto"), default="auto",
                        help="Mask preprocessing mode ('legacy', 'fine', 'none', or 'auto' from dataset name).")
    parser.add_argument("--manifest-dir", type=Path, default=None,
                        help="Directory containing train, val, and test JSON manifests.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")

    dir_str = str(args.dataset_dir).lower()
    # Auto-detect input_mode if specified as auto
    if args.input_mode == "auto":
        if "crop" in dir_str or "webcam" in dir_str or "speech" in dir_str or "masked" in dir_str or "fine" in dir_str:
            args.input_mode = "crop"
        else:
            args.input_mode = "full_frame"

    # Auto-detect grayscale if not explicitly passed
    if args.grayscale is None:
        args.grayscale = True if ("masked" in dir_str or "gray" in dir_str or "fine" in dir_str) else False

    # Auto-detect mask_mode if auto
    if args.mask_mode == "auto":
        if "fine" in dir_str:
            args.mask_mode = "fine"
        elif "masked" in dir_str or "webcam" in dir_str:
            args.mask_mode = "legacy"
        elif getattr(args, "base_checkpoint", None) and Path(args.base_checkpoint).is_file():
            base_data = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
            args.mask_mode = base_data.get("mask_mode", "legacy")
        else:
            args.mask_mode = "none"

    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset_dir} (input_mode: {args.input_mode}, grayscale: {args.grayscale}, split: {args.split_strategy})")

    loaders = make_loaders(args.batch_size, args.workers, args.dataset_dir, split_strategy=args.split_strategy, grayscale=args.grayscale, manifest_dir=args.manifest_dir)
    model_names = ("cnn", "resnet") if args.model == "all" else (args.model,)
    results = [
        train_model(name, args, loaders, loaders[3], device)
        for name in model_names
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.report_path:
        report_path = Path(args.report_path)
    elif args.checkpoint_name:
        stem = Path(args.checkpoint_name).stem.replace("isl_", "").replace("resnet_", "")
        report_path = REPORT_DIR / f"image_model_{stem}_results.json"
    else:
        report_path = REPORT_DIR / "image_model_results.json"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved training report to: {report_path}")

    # Auto-generate presentation training curves if enabled
    if not args.no_plot_curves and plot_loss_curve is not None and plot_accuracy_curve is not None:
        for result in results:
            if "history" in result and result["history"]:
                suffix = Path(args.checkpoint_name).stem.replace("isl_", "").replace("resnet_", "") if args.checkpoint_name else result["model"]
                loss_file = report_path.parent / f"training_loss_curve_{suffix}.png"
                acc_file = report_path.parent / f"training_accuracy_curve_{suffix}.png"
                plot_loss_curve(result, result["history"], loss_file)
                plot_accuracy_curve(result, result["history"], acc_file)
                print(f"Saved training loss curve:     {loss_file}")
                print(f"Saved training accuracy curve: {acc_file}")

    for result in results:
        print(
            f"\nSummary: {result['model'].upper()} -> Test Accuracy: "
            f"{result['test_accuracy'] * 100:.2f}%, Macro F1: {result['test_metrics']['macro_f1'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
