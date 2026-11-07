"""
Plot training and validation loss and accuracy curves from saved experiment history.

This script parses training history recorded in JSON format (e.g., reports/image_model_results.json)
and generates presentation-quality loss and accuracy curve figures for project documentation
and Phase I submission artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
DEFAULT_JSON_PATH = PROJECT_DIR / "reports" / "image_model_results.json"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "reports"


def load_experiment_data(json_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load and validate the experiment result entry and its epoch history.

    Parameters
    ----------
    json_path : Path
        Path to the JSON file containing experiment metrics.

    Returns
    -------
    Tuple[Dict[str, Any], List[Dict[str, Any]]]
        The experiment metadata dict and the list of epoch records.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Experiment results file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # image_model_results.json typically stores a list of model result dictionaries
    if isinstance(data, list):
        if not data:
            raise ValueError(f"Experiment results file {json_path} is an empty list.")
        # Find first entry with 'history'
        entry = next((item for item in data if "history" in item and item["history"]), None)
        if entry is None:
            raise ValueError(f"No experiment record with non-empty 'history' found in {json_path}")
    elif isinstance(data, dict):
        if "history" not in data or not data["history"]:
            raise ValueError(f"JSON dict in {json_path} does not contain non-empty 'history'.")
        entry = data
    else:
        raise TypeError(f"Unexpected JSON root type {type(data)} in {json_path}")

    history = entry["history"]
    required_keys = {"epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy"}
    for idx, rec in enumerate(history):
        missing = required_keys - set(rec.keys())
        if missing:
            raise KeyError(f"History epoch record at index {idx} missing required keys: {sorted(missing)}")
        for key in required_keys:
            if rec[key] is None:
                raise ValueError(f"Null metric found in epoch record index {idx}: {key} is None")

    return entry, history


def plot_loss_curve(
    metadata: Dict[str, Any],
    history: List[Dict[str, Any]],
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """Generate and save the training and validation loss curve.

    Parameters
    ----------
    metadata : Dict[str, Any]
        Experiment metadata including model, checkpoint, and test loss.
    history : List[Dict[str, Any]]
        Per-epoch records.
    output_path : Path
        Destination PNG path.
    dpi : int
        Figure resolution.

    Returns
    -------
    Path
        Saved output path.
    """
    epochs = [int(rec["epoch"]) for rec in history]
    train_loss = [float(rec["train_loss"]) for rec in history]
    val_loss = [float(rec["validation_loss"]) for rec in history]

    # Find minimum validation loss epoch
    min_val_idx = min(range(len(val_loss)), key=lambda i: val_loss[i])
    min_val_epoch = epochs[min_val_idx]
    min_val_val = val_loss[min_val_idx]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        epochs,
        train_loss,
        label="Training Loss",
        color="#1f77b4",
        linewidth=2.2,
        marker="o",
        markersize=6,
        alpha=0.95,
    )
    ax.plot(
        epochs,
        val_loss,
        label="Validation Loss",
        color="#d62728",
        linewidth=2.2,
        linestyle="--",
        marker="s",
        markersize=6,
        alpha=0.95,
    )

    # Highlight min validation loss
    ax.scatter(
        [min_val_epoch],
        [min_val_val],
        color="#b71540",
        s=130,
        zorder=5,
        edgecolors="black",
        linewidths=1.5,
    )
    ax.annotate(
        f"Lowest Val Loss: {min_val_val:.4f}\n(Epoch {min_val_epoch})",
        xy=(min_val_epoch, min_val_val),
        xytext=(min_val_epoch + 0.6, min_val_val + 0.28),
        arrowprops=dict(facecolor="#b71540", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9.5,
        fontweight="semibold",
        color="#2c3e50",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#b71540", alpha=0.9),
    )

    ax.set_title(
        "Training & Validation Loss — ResNet-18 (dataset_ISL_fine)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Epoch Number", fontsize=11, fontweight="semibold", labelpad=8)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="semibold", labelpad=8)

    ax.set_xticks(epochs)
    ax.set_xlim(min(epochs) - 0.5, max(epochs) + 0.5)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Experiment context badge in corner
    badge_text = (
        f"Model: ResNet-18 ({Path(metadata.get('checkpoint', 'isl_resnet_fine.pth')).name})\n"
        f"Split Strategy: {metadata.get('split_strategy', 'block')} (Seed 42, video-chunk holdout)\n"
        f"Dataset: dataset_ISL_fine (fine silhouette hand crops)\n"
        f"Held-out Test Loss: {float(metadata.get('test_loss', 1.7533)):.4f}"
    )
    ax.text(
        0.025,
        0.04,
        badge_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.92),
    )

    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=10.5)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_accuracy_curve(
    metadata: Dict[str, Any],
    history: List[Dict[str, Any]],
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """Generate and save the training and validation accuracy curve.

    Parameters
    ----------
    metadata : Dict[str, Any]
        Experiment metadata including best validation and test accuracy.
    history : List[Dict[str, Any]]
        Per-epoch records.
    output_path : Path
        Destination PNG path.
    dpi : int
        Figure resolution.

    Returns
    -------
    Path
        Saved output path.
    """
    epochs = [int(rec["epoch"]) for rec in history]
    train_acc = [float(rec["train_accuracy"]) * 100.0 for rec in history]
    val_acc = [float(rec["validation_accuracy"]) * 100.0 for rec in history]

    # Find highest validation accuracy epoch
    max_val_idx = max(range(len(val_acc)), key=lambda i: val_acc[i])
    max_val_epoch = epochs[max_val_idx]
    max_val_acc = val_acc[max_val_idx]

    test_acc = float(metadata.get("test_accuracy", 0.6765)) * 100.0

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        epochs,
        train_acc,
        label="Training Accuracy",
        color="#2ca02c",
        linewidth=2.2,
        marker="o",
        markersize=6,
        alpha=0.95,
    )
    ax.plot(
        epochs,
        val_acc,
        label="Validation Accuracy",
        color="#ff7f0e",
        linewidth=2.2,
        linestyle="--",
        marker="s",
        markersize=6,
        alpha=0.95,
    )

    # Test accuracy reference line
    ax.axhline(
        y=test_acc,
        color="#6c5ce7",
        linestyle=":",
        linewidth=1.8,
        label=f"Held-Out Test Acc ({test_acc:.2f}%)",
    )

    # Highlight best validation accuracy
    ax.scatter(
        [max_val_epoch],
        [max_val_acc],
        color="#d35400",
        s=130,
        zorder=5,
        edgecolors="black",
        linewidths=1.5,
    )
    ax.annotate(
        f"Peak Val Acc: {max_val_acc:.2f}%\n(Epoch {max_val_epoch})",
        xy=(max_val_epoch, max_val_acc),
        xytext=(max_val_epoch + 0.6, max_val_acc - 7.5),
        arrowprops=dict(facecolor="#d35400", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9.5,
        fontweight="semibold",
        color="#2c3e50",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#d35400", alpha=0.9),
    )

    ax.set_title(
        "Training & Validation Accuracy — ResNet-18 (dataset_ISL_fine)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Epoch Number", fontsize=11, fontweight="semibold", labelpad=8)
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="semibold", labelpad=8)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_xticks(epochs)
    ax.set_xlim(min(epochs) - 0.5, max(epochs) + 0.5)
    ax.set_ylim(25.0, 105.0)
    ax.grid(True, linestyle="--", alpha=0.5)

    badge_text = (
        f"Model: ResNet-18 ({Path(metadata.get('checkpoint', 'isl_resnet_fine.pth')).name})\n"
        f"Peak Validation: {max_val_acc:.2f}% (Epoch {max_val_epoch})\n"
        f"Held-Out Test Accuracy: {test_acc:.2f}%\n"
        f"Production Baseline Reference: models/isl_resnet.pth (68.19% test acc)"
    )
    ax.text(
        0.025,
        0.04,
        badge_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.92),
    )

    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.96), frameon=True, framealpha=0.95, fontsize=10.5)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot training and validation loss and accuracy curves.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help=f"Path to input JSON file (default: {DEFAULT_JSON_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save output plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--loss-filename",
        type=str,
        default="training_loss_curve.png",
        help="Filename for the loss plot (default: training_loss_curve.png)",
    )
    parser.add_argument(
        "--accuracy-filename",
        type=str,
        default="training_accuracy_curve.png",
        help="Filename for the accuracy plot (default: training_accuracy_curve.png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI for saved images (default: 300)",
    )

    args = parser.parse_args()

    metadata, history = load_experiment_data(args.input)
    print(f"Loaded {len(history)} epochs from {args.input}")
    print(f"Experiment Checkpoint: {metadata.get('checkpoint')}")
    print(f"Model: {metadata.get('model')}, Split: {metadata.get('split_strategy')}")

    loss_file = args.output_dir / args.loss_filename
    acc_file = args.output_dir / args.accuracy_filename

    saved_loss = plot_loss_curve(metadata, history, loss_file, dpi=args.dpi)
    print(f"Saved training loss curve: {saved_loss} ({saved_loss.stat().st_size} bytes)")

    saved_acc = plot_accuracy_curve(metadata, history, acc_file, dpi=args.dpi)
    print(f"Saved training accuracy curve: {saved_acc} ({saved_acc.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
