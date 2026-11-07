"""Centralized model configuration and active model manager for ISL recognition.

Allows seamless switching between models:
- 'large' / 'resnet_large': models/isl_resnet_large.pth (Trained on 34k masked crops)
- 'balanced' / 'resnet_balanced': models/isl_resnet_balanced.pth (Experiment B session-balanced)
- 'resnet' / 'baseline': models/isl_resnet.pth (Original baseline production model)
- 'cnn': models/isl_cnn.pth
- 'skeleton_cnn': models/isl_skeleton_cnn.pth
"""
import os
import sys
import json
from pathlib import Path
from typing import Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
CONFIG_FILE = MODEL_DIR / "active_model.json"

AVAILABLE_MODELS = {
    "finetuned": {
        "filename": "isl_resnet_large_finetuned.pth",
        "aliases": ["finetuned", "large_finetuned", "isl_resnet_large_finetuned"],
        "display_name": "ResNet-18 Large (Webcam Finetuned)",
        "description": "ResNet-18 Large fine-tuned on user webcam photos.",
    },
    "large": {
        "filename": "isl_resnet_large.pth",
        "aliases": ["large", "resnet_large", "isl_resnet_large"],
        "display_name": "ResNet-18 Large (34k Dataset)",
        "description": "High-capacity ResNet-18 model trained on dataset_ISL_large_masked.",
    },
    "balanced": {
        "filename": "isl_resnet_balanced.pth",
        "aliases": ["balanced", "resnet_balanced", "isl_resnet_balanced"],
        "display_name": "ResNet-18 Balanced (Experiment B)",
        "description": "Session- and demographic-balanced ResNet-18 model.",
    },
    "resnet": {
        "filename": "isl_resnet.pth",
        "aliases": ["resnet", "baseline", "production", "isl_resnet"],
        "display_name": "ResNet-18 Baseline (Production)",
        "description": "Original production baseline ResNet-18 model.",
    },
    "cnn": {
        "filename": "isl_cnn.pth",
        "aliases": ["cnn", "isl_cnn"],
        "display_name": "Small CNN Baseline",
        "description": "Lightweight convolutional neural network.",
    },
    "skeleton_cnn": {
        "filename": "isl_skeleton_cnn.pth",
        "aliases": ["skeleton_cnn", "skeleton", "isl_skeleton_cnn"],
        "display_name": "Hybrid Skeleton CNN",
        "description": "Hybrid MediaPipe skeleton image classifier.",
    },
}

# Default model when nothing is explicitly configured
DEFAULT_MODEL_KEY = "large"


def _warn(message: str) -> None:
    """Print a non-fatal configuration warning to stderr."""
    print(f"[model_config] WARNING: {message}", file=sys.stderr)


def get_model_info_by_alias(alias: str) -> Optional[dict]:
    """Find model specification by name or alias."""
    clean = str(alias).lower().strip()
    for key, info in AVAILABLE_MODELS.items():
        if clean == key or clean in info["aliases"] or clean == info["filename"].lower():
            return info
    return None


def get_active_model_info() -> Tuple[str, Path, dict]:
    """Get the currently active model key, resolved checkpoint path, and metadata dict.

    Priority:
    1. Environment variable ISL_MODEL
    2. Config file models/active_model.json
    3. Default to 'large' if models/isl_resnet_large.pth exists, otherwise 'resnet'
    """
    # 1. Environment variable
    env_choice = os.environ.get("ISL_MODEL")
    if env_choice:
        info = get_model_info_by_alias(env_choice)
        if info:
            ckpt = MODEL_DIR / info["filename"]
            if ckpt.is_file():
                return info["aliases"][0], ckpt, info
            _warn(f"ISL_MODEL='{env_choice}' checkpoint not found: {ckpt}. Falling back.")
        else:
            _warn(f"ISL_MODEL='{env_choice}' is not a known model. Falling back.")

    # 2. Config file
    if CONFIG_FILE.is_file():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            key = data.get("active_model", DEFAULT_MODEL_KEY)
            info = get_model_info_by_alias(key)
            if info:
                ckpt = MODEL_DIR / info["filename"]
                if ckpt.is_file():
                    return key, ckpt, info
                _warn(f"Configured model '{key}' checkpoint not found: {ckpt}. Falling back.")
            else:
                _warn(f"Configured model '{key}' in {CONFIG_FILE} is not a known model. Falling back.")
        except Exception as exc:
            _warn(f"Could not read {CONFIG_FILE} ({exc}). Falling back.")

    # 3. Default fallback: 'large' if available, else 'resnet'
    large_ckpt = MODEL_DIR / AVAILABLE_MODELS["large"]["filename"]
    if large_ckpt.is_file():
        return "large", large_ckpt, AVAILABLE_MODELS["large"]

    resnet_ckpt = MODEL_DIR / AVAILABLE_MODELS["resnet"]["filename"]
    return "resnet", resnet_ckpt, AVAILABLE_MODELS["resnet"]


def set_active_model(alias: str) -> Path:
    """Set the project active model in models/active_model.json.

    Returns:
        Path: Path to the selected checkpoint.
    """
    info = get_model_info_by_alias(alias)
    if not info:
        valid_keys = list(AVAILABLE_MODELS.keys())
        raise ValueError(f"Unknown model alias '{alias}'. Available options: {valid_keys}")

    ckpt_path = MODEL_DIR / info["filename"]
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint file does not exist: {ckpt_path}. "
            f"Ensure the model is trained before activating."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config_data = {
        "active_model": info["aliases"][0],
        "checkpoint": str(ckpt_path),
        "filename": info["filename"],
        "display_name": info["display_name"],
    }
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    return ckpt_path


def resolve_checkpoint(
    model_arg: Optional[str] = None,
    checkpoint_arg: Optional[Path] = None
) -> Tuple[Path, str]:
    """Resolve the effective checkpoint path and model display name.

    Priority:
    1. Explicit checkpoint_arg (e.g. --checkpoint path/to/model.pth)
    2. Explicit model_arg (e.g. --model large or --model resnet)
    3. Active configured model (from active_model.json or env var)
    """
    if checkpoint_arg is not None:
        p = Path(checkpoint_arg)
        if not p.is_file():
            raise FileNotFoundError(f"Specified checkpoint not found: {p}")
        info = get_model_info_by_alias(p.name)
        display = info["display_name"] if info else p.name
        return p, display

    if model_arg is not None:
        info = get_model_info_by_alias(model_arg)
        if not info:
            # Check if direct file under models/
            candidate = MODEL_DIR / f"isl_{model_arg}.pth"
            if candidate.is_file():
                return candidate, f"ISL {model_arg}"
            raise ValueError(f"Unknown model argument: '{model_arg}'")
        ckpt = MODEL_DIR / info["filename"]
        if not ckpt.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")
        return ckpt, info["display_name"]

    # Active configured model
    _, ckpt, info = get_active_model_info()
    return ckpt, info["display_name"]

