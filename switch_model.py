"""CLI utility to switch and inspect the active ISL recognition model.

Usage:
  python switch_model.py                # Show current active model & status
  python switch_model.py large          # Activate models/isl_resnet_large.pth
  python switch_model.py resnet         # Fall back to models/isl_resnet.pth (baseline)
  python switch_model.py balanced       # Activate models/isl_resnet_balanced.pth
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.model_config import (
    get_active_model_info,
    set_active_model,
    AVAILABLE_MODELS,
    CONFIG_FILE,
)


def print_status():
    key, ckpt, info = get_active_model_info()
    print("=" * 65)
    print("  ISL ACTIVE MODEL CONFIGURATION")
    print("=" * 65)
    print(f"  Active Model:    {info['display_name']}")
    print(f"  Checkpoint File: {ckpt}")
    print(f"  Status:          {'[EXISTS / READY]' if ckpt.is_file() else '[MISSING]'}")
    print(f"  Config Source:   {CONFIG_FILE if CONFIG_FILE.is_file() else 'Default (models/isl_resnet_large.pth)'}")
    print("\n  Available Models:")
    for k, m in AVAILABLE_MODELS.items():
        p = ckpt.parent / m["filename"]
        marker = "==> [ACTIVE]" if p.name == ckpt.name else "    "
        print(f"  {marker} {k:<12} -> {m['filename']:<26} ({'Present' if p.is_file() else 'Not found'})")
    print("\n  To switch models:")
    print("    python switch_model.py large      (Activate ResNet Large)")
    print("    python switch_model.py resnet     (Fall back to Baseline ResNet)")
    print("    python switch_model.py balanced   (Activate ResNet Balanced)")
    print("=" * 65)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("status", "--status", "-s"):
        print_status()
        return

    target = sys.argv[1].lower().strip()
    try:
        new_path = set_active_model(target)
        _, _, info = get_active_model_info()
        print("\nSuccessfully updated active model:")
        print(f"  ==> {info['display_name']}")
        print(f"  Checkpoint: {new_path}")
        print("\nAll live tools (live_image_inference.py, text_composer.py) will now use this model.")
        print("To fall back at any time, run: python switch_model.py resnet\n")
    except Exception as e:
        print(f"Error switching model: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

