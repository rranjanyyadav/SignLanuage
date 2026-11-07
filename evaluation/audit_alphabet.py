import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from data_preparation.prepare_cropped_dataset import compute_hand_box_and_mask
from app.live_image_inference import (
    build_model,
    create_transform,
    smooth_box,
    stabilize_two_hand_box,
    verify_hand_count_consistency,
    correct_n_m_prediction,
    ISL_SINGLE_HAND_CLASSES,
    ISL_TWO_HAND_CLASSES,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
MODEL_DIR = PROJECT_DIR / "models"
LANDMARKER_PATH = PROJECT_DIR / "hand_landmarker.task"
CLASSES = tuple("abcdefghijklmnopqrstuvwxyz")

LETTER_AUDIT_GUIDANCE = {
    "H": "Two hands: Place dominant flat palm across non-dominant flat palm.",
    "X": "Two hands: Cross index fingers at mid-knuckles; keep thumbs folded flat against palms.",
}


def check_letter_safeguards(target_letter, num_hands, box=None, frame_shape=None):
    """Evaluate capture safeguards, warnings, and guidance for ISL alphabet letters.

    Returns:
        status (str): 'accepted', 'uncertain', etc.
        warnings (list of str): Human-readable warnings to display.
        notes (str): Audit log notes.
    """
    letter = target_letter.upper()
    warnings = []
    status = "accepted"
    notes = "manual_confirmed"

    # H Safeguard: H strictly requires two flat hands in ISL
    if letter == "H":
        if num_hands < 2:
            status = "uncertain"
            notes = "missing_second_hand_h"
            warnings.append("H requires two hands: place one flat palm across the other.")
    elif letter.lower() in ISL_TWO_HAND_CLASSES and num_hands < 2:
        status = "uncertain"
        notes = "missing_second_hand"
        warnings.append(f"{letter} requires two hands (only {num_hands} detected).")
    elif letter.lower() in ISL_SINGLE_HAND_CLASSES and num_hands > 1:
        warnings.append(f"{letter} is a single-hand sign ({num_hands} hands detected).")

    # Crop checks (boundary clipping and native model resolution)
    if box is not None and frame_shape is not None:
        left, top, right, bottom = box
        h, w = frame_shape[:2]
        # Frame boundary clipping check: landmarks/padding cut off by camera frame edges
        if left <= 0 or top <= 0 or right >= w or bottom >= h:
            warnings.append("Hand/crop touches frame boundary - center hands in view.")
            if notes == "manual_confirmed":
                notes = "crop_clipped_boundary"
        # Native model resolution check (ResNet expects 224x224 input)
        if (right - left) < 224 or (bottom - top) < 224:
            warnings.append("Crop is smaller than 224px (model resolution) - bring hands closer.")
            if notes == "manual_confirmed":
                notes = "crop_below_native_resolution"

    return status, warnings, notes


def determine_masking_config(checkpoint, checkpoint_path, cli_flag=None):
    """Determine whether background silhouette masking was used during model training.

    Checks:
    1. Explicit CLI flag if provided (--mask-background / --no-mask-background).
    2. Explicit 'mask_background' attribute in checkpoint dictionary.
    3. Dataset path / checkpoint name containing 'masked' or 'skeleton',
       or known checkpoints trained on dataset_ISL_masked ('isl_resnet.pth', 'isl_cnn.pth').
    4. Conservative fallback: False (does not infer masking merely from grayscale).
    """
    if cli_flag is not None:
        return cli_flag

    if "mask_background" in checkpoint:
        return bool(checkpoint["mask_background"])

    path_str = str(checkpoint_path).lower()
    filename = Path(checkpoint_path).name.lower()
    if "masked" in path_str or "skeleton" in path_str or filename in ("isl_resnet.pth", "isl_cnn.pth"):
        return True

    return False


def compute_audit_metrics(records):
    """Compute comprehensive per-letter, overall, and confusion metrics from audit attempts."""
    total_attempts = len(records)
    valid_records = [r for r in records if r.get("status") != "skipped"]
    skipped_records = [r for r in records if r.get("status") == "skipped"]
    flagged_records = [r for r in records if r.get("status") == "needs_review"]

    correct_records = [r for r in valid_records if r.get("is_correct") is True]
    overall_accuracy = len(correct_records) / len(valid_records) if valid_records else 0.0

    # Per-letter breakdown
    letter_groups = defaultdict(list)
    for r in records:
        letter_groups[r["target_letter"].lower()].append(r)

    per_letter = {}
    matrix = [[0] * len(CLASSES) for _ in range(len(CLASSES))]

    for idx, char in enumerate(CLASSES):
        recs = letter_groups.get(char, [])
        valid_recs = [r for r in recs if r.get("status") != "skipped"]
        correct_count = sum(1 for r in valid_recs if r.get("is_correct") is True)
        total_valid = len(valid_recs)
        acc = correct_count / total_valid if total_valid > 0 else 0.0

        preds = [r["predicted_letter"].upper() for r in valid_recs if r.get("predicted_letter")]
        unique_preds = set(preds)
        is_inconsistent = len(unique_preds) > 1

        for r in valid_recs:
            p_char = r.get("predicted_letter", "").lower()
            if p_char in CLASSES:
                p_idx = CLASSES.index(p_char)
                matrix[idx][p_idx] += 1

        per_letter[char.upper()] = {
            "letter": char.upper(),
            "target_idx": idx,
            "total_attempts": len(recs),
            "valid_attempts": total_valid,
            "skipped_attempts": len(recs) - total_valid,
            "correct_attempts": correct_count,
            "accuracy": acc,
            "is_inconsistent": is_inconsistent,
            "predictions": preds,
            "hand_type": "1-Hand" if char in ISL_SINGLE_HAND_CLASSES else "2-Hand",
            "needs_attention": (acc < 1.0 or is_inconsistent) and total_valid > 0,
        }

    # Confusion pairs
    confusion_pairs = []
    for t_idx, t_char in enumerate(CLASSES):
        for p_idx, p_char in enumerate(CLASSES):
            if t_idx != p_idx and matrix[t_idx][p_idx] > 0:
                confusion_pairs.append({
                    "count": matrix[t_idx][p_idx],
                    "true_letter": t_char.upper(),
                    "predicted_letter": p_char.upper(),
                })
    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)

    needing_attention = [
        char for char, data in per_letter.items()
        if data["needs_attention"]
    ]

    return {
        "total_attempts": total_attempts,
        "valid_attempts": len(valid_records),
        "skipped_attempts": len(skipped_records),
        "flagged_attempts": len(flagged_records),
        "correct_attempts": len(correct_records),
        "overall_accuracy": overall_accuracy,
        "per_letter": per_letter,
        "confusion_matrix": matrix,
        "top_confusion_pairs": confusion_pairs,
        "letters_needing_attention": needing_attention,
    }


def generate_csv_report(records, output_path):
    """Write individual capture attempts to CSV."""
    fieldnames = [
        "attempt_id",
        "timestamp",
        "target_letter",
        "predicted_letter",
        "is_correct",
        "confidence",
        "status",
        "detected_hands",
        "top1_letter",
        "top1_prob",
        "top2_letter",
        "top2_prob",
        "top3_letter",
        "top3_prob",
        "raw_image_path",
        "processed_image_path",
        "notes",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, r in enumerate(records, start=1):
            top3 = r.get("top3", [])
            writer.writerow({
                "attempt_id": idx,
                "timestamp": r.get("timestamp", ""),
                "target_letter": r.get("target_letter", "").upper(),
                "predicted_letter": r.get("predicted_letter", "").upper(),
                "is_correct": r.get("is_correct", ""),
                "confidence": f"{r.get('confidence', 0.0):.4f}" if r.get("confidence") is not None else "",
                "status": r.get("status", ""),
                "detected_hands": r.get("detected_hands", 0),
                "top1_letter": top3[0][0].upper() if len(top3) > 0 else "",
                "top1_prob": f"{top3[0][1]:.4f}" if len(top3) > 0 else "",
                "top2_letter": top3[1][0].upper() if len(top3) > 1 else "",
                "top2_prob": f"{top3[1][1]:.4f}" if len(top3) > 1 else "",
                "top3_letter": top3[2][0].upper() if len(top3) > 2 else "",
                "top3_prob": f"{top3[2][1]:.4f}" if len(top3) > 2 else "",
                "raw_image_path": r.get("raw_image_path", ""),
                "processed_image_path": r.get("processed_image_path", ""),
                "notes": r.get("notes", ""),
            })


def generate_html_report(records, metrics, metadata, output_path):
    """Generate self-contained HTML audit dashboard."""
    acc_pct = f"{metrics['overall_accuracy'] * 100:.1f}%"
    total_v = metrics["valid_attempts"]
    corr_v = metrics["correct_attempts"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ISL Alphabet Webcam Audit Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
  h1, h2, h3 {{ color: #38bdf8; margin-top: 0; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .card {{ background: #1e293b; padding: 18px; border-radius: 8px; border: 1px solid #334155; }}
  .card .val {{ font-size: 2rem; font-weight: bold; color: #38bdf8; }}
  .card .sub {{ font-size: 0.85rem; color: #94a3b8; }}
  .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; }}
  .badge-success {{ background: #065f46; color: #6ee7b7; }}
  .badge-danger {{ background: #7f1d1d; color: #fca5a5; }}
  .badge-warning {{ background: #78350f; color: #fcd34d; }}
  .badge-info {{ background: #1e3a8a; color: #93c5fd; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 24px; background: #1e293b; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #334155; }}
  th {{ background: #0f172a; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; }}
  tr:hover {{ background: #283548; }}
  .grid-thumbs {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; margin-top: 16px; }}
  .thumb-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 8px; text-align: center; }}
  .thumb-card img {{ width: 100%; height: 110px; object-fit: contain; background: #000; border-radius: 4px; }}
  .thumb-title {{ font-weight: bold; font-size: 0.9rem; margin-top: 6px; }}
  .meta-tag {{ font-size: 0.75rem; color: #94a3b8; }}
  .matrix-table {{ font-size: 0.75rem; text-align: center; }}
  .matrix-table th, .matrix-table td {{ padding: 4px 6px; text-align: center; }}
  .mat-hit {{ background: rgba(16, 185, 129, 0.3); font-weight: bold; color: #10b981; }}
  .mat-err {{ background: rgba(239, 68, 68, 0.4); font-weight: bold; color: #ef4444; }}
</style>
</head>
<body>
<div class="container">
  <h1>ISL Alphabet Webcam Audit Report</h1>
  <p class="meta-tag">Session: {metadata.get('timestamp')} | Checkpoint: {metadata.get('checkpoint_path')} | Model: {metadata.get('model_type').upper()} | Mask: {metadata.get('mask_background')} | Grayscale: {metadata.get('grayscale')}</p>

  <div class="card-grid">
    <div class="card">
      <div class="val">{acc_pct}</div>
      <div class="sub">Overall Accuracy ({corr_v}/{total_v} valid)</div>
    </div>
    <div class="card">
      <div class="val">{metrics['valid_attempts']} / {metrics['total_attempts']}</div>
      <div class="sub">Attempts Evaluated (Skipped: {metrics['skipped_attempts']})</div>
    </div>
    <div class="card">
      <div class="val">{len(metrics['letters_needing_attention'])} / 26</div>
      <div class="sub">Letters Needing Attention</div>
    </div>
    <div class="card">
      <div class="val">{metrics['flagged_attempts']}</div>
      <div class="sub">Flagged for Review</div>
    </div>
  </div>

  <h2>Letters Needing Attention</h2>
  <p class="meta-tag">Letters with accuracy under 100%, inconsistent predictions across attempts, or hand-count warnings.</p>
  """

    if metrics["letters_needing_attention"]:
        html += """<table>
    <thead><tr><th>Letter</th><th>Type</th><th>Accuracy</th><th>Attempts</th><th>Predictions Across Attempts</th><th>Status</th></tr></thead>
    <tbody>"""
        for char in metrics["letters_needing_attention"]:
            data = metrics["per_letter"][char]
            acc = f"{data['accuracy'] * 100:.1f}%"
            preds = ", ".join(data["predictions"]) or "None"
            badge = '<span class="badge badge-danger">Inconsistent</span>' if data["is_inconsistent"] else '<span class="badge badge-warning">Low Accuracy</span>'
            html += f"<tr><td><strong>{char}</strong></td><td>{data['hand_type']}</td><td>{acc}</td><td>{data['correct_attempts']}/{data['valid_attempts']}</td><td>{preds}</td><td>{badge}</td></tr>"
        html += "</tbody></table>"
    else:
        html += "<p>All evaluated letters achieved 100% agreement!</p>"

    html += """<h2>Top Confusion Pairs</h2>"""
    if metrics["top_confusion_pairs"]:
        html += """<table><thead><tr><th>True Letter</th><th>Predicted As</th><th>Occurrences</th></tr></thead><tbody>"""
        for pair in metrics["top_confusion_pairs"][:10]:
            html += f"<tr><td><strong>{pair['true_letter']}</strong></td><td><strong style='color:#ef4444;'>{pair['predicted_letter']}</strong></td><td>{pair['count']}</td></tr>"
        html += "</tbody></table>"
    else:
        html += "<p>No cross-class confusions observed!</p>"

    html += """<h2>Full Per-Letter Summary</h2>
  <table>
    <thead><tr><th>Letter</th><th>Type</th><th>Valid Attempts</th><th>Correct</th><th>Accuracy</th><th>Consistency</th></tr></thead>
    <tbody>"""
    for char in CLASSES:
        data = metrics["per_letter"][char.upper()]
        acc = f"{data['accuracy'] * 100:.1f}%" if data["valid_attempts"] > 0 else "N/A"
        cons = '<span class="badge badge-danger">Inconsistent</span>' if data["is_inconsistent"] else '<span class="badge badge-success">Consistent</span>'
        if data["valid_attempts"] == 0:
            cons = '<span class="badge badge-info">Skipped</span>'
        html += f"<tr><td><strong>{char.upper()}</strong></td><td>{data['hand_type']}</td><td>{data['valid_attempts']}</td><td>{data['correct_attempts']}</td><td>{acc}</td><td>{cons}</td></tr>"
    html += "</tbody></table>"

    html += """<h2>Visual Attempt Contact Sheet</h2>
  <p class="meta-tag">Processed input crops evaluated by the neural network.</p>
  <div class="grid-thumbs">"""

    for r in records:
        if r.get("status") == "skipped":
            continue
        rel_img = r.get("processed_image_rel_path") or Path(r.get("processed_image_path", "")).name
        target = r.get("target_letter", "").upper()
        pred = r.get("predicted_letter", "").upper()
        conf = f"{r.get('confidence', 0.0) * 100:.0f}%"
        is_corr = r.get("is_correct")
        border_color = "#10b981" if is_corr else "#ef4444"
        badge = '<span class="badge badge-success">OK</span>' if is_corr else '<span class="badge badge-danger">WRONG</span>'
        if r.get("status") == "needs_review":
            badge = '<span class="badge badge-warning">REVIEW</span>'

        html += f"""
    <div class="thumb-card" style="border-color:{border_color};">
      <img src="{rel_img}" alt="Attempt {target}"/>
      <div class="thumb-title">Target: {target}</div>
      <div class="meta-tag">Pred: {pred} ({conf})</div>
      <div style="margin-top:4px;">{badge}</div>
    </div>"""

    html += """
  </div>
</div>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")


def generate_contact_sheet_image(records, output_path, cols=6, thumb_size=160):
    """Generate a combined JPG contact sheet of all non-skipped processed crops."""
    valid_records = [r for r in records if r.get("status") != "skipped" and r.get("processed_image_path")]
    if not valid_records:
        return

    n = len(valid_records)
    rows = (n + cols - 1) // cols
    header_h = 40
    cell_h = thumb_size + header_h
    cell_w = thumb_size

    sheet = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for idx, r in enumerate(valid_records):
        row = idx // cols
        col = idx % cols
        x = col * cell_w
        y = row * cell_h

        img_path = r["processed_image_path"]
        crop = cv2.imread(str(img_path))
        if crop is not None:
            crop_resized = cv2.resize(crop, (cell_w, thumb_size))
            sheet[y:y+thumb_size, x:x+cell_w] = crop_resized

        # Banner background
        target = r.get("target_letter", "").upper()
        pred = r.get("predicted_letter", "").upper()
        is_corr = r.get("is_correct")
        bg_color = (0, 160, 0) if is_corr else (0, 0, 180)
        cv2.rectangle(sheet, (x, y + thumb_size), (x + cell_w, y + cell_h), bg_color, -1)

        conf = f"{r.get('confidence', 0.0)*100:.0f}%"
        label = f"{target} -> {pred} ({conf})"
        cv2.putText(sheet, label, (x + 8, y + thumb_size + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), sheet)


def run_audit(
    checkpoint_path,
    output_base_dir=PROJECT_DIR / "reports" / "alphabet_audit",
    camera_id=0,
    attempts_per_letter=3,
    mirror=False,
    mask_background=None,
    mock_frames=None,
):
    """Execute guided webcam alphabet audit and generate reports."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    # Unique timestamped session directory (NEVER overwriting previous audits)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = output_base_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model_type = checkpoint.get("model_type", "resnet")
    is_gray = checkpoint.get("grayscale", False)
    mask_mode = checkpoint.get("mask_mode", "legacy")
    should_mask = determine_masking_config(checkpoint, checkpoint_path, mask_background)

    model = build_model(model_type, checkpoint)
    transform = create_transform(checkpoint, force_grayscale=is_gray)

    print("\n" + "=" * 65)
    print("      ISL ALPHABET WEBCAM AUDIT & DIAGNOSTIC SESSION")
    print("=" * 65)
    print(f"Model:           {model_type.upper()} ({checkpoint_path.name})")
    print(f"Mask Background: {should_mask}")
    print(f"Grayscale:       {is_gray}")
    print(f"Mirroring:       {mirror} (disabled by default for asymmetric ISL signs)")
    print(f"Attempts/Letter: {attempts_per_letter}")
    print(f"Audit Folder:    {session_dir}")
    print("=" * 65)
    print("Controls: SPACE=Capture | R=Retry | S=Skip | F=Flag Review | Q=Finish & Save\n")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "checkpoint_path": str(checkpoint_path),
        "model_type": model_type,
        "grayscale": is_gray,
        "mask_background": should_mask,
        "mirror": mirror,
        "attempts_per_letter": attempts_per_letter,
    }

    records = []

    # If mock_frames provided (for unit testing), process headlessly
    if mock_frames is not None:
        for letter_idx, letter in enumerate(CLASSES):
            target_letter = letter.upper()
            letter_dir = session_dir / f"letter_{target_letter}"
            letter_dir.mkdir(parents=True, exist_ok=True)

            for att in range(1, attempts_per_letter + 1):
                raw_frame = mock_frames(target_letter, att)
                mock_det_hands = 2 if letter in ISL_TWO_HAND_CLASSES else 1
                if isinstance(raw_frame, tuple):
                    raw_frame, mock_det_hands = raw_frame

                raw_path = letter_dir / f"attempt_{att}_raw.jpg"
                proc_path = letter_dir / f"attempt_{att}_processed.jpg"

                cv2.imwrite(str(raw_path), raw_frame)

                # Simulate center crop if mock
                h, w = raw_frame.shape[:2]
                crop = cv2.resize(raw_frame, (224, 224))
                if is_gray:
                    crop = cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                cv2.imwrite(str(proc_path), crop)

                inputs = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0)
                with torch.inference_mode():
                    probs = torch.softmax(model(inputs), dim=1)[0]
                    top3_p, top3_i = torch.topk(probs, k=min(3, len(classes)))

                pred_char = classes[int(top3_i[0])].upper()
                top3 = [(classes[int(top3_i[k])], float(top3_p[k])) for k in range(len(top3_i))]
                conf = float(top3_p[0])
                is_correct = (pred_char.lower() == letter.lower())

                guard_status, guard_warnings, guard_notes = check_letter_safeguards(
                    target_letter, mock_det_hands, box=(10, 10, w - 10, h - 10), frame_shape=(h, w)
                )

                records.append({
                    "target_letter": target_letter,
                    "attempt": att,
                    "timestamp": datetime.now().isoformat(),
                    "predicted_letter": pred_char,
                    "is_correct": is_correct,
                    "confidence": conf,
                    "status": guard_status,
                    "detected_hands": mock_det_hands,
                    "top3": top3,
                    "raw_image_path": str(raw_path),
                    "processed_image_path": str(proc_path),
                    "processed_image_rel_path": f"letter_{target_letter}/{proc_path.name}",
                    "notes": guard_notes if guard_status != "accepted" else "mock_test_capture",
                })

        metrics = compute_audit_metrics(records)
        generate_csv_report(records, session_dir / "results.csv")
        (session_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        generate_html_report(records, metrics, metadata, session_dir / "summary.html")
        generate_contact_sheet_image(records, session_dir / "contact_sheet.jpg")
        return session_dir, metrics

    # Live Camera Execution
    camera = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_id)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {camera_id}. Please ensure camera is connected and not in use by another app.")

    if not LANDMARKER_PATH.is_file():
        raise FileNotFoundError(f"MediaPipe hand landmarker task file missing: {LANDMARKER_PATH}")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )

    letter_idx = 0
    attempt_num = 1
    last_timestamp_ms = 0
    smooth_box_coords = None

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while letter_idx < len(CLASSES):
                current_char = CLASSES[letter_idx]
                target_letter = current_char.upper()
                letter_dir = session_dir / f"letter_{target_letter}"
                letter_dir.mkdir(parents=True, exist_ok=True)

                hand_req = "1-Hand" if current_char in ISL_SINGLE_HAND_CLASSES else "2-Hand"

                while attempt_num <= attempts_per_letter:
                    ok, frame = camera.read()
                    if not ok:
                        print("Warning: Failed to read frame from camera.")
                        continue
                    if mirror:
                        frame = cv2.flip(frame, 1)

                    h, w = frame.shape[:2]
                    timestamp_ms = int(time.time() * 1000)
                    if timestamp_ms <= last_timestamp_ms:
                        timestamp_ms = last_timestamp_ms + 1
                    last_timestamp_ms = timestamp_ms

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect_for_video(mp_img, timestamp_ms)

                    display = frame.copy()
                    num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0

                    raw_box, crop_mask = None, None
                    if num_hands > 0:
                        raw_box, crop_mask = compute_hand_box_and_mask(
                            result.hand_landmarks, w, h, padding=0.25, square=True,
                            mask_mode=mask_mode,
                        )

                    preview_crop = None
                    if raw_box is not None:
                        smooth_box_coords = smooth_box(raw_box, smooth_box_coords, alpha=0.45)
                        left, top, right, bottom = smooth_box_coords
                        crop = frame[top:bottom, left:right]
                        if crop.size > 0:
                            if should_mask and crop_mask is not None:
                                if crop_mask.shape != crop.shape[:2]:
                                    crop_mask = cv2.resize(crop_mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
                                crop = cv2.bitwise_and(crop, crop, mask=crop_mask)

                            if is_gray:
                                preview_crop = cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                            else:
                                preview_crop = crop.copy()

                            cv2.rectangle(display, (left, top), (right, bottom), (0, 220, 0), 2)

                    # Top Banner
                    cv2.rectangle(display, (0, 0), (w, 88), (20, 25, 35), -1)
                    banner_text = f"TARGET: [{target_letter}] ({hand_req})  --  Attempt {attempt_num}/{attempts_per_letter}"
                    cv2.putText(display, banner_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2, cv2.LINE_AA)

                    guidance = LETTER_AUDIT_GUIDANCE.get(target_letter, "")
                    if guidance:
                        cv2.putText(display, f"Hint: {guidance}", (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 230, 120), 1, cv2.LINE_AA)

                    guard_status, guard_warnings, guard_notes = check_letter_safeguards(
                        target_letter, num_hands, box=smooth_box_coords, frame_shape=(h, w)
                    )
                    if guard_warnings:
                        warn_color = (0, 0, 255) if (target_letter == "H" and num_hands < 2) else (0, 165, 255)
                        warn_y = 76 if guidance else 56
                        cv2.putText(display, f"Warning: {guard_warnings[0]}", (20, warn_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, warn_color, 1, cv2.LINE_AA)

                    # Picture-in-picture preview of what will be evaluated
                    if preview_crop is not None:
                        pip_size = 140
                        pip_img = cv2.resize(preview_crop, (pip_size, pip_size))
                        display[95:95+pip_size, w - pip_size - 15 : w - 15] = pip_img
                        cv2.rectangle(display, (w - pip_size - 15, 95), (w - 15, 95 + pip_size), (0, 220, 255), 2)
                        cv2.putText(display, "MODEL INPUT", (w - pip_size - 15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

                    # Footer Instructions
                    cv2.rectangle(display, (0, h - 45), (w, h), (20, 25, 35), -1)
                    cv2.putText(display, "SPACE: Confirm & Capture | R: Retake | S: Skip | F: Flag Review | Q: Finish",
                                (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

                    cv2.imshow("ISL Alphabet Audit", display)
                    key = cv2.waitKey(1) & 0xFF

                    # Actions
                    if key in (ord("q"), 27):
                        print("\nAudit terminated early by user. Generating report for captured letters...")
                        letter_idx = len(CLASSES)
                        break

                    elif key == 32:  # SPACE: Confirm & Capture
                        if preview_crop is None:
                            print("No hand detected. Please position your hand in frame before pressing SPACE.")
                            continue

                        raw_path = letter_dir / f"attempt_{attempt_num}_raw.jpg"
                        proc_path = letter_dir / f"attempt_{attempt_num}_processed.jpg"

                        cv2.imwrite(str(raw_path), frame)
                        save_crop = cv2.resize(preview_crop, (224, 224))
                        cv2.imwrite(str(proc_path), save_crop)

                        # Inference
                        inputs = transform(cv2.cvtColor(save_crop, cv2.COLOR_BGR2RGB)).unsqueeze(0)
                        with torch.inference_mode():
                            probs = torch.softmax(model(inputs), dim=1)[0]
                            top3_p, top3_i = torch.topk(probs, k=min(3, len(classes)))

                        pred_char = classes[int(top3_i[0])].upper()
                        # Guarded N vs M landmark disambiguation reusing existing landmarks
                        effective_char, nm_info = correct_n_m_prediction(
                            pred_char,
                            result.hand_landmarks if result else None,
                        )
                        pred_char = effective_char.upper()
                        conf = float(top3_p[0])
                        top3 = [(classes[int(top3_i[k])], float(top3_p[k])) for k in range(len(top3_i))]
                        is_corr = (pred_char.lower() == current_char.lower())

                        guard_status, guard_warnings, guard_notes = check_letter_safeguards(
                            target_letter, num_hands, box=smooth_box_coords, frame_shape=(h, w)
                        )
                        status = guard_status
                        notes = guard_notes
                        if nm_info.get("applied") and nm_info.get("changed"):
                            notes = f"{notes};nm_corrected"

                        if target_letter == "H" and num_hands < 2:
                            print(f"Warning: H requires two hands ({num_hands} detected). Marked capture as uncertain. Press R to retake if desired.")

                        records.append({
                            "target_letter": target_letter,
                            "attempt": attempt_num,
                            "timestamp": datetime.now().isoformat(),
                            "predicted_letter": pred_char,
                            "is_correct": is_corr,
                            "confidence": conf,
                            "status": status,
                            "detected_hands": num_hands,
                            "top3": top3,
                            "raw_image_path": str(raw_path),
                            "processed_image_path": str(proc_path),
                            "processed_image_rel_path": f"letter_{target_letter}/{proc_path.name}",
                            "notes": notes,
                        })

                        result_sym = "[OK]" if is_corr else "[FAIL]"
                        print(f"Captured {target_letter} #{attempt_num}: Predicted {pred_char} ({conf*100:.1f}%) {result_sym}")
                        attempt_num += 1

                    elif key in (ord("s"), ord("S")):  # S: Skip
                        records.append({
                            "target_letter": target_letter,
                            "attempt": attempt_num,
                            "timestamp": datetime.now().isoformat(),
                            "predicted_letter": "",
                            "is_correct": None,
                            "confidence": None,
                            "status": "skipped",
                            "detected_hands": num_hands,
                            "top3": [],
                            "raw_image_path": "",
                            "processed_image_path": "",
                            "processed_image_rel_path": "",
                            "notes": "user_skipped",
                        })
                        print(f"Skipped attempt #{attempt_num} for letter {target_letter}.")
                        attempt_num += 1

                    elif key in (ord("r"), ord("R")):  # R: Retry current attempt
                        print(f"Retrying attempt #{attempt_num} for letter {target_letter}...")

                    elif key in (ord("f"), ord("F")):  # F: Flag review
                        if preview_crop is not None:
                            raw_path = letter_dir / f"attempt_{attempt_num}_raw.jpg"
                            proc_path = letter_dir / f"attempt_{attempt_num}_processed.jpg"
                            cv2.imwrite(str(raw_path), frame)
                            save_crop = cv2.resize(preview_crop, (224, 224))
                            cv2.imwrite(str(proc_path), save_crop)

                            inputs = transform(cv2.cvtColor(save_crop, cv2.COLOR_BGR2RGB)).unsqueeze(0)
                            with torch.inference_mode():
                                probs = torch.softmax(model(inputs), dim=1)[0]
                                top3_p, top3_i = torch.topk(probs, k=min(3, len(classes)))

                            pred_char = classes[int(top3_i[0])].upper()
                            conf = float(top3_p[0])
                            top3 = [(classes[int(top3_i[k])], float(top3_p[k])) for k in range(len(top3_i))]

                            records.append({
                                "target_letter": target_letter,
                                "attempt": attempt_num,
                                "timestamp": datetime.now().isoformat(),
                                "predicted_letter": pred_char,
                                "is_correct": (pred_char.lower() == current_char.lower()),
                                "confidence": conf,
                                "status": "needs_review",
                                "detected_hands": num_hands,
                                "top3": top3,
                                "raw_image_path": str(raw_path),
                                "processed_image_path": str(proc_path),
                                "processed_image_rel_path": f"letter_{target_letter}/{proc_path.name}",
                                "notes": "flagged_by_user",
                            })
                            print(f"Flagged attempt #{attempt_num} of {target_letter} for manual review.")
                            attempt_num += 1

                # Advance to next letter
                letter_idx += 1
                attempt_num = 1

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("\nCompiling audit reports...")
    metrics = compute_audit_metrics(records)
    generate_csv_report(records, session_dir / "results.csv")
    (session_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    generate_html_report(records, metrics, metadata, session_dir / "summary.html")
    generate_contact_sheet_image(records, session_dir / "contact_sheet.jpg")

    print("\n" + "=" * 65)
    print("                AUDIT COMPLETE")
    print("=" * 65)
    print(f"Total Attempts:      {metrics['total_attempts']}")
    print(f"Evaluated Attempts:  {metrics['valid_attempts']} (Skipped: {metrics['skipped_attempts']})")
    print(f"Overall Accuracy:    {metrics['overall_accuracy'] * 100:.1f}%")
    print(f"HTML Dashboard:      {session_dir / 'summary.html'}")
    print(f"CSV Attempt Log:     {session_dir / 'results.csv'}")
    print(f"JSON Machine Report: {session_dir / 'summary.json'}")
    print("=" * 65)
    return session_dir, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run ISL Alphabet Webcam Audit & Reporting Mode.")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "isl_resnet.pth",
                        help="Path to trained model checkpoint.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--attempts", type=int, default=3, help="Number of attempts per alphabet letter.")
    parser.add_argument("--mirror", action="store_true", default=False,
                        help="Mirror preview horizontally (DEFAULT: False). Note: ISL signs are asymmetric; leave off for normal signing.")
    parser.add_argument("--mask-background", action="store_true", default=None,
                        help="Explicitly enable silhouette background masking.")
    parser.add_argument("--no-mask-background", dest="mask_background", action="store_false",
                        help="Explicitly disable silhouette background masking.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "reports" / "alphabet_audit",
                        help="Directory to store audit sessions.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_audit(
        checkpoint_path=args.checkpoint,
        output_base_dir=args.output_dir,
        camera_id=args.camera,
        attempts_per_letter=args.attempts,
        mirror=args.mirror,
        mask_background=args.mask_background,
    )


if __name__ == "__main__":
    main()
