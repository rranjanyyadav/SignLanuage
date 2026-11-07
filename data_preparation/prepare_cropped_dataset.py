import sys
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm.auto import tqdm


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
LANDMARKER_PATH = PROJECT_DIR / "hand_landmarker.task"
CLASSES = tuple("abcdefghijklmnopqrstuvwxyz")


def compute_hand_box_and_mask(landmarks_list, width, height, padding=0.25, square=True,
                              mask_mode="legacy", finger_thickness_ratio=0.06,
                              dilation_kernel_size=5, dilate_iterations=1):
    """Compute square bounding box and binary hand silhouette mask from 21 landmarks.
    
    Supports two mask modes:
      - 'legacy': Full-hand convex hull + 0.14 bone thickness + 15x15 double dilation.
      - 'fine': Palm-only convex hull + fine bone thickness + fingertip caps + minimal dilation.
    """
    all_pts = []
    for hand in landmarks_list:
        for pt in hand:
            all_pts.append([int(pt.x * width), int(pt.y * height)])
    if not all_pts:
        return None, None

    all_pts = np.array(all_pts, dtype=np.int32)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    # Compute square bounding box with padding
    w_box = x_max - x_min
    h_box = y_max - y_min
    pad = int(max(w_box, h_box) * padding)

    left = max(0, x_min - pad)
    top = max(0, y_min - pad)
    right = min(width, x_max + pad)
    bottom = min(height, y_max + pad)

    if square:
        size = max(right - left, bottom - top)
        cx, cy = (left + right) // 2, (top + bottom) // 2
        left = max(0, cx - size // 2)
        top = max(0, cy - size // 2)
        right = min(width, left + size)
        bottom = min(height, top + size)

    if right <= left or bottom <= top:
        return None, None

    # Construct hand silhouette mask
    mask = np.zeros((height, width), dtype=np.uint8)
    if mask_mode == "legacy":
        for hand in landmarks_list:
            pts = np.array([[int(pt.x * width), int(pt.y * height)] for pt in hand], dtype=np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)

            # Draw bone connections with realistic finger radius
            if len(hand) == 21:
                thickness = max(8, int(max(w_box, h_box) * 0.14))
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (17, 18), (18, 19), (19, 20),
                    (0, 17),
                ]
                for c1, c2 in connections:
                    p1 = (int(hand[c1].x * width), int(hand[c1].y * height))
                    p2 = (int(hand[c2].x * width), int(hand[c2].y * height))
                    cv2.line(mask, p1, p2, 255, thickness=thickness)

        # Smooth the mask boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel, iterations=2)

    elif mask_mode == "fine":
        palm_indices = [0, 1, 2, 5, 9, 13, 17]
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]
        fingertip_indices = [4, 8, 12, 16, 20]

        for hand in landmarks_list:
            # 1. Fill ONLY the palm polygon, not the full hand convex hull
            if len(hand) >= 18:
                palm_pts = np.array([[int(hand[i].x * width), int(hand[i].y * height)] for i in palm_indices], dtype=np.int32)
                palm_hull = cv2.convexHull(palm_pts)
                cv2.fillConvexPoly(mask, palm_hull, 255)
            else:
                pts = np.array([[int(pt.x * width), int(pt.y * height)] for pt in hand], dtype=np.int32)
                cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)

            # 2. Draw bone connections with configurable thinner finger radius
            thickness = max(4, int(max(w_box, h_box) * finger_thickness_ratio))
            if len(hand) == 21:
                for c1, c2 in connections:
                    p1 = (int(hand[c1].x * width), int(hand[c1].y * height))
                    p2 = (int(hand[c2].x * width), int(hand[c2].y * height))
                    cv2.line(mask, p1, p2, 255, thickness=thickness)

                # 3. Draw circular caps at fingertips
                for tip_idx in fingertip_indices:
                    pt = (int(hand[tip_idx].x * width), int(hand[tip_idx].y * height))
                    cv2.circle(mask, pt, max(2, thickness // 2), 255, -1)

        # 4. Minimal dilation only if necessary to smooth antialiased edges
        if dilation_kernel_size > 0 and dilate_iterations > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
            mask = cv2.dilate(mask, k, iterations=dilate_iterations)

    elif mask_mode in ("none", None, "raw"):
        return (left, top, right, bottom), None
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode}. Expected 'legacy' or 'fine'.")

    return (left, top, right, bottom), mask[top:bottom, left:right]


def collect_diverse_images(source_dir, samples_per_class=150):
    """Collect diverse images per class across Kids, Teens, Adults in full/half sleeves.

    If samples_per_class is None or <= 0, collects all available images.
    Returns list of tuples: (letter, img_path, cohort_tag).
    """
    source_dir = Path(source_dir)
    class_groups = defaultdict(list)

    # Search for English Alphabet folders
    for age in sorted(source_dir.iterdir()):
        if not age.is_dir(): continue
        age_slug = "kids" if "kid" in age.name.lower() else ("teens" if "teen" in age.name.lower() else "adults")
        for sleeve in sorted(age.iterdir()):
            if not sleeve.is_dir(): continue
            sleeve_slug = "full" if "full" in sleeve.name.lower() else "half"
            cohort_tag = f"{age_slug}_{sleeve_slug}"
            for cat in sorted(sleeve.iterdir()):
                if 'alphabet' not in cat.name.lower(): continue
                for letter_dir in sorted(cat.iterdir()):
                    if not letter_dir.is_dir(): continue
                    raw_name = letter_dir.name.upper()
                    mapped_letter = "E" if raw_name in ("E1", "E2") else raw_name
                    if len(mapped_letter) == 1 and mapped_letter.isalpha():
                        files = list(letter_dir.glob("*.jpg")) + list(letter_dir.glob("*.jpeg")) + list(letter_dir.glob("*.png"))
                        for f in files:
                            class_groups[mapped_letter.lower()].append((f, cohort_tag))

    selected = []
    for letter in CLASSES:
        items = class_groups[letter]
        if not items:
            continue
        if samples_per_class is not None and samples_per_class > 0 and len(items) > samples_per_class:
            step = max(1, len(items) // samples_per_class)
            sampled = items[::step][:samples_per_class]
        else:
            sampled = items
        for f, tag in sampled:
            selected.append((letter, f, tag))

    return selected


def process_dataset(source_dir, output_dir, landmarker_path=LANDMARKER_PATH,
                    samples_per_class=150, grayscale=True, mask_background=True,
                    target_size=224, padding=0.25,
                    mask_mode="legacy", finger_thickness_ratio=0.06,
                    dilation_kernel_size=5, dilate_iterations=1,
                    skip_failed=True, skip_existing=False, summary_path=None):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if not Path(landmarker_path).is_file():
        raise FileNotFoundError(f"MediaPipe task file not found: {landmarker_path}")

    start_time = time.time()
    sample_desc = f"{samples_per_class} per class" if (samples_per_class and samples_per_class > 0) else "ALL available"
    print(f"Sampling {sample_desc} images from {source_dir}...")
    image_items = collect_diverse_images(source_dir, samples_per_class=samples_per_class)
    print(f"Collected {len(image_items)} total diverse images across all 26 alphabet classes.")
    print(f"Preprocessing options: grayscale={grayscale}, mask_background={mask_background}, target_size={target_size}, mask_mode={mask_mode}, skip_failed={skip_failed}")
    print(f"Saving prepared dataset to: {output_dir}")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(landmarker_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )

    success_count = 0
    fallback_count = 0
    class_success = defaultdict(int)
    class_failed = defaultdict(int)
    cohort_success = defaultdict(int)
    cohort_failed = defaultdict(int)
    skipped_records = []

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        pbar = tqdm(image_items, desc="Extracting masked crops", unit="img", dynamic_ncols=True)
        for idx, (letter, img_path, cohort_tag) in enumerate(pbar):
            out_class_dir = output_dir / letter
            out_class_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_class_dir / f"{letter}_{cohort_tag}_{img_path.stem}.jpg"

            if skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
                success_count += 1
                class_success[letter] += 1
                cohort_success[cohort_tag] += 1
                if (idx + 1) % 25 == 0:
                    pbar.set_postfix({"ok": success_count, "skip": fallback_count})
                continue

            bgr = cv2.imread(str(img_path))
            if bgr is None:
                class_failed[letter] += 1
                cohort_failed[cohort_tag] += 1
                fallback_count += 1
                skipped_records.append({"letter": letter, "path": str(img_path), "reason": "unreadable_image"})
                if (idx + 1) % 25 == 0:
                    pbar.set_postfix({"ok": success_count, "skip": fallback_count})
                continue

            h, w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            box, crop_mask = None, None
            if result.hand_landmarks:
                box, crop_mask = compute_hand_box_and_mask(
                    result.hand_landmarks, w, h, padding=padding, square=True,
                    mask_mode=mask_mode,
                    finger_thickness_ratio=finger_thickness_ratio,
                    dilation_kernel_size=dilation_kernel_size,
                    dilate_iterations=dilate_iterations,
                )

            if box is not None and crop_mask is not None:
                left, top, right, bottom = box
                crop = bgr[top:bottom, left:right]
                if mask_background:
                    crop = cv2.bitwise_and(crop, crop, mask=crop_mask)
                success_count += 1
                class_success[letter] += 1
                cohort_success[cohort_tag] += 1
            else:
                class_failed[letter] += 1
                cohort_failed[cohort_tag] += 1
                fallback_count += 1
                skipped_records.append({"letter": letter, "path": str(img_path), "reason": "no_hand_landmarks"})
                if (idx + 1) % 20 == 0:
                    pbar.set_postfix({"ok": success_count, "skip": fallback_count})
                if skip_failed:
                    continue
                # Fallback: square center crop if hands not recognized
                side = min(w, h)
                cx, cy = w // 2, h // 2
                crop = bgr[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2]

            if grayscale:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            if target_size:
                crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)

            cv2.imwrite(str(out_path), crop)
            if (idx + 1) % 20 == 0:
                pbar.set_postfix({"ok": success_count, "skip": fallback_count})

    elapsed = time.time() - start_time
    print(f"\nProcessing complete in {elapsed:.2f}s ({len(image_items) / max(elapsed, 0.001):.1f} fps)!")
    print(f"Hand crops extracted successfully: {success_count} ({success_count / len(image_items):.1%})")
    print(f"Failed / Skipped: {fallback_count} ({fallback_count / len(image_items):.1%})")
    print(f"Dataset ready at: {output_dir}")

    summary_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "total_source_images": len(image_items),
        "total_successful_crops": success_count,
        "total_skipped_crops": fallback_count,
        "overall_success_rate": success_count / max(len(image_items), 1),
        "output_dimensions": [target_size, target_size],
        "grayscale": grayscale,
        "mask_mode": mask_mode,
        "elapsed_seconds": elapsed,
        "per_class_counts": {c: class_success[c] for c in CLASSES},
        "per_class_failed": {c: class_failed[c] for c in CLASSES},
        "per_cohort_success": dict(cohort_success),
        "per_cohort_failed": dict(cohort_failed),
        "total_skipped_records": len(skipped_records),
    }

    if summary_path is None:
        summary_path = PROJECT_DIR / "reports" / "preprocessing_large_masked_summary.json"
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"Saved extraction summary report to: {summary_path}")

    skipped_log_path = summary_path.parent / (summary_path.stem + "_skipped.json")
    skipped_log_path.write_text(json.dumps(skipped_records, indent=2), encoding="utf-8")
    print(f"Saved skipped images record ({len(skipped_records)} items) to: {skipped_log_path}")

    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Extract Grayscale Masked Hand Crops from diverse ISL images.")
    parser.add_argument("--source-dir", type=Path, default=PROJECT_DIR / "ISL Images")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "dataset_ISL_masked")
    parser.add_argument("--samples-per-class", type=int, default=150,
                        help="Number of diverse signers/sessions to sample per letter (0 or negative for all available).")
    parser.add_argument("--no-grayscale", action="store_true", help="Keep 3-channel BGR instead of grayscale.")
    parser.add_argument("--no-mask", action="store_true", help="Do not mask background to black.")
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--mask-mode", choices=["legacy", "fine"], default="legacy",
                        help="Mask generation mode: 'legacy' (full convex hull + 15x15 dilation) or 'fine' (palm-only hull + fine finger bones).")
    parser.add_argument("--finger-thickness-ratio", type=float, default=0.06,
                        help="Finger stroke thickness as a fraction of bounding box dimension in fine mode (default: 0.06).")
    parser.add_argument("--dilation-kernel-size", type=int, default=5,
                        help="Dilation kernel size in fine mode (default: 5; use 0 for no dilation).")
    parser.add_argument("--dilate-iterations", type=int, default=1,
                        help="Dilation iterations in fine mode (default: 1).")
    parser.add_argument("--keep-failed-crops", action="store_true",
                        help="Substitute center crops for failed detections instead of skipping them.")
    parser.add_argument("--skip-existing", "--resume", dest="skip_existing", action="store_true",
                        help="Skip images that are already extracted in the output directory to resume an interrupted run.")
    parser.add_argument("--summary-path", type=Path, default=None,
                        help="Path to save summary report JSON.")
    args = parser.parse_args()

    process_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        samples_per_class=args.samples_per_class,
        grayscale=not args.no_grayscale,
        mask_background=not args.no_mask,
        target_size=args.size,
        padding=args.padding,
        mask_mode=args.mask_mode,
        finger_thickness_ratio=args.finger_thickness_ratio,
        dilation_kernel_size=args.dilation_kernel_size,
        dilate_iterations=args.dilate_iterations,
        skip_failed=not args.keep_failed_crops,
        skip_existing=args.skip_existing,
        summary_path=args.summary_path,
    )


if __name__ == "__main__":
    main()
