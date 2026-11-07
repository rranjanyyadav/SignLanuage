import sys
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
CSV_PATH = PROJECT_DIR / "Indian Sign Language Gesture Landmarks.csv"
OUTPUT_DIR = PROJECT_DIR / "dataset_skeleton"
CLASSES = list("abcdefghijklmnopqrstuvwxyz")

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def render_skeleton_image(landmarks_list, canvas_size=224, padding=0.18):
    """Render a clean normalized skeleton map onto a black canvas.

    Args:
        landmarks_list: List of hands, where each hand has 21 [x, y] or [x, y, z] points.
        canvas_size: Square canvas pixel dimension (default 224).
        padding: Margin fraction to keep hands centered without touching edges.

    Returns:
        RGB image (canvas_size, canvas_size, 3) with anti-aliased bones and joints.
    """
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    all_pts = []
    for hand in landmarks_list:
        for p in hand:
            if hasattr(p, "x"):
                all_pts.append([p.x, p.y])
            else:
                all_pts.append([float(p[0]), float(p[1])])

    if not all_pts:
        return canvas

    all_pts = np.array(all_pts, dtype=np.float32)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)

    span = max(float(max_xy[0] - min_xy[0]), float(max_xy[1] - min_xy[1]), 1e-5)
    usable_size = canvas_size * (1.0 - 2 * padding)
    scale = usable_size / span

    center_data = (min_xy + max_xy) / 2.0
    center_canvas = canvas_size / 2.0

    for hand in landmarks_list:
        pts_canvas = []
        for p in hand:
            px = p.x if hasattr(p, "x") else float(p[0])
            py = p.y if hasattr(p, "y") else float(p[1])
            cx = int(center_canvas + (px - center_data[0]) * scale)
            cy = int(center_canvas + (py - center_data[1]) * scale)
            # Clamp to canvas boundaries
            cx = max(2, min(canvas_size - 3, cx))
            cy = max(2, min(canvas_size - 3, cy))
            pts_canvas.append((cx, cy))

        # Draw bone lines with anti-aliasing
        for c1, c2 in CONNECTIONS:
            cv2.line(canvas, pts_canvas[c1], pts_canvas[c2], (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)

        # Draw joint nodes
        for pt in pts_canvas:
            cv2.circle(canvas, pt, 4, (0, 220, 255), -1, lineType=cv2.LINE_AA)

    return canvas


def generate_dataset(csv_path=CSV_PATH, output_dir=OUTPUT_DIR, samples_per_class=200, canvas_size=224):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Landmarks CSV not found: {csv_path}")

    print(f"Reading landmarks from {csv_path}...")
    rows_by_class = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            target = int(row["target"])
            if 0 <= target < len(CLASSES):
                rows_by_class[target].append(row)

    print(f"Found {sum(len(v) for v in rows_by_class.values())} total rows across {len(rows_by_class)} classes.")
    print(f"Generating {samples_per_class} skeleton images per class into {output_dir}...")

    total_saved = 0
    for target in range(len(CLASSES)):
        letter = CLASSES[target]
        letter_dir = output_dir / letter
        letter_dir.mkdir(parents=True, exist_ok=True)

        rows = rows_by_class[target]
        # Evenly stride through the session rows
        step = max(1, len(rows) // samples_per_class)
        sampled = rows[::step][:samples_per_class]

        for idx, row in enumerate(sampled):
            u2 = float(row["uses_two_hands"])
            left_valid = float(row["left_hand_x_0"]) > -0.5
            right_valid = float(row["right_hand_x_0"]) > -0.5

            hands = []
            if left_valid:
                hands.append([[float(row[f"left_hand_x_{i}"]), float(row[f"left_hand_y_{i}"])] for i in range(21)])
            if u2 == 1 and right_valid:
                hands.append([[float(row[f"right_hand_x_{i}"]), float(row[f"right_hand_y_{i}"])] for i in range(21)])

            img = render_skeleton_image(hands, canvas_size=canvas_size)
            out_path = letter_dir / f"{letter}_{idx:04d}.png"
            cv2.imwrite(str(out_path), img)
            total_saved += 1

    print(f"Done! Generated {total_saved} clean skeleton maps at {output_dir}.")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic normalized hand skeleton dataset from CSV.")
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    generate_dataset(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        samples_per_class=args.samples_per_class,
        canvas_size=args.size,
    )


if __name__ == "__main__":
    main()

