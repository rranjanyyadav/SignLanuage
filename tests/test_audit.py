import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from audit_alphabet import (
    compute_audit_metrics,
    generate_csv_report,
    generate_html_report,
    determine_masking_config,
    run_audit,
    LETTER_AUDIT_GUIDANCE,
    check_letter_safeguards,
)


class TestAlphabetAudit(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compute_audit_metrics_basic(self):
        """Metrics correctly compute accuracy, counts, and per-letter stats."""
        records = [
            {"target_letter": "A", "predicted_letter": "A", "is_correct": True, "status": "accepted"},
            {"target_letter": "A", "predicted_letter": "A", "is_correct": True, "status": "accepted"},
            {"target_letter": "B", "predicted_letter": "B", "is_correct": True, "status": "accepted"},
            {"target_letter": "B", "predicted_letter": "C", "is_correct": False, "status": "accepted"},
        ]
        metrics = compute_audit_metrics(records)

        self.assertEqual(metrics["total_attempts"], 4)
        self.assertEqual(metrics["valid_attempts"], 4)
        self.assertEqual(metrics["correct_attempts"], 3)
        self.assertAlmostEqual(metrics["overall_accuracy"], 0.75)
        self.assertEqual(metrics["per_letter"]["A"]["accuracy"], 1.0)
        self.assertEqual(metrics["per_letter"]["B"]["accuracy"], 0.5)
        self.assertTrue(metrics["per_letter"]["B"]["is_inconsistent"])
        self.assertIn("B", metrics["letters_needing_attention"])

    def test_skipped_attempts_handling(self):
        """Skipped attempts are not counted in accuracy denominator."""
        records = [
            {"target_letter": "A", "predicted_letter": "A", "is_correct": True, "status": "accepted"},
            {"target_letter": "A", "predicted_letter": "", "is_correct": None, "status": "skipped"},
            {"target_letter": "B", "predicted_letter": "", "is_correct": None, "status": "skipped"},
        ]
        metrics = compute_audit_metrics(records)

        self.assertEqual(metrics["total_attempts"], 3)
        self.assertEqual(metrics["valid_attempts"], 1)
        self.assertEqual(metrics["skipped_attempts"], 2)
        self.assertEqual(metrics["correct_attempts"], 1)
        self.assertEqual(metrics["overall_accuracy"], 1.0)
        self.assertEqual(metrics["per_letter"]["A"]["valid_attempts"], 1)
        self.assertEqual(metrics["per_letter"]["A"]["skipped_attempts"], 1)
        self.assertEqual(metrics["per_letter"]["B"]["valid_attempts"], 0)

    def test_determine_masking_config(self):
        """Masking config checks CLI flags, metadata, and path without assuming grayscale implies masking."""
        # Case 1: CLI override True
        self.assertTrue(determine_masking_config({"grayscale": False}, Path("models/model.pth"), cli_flag=True))
        # Case 2: CLI override False
        self.assertFalse(determine_masking_config({"grayscale": True, "mask_background": True}, Path("models/model.pth"), cli_flag=False))
        # Case 3: Explicit checkpoint metadata
        self.assertTrue(determine_masking_config({"mask_background": True}, Path("models/model.pth")))
        self.assertFalse(determine_masking_config({"mask_background": False}, Path("models/model.pth")))
        # Case 4: Name contains 'masked' or known masked models
        self.assertTrue(determine_masking_config({}, Path("models/isl_masked_resnet.pth")))
        self.assertTrue(determine_masking_config({}, Path("models/isl_resnet.pth")))
        # Case 5: Grayscale alone does NOT infer masking for unknown/webcam models
        self.assertFalse(determine_masking_config({"grayscale": True}, Path("models/isl_webcam.pth")))

    def test_report_generation(self):
        """CSV, JSON, and HTML reports are generated with complete fields."""
        records = [
            {
                "target_letter": "A",
                "attempt": 1,
                "timestamp": "2026-10-28T01:00:00",
                "predicted_letter": "A",
                "is_correct": True,
                "confidence": 0.95,
                "status": "accepted",
                "detected_hands": 2,
                "top3": [("A", 0.95), ("B", 0.03), ("E", 0.01)],
                "raw_image_path": "path/raw.jpg",
                "processed_image_path": "path/proc.jpg",
                "notes": "confirmed",
            }
        ]
        metrics = compute_audit_metrics(records)
        metadata = {
            "timestamp": "2026-10-28T01:00:00",
            "checkpoint_path": "models/isl_resnet.pth",
            "model_type": "resnet",
            "grayscale": True,
            "mask_background": True,
        }

        csv_path = self.output_dir / "results.csv"
        html_path = self.output_dir / "summary.html"
        json_path = self.output_dir / "summary.json"

        generate_csv_report(records, csv_path)
        generate_html_report(records, metrics, metadata, html_path)
        json_path.write_text(json.dumps(metrics, indent=2))

        self.assertTrue(csv_path.is_file())
        self.assertTrue(html_path.is_file())
        self.assertTrue(json_path.is_file())

        html_content = html_path.read_text(encoding="utf-8")
        self.assertIn("ISL Alphabet Webcam Audit Report", html_content)
        self.assertIn("TARGET: A", html_content.upper())

    def test_h_missing_hand_guard(self):
        """H strictly requires 2 flat hands; 1 hand triggers uncertain status and warning."""
        status_1, warns_1, note_1 = check_letter_safeguards("H", num_hands=1)
        self.assertEqual(status_1, "uncertain")
        self.assertEqual(note_1, "missing_second_hand_h")
        self.assertTrue(any("two hands" in w.lower() for w in warns_1))

        status_2, warns_2, note_2 = check_letter_safeguards("H", num_hands=2)
        self.assertEqual(status_2, "accepted")
        self.assertEqual(note_2, "manual_confirmed")
        self.assertEqual(len(warns_2), 0)

    def test_x_guidance_and_crop_safeguards(self):
        """X includes index and thumb guidance; clipping and resolution checks behave correctly."""
        # Guidance content check
        self.assertIn("X", LETTER_AUDIT_GUIDANCE)
        x_guidance = LETTER_AUDIT_GUIDANCE["X"].lower()
        self.assertIn("index", x_guidance)
        self.assertIn("thumb", x_guidance)

        # Well-framed crop: 250x250 inside 480x640 frame
        status_ok, warns_ok, note_ok = check_letter_safeguards(
            "X", num_hands=2, box=(50, 50, 300, 300), frame_shape=(480, 640)
        )
        self.assertEqual(status_ok, "accepted")
        self.assertEqual(note_ok, "manual_confirmed")
        self.assertEqual(len(warns_ok), 0)

        # Boundary clipping: left=0
        status_clip, warns_clip, note_clip = check_letter_safeguards(
            "X", num_hands=2, box=(0, 50, 250, 300), frame_shape=(480, 640)
        )
        self.assertEqual(status_clip, "accepted")
        self.assertEqual(note_clip, "crop_clipped_boundary")
        self.assertTrue(any("boundary" in w.lower() for w in warns_clip))

        # Too small: 100x100 (< 224 native model resolution)
        status_small, warns_small, note_small = check_letter_safeguards(
            "X", num_hands=2, box=(50, 50, 150, 150), frame_shape=(480, 640)
        )
        self.assertEqual(status_small, "accepted")
        self.assertEqual(note_small, "crop_below_native_resolution")
        self.assertTrue(any("224" in w for w in warns_small))

    def test_other_letters_safeguards_preservation(self):
        """Single-hand and two-hand letters other than H preserve standard status rules."""
        # Single hand letter: C
        status_c, warns_c, note_c = check_letter_safeguards("C", num_hands=1)
        self.assertEqual(status_c, "accepted")
        self.assertEqual(note_c, "manual_confirmed")

        # Two-hand letter with 2 hands: A
        status_a, warns_a, note_a = check_letter_safeguards("A", num_hands=2)
        self.assertEqual(status_a, "accepted")
        self.assertEqual(note_a, "manual_confirmed")

        # Two-hand letter with 1 hand: A
        status_a1, warns_a1, note_a1 = check_letter_safeguards("A", num_hands=1)
        self.assertEqual(status_a1, "uncertain")
        self.assertEqual(note_a1, "missing_second_hand")

    def test_mock_audit_execution(self):
        """Audit runs end-to-end with mock camera frames and generates reports."""
        ckpt_path = Path("models/isl_resnet.pth")
        if not ckpt_path.is_file():
            self.skipTest("models/isl_resnet.pth not available")

        # Mock frame provider returns simple 240x320 RGB frames
        def mock_provider(letter, attempt):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[50:180, 80:220] = 180  # Dummy bright hand patch
            return frame

        session_dir, metrics = run_audit(
            checkpoint_path=ckpt_path,
            output_base_dir=self.output_dir,
            attempts_per_letter=1,
            mock_frames=mock_provider,
        )

        self.assertTrue(session_dir.is_dir())
        self.assertTrue((session_dir / "summary.html").is_file())
        self.assertTrue((session_dir / "results.csv").is_file())
        self.assertTrue((session_dir / "summary.json").is_file())
        self.assertTrue((session_dir / "contact_sheet.jpg").is_file())
        self.assertEqual(metrics["total_attempts"], 26)


if __name__ == "__main__":
    unittest.main()
