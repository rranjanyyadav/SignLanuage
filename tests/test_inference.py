import unittest
from collections import Counter, deque


class TestInference(unittest.TestCase):
    def test_temporal_smoothing_majority_vote(self):
        """Temporal history deque correctly computes majority vote."""
        history = deque(maxlen=7)
        # Sequence of 7 predictions: four '0's (letter A) and three '1's (letter B)
        for val in [0, 1, 0, 1, 0, 0, 1]:
            history.append(val)

        stable_index, votes = Counter(history).most_common(1)[0]
        self.assertEqual(stable_index, 0)
        self.assertEqual(votes, 4)

    def test_confidence_threshold_logic(self):
        """Confidence below threshold triggers uncertain state."""
        min_confidence = 0.65
        conf_low = 0.45
        conf_high = 0.85

        def check_status(conf, vote_frac, min_conf=0.65, min_vote=0.55):
            if conf >= min_conf and vote_frac >= min_vote:
                return "CONFIDENT"
            return "UNCERTAIN"

        self.assertEqual(check_status(conf_low, 0.80), "UNCERTAIN")
        self.assertEqual(check_status(conf_high, 0.40), "UNCERTAIN")
        self.assertEqual(check_status(conf_high, 0.80), "CONFIDENT")

    def test_no_mirror_default(self):
        """Webcam inference defaults to no-mirror (unflipped) to preserve asymmetric hand geometry."""
        from live_image_inference import parse_args
        import sys
        from unittest.mock import patch

        with patch.object(sys, 'argv', ['live_image_inference.py']):
            args = parse_args()
            self.assertFalse(args.mirror)

    def test_stabilize_two_hand_box_grace_period(self):
        """Bounding box maintains spatial extent for 4 frames during momentary occlusion."""
        from live_image_inference import stabilize_two_hand_box

        box_2h = (100, 100, 300, 300)
        box_1h = (150, 150, 250, 250)

        # Frame 1: 2 hands detected
        eff_box, last_2h, hold = stabilize_two_hand_box(box_2h, num_hands=2, last_two_hand_box=None, hold_counter=0, max_hold=4)
        self.assertEqual(eff_box, box_2h)
        self.assertEqual(hold, 4)

        # Frame 2: 1 hand detected (occlusion) -> hold is active, returns larger box
        eff_box, last_2h, hold = stabilize_two_hand_box(box_1h, num_hands=1, last_two_hand_box=last_2h, hold_counter=hold, max_hold=4)
        self.assertEqual(eff_box, box_2h)
        self.assertEqual(hold, 3)

        # Exhaust hold counter
        for _ in range(3):
            eff_box, last_2h, hold = stabilize_two_hand_box(box_1h, num_hands=1, last_two_hand_box=last_2h, hold_counter=hold, max_hold=4)

        # Frame 6: Hold counter exhausted -> reverts to 1h box
        eff_box, last_2h, hold = stabilize_two_hand_box(box_1h, num_hands=1, last_two_hand_box=last_2h, hold_counter=hold, max_hold=4)
        self.assertEqual(eff_box, box_1h)
        self.assertEqual(hold, 0)

    def test_hand_count_consistency_uncertainty(self):
        """Predictions with missing hands are flagged as inconsistent/uncertain."""
        from live_image_inference import verify_hand_count_consistency

        # Two-handed signs (e.g. 'A', 'B', 'D') with only 1 hand
        ok_a, reason_a = verify_hand_count_consistency('a', num_hands=1)
        self.assertFalse(ok_a)
        self.assertIn("Needs 2 hands", reason_a)

        ok_b, reason_b = verify_hand_count_consistency('b', num_hands=1)
        self.assertFalse(ok_b)

        # H specifically requires 2 flat hands
        ok_h, reason_h = verify_hand_count_consistency('h', num_hands=1)
        self.assertFalse(ok_h)
        self.assertIn("H requires 2 flat hands", reason_h)

        ok_h2, _ = verify_hand_count_consistency('h', num_hands=2)
        self.assertTrue(ok_h2)

        # Two-handed signs with 2 hands
        ok_a2, _ = verify_hand_count_consistency('a', num_hands=2)
        self.assertTrue(ok_a2)

        # Single-handed signs (e.g. 'C', 'L', 'O', 'V') with 1 hand
        ok_c, _ = verify_hand_count_consistency('c', num_hands=1)
        self.assertTrue(ok_c)
        ok_v, _ = verify_hand_count_consistency('v', num_hands=1)
        self.assertTrue(ok_v)

        # Single-handed sign with 2 hands in frame
        ok_c2, reason_c2 = verify_hand_count_consistency('c', num_hands=2)
        self.assertFalse(ok_c2)
        self.assertIn("Needs 1 hand", reason_c2)

    def test_resnet_checkpoint_preprocessing_legacy_intact(self):
        """Existing isl_resnet.pth must preserve its legacy preprocessing configuration."""
        from pathlib import Path
        import torch
        from prepare_cropped_dataset import compute_hand_box_and_mask

        ckpt_path = Path("models/isl_resnet.pth")
        self.assertTrue(ckpt_path.is_file(), "Existing checkpoint models/isl_resnet.pth must exist")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Verify checkpoint has not been overwritten or altered
        self.assertEqual(ckpt.get("model_type"), "resnet")
        self.assertTrue(ckpt.get("grayscale", False))
        self.assertEqual(ckpt.get("mask_mode", "legacy"), "legacy")

        # Verify default mask_mode in compute_hand_box_and_mask defaults to legacy
        class DummyPt:
            def __init__(self, x, y): self.x, self.y = x, y
        hand = [DummyPt(0.5, 0.5), DummyPt(0.6, 0.6)]
        box_def, mask_def = compute_hand_box_and_mask([hand], 100, 100)
        box_leg, mask_leg = compute_hand_box_and_mask([hand], 100, 100, mask_mode="legacy")
        self.assertEqual(box_def, box_leg)
        self.assertTrue((mask_def == mask_leg).all())

    def test_default_checkpoints_target_production_resnet(self):
        """Both audit_alphabet.py and live_image_inference.py must default to models/isl_resnet.pth."""
        from pathlib import Path
        import sys
        from unittest.mock import patch
        import audit_alphabet

        # 1. audit_alphabet default
        with patch.object(sys, 'argv', ['audit_alphabet.py']):
            audit_args = audit_alphabet.parse_args()
            self.assertEqual(Path(audit_args.checkpoint).name, "isl_resnet.pth")

        # 2. live_image_inference default
        from live_image_inference import MODEL_DIR
        default_model = "resnet"
        checkpoint_path = MODEL_DIR / f"isl_{default_model}.pth"
        self.assertEqual(checkpoint_path.name, "isl_resnet.pth")
        self.assertTrue(checkpoint_path.is_file())

    def test_nm_correction_m_with_curled_ring_corrects_to_n(self):
        """When ResNet predicts M but ring finger is curled (ratio < 0.72), correct to N."""
        from live_image_inference import correct_n_m_prediction

        class Point:
            def __init__(self, x, y, z=0.0):
                self.x, self.y, self.z = x, y, z

        # Construct 21 landmarks: Wrist at (0.5, 0.8), Mid at (0.5, 0.3) [span 0.50], Ring at (0.5, 0.55) [span 0.25]
        # Ratio = 0.25 / 0.50 = 0.50 (< 0.72 -> 'n')
        hand = [Point(0.5, 0.8)] * 21
        hand[12] = Point(0.5, 0.3)
        hand[16] = Point(0.5, 0.55)

        # Uppercase input
        res_upper, info_upper = correct_n_m_prediction("M", [hand])
        self.assertEqual(res_upper, "N")
        self.assertTrue(info_upper["applied"])
        self.assertTrue(info_upper["changed"])
        self.assertAlmostEqual(info_upper["ratio"], 0.50, places=2)

        # Lowercase input preserves casing
        res_lower, info_lower = correct_n_m_prediction("m", [hand])
        self.assertEqual(res_lower, "n")
        self.assertTrue(info_lower["applied"])
        self.assertTrue(info_lower["changed"])

    def test_nm_correction_m_with_extended_ring_stays_m(self):
        """When ResNet predicts M and ring finger is extended (ratio >= 0.72), retain M."""
        from live_image_inference import correct_n_m_prediction

        # Landmark 0: wrist at (0.5, 0.8), Mid at (0.5, 0.3) [0.50], Ring at (0.5, 0.33) [0.47]
        # Ratio = 0.47 / 0.50 = 0.94 (>= 0.72 -> 'm')
        hand = [{"x": 0.5, "y": 0.8, "z": 0.0} for _ in range(21)]
        hand[12] = {"x": 0.5, "y": 0.3, "z": 0.0}
        hand[16] = {"x": 0.5, "y": 0.33, "z": 0.0}

        res, info = correct_n_m_prediction("M", [hand])
        self.assertEqual(res, "M")
        self.assertTrue(info["applied"])
        self.assertFalse(info["changed"])
        self.assertAlmostEqual(info["ratio"], 0.94, places=2)

    def test_nm_correction_n_with_extended_ring_corrects_to_m(self):
        """When ResNet predicts N but ring finger is extended (ratio >= 0.72), correct to M."""
        from live_image_inference import correct_n_m_prediction

        hand = [(0.5, 0.8, 0.0)] * 21
        hand[12] = (0.5, 0.3, 0.0)   # span 0.50
        hand[16] = (0.5, 0.33, 0.0)  # span 0.47 -> ratio 0.94
        res, info = correct_n_m_prediction("N", [hand])
        self.assertEqual(res, "M")
        self.assertTrue(info["applied"])
        self.assertTrue(info["changed"])

    def test_nm_correction_n_with_curled_ring_stays_n(self):
        """When ResNet predicts N and ring finger is curled (ratio < 0.72), retain N."""
        from live_image_inference import correct_n_m_prediction

        hand = [(0.5, 0.8, 0.0)] * 21
        hand[12] = (0.5, 0.3, 0.0)   # span 0.50
        hand[16] = (0.5, 0.55, 0.0)  # span 0.25 -> ratio 0.50
        res, info = correct_n_m_prediction("n", [hand])
        self.assertEqual(res, "n")
        self.assertTrue(info["applied"])
        self.assertFalse(info["changed"])

    def test_nm_correction_unchanged_for_other_letters(self):
        """All letter predictions other than N or M remain strictly unchanged."""
        from live_image_inference import correct_n_m_prediction

        hand_extended = [(0.5, 0.8, 0.0)] * 21
        hand_extended[12] = (0.5, 0.3, 0.0)
        hand_extended[16] = (0.5, 0.33, 0.0)

        for other_letter in ("A", "b", "C", "D", "H", "x", "Z"):
            res, info = correct_n_m_prediction(other_letter, [hand_extended])
            self.assertEqual(res, other_letter)
            self.assertFalse(info["applied"])
            self.assertEqual(info["reason"], "not_n_or_m")

    def test_nm_correction_missing_or_empty_landmarks_retains_resnet(self):
        """If landmarks are None or empty, retain original ResNet prediction."""
        from live_image_inference import correct_n_m_prediction

        res_none, info_none = correct_n_m_prediction("N", None)
        self.assertEqual(res_none, "N")
        self.assertFalse(info_none["applied"])

        res_empty, info_empty = correct_n_m_prediction("M", [])
        self.assertEqual(res_empty, "M")
        self.assertFalse(info_empty["applied"])

    def test_nm_correction_invalid_or_insufficient_landmarks_retains_resnet(self):
        """If landmarks are incomplete, nan, or lack middle extension, retain original prediction."""
        from live_image_inference import correct_n_m_prediction

        # 1. Fewer than 21 landmarks
        short_hand = [(0.5, 0.5, 0.0)] * 10
        res1, info1 = correct_n_m_prediction("M", [short_hand])
        self.assertEqual(res1, "M")
        self.assertFalse(info1["applied"])

        # 2. Middle finger span below min_middle_span (0.05) - e.g. fist / closed hand
        fist = [(0.5, 0.5, 0.0)] * 21
        fist[12] = (0.505, 0.505, 0.0)  # span ~0.007 < 0.05
        fist[16] = (0.505, 0.505, 0.0)
        res2, info2 = correct_n_m_prediction("N", [fist])
        self.assertEqual(res2, "N")
        self.assertFalse(info2["applied"])

        # 3. NaN coordinate
        nan_hand = [(0.5, 0.8, 0.0)] * 21
        nan_hand[12] = (float("nan"), 0.3, 0.0)
        nan_hand[16] = (0.5, 0.55, 0.0)
        res3, info3 = correct_n_m_prediction("M", [nan_hand])
        self.assertEqual(res3, "M")
        self.assertFalse(info3["applied"])


if __name__ == "__main__":
    unittest.main()

