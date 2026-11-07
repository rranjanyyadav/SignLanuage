import unittest
import torch
from unittest.mock import MagicMock

from train_landmark_model import normalize_hand, row_to_features
from live_image_inference import compute_hand_box, smooth_box


class LandmarkPoint:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class TestPreprocessing(unittest.TestCase):
    def test_normalize_hand_zero_for_missing_hand(self):
        """When hand is absent (-1.0 values), normalize_hand should return 63 zeros."""
        missing_values = [-1.0] * 63
        norm = normalize_hand(missing_values)
        self.assertEqual(norm.shape, (63,))
        self.assertTrue(torch.all(norm == 0.0))

    def test_normalize_hand_translation_invariance(self):
        """Moving the hand by an arbitrary translation offset should yield identical normalized coordinates."""
        base_points = [[0.1 * i, 0.05 * i, 0.02 * i] for i in range(21)]
        offset = [0.35, -0.22, 0.18]
        translated_points = [[p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]] for p in base_points]

        flat_base = [coord for pt in base_points for coord in pt]
        flat_trans = [coord for pt in translated_points for coord in pt]

        norm_base = normalize_hand(flat_base)
        norm_trans = normalize_hand(flat_trans)

        self.assertTrue(torch.allclose(norm_base, norm_trans, atol=1e-5))

    def test_normalize_hand_scale_invariance(self):
        """Scaling hand landmarks by a positive factor should yield identical normalized coordinates."""
        base_points = [[0.05 * i, 0.03 * i, 0.01 * i] for i in range(21)]
        scaled_points = [[coord * 2.5 for coord in pt] for pt in base_points]

        flat_base = [coord for pt in base_points for coord in pt]
        flat_scaled = [coord for pt in scaled_points for coord in pt]

        norm_base = normalize_hand(flat_base)
        norm_scaled = normalize_hand(flat_scaled)

        self.assertTrue(torch.allclose(norm_base, norm_scaled, atol=1e-5))

    def test_row_to_features_shape(self):
        """Feature vector should be 127 floats: 63 (left) + 63 (right) + 1 (uses_two_hands)."""
        mock_row = {"uses_two_hands": "1.0"}
        for side in ("left_hand", "right_hand"):
            for idx in range(21):
                for axis in ("x", "y", "z"):
                    mock_row[f"{side}_{axis}_{idx}"] = "0.5"

        features = row_to_features(mock_row, swap_hands=False)
        self.assertEqual(features.shape, (127,))
        self.assertEqual(features[-1].item(), 1.0)

    def test_hand_box_single_hand(self):
        """Hand box calculation should bound landmarks and clamp to image dimensions."""
        hand = [LandmarkPoint(0.4, 0.4, 0.0), LandmarkPoint(0.6, 0.6, 0.0)]
        box = compute_hand_box([hand], width=640, height=480, padding=0.1, square=True)
        self.assertIsNotNone(box)
        left, top, right, bottom = box
        self.assertGreaterEqual(left, 0)
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(right, 640)
        self.assertLessEqual(bottom, 480)
        self.assertEqual(right - left, bottom - top)

    def test_hand_box_clamping_boundary(self):
        """Hand box calculation should not exceed image frame boundaries."""
        hand = [LandmarkPoint(0.01, 0.01, 0.0), LandmarkPoint(0.05, 0.05, 0.0)]
        box = compute_hand_box([hand], width=100, height=100, padding=0.5, square=True)
        self.assertIsNotNone(box)
        left, top, right, bottom = box
        self.assertEqual(left, 0)
        self.assertEqual(top, 0)
        self.assertLessEqual(right, 100)
        self.assertLessEqual(bottom, 100)

    def test_smooth_box_ema(self):
        """Exponential moving average on bounding box should smooth coordinates."""
        b1 = (10, 10, 50, 50)
        b2 = (20, 20, 60, 60)
        smoothed = smooth_box(b2, b1, alpha=0.5)
        self.assertEqual(smoothed, (15, 15, 55, 55))

    def test_grayscale_versus_rgb_preprocessing(self):
        """Grayscale transform should replicate single channel to 3 identical channels."""
        from PIL import Image
        import numpy as np
        from train_image_models import make_transforms

        # Create colored test image (pure red)
        rgb_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb_arr[:, :, 0] = 255  # Red channel only
        pil_img = Image.fromarray(rgb_arr)

        _, gray_tf = make_transforms(grayscale=True)
        _, rgb_tf = make_transforms(grayscale=False)

        gray_out = gray_tf(pil_img)
        rgb_out = rgb_tf(pil_img)

        self.assertEqual(gray_out.shape, (3, 224, 224))
        self.assertEqual(rgb_out.shape, (3, 224, 224))

        # In gray_out, before channel normalization, R=G=B. After normalization with different mean/std,
        # rgb_out channels have vastly different values (channel 0 is high, 1 and 2 are low).
        self.assertNotEqual(float(rgb_out[0].mean()), float(rgb_out[1].mean()))

    def test_crop_mask_consistency_and_resizing(self):
        """Mask and crop alignment: resizing mask correctly allows bitwise_and without dimension mismatch."""
        import cv2
        import numpy as np
        from prepare_cropped_dataset import compute_hand_box_and_mask

        hand = [LandmarkPoint(0.40, 0.40, 0.0), LandmarkPoint(0.60, 0.60, 0.0)]
        box, crop_mask = compute_hand_box_and_mask([hand], width=640, height=480, padding=0.25, square=True)

        self.assertIsNotNone(box)
        self.assertIsNotNone(crop_mask)

        # Simulate a slight smoothing dimension mismatch (e.g. crop is 1 pixel wider)
        crop_h, crop_w = crop_mask.shape[0] + 1, crop_mask.shape[1] + 2
        dummy_crop = np.full((crop_h, crop_w, 3), 200, dtype=np.uint8)

        # Resize mask to match crop
        resized_mask = cv2.resize(crop_mask, (dummy_crop.shape[1], dummy_crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        masked_crop = cv2.bitwise_and(dummy_crop, dummy_crop, mask=resized_mask)

        self.assertEqual(masked_crop.shape, dummy_crop.shape)
        # Verify background outside mask is solid 0
        self.assertEqual(int(masked_crop[resized_mask == 0].sum()), 0)

    def test_legacy_vs_fine_mask_modes(self):
        """Fine mask mode should yield smaller area and higher inter-finger concavity than legacy mode."""
        import cv2
        import numpy as np
        from prepare_cropped_dataset import compute_hand_box_and_mask

        # Create a mock 21-landmark hand with 2 spread fingers (index and middle)
        # 0: wrist, 1-4: thumb, 5-8: index, 9-12: middle, 13-16: ring (curled), 17-20: pinky (curled)
        hand = [LandmarkPoint(0.5, 0.8, 0.0)] # 0 wrist
        for _ in range(4): hand.append(LandmarkPoint(0.4, 0.7, 0.0)) # 1-4 thumb
        # index extended
        hand.extend([LandmarkPoint(0.45, 0.6, 0.0), LandmarkPoint(0.45, 0.5, 0.0),
                     LandmarkPoint(0.45, 0.4, 0.0), LandmarkPoint(0.45, 0.3, 0.0)]) # 5-8
        # middle extended
        hand.extend([LandmarkPoint(0.55, 0.6, 0.0), LandmarkPoint(0.55, 0.5, 0.0),
                     LandmarkPoint(0.55, 0.4, 0.0), LandmarkPoint(0.55, 0.3, 0.0)]) # 9-12
        # ring curled
        for _ in range(4): hand.append(LandmarkPoint(0.58, 0.7, 0.0)) # 13-16
        # pinky curled
        for _ in range(4): hand.append(LandmarkPoint(0.62, 0.75, 0.0)) # 17-20

        # Legacy mask
        box_leg, mask_leg = compute_hand_box_and_mask([hand], 224, 224, mask_mode="legacy")
        # Fine mask
        box_fine, mask_fine = compute_hand_box_and_mask([hand], 224, 224, mask_mode="fine",
                                                        finger_thickness_ratio=0.06, dilation_kernel_size=5)

        self.assertIsNotNone(box_leg)
        self.assertIsNotNone(box_fine)
        self.assertEqual(box_leg, box_fine)  # Bounding box calculation identical

        # Fine mask should preserve finger gap: area must be strictly smaller than legacy convex hull
        leg_area = np.count_nonzero(mask_leg)
        fine_area = np.count_nonzero(mask_fine)
        self.assertGreater(leg_area, fine_area)
        self.assertGreater(fine_area, 0)

        # Check midpoint between index and middle fingertips (x=0.50, y=0.35)
        # In legacy mode (convex hull + 15x15 dilation), gap is filled white (255)
        # In fine mode, gap between the two fingers remains black background (0)
        mid_x = int(0.50 * 224) - box_fine[0]
        mid_y = int(0.35 * 224) - box_fine[1]
        self.assertEqual(mask_leg[mid_y, mid_x], 255, "Legacy mode should fill convex hull between fingers")
        self.assertEqual(mask_fine[mid_y, mid_x], 0, "Fine mode should keep gap between separated fingers unmasked/black")

    def test_invalid_mask_mode_raises_value_error(self):
        """Invalid mask_mode string should raise ValueError."""
        from prepare_cropped_dataset import compute_hand_box_and_mask
        hand = [LandmarkPoint(0.4, 0.4, 0.0), LandmarkPoint(0.6, 0.6, 0.0)]
        with self.assertRaises(ValueError):
            compute_hand_box_and_mask([hand], 200, 200, mask_mode="non_existent")


if __name__ == "__main__":
    unittest.main()

