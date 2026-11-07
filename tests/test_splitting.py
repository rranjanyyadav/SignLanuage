import unittest
from collections import defaultdict
import torch

from train_landmark_model import split_sample_indices
from train_image_models import split_dataset_indices


class TestDatasetSplitting(unittest.TestCase):
    def test_landmark_split_no_leakage_block(self):
        """Block split must produce disjoint train, validation, and test sets."""
        class_map = defaultdict(list)
        # Create 100 samples for each of 26 classes
        for target in range(26):
            class_map[target] = list(range(target * 100, (target + 1) * 100))

        train_idx, val_idx, test_idx = split_sample_indices(
            class_map, val_fraction=0.15, test_fraction=0.15, strategy="block", seed=42
        )

        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)

        # Disjointness check
        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(train_set.intersection(test_set)), 0)
        self.assertEqual(len(val_set.intersection(test_set)), 0)

        # Completeness check
        total_samples = 26 * 100
        self.assertEqual(len(train_idx) + len(val_idx) + len(test_idx), total_samples)

    def test_landmark_split_no_leakage_random(self):
        """Random stratified split must produce disjoint sets."""
        class_map = defaultdict(list)
        for target in range(26):
            class_map[target] = list(range(target * 50, (target + 1) * 50))

        train_idx, val_idx, test_idx = split_sample_indices(
            class_map, val_fraction=0.20, test_fraction=0.10, strategy="random", seed=42
        )

        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)

        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(train_set.intersection(test_set)), 0)
        self.assertEqual(len(val_set.intersection(test_set)), 0)

    def test_image_split_block_contiguous(self):
        """Block split in image pipeline should hold out end slices per class without overlap."""
        # 10 images per class for 5 classes
        targets = [c for c in range(5) for _ in range(10)]
        train_idx, val_idx, test_idx = split_dataset_indices(
            targets, validation_fraction=0.20, test_fraction=0.20, strategy="block", seed=42
        )

        train_set = set(train_idx.tolist())
        val_set = set(val_idx.tolist())
        test_set = set(test_idx.tolist())

        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(train_set.intersection(test_set)), 0)
        self.assertEqual(len(val_set.intersection(test_set)), 0)
        self.assertEqual(len(train_set) + len(val_set) + len(test_set), len(targets))


if __name__ == "__main__":
    unittest.main()

