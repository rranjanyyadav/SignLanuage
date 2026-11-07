import json
import tempfile
import unittest
from pathlib import Path

from plot_training_curves import load_experiment_data, plot_loss_curve, plot_accuracy_curve


class TestPlotTrainingCurves(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.sample_data = [
            {
                "model": "resnet",
                "checkpoint": "models/isl_resnet_fine.pth",
                "split_strategy": "block",
                "input_mode": "crop",
                "best_validation_accuracy": 0.5849,
                "test_loss": 1.7533,
                "test_accuracy": 0.6765,
                "history": [
                    {
                        "epoch": 1,
                        "train_loss": 1.6747,
                        "train_accuracy": 0.5387,
                        "validation_loss": 2.5230,
                        "validation_accuracy": 0.3477,
                    },
                    {
                        "epoch": 2,
                        "train_loss": 0.7490,
                        "train_accuracy": 0.7728,
                        "validation_loss": 2.4831,
                        "validation_accuracy": 0.4177,
                    },
                ],
            }
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_valid_experiment_data(self):
        json_file = self.temp_path / "results.json"
        json_file.write_text(json.dumps(self.sample_data), encoding="utf-8")

        meta, history = load_experiment_data(json_file)
        self.assertEqual(meta["model"], "resnet")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["epoch"], 1)

    def test_load_missing_field_raises(self):
        invalid_data = [
            {
                "model": "resnet",
                "history": [
                    {
                        "epoch": 1,
                        "train_loss": 1.0,
                        # missing train_accuracy, validation_loss, validation_accuracy
                    }
                ],
            }
        ]
        json_file = self.temp_path / "invalid.json"
        json_file.write_text(json.dumps(invalid_data), encoding="utf-8")

        with self.assertRaises(KeyError):
            load_experiment_data(json_file)

    def test_generate_loss_and_accuracy_curves(self):
        meta = self.sample_data[0]
        history = meta["history"]

        loss_out = self.temp_path / "loss.png"
        acc_out = self.temp_path / "acc.png"

        saved_loss = plot_loss_curve(meta, history, loss_out, dpi=100)
        saved_acc = plot_accuracy_curve(meta, history, acc_out, dpi=100)

        self.assertTrue(saved_loss.exists())
        self.assertGreater(saved_loss.stat().st_size, 1000)

        self.assertTrue(saved_acc.exists())
        self.assertGreater(saved_acc.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()

