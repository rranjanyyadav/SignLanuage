import unittest
import torch

from train_landmark_model import LandmarkMLP
from train_image_models import SmallCNN, LegacySmallCNN, build_model


class TestModels(unittest.TestCase):
    def test_landmark_mlp_forward_and_backward(self):
        """LandmarkMLP accepts (B, 127) and outputs (B, 26)."""
        model = LandmarkMLP(num_classes=26)
        x = torch.randn(8, 127)
        out = model(x)
        self.assertEqual(out.shape, (8, 26))

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(model.network[0].weight.grad)

    def test_small_cnn_forward_and_backward(self):
        """SmallCNN accepts (B, 3, 224, 224) and outputs (B, 26)."""
        model = SmallCNN(num_classes=26)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (2, 26))

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(model.classifier[1].weight.grad)

    def test_legacy_small_cnn_forward(self):
        """LegacySmallCNN accepts (B, 3, 224, 224) and outputs (B, 26)."""
        model = LegacySmallCNN(num_classes=26)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (2, 26))

    def test_resnet_build(self):
        """build_model('resnet') returns ResNet-18 with 26-class linear head."""
        model = build_model("resnet", use_pretrained=False)
        self.assertEqual(model.fc.out_features, 26)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (2, 26))

    def test_load_image_model_strict_architecture_check(self):
        """load_image_model verifies architecture compatibility and raises on mismatch."""
        from evaluate_models import load_image_model
        from pathlib import Path

        cnn_path = Path("models/isl_cnn.pth")
        resnet_path = Path("models/isl_resnet.pth")

        if cnn_path.is_file():
            # Loading with correct model name succeeds
            m, ckpt = load_image_model("cnn", cnn_path)
            self.assertIsNotNone(m)
            # Mismatch: asking for resnet when file is CNN must raise ValueError
            with self.assertRaises(ValueError):
                load_image_model("resnet", cnn_path)

        if resnet_path.is_file():
            # Loading with correct model name succeeds
            m, ckpt = load_image_model("resnet", resnet_path)
            self.assertIsNotNone(m)
            # Mismatch: asking for cnn when file is ResNet must raise ValueError
            with self.assertRaises(ValueError):
                load_image_model("cnn", resnet_path)


if __name__ == "__main__":
    unittest.main()

