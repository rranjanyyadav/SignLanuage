import csv
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "Indian Sign Language Gesture Landmarks.csv"
MODEL_PATH = PROJECT_DIR / "models" / "isl_landmark_mlp.pth"
CLASSES = list("abcdefghijklmnopqrstuvwxyz")
SEED = 42
VALIDATION_FRACTION = 0.20


def normalize_hand(values):
    points = torch.tensor(values, dtype=torch.float32).reshape(21, 3)
    if points[0, 0] < -0.5:
        return torch.zeros(63)

    wrist = points[0].clone()
    points = points - wrist
    scale = torch.linalg.vector_norm(points[:, :2], dim=1).max().clamp_min(1e-6)
    return (points / scale).reshape(-1)


def row_to_features(row, swap_hands=False):
    features = []
    sides = ("right_hand", "left_hand") if swap_hands else ("left_hand", "right_hand")
    for side in sides:
        values = [
            float(row[f"{side}_{axis}_{index}"])
            for index in range(21)
            for axis in ("x", "y", "z")
        ]
        features.append(normalize_hand(values))
    features.append(torch.tensor([float(row["uses_two_hands"])], dtype=torch.float32))
    return torch.cat(features)


class LandmarkMLP(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(127, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def load_data():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    features, labels = [], []
    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            target = int(row["target"])
            if not 0 <= target < len(CLASSES):
                continue
            features.append(row_to_features(row))
            labels.append(target)
            features.append(row_to_features(row, swap_hands=True))
            labels.append(target)

    data = torch.stack(features)
    targets = torch.tensor(labels, dtype=torch.long)
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(len(targets), generator=generator)
    split = int(len(targets) * (1 - VALIDATION_FRACTION))
    train_order, validation_order = order[:split], order[split:]
    return data[train_order], targets[train_order], data[validation_order], targets[validation_order]


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    train_x, train_y, validation_x, validation_y = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LandmarkMLP(len(CLASSES)).to(device)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=512,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    best_state = None
    for epoch in range(1, 51):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            predictions = model(validation_x.to(device)).argmax(dim=1).cpu()
        accuracy = (predictions == validation_y).float().mean().item()
        print(f"Epoch {epoch:02d}: validation accuracy {accuracy * 100:.2f}%")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "classes": CLASSES,
            "feature_size": 127,
            "normalization": "wrist-relative, max-xy-distance scale",
            "validation_accuracy": best_accuracy,
        },
        MODEL_PATH,
    )
    print(f"Saved: {MODEL_PATH}")
    print(f"Best validation accuracy: {best_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
