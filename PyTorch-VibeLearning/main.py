import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate
from datasets import Dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core import vibelearn, VibeTrainer


@vibelearn
class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def load_data() -> tuple[Dataset, Dataset]:
    raw = load_iris()
    X = StandardScaler().fit_transform(raw.data).astype(np.float32)
    y = raw.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return (
        Dataset.from_dict({"features": X_train.tolist(), "label": y_train.tolist()}),
        Dataset.from_dict({"features": X_test.tolist(), "label": y_test.tolist()}),
    )


def main():
    print("=== VibeLearning - Iris Classification ===\n")

    train_ds, test_ds = load_data()
    print(f"Train: {len(train_ds)}  |  Test: {len(test_ds)}\n")

    X_train = torch.tensor(train_ds["features"])
    y_train = torch.tensor(train_ds["label"])
    X_test  = torch.tensor(test_ds["features"])
    y_test  = test_ds["label"]

    trainer = VibeTrainer(
        IrisNet(),
        metric=evaluate.load("accuracy"),
        context_samples=20,
    )

    trainer.fit(X_train, y_train, epochs=8, eval_inputs=X_test, eval_targets=y_test)

    best = max(trainer.history, key=lambda r: r.get("accuracy", 0))
    print(f"\nbest  accuracy={best['accuracy']:.4f}")


if __name__ == "__main__":
    main()
