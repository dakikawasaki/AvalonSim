import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.torch_models import TorchMLP


Example = Tuple[List[float], float]


def load_examples(path: Path) -> Tuple[torch.Tensor, torch.Tensor, int]:
    features = []
    labels = []

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        feature_size = len(header) - 1

        for row in reader:
            values = [float(value) for value in row]
            features.append(values[:feature_size])
            labels.append(values[feature_size])

    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
        feature_size,
    )


def split_tensors(
    features: torch.Tensor,
    labels: torch.Tensor,
    test_ratio: float,
    seed: int,
) -> Tuple[TensorDataset, TensorDataset]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(features), generator=generator)
    test_count = int(len(indices) * test_ratio)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    return (
        TensorDataset(features[train_indices], labels[train_indices]),
        TensorDataset(features[test_indices], labels[test_indices]),
    )


def evaluate(model: TorchMLP, dataset: TensorDataset, batch_size: int) -> Tuple[float, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    correct = 0
    total = 0
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features)
            loss = criterion(logits, labels)
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.numel()
            total_loss += loss.item() * labels.numel()

    return correct / total, total_loss / total


def train(
    train_dataset: TensorDataset,
    test_dataset: TensorDataset,
    feature_size: int,
    epochs: int,
    learning_rate: float,
    hidden_size: int,
    batch_size: int,
    positive_weight: float,
) -> TorchMLP:
    model = TorchMLP(input_size=feature_size, hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    pos_weight = torch.tensor([positive_weight], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(1, epochs + 1):
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model.train()

        for features, labels in loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        train_accuracy, train_loss = evaluate(model, train_dataset, batch_size)
        test_accuracy, test_loss = evaluate(model, test_dataset, batch_size)
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_accuracy:.1%} | "
            f"test loss {test_loss:.4f} | "
            f"test acc {test_accuracy:.1%}"
        )

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary action model with PyTorch.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--positive-weight",
        default="1.0",
        help="Positive class weight, or 'auto' to use negatives/positives from the train split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    features, labels, feature_size = load_examples(args.data)
    train_dataset, test_dataset = split_tensors(
        features=features,
        labels=labels,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    if args.positive_weight == "auto":
        train_labels = train_dataset.tensors[1]
        positive_count = train_labels.sum().item()
        negative_count = len(train_labels) - positive_count
        positive_weight = negative_count / positive_count if positive_count else 1.0
    else:
        positive_weight = float(args.positive_weight)

    print(f"Loaded examples: {len(features)}")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    print(f"Feature size: {feature_size}")
    print(f"Positive weight: {positive_weight:.3f}")

    model = train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        feature_size=feature_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        positive_weight=positive_weight,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_size": feature_size,
            "hidden_size": args.hidden_size,
            "state_dict": model.state_dict(),
        },
        args.output,
    )
    print(f"Saved model: {args.output}")


if __name__ == "__main__":
    main()
