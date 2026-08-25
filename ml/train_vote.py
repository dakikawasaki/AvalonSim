import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple

from ml.features import vote_feature_size
from ml.models import VoteMLP


Example = Tuple[List[float], float]


def load_vote_examples(path: Path) -> List[Example]:
    examples: List[Example] = []
    expected_size = vote_feature_size()

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        if len(header) != expected_size + 1:
            raise ValueError(
                f"Expected {expected_size} features plus label, got {len(header)} columns"
            )

        for row in reader:
            values = [float(value) for value in row]
            features = values[:expected_size]
            label = values[expected_size]
            examples.append((features, label))

    return examples


def split_examples(
    examples: List[Example],
    test_ratio: float,
    seed: int,
) -> Tuple[List[Example], List[Example]]:
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    test_count = int(len(shuffled) * test_ratio)
    test_examples = shuffled[:test_count]
    train_examples = shuffled[test_count:]
    return train_examples, test_examples


def evaluate(model: VoteMLP, examples: List[Example]) -> Tuple[float, float]:
    if not examples:
        return 0.0, 0.0

    correct = 0
    total_loss = 0.0
    for features, label in examples:
        probability = model.predict_probability(features)
        prediction = 1.0 if probability >= 0.5 else 0.0
        if prediction == label:
            correct += 1
        total_loss += _binary_cross_entropy(probability, label)

    return correct / len(examples), total_loss / len(examples)


def train(
    train_examples: List[Example],
    test_examples: List[Example],
    epochs: int,
    learning_rate: float,
    hidden_size: int,
    seed: int,
) -> VoteMLP:
    model = VoteMLP(
        input_size=vote_feature_size(),
        hidden_size=hidden_size,
        seed=seed,
    )
    rng = random.Random(seed)

    for epoch in range(1, epochs + 1):
        rng.shuffle(train_examples)
        total_loss = 0.0

        for features, label in train_examples:
            total_loss += model.train_example(features, label, learning_rate)

        train_accuracy, train_eval_loss = evaluate(model, train_examples)
        test_accuracy, test_loss = evaluate(model, test_examples)
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_eval_loss:.4f} | "
            f"train acc {train_accuracy:.1%} | "
            f"test loss {test_loss:.4f} | "
            f"test acc {test_accuracy:.1%}"
        )

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the vote approve/reject model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/vote_smart.csv"),
        help="CSV dataset path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/vote_model.json"),
        help="Where to save the trained model.",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_vote_examples(args.data)
    train_examples, test_examples = split_examples(
        examples=examples,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"Loaded examples: {len(examples)}")
    print(f"Train examples: {len(train_examples)}")
    print(f"Test examples: {len(test_examples)}")
    print(f"Feature size: {vote_feature_size()}")

    model = train(
        train_examples=train_examples,
        test_examples=test_examples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        seed=args.seed,
    )
    model.save(args.output)
    print(f"Saved model: {args.output}")


def _binary_cross_entropy(probability: float, label: float) -> float:
    epsilon = 1e-8
    probability = min(1.0 - epsilon, max(epsilon, probability))
    if label == 1.0:
        return -_log(probability)
    return -_log(1.0 - probability)


def _log(value: float) -> float:
    import math

    return math.log(value)


if __name__ == "__main__":
    main()
