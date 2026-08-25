import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple

from ml.models import VoteMLP


Example = Tuple[List[float], float]


def load_examples(path: Path) -> Tuple[List[Example], int]:
    examples: List[Example] = []

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        feature_size = len(header) - 1

        for row in reader:
            values = [float(value) for value in row]
            examples.append((values[:feature_size], values[feature_size]))

    return examples, feature_size


def split_examples(
    examples: List[Example],
    test_ratio: float,
    seed: int,
) -> Tuple[List[Example], List[Example]]:
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    test_count = int(len(shuffled) * test_ratio)
    return shuffled[test_count:], shuffled[:test_count]


def sample_examples(
    examples: List[Example],
    max_examples: int,
    seed: int,
) -> List[Example]:
    if max_examples <= 0 or len(examples) <= max_examples:
        return examples

    positives = [example for example in examples if example[1] >= 0.5]
    negatives = [example for example in examples if example[1] < 0.5]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    positive_target = round(max_examples * len(positives) / len(examples))
    negative_target = max_examples - positive_target
    sampled = positives[:positive_target] + negatives[:negative_target]
    rng.shuffle(sampled)
    return sampled


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
    feature_size: int,
    epochs: int,
    learning_rate: float,
    hidden_size: int,
    seed: int,
) -> VoteMLP:
    model = VoteMLP(input_size=feature_size, hidden_size=hidden_size, seed=seed)
    rng = random.Random(seed)

    for epoch in range(1, epochs + 1):
        rng.shuffle(train_examples)
        for features, label in train_examples:
            model.train_example(features, label, learning_rate)

        train_accuracy, train_loss = evaluate(model, train_examples)
        test_accuracy, test_loss = evaluate(model, test_examples)
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_accuracy:.1%} | "
            f"test loss {test_loss:.4f} | "
            f"test acc {test_accuracy:.1%}"
        )

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary action model.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Optional stratified sample cap. 0 uses every example.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples, feature_size = load_examples(args.data)
    full_count = len(examples)
    examples = sample_examples(examples, args.max_examples, args.seed)
    train_examples, test_examples = split_examples(examples, args.test_ratio, args.seed)

    print(f"Loaded examples: {full_count}")
    if len(examples) != full_count:
        print(f"Sampled examples: {len(examples)}")
    print(f"Train examples: {len(train_examples)}")
    print(f"Test examples: {len(test_examples)}")
    print(f"Feature size: {feature_size}")

    model = train(
        train_examples=train_examples,
        test_examples=test_examples,
        feature_size=feature_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        seed=args.seed,
    )
    model.save(args.output)
    print(f"Saved model: {args.output}")


def _binary_cross_entropy(probability: float, label: float) -> float:
    import math

    epsilon = 1e-8
    probability = min(1.0 - epsilon, max(epsilon, probability))
    return -(label * math.log(probability) + (1.0 - label) * math.log(1.0 - probability))


if __name__ == "__main__":
    main()
