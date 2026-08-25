import json
import math
import random
from pathlib import Path
from typing import Dict, List


class VoteMLP:
    """Small dependency-free MLP for approve/reject vote prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 16,
        seed: int = 0,
        weights: Dict[str, List] = None,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size

        if weights is None:
            rng = random.Random(seed)
            self.w1 = [
                [rng.uniform(-0.15, 0.15) for _ in range(input_size)]
                for _ in range(hidden_size)
            ]
            self.b1 = [0.0 for _ in range(hidden_size)]
            self.w2 = [rng.uniform(-0.15, 0.15) for _ in range(hidden_size)]
            self.b2 = 0.0
        else:
            self.w1 = weights["w1"]
            self.b1 = weights["b1"]
            self.w2 = weights["w2"]
            self.b2 = weights["b2"]

    def predict_probability(self, features: List[float]) -> float:
        hidden = self._hidden_forward(features)
        output = self.b2
        for hidden_index in range(self.hidden_size):
            output += self.w2[hidden_index] * hidden[hidden_index]
        return _sigmoid(output)

    def predict(self, features: List[float], threshold: float = 0.5) -> bool:
        return self.predict_probability(features) >= threshold

    def train_example(self, features: List[float], label: float, learning_rate: float) -> float:
        hidden_raw = []
        hidden = []
        for hidden_index in range(self.hidden_size):
            value = self.b1[hidden_index]
            weights = self.w1[hidden_index]
            for feature_index in range(self.input_size):
                value += weights[feature_index] * features[feature_index]
            hidden_raw.append(value)
            hidden.append(_relu(value))

        output_raw = self.b2
        for hidden_index in range(self.hidden_size):
            output_raw += self.w2[hidden_index] * hidden[hidden_index]

        probability = _sigmoid(output_raw)
        loss = _binary_cross_entropy(probability, label)

        output_error = probability - label

        for hidden_index in range(self.hidden_size):
            old_w2 = self.w2[hidden_index]
            self.w2[hidden_index] -= learning_rate * output_error * hidden[hidden_index]

            hidden_error = output_error * old_w2 * _relu_derivative(hidden_raw[hidden_index])
            for feature_index in range(self.input_size):
                self.w1[hidden_index][feature_index] -= (
                    learning_rate * hidden_error * features[feature_index]
                )
            self.b1[hidden_index] -= learning_rate * hidden_error

        self.b2 -= learning_rate * output_error
        return loss

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "VoteMLP",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "VoteMLP":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            input_size=payload["input_size"],
            hidden_size=payload["hidden_size"],
            weights={
                "w1": payload["w1"],
                "b1": payload["b1"],
                "w2": payload["w2"],
                "b2": payload["b2"],
            },
        )

    def _hidden_forward(self, features: List[float]) -> List[float]:
        hidden = []
        for hidden_index in range(self.hidden_size):
            value = self.b1[hidden_index]
            weights = self.w1[hidden_index]
            for feature_index in range(self.input_size):
                value += weights[feature_index] * features[feature_index]
            hidden.append(_relu(value))
        return hidden


def _relu(value: float) -> float:
    return max(0.0, value)


def _relu_derivative(value: float) -> float:
    return 1.0 if value > 0.0 else 0.0


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def _binary_cross_entropy(probability: float, label: float) -> float:
    epsilon = 1e-8
    probability = min(1.0 - epsilon, max(epsilon, probability))
    return -(label * math.log(probability) + (1.0 - label) * math.log(1.0 - probability))
