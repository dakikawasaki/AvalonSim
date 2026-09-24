from pathlib import Path
from typing import List

import torch
from torch import nn


class TorchMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)

class AssassinDQN(nn.Module):
    def __init__(
        self,
        input_size: int = 87,
        hidden_size: int = 64,
        num_actions: int = 7,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, features):
        return self.network(features)


class TorchActionModel:
    def __init__(self, network: TorchMLP):
        self.network = network
        self.network.eval()

    def predict_probability(self, features: List[float]) -> float:
        with torch.no_grad():
            tensor = torch.tensor([features], dtype=torch.float32)
            logits = self.network(tensor)
            probability = torch.sigmoid(logits)[0].item()
        return probability

    def predict(self, features: List[float], threshold: float = 0.5) -> bool:
        return self.predict_probability(features) >= threshold

    @classmethod
    def load(cls, path: Path) -> "TorchActionModel":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        network = TorchMLP(
            input_size=payload["input_size"],
            hidden_size=payload["hidden_size"],
        )
        network.load_state_dict(payload["state_dict"])
        return cls(network)
