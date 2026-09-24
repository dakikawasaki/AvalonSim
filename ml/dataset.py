import csv
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import List, Type

from avalon.simulator import AvalonSimulator
from bots.base_bot import BaseBot
from ml.features import (
    build_assassination_features,
    build_mission_features,
    build_team_features,
    build_vote_features,
    assassination_feature_size,
    mission_feature_size,
    team_feature_size,
    vote_feature_size,
)

@dataclass
class BinaryDataset:
    feature_size: int
    rows: List[List[float]] = field(default_factory=list)
    positive_count: int = 0
    negative_count: int = 0

    def add_example(self, features: List[float], label: bool) -> None:
        if len(features) != self.feature_size:
            raise ValueError(f"Expected {self.feature_size} features, got {len(features)}")

        numeric_label = 1.0 if label else 0.0
        self.rows.append([*features, numeric_label])

        if label:
            self.positive_count += 1
        else:
            self.negative_count += 1

    @property
    def example_count(self) -> int:
        return len(self.rows)

    @property
    def positive_rate(self) -> float:
        if self.example_count == 0:
            return 0.0
        return self.positive_count / self.example_count

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = [f"f{i}" for i in range(self.feature_size)] + ["label"]

        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(self.rows)


@dataclass
class ActionDatasets:
    team: BinaryDataset = field(default_factory=lambda: BinaryDataset(team_feature_size()))
    vote: BinaryDataset = field(default_factory=lambda: BinaryDataset(vote_feature_size()))
    mission: BinaryDataset = field(default_factory=lambda: BinaryDataset(mission_feature_size()))
    assassinate: BinaryDataset = field(
        default_factory=lambda: BinaryDataset(assassination_feature_size())
    )

