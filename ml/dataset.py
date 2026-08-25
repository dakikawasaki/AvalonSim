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
class VoteDataset:
    rows: List[List[float]] = field(default_factory=list)
    approve_count: int = 0
    reject_count: int = 0

    def add_example(self, features: List[float], vote: bool) -> None:
        if len(features) != vote_feature_size():
            raise ValueError(
                f"Expected {vote_feature_size()} features, got {len(features)}"
            )

        label = 1.0 if vote else 0.0
        self.rows.append([*features, label])

        if vote:
            self.approve_count += 1
        else:
            self.reject_count += 1

    @property
    def example_count(self) -> int:
        return len(self.rows)

    @property
    def approve_rate(self) -> float:
        if self.example_count == 0:
            return 0.0
        return self.approve_count / self.example_count

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = [f"f{i}" for i in range(vote_feature_size())] + ["label"]

        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(self.rows)


def generate_vote_dataset(bot_cls: Type[BaseBot], games: int) -> VoteDataset:
    dataset = VoteDataset()

    def record_vote(observation, proposed_team, vote):
        features = build_vote_features(observation, proposed_team)
        dataset.add_example(features, vote)

    for _ in range(games):
        simulator = AvalonSimulator(
            bot_cls=bot_cls,
            verbose=False,
            vote_callback=record_vote,
        )
        simulator.play_game()

    return dataset


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


def generate_action_datasets(bot_cls: Type[BaseBot], games: int) -> ActionDatasets:
    datasets = ActionDatasets()

    def record_team(observation, team_size, proposed_team):
        proposed_team_set = set(proposed_team)
        negative_examples = []

        for candidate_team in combinations(observation["all_player_ids"], team_size):
            candidate_team = list(candidate_team)
            features = build_team_features(observation, candidate_team)
            is_selected = set(candidate_team) == proposed_team_set
            if is_selected:
                datasets.team.add_example(features, True)
            else:
                negative_examples.append(features)
                datasets.team.add_example(features, False)

        selected_features = build_team_features(observation, proposed_team)
        for _ in negative_examples:
            datasets.team.add_example(selected_features, True)

    def record_vote(observation, proposed_team, vote):
        features = build_vote_features(observation, proposed_team)
        datasets.vote.add_example(features, vote)

    def record_mission(observation, proposed_team, choice):
        features = build_mission_features(observation, proposed_team)
        datasets.mission.add_example(features, choice)

    def record_assassinate(observation, players, target_id):
        known_evil = set(observation["known_evil_ids"])
        possible_targets = [player_id for player_id in players if player_id not in known_evil]
        for candidate_id in possible_targets:
            features = build_assassination_features(observation, candidate_id)
            datasets.assassinate.add_example(features, candidate_id == target_id)

    for _ in range(games):
        simulator = AvalonSimulator(
            bot_cls=bot_cls,
            verbose=False,
            team_callback=record_team,
            vote_callback=record_vote,
            mission_callback=record_mission,
            assassinate_callback=record_assassinate,
        )
        simulator.play_game()

    return datasets
