import json
from pathlib import Path
from itertools import combinations
import random
from typing import Any, Dict, List

from bots.smart_bot import GOOD_ROLES, SmartBot
from ml.features import (
    build_assassination_features,
    build_mission_features,
    build_team_features,
    build_vote_features,
)
from ml.torch_models import TorchActionModel


DEFAULT_TEAM_MODEL_PATH = Path("models/team_human_model.pt")
DEFAULT_VOTE_MODEL_PATH = Path("models/vote_human_model.pt")
DEFAULT_MISSION_MODEL_PATH = Path("models/mission_human_model.pt")
DEFAULT_ASSASSINATE_MODEL_PATH = Path("models/assassinate_human_model.pt")
DEFAULT_RL_CONFIG_PATH = Path("models/rl_config.json")


class MLBot(SmartBot):

    _models = {}
    mission_success_threshold = 0.55
    self_team_multiplier = 1.2
    good_known_evil_multiplier = 0.02
    percival_double_candidate_multiplier = 0.55
    evil_known_evil_multiplier = 1.05
    assassin_min_weight = 0.2
    _rl_config_loaded = False

    def __init__(
        self,
        player_id: int,
        team_model_path: Path = DEFAULT_TEAM_MODEL_PATH,
        vote_model_path: Path = DEFAULT_VOTE_MODEL_PATH,
        mission_model_path: Path = DEFAULT_MISSION_MODEL_PATH,
        assassinate_model_path: Path = DEFAULT_ASSASSINATE_MODEL_PATH,
        rl_config_path: Path = DEFAULT_RL_CONFIG_PATH,
    ):
        super().__init__(player_id)
        if type(self) is MLBot:
            self._load_rl_config_if_available(rl_config_path)
        self.team_model_path = team_model_path
        self.vote_model_path = vote_model_path
        self.mission_model_path = mission_model_path
        self.assassinate_model_path = assassinate_model_path

    def propose_team(self, observation: Dict[str, Any], team_size: int) -> List[int]:
        try:
            model = self._get_model("team", self.team_model_path)
        except Exception:
            return super().propose_team(observation, team_size)

        scored_teams = []
        for team in combinations(observation["all_player_ids"], team_size):
            team = list(team)
            features = build_team_features(observation, team)
            try:
                score = model.predict_probability(features)
            except Exception:
                return super().propose_team(observation, team_size)
            score = self._calibrate_team_score(score, observation, team)
            scored_teams.append((score, team))

        scored_teams.sort(key=lambda item: item[0], reverse=True)
        return scored_teams[0][1]

    def vote_team(self, observation: Dict[str, Any], proposed_team: List[int]) -> bool:
        features = build_vote_features(observation, proposed_team)
        try:
            model = self._get_model("vote", self.vote_model_path)
            return model.predict(features)
        except Exception:
            return super().vote_team(observation, proposed_team)

    def play_mission(self, observation: Dict[str, Any], proposed_team: List[int]) -> bool:
        if observation["role"] in GOOD_ROLES:
            return True

        evil_on_team = [
            player_id
            for player_id in proposed_team
            if player_id in observation["known_evil_ids"]
        ]
        if len(evil_on_team) > 1 and self.player_id != min(evil_on_team):
            return True

        features = build_mission_features(observation, proposed_team)
        threshold = max(self.mission_success_threshold, 0.55)
        try:
            model = self._get_model("mission", self.mission_model_path)
            return model.predict_probability(features) >= threshold
        except Exception:
            return super().play_mission(observation, proposed_team)

    def assassinate(self, observation: Dict[str, Any], players: List[int]) -> int:
        known_evil = set(observation["known_evil_ids"])
        possible_targets = [player_id for player_id in players if player_id not in known_evil]
        try:
            model = self._get_model("assassinate", self.assassinate_model_path)
        except Exception:
            return super().assassinate(observation, players)

        scored_targets = []
        for player_id in possible_targets:
            features = build_assassination_features(observation, player_id)
            try:
                score = model.predict_probability(features)
            except Exception:
                return super().assassinate(observation, players)
            scored_targets.append((score, player_id))

        targets = [player_id for _, player_id in scored_targets]
        weights = [max(self.assassin_min_weight, score) for score, _ in scored_targets]
        return random.choices(targets, weights=weights, k=1)[0]

    def _calibrate_team_score(
        self,
        score: float,
        observation: Dict[str, Any],
        team: List[int],
    ) -> float:
        if observation["self_id"] in team:
            score *= self.self_team_multiplier

        if observation["role"] in GOOD_ROLES:
            if any(player_id in observation["known_evil_ids"] for player_id in team):
                score *= self.good_known_evil_multiplier

            merlin_candidates_on_team = set(team) & set(observation["merlin_candidates"])
            if observation["role"] == "Percival" and len(merlin_candidates_on_team) > 1:
                score *= self.percival_double_candidate_multiplier
        else:
            if any(player_id in observation["known_evil_ids"] for player_id in team):
                score *= self.evil_known_evil_multiplier

        return score

    def _get_model(self, name: str, path: Path) -> TorchActionModel:
        cache_key = (name, str(path))

        if cache_key not in MLBot._models:
            if not path.exists():
                raise FileNotFoundError(
                    f"{name.title()} model not found at {path}. "
                    "Run: python generate_human_log_data.py, then python train_all_actions.py"
                )

            MLBot._models[cache_key] = TorchActionModel.load(path)

        return MLBot._models[cache_key]

    @classmethod
    def _load_rl_config_if_available(cls, path: Path) -> None:
        if cls._rl_config_loaded:
            return

        cls._rl_config_loaded = True
        if not path.exists():
            return

        config = json.loads(path.read_text(encoding="utf-8"))
        for key in [
            "mission_success_threshold",
            "self_team_multiplier",
            "good_known_evil_multiplier",
            "percival_double_candidate_multiplier",
            "evil_known_evil_multiplier",
            "assassin_min_weight",
        ]:
            if key in config:
                setattr(cls, key, float(config[key]))
