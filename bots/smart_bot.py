import random
from typing import Any, Dict, List

from bots.base_bot import BaseBot


GOOD_ROLES = {"Merlin", "Percival", "Loyal Servant"}


class SmartBot(BaseBot):
    """Simple Avalon bot that uses role knowledge and public mission history."""

    def propose_team(self, observation: Dict[str, Any], team_size: int) -> List[int]:
        if self._is_evil(observation):
            return self._propose_evil_team(observation, team_size)
        return self._propose_good_team(observation, team_size)

    def vote_team(self, observation: Dict[str, Any], proposed_team: List[int]) -> bool:
        if observation["attempt_number"] >= 5:
            return self._is_good(observation)

        if self._is_evil(observation):
            return self._vote_as_evil(observation, proposed_team)
        return self._vote_as_good(observation, proposed_team)

    def play_mission(self, observation: Dict[str, Any], proposed_team: List[int]) -> bool:
        if self._is_good(observation):
            return True

        evil_on_team = [
            player_id
            for player_id in proposed_team
            if player_id in observation["known_evil_ids"]
        ]

        if len(evil_on_team) > 1 and self.player_id != min(evil_on_team):
            return True
        return False

    def assassinate(self, observation: Dict[str, Any], players: List[int]) -> int:
        known_evil = set(observation["known_evil_ids"])
        possible_targets = [player_id for player_id in players if player_id not in known_evil]
        suspicion = self._suspicion_scores(observation)

        def merlin_score(player_id: int) -> float:
            successful_team_count = sum(
                1
                for record in observation["history"]
                if record["mission_result"] is True and player_id in record["proposed_team"]
            )
            approve_success_count = sum(
                1
                for record in observation["history"]
                if record["mission_result"] is True and record["votes"][player_id]
            )
            return successful_team_count + approve_success_count - suspicion[player_id]

        if random.random() < 0.75:
            return random.choice(possible_targets)
        return max(possible_targets, key=merlin_score)

    def _propose_good_team(self, observation: Dict[str, Any], team_size: int) -> List[int]:
        suspicion = self._suspicion_scores(observation)
        all_players = observation["all_player_ids"]

        preferred = [self.player_id]
        if observation["role"] == "Percival":
            candidates = [
                player_id
                for player_id in observation["merlin_candidates"]
                if player_id != self.player_id
            ]
            candidates.sort(key=lambda player_id: (suspicion[player_id], player_id))
            preferred.extend(candidates[:1])

        team = self._unique_take(preferred, team_size)
        remaining = [player_id for player_id in all_players if player_id not in team]
        remaining.sort(key=lambda player_id: (suspicion[player_id], player_id))
        team.extend(remaining[: team_size - len(team)])
        return team

    def _propose_evil_team(self, observation: Dict[str, Any], team_size: int) -> List[int]:
        known_evil = [player_id for player_id in observation["known_evil_ids"]]
        team = self._unique_take([self.player_id], team_size)

        evil_candidates = [player_id for player_id in known_evil if player_id not in team]
        random.shuffle(evil_candidates)
        team = self._unique_take(team + evil_candidates[:1], team_size)

        suspicion = self._suspicion_scores(observation)
        fillers = [player_id for player_id in observation["all_player_ids"] if player_id not in team]
        fillers.sort(key=lambda player_id: (-suspicion[player_id], player_id))
        team.extend(fillers[: team_size - len(team)])
        return team

    def _vote_as_good(self, observation: Dict[str, Any], proposed_team: List[int]) -> bool:
        suspicion = self._suspicion_scores(observation)
        known_evil = set(observation["known_evil_ids"])
        if any(player_id in known_evil for player_id in proposed_team):
            return False

        average_suspicion = sum(suspicion[player_id] for player_id in proposed_team) / len(
            proposed_team
        )
        includes_self = self.player_id in proposed_team
        threshold = 0.62 if includes_self else 0.48
        return average_suspicion <= threshold

    def _vote_as_evil(self, observation: Dict[str, Any], proposed_team: List[int]) -> bool:
        known_evil = set(observation["known_evil_ids"])
        has_evil = any(player_id in known_evil for player_id in proposed_team)
        if has_evil:
            return True

        if observation["failed_missions"] >= 2:
            return False
        return random.random() < 0.25

    def _suspicion_scores(self, observation: Dict[str, Any]) -> Dict[int, float]:
        scores = {player_id: 0.5 for player_id in observation["all_player_ids"]}
        scores[self.player_id] = 0.0 if self._is_good(observation) else 1.0

        for player_id in observation["known_evil_ids"]:
            scores[player_id] = 1.0

        if observation["role"] == "Percival":
            for player_id in observation["merlin_candidates"]:
                scores[player_id] = min(scores[player_id], 0.42)

        for record in observation["history"]:
            if not record["team_approved"]:
                continue

            team = record["proposed_team"]
            if record["mission_result"] is False:
                fail_pressure = 0.28 * max(record["fail_count"], 1)
                for player_id in team:
                    scores[player_id] += fail_pressure
                for player_id, vote in enumerate(record["votes"]):
                    scores[player_id] += 0.08 if vote else -0.04
            elif record["mission_result"] is True:
                for player_id in team:
                    scores[player_id] -= 0.18
                for player_id, vote in enumerate(record["votes"]):
                    scores[player_id] += -0.03 if vote else 0.04

        return {
            player_id: max(0.0, min(1.0, score))
            for player_id, score in scores.items()
        }

    def _is_good(self, observation: Dict[str, Any]) -> bool:
        return observation["role"] in GOOD_ROLES

    def _is_evil(self, observation: Dict[str, Any]) -> bool:
        return not self._is_good(observation)

    def _unique_take(self, players: List[int], team_size: int) -> List[int]:
        team = []
        for player_id in players:
            if player_id not in team:
                team.append(player_id)
            if len(team) == team_size:
                break
        return team
