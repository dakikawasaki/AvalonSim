from typing import Any, Dict, List


ROLE_ORDER = ["Merlin", "Percival", "Loyal Servant", "Assassin", "Mordred", "Morgana"]
ALIGNMENT_ORDER = ["Good", "Evil"]
MAX_PLAYERS = 7


def build_vote_features(observation: Dict[str, Any], proposed_team: List[int]) -> List[float]:
    """Convert a vote decision state into fixed-size numeric features."""
    return _build_decision_features(observation, proposed_team, [])


def build_team_features(
    observation: Dict[str, Any],
    proposed_team: List[int],
) -> List[float]:
    """Convert a complete team proposal into fixed-size numeric features."""
    return _build_decision_features(observation, proposed_team, [])


def build_mission_features(
    observation: Dict[str, Any],
    proposed_team: List[int],
) -> List[float]:
    """Convert a mission card decision into fixed-size numeric features."""
    return _build_decision_features(observation, proposed_team, [])


def build_assassination_features(
    observation: Dict[str, Any],
    target_id: int,
) -> List[float]:
    """Convert an assassination target decision into fixed-size numeric features."""
    return _build_decision_features(
        observation,
        [target_id],
        [
            target_id / (MAX_PLAYERS - 1),
            _as_float(target_id in observation["known_evil_ids"]),
        ],
    )

def build_assassination_state(
    observation: Dict[str, Any],
) -> List[float]:
    """Convert the current game state into DQN state features."""
    return _build_decision_features(
        observation,
        [],
        [],
    )

def _build_decision_features(
    observation: Dict[str, Any],
    proposed_team: List[int],
    extra_features: List[float],
) -> List[float]:
    features: List[float] = []
    all_players = observation["all_player_ids"]
    proposed_team_set = set(proposed_team)

    features.extend(
        [
            observation["mission_number"] / 5,
            observation["attempt_number"] / 5,
            observation["successful_missions"] / 3,
            observation["failed_missions"] / 3,
            observation["consecutive_rejections"] / 5,
            observation["history_length"] / 25,
            _as_float(observation["leader_id"] == observation["self_id"]),
            _as_float(observation["self_id"] in proposed_team_set),
            len(proposed_team) / MAX_PLAYERS,
        ]
    )

    features.extend(_one_hot(observation["role"], ROLE_ORDER))
    features.extend(_one_hot(observation["alignment"], ALIGNMENT_ORDER))

    player_stats = _build_player_stats(observation)
    for player_id in range(MAX_PLAYERS):
        if player_id in all_players:
            stats = player_stats[player_id]
            features.extend(
                [
                    _as_float(player_id == observation["self_id"]),
                    _as_float(player_id == observation["leader_id"]),
                    _as_float(player_id in proposed_team_set),
                    _as_float(player_id in observation["known_evil_ids"]),
                    _as_float(player_id in observation["merlin_candidates"]),
                    stats["approve_count"] / 25,
                    stats["reject_count"] / 25,
                    stats["successful_team_count"] / 5,
                    stats["failed_team_count"] / 5,
                    stats["approved_failed_team_count"] / 5,
                ]
            )
        else:
            features.extend([0.0] * 10)

    features.extend(extra_features)
    return features


def vote_feature_size() -> int:
    empty_observation = {
        "self_id": 0,
        "role": "Merlin",
        "alignment": "Good",
        "all_player_ids": list(range(MAX_PLAYERS)),
        "known_evil_ids": [],
        "merlin_candidates": [],
        "mission_number": 1,
        "attempt_number": 1,
        "leader_id": 0,
        "history_length": 0,
        "successful_missions": 0,
        "failed_missions": 0,
        "consecutive_rejections": 0,
        "history": [],
    }
    return len(build_vote_features(empty_observation, [0, 1]))


def team_feature_size() -> int:
    return len(build_team_features(_empty_observation(), [0, 1]))


def mission_feature_size() -> int:
    return len(build_mission_features(_empty_observation(), [0, 1]))


def assassination_feature_size() -> int:
    return len(build_assassination_features(_empty_observation(), 0))


def _empty_observation() -> Dict[str, Any]:
    return {
        "self_id": 0,
        "role": "Merlin",
        "alignment": "Good",
        "all_player_ids": list(range(MAX_PLAYERS)),
        "known_evil_ids": [],
        "merlin_candidates": [],
        "mission_number": 1,
        "attempt_number": 1,
        "leader_id": 0,
        "history_length": 0,
        "successful_missions": 0,
        "failed_missions": 0,
        "consecutive_rejections": 0,
        "history": [],
    }


def _build_player_stats(observation: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    stats = {
        player_id: {
            "approve_count": 0.0,
            "reject_count": 0.0,
            "successful_team_count": 0.0,
            "failed_team_count": 0.0,
            "approved_failed_team_count": 0.0,
        }
        for player_id in observation["all_player_ids"]
    }

    for record in observation["history"]:
        for player_id, vote in enumerate(record["votes"]):
            if player_id not in stats:
                continue
            if vote:
                stats[player_id]["approve_count"] += 1
            else:
                stats[player_id]["reject_count"] += 1

        if record["mission_result"] is True:
            for player_id in record["proposed_team"]:
                stats[player_id]["successful_team_count"] += 1
        elif record["mission_result"] is False:
            for player_id in record["proposed_team"]:
                stats[player_id]["failed_team_count"] += 1
                if record["votes"][player_id]:
                    stats[player_id]["approved_failed_team_count"] += 1

    return stats


def _one_hot(value: str, choices: List[str]) -> List[float]:
    return [_as_float(value == choice) for choice in choices]


def _as_float(value: bool) -> float:
    return 1.0 if value else 0.0
