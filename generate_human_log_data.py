import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from avalon.game_state import MissionRecord
from avalon.player import Player
from avalon.roles import Role
from ml.dataset import ActionDatasets
from ml.features import (
    build_assassination_features,
    build_mission_features,
    build_team_features,
    build_vote_features,
)


ROLE_MAP = {
    "MERLIN": Role.MERLIN,
    "PERCIVAL": Role.PERCIVAL,
    "MORGANA": Role.MORGANA,
    "MORDRED": Role.MORDRED,
    "LOYAL FOLLOWER": Role.LOYAL_SERVANT,
    "LOYAL SERVANT": Role.LOYAL_SERVANT,
}


class HumanLogParser:
    def __init__(self, strict_roles: bool = False) -> None:
        self.strict_roles = strict_roles
        self.stats = {
            "files_seen": 0,
            "games_used": 0,
            "skipped_not_7_players": 0,
            "skipped_bad_json": 0,
            "skipped_roles": 0,
            "skipped_malformed": 0,
        }

    def parse_directory(self, path: Path, limit: Optional[int] = None) -> ActionDatasets:
        datasets = ActionDatasets()
        used = 0

        for log_path in sorted(path.iterdir()):
            if not log_path.is_file():
                continue

            self.stats["files_seen"] += 1
            try:
                data = json.loads(log_path.read_text(encoding="utf-8"))
            except Exception:
                self.stats["skipped_bad_json"] += 1
                continue

            if len(data.get("players", [])) != 7:
                self.stats["skipped_not_7_players"] += 1
                continue

            try:
                if self.parse_game(data, datasets):
                    used += 1
                    self.stats["games_used"] += 1
            except (KeyError, TypeError, ValueError, IndexError):
                self.stats["skipped_malformed"] += 1
                continue

            if limit is not None and used >= limit:
                break

        return datasets

    def parse_game(self, data: Dict[str, Any], datasets: ActionDatasets) -> bool:
        names = [player["name"] for player in data["players"]]
        name_to_id = {name: index for index, name in enumerate(names)}
        roles = self._normalise_roles(data.get("outcome", {}).get("roles", []), name_to_id)
        if roles is None:
            self.stats["skipped_roles"] += 1
            return False

        players = [Player(player_id=index, role=roles[name]) for index, name in enumerate(names)]
        history: List[MissionRecord] = []
        successes = 0
        failures = 0
        consecutive_rejections = 0
        mission_cards = data.get("outcome", {}).get("votes", [])

        for mission_index, mission in enumerate(data.get("missions", []), start=1):
            attempt = 1
            approved_record: Optional[MissionRecord] = None

            for proposal in mission.get("proposals", []):
                leader_id = name_to_id[proposal["proposer"]]
                team = self._ids(proposal.get("team", []), name_to_id)
                approved_names = set(proposal.get("votes", []))
                votes = [name in approved_names for name in names]
                is_approved = proposal.get("state") == "APPROVED"

                observation = self._observation(
                    players=players,
                    viewer=players[leader_id],
                    history=history,
                    mission_number=mission_index,
                    attempt_number=attempt,
                    leader_id=leader_id,
                    successes=successes,
                    failures=failures,
                    consecutive_rejections=consecutive_rejections,
                )
                self._add_team_examples(datasets, observation, team, int(mission["teamSize"]))

                for voter in players:
                    observation = self._observation(
                        players=players,
                        viewer=voter,
                        history=history,
                        mission_number=mission_index,
                        attempt_number=attempt,
                        leader_id=leader_id,
                        successes=successes,
                        failures=failures,
                        consecutive_rejections=consecutive_rejections,
                    )
                    datasets.vote.add_example(build_vote_features(observation, team), votes[voter.player_id])

                record = MissionRecord(
                    mission_number=mission_index,
                    attempt_number=attempt,
                    leader_id=leader_id,
                    proposed_team=team,
                    votes=votes,
                    team_approved=is_approved,
                )

                if is_approved:
                    record.mission_result = mission.get("state") == "SUCCESS"
                    record.fail_count = int(mission.get("numFails", 0))
                    approved_record = record
                    self._add_mission_examples(
                        datasets=datasets,
                        mission_cards=mission_cards,
                        mission_index=mission_index,
                        players=players,
                        names=names,
                        team=team,
                        history=history,
                        attempt_number=attempt,
                        leader_id=leader_id,
                        successes=successes,
                        failures=failures,
                        consecutive_rejections=consecutive_rejections,
                    )
                    history.append(record)
                    if record.mission_result:
                        successes += 1
                    else:
                        failures += 1
                    consecutive_rejections = 0
                    break

                history.append(record)
                consecutive_rejections += 1
                attempt += 1

            if approved_record is None:
                return False

            if successes >= 3 or failures >= 3:
                break

        self._add_assassination_examples(data, datasets, players, names, history, successes, failures)
        return True

    def _normalise_roles(
        self,
        role_entries: List[Dict[str, Any]],
        name_to_id: Dict[str, int],
    ) -> Optional[Dict[str, Role]]:
        if len(role_entries) != 7:
            return None

        roles: Dict[str, Role] = {}
        spare_evil: List[str] = []

        for entry in role_entries:
            name = entry.get("name")
            if name not in name_to_id:
                return None

            raw_role = str(entry.get("role", "")).upper()
            if entry.get("assassin"):
                roles[name] = Role.ASSASSIN
            elif raw_role in ROLE_MAP:
                roles[name] = ROLE_MAP[raw_role]
            elif raw_role in {"EVIL MINION", "OBERON"} and not self.strict_roles:
                spare_evil.append(name)
            else:
                return None

        for needed_role in [Role.MORDRED, Role.MORGANA]:
            if needed_role in roles.values():
                continue
            if not spare_evil:
                return None
            roles[spare_evil.pop(0)] = needed_role

        for name in spare_evil:
            if Role.MORDRED not in roles.values():
                roles[name] = Role.MORDRED
            elif Role.MORGANA not in roles.values():
                roles[name] = Role.MORGANA
            else:
                return None

        required_counts = {
            Role.MERLIN: 1,
            Role.PERCIVAL: 1,
            Role.ASSASSIN: 1,
            Role.MORDRED: 1,
            Role.MORGANA: 1,
            Role.LOYAL_SERVANT: 2,
        }
        actual_counts = {role: list(roles.values()).count(role) for role in required_counts}
        if actual_counts != required_counts:
            return None
        return roles

    def _observation(
        self,
        players: List[Player],
        viewer: Player,
        history: List[MissionRecord],
        mission_number: int,
        attempt_number: int,
        leader_id: int,
        successes: int,
        failures: int,
        consecutive_rejections: int,
    ) -> Dict[str, Any]:
        evil_ids = [player.player_id for player in players if player.is_evil()]
        visible_evil_ids = [
            player.player_id
            for player in players
            if player.is_evil() and player.role != Role.MORDRED
        ]
        known_evil_ids: List[int] = []
        if viewer.is_evil():
            known_evil_ids = evil_ids
        elif viewer.role == Role.MERLIN:
            known_evil_ids = visible_evil_ids

        merlin_candidates = []
        if viewer.role == Role.PERCIVAL:
            merlin_candidates = [
                player.player_id
                for player in players
                if player.role in {Role.MERLIN, Role.MORGANA}
            ]

        return {
            "self_id": viewer.player_id,
            "role": viewer.role.value,
            "alignment": viewer.alignment.value,
            "all_player_ids": [player.player_id for player in players],
            "known_evil_ids": known_evil_ids,
            "merlin_candidates": merlin_candidates,
            "mission_number": mission_number,
            "attempt_number": attempt_number,
            "leader_id": leader_id,
            "history_length": len(history),
            "successful_missions": successes,
            "failed_missions": failures,
            "consecutive_rejections": consecutive_rejections,
            "history": [
                {
                    "mission_number": record.mission_number,
                    "attempt_number": record.attempt_number,
                    "leader_id": record.leader_id,
                    "proposed_team": record.proposed_team,
                    "votes": record.votes,
                    "team_approved": record.team_approved,
                    "mission_result": record.mission_result,
                    "fail_count": record.fail_count,
                }
                for record in history
            ],
        }

    def _add_team_examples(
        self,
        datasets: ActionDatasets,
        observation: Dict[str, Any],
        selected_team: List[int],
        team_size: int,
    ) -> None:
        selected = set(selected_team)
        selected_features = build_team_features(observation, selected_team)
        negative_count = 0

        for candidate_team in combinations(observation["all_player_ids"], team_size):
            candidate_team = list(candidate_team)
            is_selected = set(candidate_team) == selected
            datasets.team.add_example(build_team_features(observation, candidate_team), is_selected)
            if not is_selected:
                negative_count += 1

        for _ in range(negative_count):
            datasets.team.add_example(selected_features, True)

    def _add_mission_examples(
        self,
        datasets: ActionDatasets,
        mission_cards: List[Dict[str, bool]],
        mission_index: int,
        players: List[Player],
        names: List[str],
        team: List[int],
        history: List[MissionRecord],
        attempt_number: int,
        leader_id: int,
        successes: int,
        failures: int,
        consecutive_rejections: int,
    ) -> None:
        if mission_index - 1 >= len(mission_cards):
            return

        cards = mission_cards[mission_index - 1]
        for player_id in team:
            if names[player_id] not in cards:
                continue
            viewer = players[player_id]
            observation = self._observation(
                players=players,
                viewer=viewer,
                history=history,
                mission_number=mission_index,
                attempt_number=attempt_number,
                leader_id=leader_id,
                successes=successes,
                failures=failures,
                consecutive_rejections=consecutive_rejections,
            )
            datasets.mission.add_example(
                build_mission_features(observation, team),
                bool(cards[names[player_id]]),
            )

    def _add_assassination_examples(
        self,
        data: Dict[str, Any],
        datasets: ActionDatasets,
        players: List[Player],
        names: List[str],
        history: List[MissionRecord],
        successes: int,
        failures: int,
    ) -> None:
        outcome = data.get("outcome", {})
        target_name = outcome.get("assassinated")
        if outcome.get("state") != "EVIL_WIN" or not target_name:
            return

        target_id = self._maybe_id(target_name, {name: index for index, name in enumerate(names)})
        if target_id is None:
            return

        assassins = [player for player in players if player.role == Role.ASSASSIN]
        if not assassins:
            return

        assassin = assassins[0]
        observation = self._observation(
            players=players,
            viewer=assassin,
            history=history,
            mission_number=min(len(history) + 1, 5),
            attempt_number=1,
            leader_id=assassin.player_id,
            successes=successes,
            failures=failures,
            consecutive_rejections=0,
        )
        known_evil = set(observation["known_evil_ids"])
        for player_id in observation["all_player_ids"]:
            if player_id in known_evil:
                continue
            datasets.assassinate.add_example(
                build_assassination_features(observation, player_id),
                player_id == target_id,
            )

    def _ids(self, names: List[str], name_to_id: Dict[str, int]) -> List[int]:
        return [name_to_id[name] for name in names]

    def _maybe_id(self, name: str, name_to_id: Dict[str, int]) -> Optional[int]:
        return name_to_id.get(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Avalon action datasets from human avalonlogs JSON files."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("external/avalonlogs-master/logs"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--prefix", default="human")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--strict-roles",
        action="store_true",
        help="Skip logs with Oberon/Evil Minion variants that need normalisation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parser = HumanLogParser(strict_roles=args.strict_roles)
    datasets = parser.parse_directory(args.logs_dir, limit=args.limit)

    outputs = {
        "team": args.output_dir / f"team_{args.prefix}.csv",
        "vote": args.output_dir / f"vote_{args.prefix}.csv",
        "mission": args.output_dir / f"mission_{args.prefix}.csv",
        "assassinate": args.output_dir / f"assassinate_{args.prefix}.csv",
    }
    print("OUTPUT TEAM:", repr(outputs["team"]))
    print("ABSOLUTE PATH:", outputs["team"].resolve())
    print("PARENT EXISTS:", outputs["team"].parent.exists())

    datasets.team.write_csv(outputs["team"])
    datasets.team.write_csv(outputs["team"])
    datasets.vote.write_csv(outputs["vote"])
    datasets.mission.write_csv(outputs["mission"])
    datasets.assassinate.write_csv(outputs["assassinate"])

    print("Human log parsing complete.")
    for key, value in parser.stats.items():
        print(f"{key}: {value}")

    for name, dataset in [
        ("team", datasets.team),
        ("vote", datasets.vote),
        ("mission", datasets.mission),
        ("assassinate", datasets.assassinate),
    ]:
        print(f"\n{name.upper()}")
        print(f"Wrote: {outputs[name]}")
        print(f"Examples: {dataset.example_count}")
        print(f"Positive: {dataset.positive_count} ({dataset.positive_rate:.1%})")
        print(f"Negative: {dataset.negative_count} ({1 - dataset.positive_rate:.1%})")


if __name__ == "__main__":
    main()
