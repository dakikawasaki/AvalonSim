import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Type

from avalon.simulator import AvalonSimulator
from bots.base_bot import BaseBot
from bots.ml_bot import MLBot
from bots.random_bot import RandomBot
from bots.smart_bot import SmartBot


BOT_TYPES: Dict[str, Type[BaseBot]] = {
    "ml": MLBot,
    "random": RandomBot,
    "smart": SmartBot,
}


@dataclass
class EvaluationResult:
    bot_name: str
    games: int

    good_wins: int = 0
    evil_wins: int = 0

    total_missions: int = 0
    total_rounds: int = 0

    approved_teams: int = 0
    rejected_teams: int = 0

    total_votes: int = 0
    approve_votes: int = 0

    evil_mission_actions: int = 0
    evil_fail_actions: int = 0

    total_fail_cards: int = 0

    assassination_attempts: int = 0
    assassination_merlin_hits: int = 0

    five_rejection_games: int = 0

    @property
    def good_win_rate(self) -> float:
        return self.good_wins / self.games

    @property
    def evil_win_rate(self) -> float:
        return self.evil_wins / self.games

    @property
    def average_missions(self) -> float:
        return self.total_missions / self.games

    @property
    def average_rounds(self) -> float:
        return self.total_rounds / self.games

    @property
    def approval_rate(self) -> float:
        if self.total_votes == 0:
            return 0.0
        return self.approve_votes / self.total_votes

    @property
    def team_approval_rate(self) -> float:
        total = self.approved_teams + self.rejected_teams
        if total == 0:
            return 0.0
        return self.approved_teams / total

    @property
    def evil_fail_rate(self) -> float:
        if self.evil_mission_actions == 0:
            return 0.0
        return self.evil_fail_actions / self.evil_mission_actions

    @property
    def assassination_accuracy(self) -> float:
        if self.assassination_attempts == 0:
            return 0.0
        return self.assassination_merlin_hits / self.assassination_attempts

    @property
    def five_rejection_rate(self) -> float:
        return self.five_rejection_games / self.games


def save_game_row(path: Path, row: Dict) -> None:
    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_action_row(path: Path, row: Dict) -> None:
    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def evaluate_bot(bot_name: str, games: int, output_dir: Path) -> EvaluationResult:
    bot_cls = BOT_TYPES[bot_name]
    result = EvaluationResult(bot_name=bot_name, games=games)

    output_dir.mkdir(parents=True, exist_ok=True)

    games_path = output_dir / f"{bot_name}_games.csv"
    actions_path = output_dir / f"{bot_name}_actions.csv"

    # Omogućava da novo pokretanje evaluacije ne dodaje
    # rezultate na staru evaluaciju.
    if games_path.exists():
        games_path.unlink()

    if actions_path.exists():
        actions_path.unlink()

    for game_id in range(1, games + 1):

        game_data = {
            "game_id": game_id,
            "bot": bot_name,
            "proposals": 0,
            "approved_teams": 0,
            "rejected_teams": 0,
            "votes": 0,
            "approve_votes": 0,
            "evil_mission_actions": 0,
            "evil_fail_actions": 0,
            "fail_cards": 0,
            "assassin_target": "",
            "assassin_target_role": "",
            "winner": "",
            "missions": 0,
            "rounds": 0,
            "five_rejections": False,
        }

        def team_callback(observation, team_size, proposed_team):
            game_data["proposals"] += 1

            save_action_row(
                actions_path,
                {
                    "game_id": game_id,
                    "action": "proposal",
                    "player_id": observation["self_id"],
                    "role": observation["role"],
                    "alignment": observation["alignment"],
                    "mission": observation["mission_number"],
                    "attempt": observation["attempt_number"],
                    "team": str(proposed_team),
                    "decision": "",
                },
            )

        def vote_callback(observation, proposed_team, vote):
            game_data["votes"] += 1

            if vote:
                game_data["approve_votes"] += 1

            save_action_row(
                actions_path,
                {
                    "game_id": game_id,
                    "action": "vote",
                    "player_id": observation["self_id"],
                    "role": observation["role"],
                    "alignment": observation["alignment"],
                    "mission": observation["mission_number"],
                    "attempt": observation["attempt_number"],
                    "team": str(proposed_team),
                    "decision": "APPROVE" if vote else "REJECT",
                },
            )

        def mission_callback(observation, team, choice):
            if observation["alignment"] == "Evil":
                game_data["evil_mission_actions"] += 1

                if not choice:
                    game_data["evil_fail_actions"] += 1

            save_action_row(
                actions_path,
                {
                    "game_id": game_id,
                    "action": "mission",
                    "player_id": observation["self_id"],
                    "role": observation["role"],
                    "alignment": observation["alignment"],
                    "mission": observation["mission_number"],
                    "attempt": observation["attempt_number"],
                    "team": str(team),
                    "decision": "SUCCESS" if choice else "FAIL",
                },
            )

        assassin_data = {}

        def assassinate_callback(observation, all_player_ids, target_id):
            assassin_data["assassin_id"] = observation["self_id"]
            assassin_data["target_id"] = target_id

            save_action_row(
                actions_path,
                {
                    "game_id": game_id,
                    "action": "assassination",
                    "player_id": observation["self_id"],
                    "role": observation["role"],
                    "alignment": observation["alignment"],
                    "mission": observation["mission_number"],
                    "attempt": observation["attempt_number"],
                    "team": "",
                    "decision": str(target_id),
                },
            )

        simulator = AvalonSimulator(
            bot_cls=bot_cls,
            verbose=False,
            team_callback=team_callback,
            vote_callback=vote_callback,
            mission_callback=mission_callback,
            assassinate_callback=assassinate_callback,
        )

        state = simulator.play_game()

        if assassin_data:
            target_id = assassin_data["target_id"]
            target_player = state.players[target_id]

            game_data["assassin_target"] = target_id
            game_data["assassin_target_role"] = target_player.role.value

            result.assassination_attempts += 1

            if target_player.role.value == "Merlin":
                result.assassination_merlin_hits += 1

        # Game-level statistika
        total_rounds = len(state.history)
        total_missions = state.successful_missions + state.failed_missions

        result.total_rounds += total_rounds
        result.total_missions += total_missions

        game_data["rounds"] = total_rounds
        game_data["missions"] = total_missions
        game_data["winner"] = state.winner

        if state.winner == "Good":
            result.good_wins += 1
        elif state.winner == "Evil":
            result.evil_wins += 1
        else:
            raise RuntimeError(
                f"Game ended without a winner: {state.winner}"
            )

        # Analiza history-ja
        for record in state.history:
            if record.team_approved:
                result.approved_teams += 1
                game_data["approved_teams"] += 1
            else:
                result.rejected_teams += 1
                game_data["rejected_teams"] += 1

            result.total_fail_cards += record.fail_count
            game_data["fail_cards"] += record.fail_count

        # Ukupno glasova
        result.total_votes += game_data["votes"]
        result.approve_votes += game_data["approve_votes"]

        # Evil fail statistika
        result.evil_mission_actions += game_data["evil_mission_actions"]
        result.evil_fail_actions += game_data["evil_fail_actions"]

        # Ako je igra završila zbog 5 rejectiona.
        if state.consecutive_rejections >= 5:
            result.five_rejection_games += 1
            game_data["five_rejections"] = True

        save_game_row(games_path, game_data)

    return result


def print_result(result: EvaluationResult) -> None:
    print(f"\n=== {result.bot_name.upper()} BOT ===")
    print(f"Games: {result.games}")

    print(
        f"Good wins: {result.good_wins} "
        f"({result.good_win_rate:.1%})"
    )

    print(
        f"Evil wins: {result.evil_wins} "
        f"({result.evil_win_rate:.1%})"
    )

    print(
        f"Average completed missions: "
        f"{result.average_missions:.2f}"
    )

    print(
        f"Average proposal rounds: "
        f"{result.average_rounds:.2f}"
    )

    print(
        f"Team approval rate: "
        f"{result.team_approval_rate:.1%}"
    )

    print(
        f"Vote approval rate: "
        f"{result.approval_rate:.1%}"
    )

    print(
        f"Evil FAIL rate on missions: "
        f"{result.evil_fail_rate:.1%}"
    )

    print(
        f"Average FAIL cards per game: "
        f"{result.total_fail_cards / result.games:.2f}"
    )

    print(
        f"Assassin -> Merlin accuracy: "
        f"{result.assassination_accuracy:.1%}"
    )

    print(
        f"Games ending by 5 rejections: "
        f"{result.five_rejection_games} "
        f"({result.five_rejection_rate:.1%})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Avalon bots over many games."
    )

    parser.add_argument(
        "--bot",
        choices=[*BOT_TYPES.keys(), "all"],
        default="all",
        help="Which bot to evaluate.",
    )

    parser.add_argument(
        "--games",
        type=int,
        default=1000,
        help="Number of games to simulate per bot.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Directory for evaluation CSV files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.games <= 0:
        raise ValueError("--games must be greater than 0")

    if args.seed is not None:
        random.seed(args.seed)

    bot_names = (
        BOT_TYPES.keys()
        if args.bot == "all"
        else [args.bot]
    )

    for bot_name in bot_names:
        result = evaluate_bot(
            bot_name,
            args.games,
            args.output,
        )

        print_result(result)


if __name__ == "__main__":
    main()
