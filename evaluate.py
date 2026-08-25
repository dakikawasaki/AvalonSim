import argparse
import random
from dataclasses import dataclass
from typing import Dict, Type

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


def evaluate_bot(bot_name: str, games: int) -> EvaluationResult:
    bot_cls = BOT_TYPES[bot_name]
    result = EvaluationResult(bot_name=bot_name, games=games)

    for _ in range(games):
        simulator = AvalonSimulator(bot_cls=bot_cls, verbose=False)
        state = simulator.play_game()

        if state.winner == "Good":
            result.good_wins += 1
        elif state.winner == "Evil":
            result.evil_wins += 1
        else:
            raise RuntimeError(f"Game ended without a winner: {state.winner}")

        result.total_missions += state.successful_missions + state.failed_missions
        result.total_rounds += len(state.history)

    return result


def print_result(result: EvaluationResult) -> None:
    print(f"\n=== {result.bot_name.upper()} BOT ===")
    print(f"Games: {result.games}")
    print(f"Good wins: {result.good_wins} ({result.good_win_rate:.1%})")
    print(f"Evil wins: {result.evil_wins} ({result.evil_win_rate:.1%})")
    print(f"Average completed missions: {result.average_missions:.2f}")
    print(f"Average proposal rounds: {result.average_rounds:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Avalon bots over many games.")
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
        help="Optional random seed for repeatable results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.games <= 0:
        raise ValueError("--games must be greater than 0")

    if args.seed is not None:
        random.seed(args.seed)

    bot_names = BOT_TYPES.keys() if args.bot == "all" else [args.bot]
    for bot_name in bot_names:
        result = evaluate_bot(bot_name, args.games)
        print_result(result)


if __name__ == "__main__":
    main()
