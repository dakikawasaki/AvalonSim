import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Type

from avalon.simulator import AvalonSimulator
from bots.ml_bot import MLBot


@dataclass
class RLConfig:
    mission_success_threshold: float = 0.28
    self_team_multiplier: float = 1.2
    good_known_evil_multiplier: float = 0.02
    percival_double_candidate_multiplier: float = 0.55
    evil_known_evil_multiplier: float = 1.05
    assassin_min_weight: float = 0.2


@dataclass
class RLEvaluation:
    games: int
    good_wins: int
    evil_wins: int

    @property
    def good_win_rate(self) -> float:
        return self.good_wins / self.games

    @property
    def balance_score(self) -> float:
        return 1.0 - abs(0.5 - self.good_win_rate)


def make_bot_cls(config: RLConfig) -> Type[MLBot]:
    class TunedMLBot(MLBot):
        mission_success_threshold = config.mission_success_threshold
        self_team_multiplier = config.self_team_multiplier
        good_known_evil_multiplier = config.good_known_evil_multiplier
        percival_double_candidate_multiplier = config.percival_double_candidate_multiplier
        evil_known_evil_multiplier = config.evil_known_evil_multiplier
        assassin_min_weight = config.assassin_min_weight

    return TunedMLBot


def evaluate_config(config: RLConfig, games: int) -> RLEvaluation:
    bot_cls = make_bot_cls(config)
    good_wins = 0
    evil_wins = 0

    for _ in range(games):
        state = AvalonSimulator(bot_cls=bot_cls, verbose=False).play_game()
        if state.winner == "Good":
            good_wins += 1
        elif state.winner == "Evil":
            evil_wins += 1
        else:
            raise RuntimeError(f"Game ended without a winner: {state.winner}")

    return RLEvaluation(games=games, good_wins=good_wins, evil_wins=evil_wins)


def mutate_config(config: RLConfig, rng: random.Random, scale: float) -> RLConfig:
    return RLConfig(
        mission_success_threshold=_mutate(
            config.mission_success_threshold, rng, scale, 0.02, 0.8
        ),
        self_team_multiplier=_mutate(config.self_team_multiplier, rng, scale, 0.8, 1.8),
        good_known_evil_multiplier=_mutate(
            config.good_known_evil_multiplier, rng, scale, 0.001, 0.25
        ),
        percival_double_candidate_multiplier=_mutate(
            config.percival_double_candidate_multiplier, rng, scale, 0.2, 1.0
        ),
        evil_known_evil_multiplier=_mutate(
            config.evil_known_evil_multiplier, rng, scale, 0.8, 1.5
        ),
        assassin_min_weight=_mutate(config.assassin_min_weight, rng, scale, 0.05, 0.8),
    )


def tune(iterations: int, games: int, seed: int) -> RLConfig:
    rng = random.Random(seed)
    best_config = RLConfig()
    best_eval = evaluate_config(best_config, games)
    print_result(0, best_config, best_eval, best_eval)

    for iteration in range(1, iterations + 1):
        scale = max(0.05, 0.25 * (1.0 - iteration / max(iterations, 1)))
        candidate = mutate_config(best_config, rng, scale)
        candidate_eval = evaluate_config(candidate, games)

        if candidate_eval.balance_score >= best_eval.balance_score:
            best_config = candidate
            best_eval = candidate_eval

        print_result(iteration, candidate, candidate_eval, best_eval)

    return best_config


def print_result(
    iteration: int,
    config: RLConfig,
    evaluation: RLEvaluation,
    best_evaluation: RLEvaluation,
) -> None:
    print(
        f"Iter {iteration:02d} | "
        f"Good {evaluation.good_win_rate:.1%} | "
        f"Balance {evaluation.balance_score:.3f} | "
        f"Best {best_evaluation.good_win_rate:.1%}"
    )
    print(f"  {asdict(config)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-play tune MLBot calibration parameters.")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("models/rl_config.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best_config = tune(args.iterations, args.games, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(best_config), indent=2), encoding="utf-8")
    print(f"\nSaved best config: {args.output}")


def _mutate(
    value: float,
    rng: random.Random,
    scale: float,
    minimum: float,
    maximum: float,
) -> float:
    mutated = value * (1.0 + rng.uniform(-scale, scale))
    return max(minimum, min(maximum, mutated))


if __name__ == "__main__":
    main()
