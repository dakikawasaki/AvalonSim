import argparse
import random
from pathlib import Path

from bots.random_bot import RandomBot
from bots.smart_bot import SmartBot
from ml.dataset import generate_vote_dataset


BOT_TYPES = {
    "random": RandomBot,
    "smart": SmartBot,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate vote training data.")
    parser.add_argument(
        "--bot",
        choices=BOT_TYPES.keys(),
        default="smart",
        help="Bot used as the teacher.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1000,
        help="Number of simulated games to record.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/vote_smart.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for repeatable data generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.games <= 0:
        raise ValueError("--games must be greater than 0")

    if args.seed is not None:
        random.seed(args.seed)

    dataset = generate_vote_dataset(BOT_TYPES[args.bot], args.games)
    dataset.write_csv(args.output)

    print(f"Wrote: {args.output}")
    print(f"Games: {args.games}")
    print(f"Vote examples: {dataset.example_count}")
    print(f"Approve: {dataset.approve_count} ({dataset.approve_rate:.1%})")
    print(f"Reject: {dataset.reject_count} ({1 - dataset.approve_rate:.1%})")


if __name__ == "__main__":
    main()
