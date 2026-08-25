import argparse
import random
from pathlib import Path

from bots.random_bot import RandomBot
from bots.smart_bot import SmartBot
from ml.dataset import generate_action_datasets


BOT_TYPES = {
    "random": RandomBot,
    "smart": SmartBot,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate training data for all bot actions.")
    parser.add_argument("--bot", choices=BOT_TYPES.keys(), default="smart")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.games <= 0:
        raise ValueError("--games must be greater than 0")

    if args.seed is not None:
        random.seed(args.seed)

    datasets = generate_action_datasets(BOT_TYPES[args.bot], args.games)
    outputs = {
        "team": args.output_dir / "team_smart.csv",
        "vote": args.output_dir / "vote_smart.csv",
        "mission": args.output_dir / "mission_smart.csv",
        "assassinate": args.output_dir / "assassinate_smart.csv",
    }

    datasets.team.write_csv(outputs["team"])
    datasets.vote.write_csv(outputs["vote"])
    datasets.mission.write_csv(outputs["mission"])
    datasets.assassinate.write_csv(outputs["assassinate"])

    print(f"Games: {args.games}")
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
