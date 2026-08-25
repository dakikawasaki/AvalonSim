import argparse
import subprocess
import sys


TRAINING_STEPS = [
    {
        "name": "team",
        "data_template": "data/team_{prefix}.csv",
        "output_stem_template": "models/team{suffix}_model",
        "epochs": 3,
    },
    {
        "name": "vote",
        "data_template": "data/vote_{prefix}.csv",
        "output_stem_template": "models/vote{suffix}_model",
        "epochs": 5,
    },
    {
        "name": "mission",
        "data_template": "data/mission_{prefix}.csv",
        "output_stem_template": "models/mission{suffix}_model",
        "epochs": 5,
    },
    {
        "name": "assassinate",
        "data_template": "data/assassinate_{prefix}.csv",
        "output_stem_template": "models/assassinate{suffix}_model",
        "epochs": 8,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all Avalon action models.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer epochs for a faster local training pass.",
    )
    parser.add_argument(
        "--prefix",
        default="human",
        help="Dataset suffix to read from data/<action>_<prefix>.csv.",
    )
    parser.add_argument(
        "--model-suffix",
        default="",
        help="Optional suffix for model files, for example 'human' writes team_human_model.pt.",
    )
    parser.add_argument(
        "--backend",
        choices=["torch", "json"],
        default="torch",
        help="Training backend. 'json' uses the dependency-free MLP trainer.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Optional stratified sample cap per action for the json backend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = f"_{args.model_suffix}" if args.model_suffix else ""

    for step in TRAINING_STEPS:
        epochs = max(1, step["epochs"] - 2) if args.quick else step["epochs"]
        output_suffix = ".pt" if args.backend == "torch" else ".json"
        command = [
            sys.executable,
            "-m",
            "ml.train_action_torch" if args.backend == "torch" else "ml.train_action",
            "--data",
            step["data_template"].format(prefix=args.prefix),
            "--output",
            step["output_stem_template"].format(suffix=suffix) + output_suffix,
            "--epochs",
            str(epochs),
            "--seed",
            str(args.seed),
        ]
        if args.backend == "json" and args.max_examples > 0:
            command.extend(["--max-examples", str(args.max_examples)])

        print(f"\n=== Training {step['name']} model ===")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
