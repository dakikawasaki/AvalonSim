import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from avalon.simulator import AvalonSimulator
from ml.features import build_assassination_state, MAX_PLAYERS
from ml.torch_models import AssassinDQN
from bots.random_bot import RandomBot
from bots.ml_bot import MLBot


def select_action(
    model: AssassinDQN,
    state,
    valid_actions,
    epsilon: float,
) -> int:
    if random.random() < epsilon:
        return random.choice(valid_actions)

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor)[0]

    masked_q_values = q_values.clone()
    invalid_actions = [
        player_id
        for player_id in range(MAX_PLAYERS)
        if player_id not in valid_actions
    ]

    masked_q_values[invalid_actions] = float("-inf")

    return int(torch.argmax(masked_q_values).item())


def train_episode(
    model: AssassinDQN,
    optimizer,
    epsilon: float,
) -> float:
    captured = {}

    def on_assassination(observation, players, _original_target):
        valid_actions = [
            player_id
            for player_id in players
            if player_id not in observation["known_evil_ids"]
        ]

        state = build_assassination_state(observation)

        target_id = select_action(
            model,
            state,
            valid_actions,
            epsilon,
        )

        captured["state"] = state
        captured["target_id"] = target_id

        return target_id

    simulator = AvalonSimulator(
        bot_cls=MLBot,
        verbose=False,
        assassinate_callback=on_assassination,
    )

    state = simulator.play_game()

    if "state" not in captured:
        return 0.0

    reward = 1.0 if state.winner == "Evil" else -1.0

    state_tensor = torch.tensor(
        captured["state"],
        dtype=torch.float32,
    ).unsqueeze(0)

    q_values = model(state_tensor)
    q_value = q_values[0, captured["target_id"]]

    target = torch.tensor(
        reward,
        dtype=torch.float32,
    )

    loss = F.smooth_l1_loss(q_value, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return reward


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Assassin DQN with one-step Q-learning."
    )
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = AssassinDQN(
        input_size=87,
        hidden_size=args.hidden_size,
        num_actions=MAX_PLAYERS,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    epsilon = args.epsilon_start

    rewards = []

    for episode in range(1, args.episodes + 1):
        reward = train_episode(
            model=model,
            optimizer=optimizer,
            epsilon=epsilon,
        )

        rewards.append(reward)

        epsilon = max(
            args.epsilon_end,
            epsilon * args.epsilon_decay,
        )

        if episode % 100 == 0:
            average_reward = sum(rewards[-100:]) / 100

            print(
                f"Episode {episode:05d} | "
                f"avg reward {average_reward:.3f} | "
                f"epsilon {epsilon:.3f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "input_size": 87,
            "hidden_size": args.hidden_size,
            "num_actions": MAX_PLAYERS,
            "state_dict": model.state_dict(),
        },
        args.output,
    )

    print(f"Saved model: {args.output}")


if __name__ == "__main__":
    main()