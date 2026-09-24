import argparse

from avalon.simulator import AvalonSimulator
from bots.ml_bot import MLBot
from bots.dqn_assassin_bot import DQNAssassinBot

class DQNRandomBot(DQNAssassinBot):
    def __init__(self, player_id):
        super().__init__(
            player_id,
            model_path="models/assassin_dqn.pt",
        )


class DQNMLBot(DQNAssassinBot):
    def __init__(self, player_id):
        super().__init__(
            player_id,
            model_path="models/assassin_MLdqn.pt",
        )

def evaluate(bot_cls, games: int):
    merlin_kills = 0
    evil_wins = 0
    assassination_games = 0

    for _ in range(games):
        simulator = AvalonSimulator(
            bot_cls=bot_cls,
            verbose=False,
        )

        state = simulator.play_game()

        if state.winner == "Evil":
            evil_wins += 1

        # Ako je Evil pobedio nakon faze atentata, assassin je pogodio Merlina.
        if state.successful_missions >= 3:
            assassination_games += 1
            if state.winner == "Evil":
                merlin_kills += 1

    return merlin_kills,assassination_games, evil_wins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1000)
    args = parser.parse_args()

    print(f"Evaluacija: {args.games} igara\n")

    print("Supervised Assassin...")
    supervised_kills, supervised_games, supervised_wins = evaluate(
        MLBot,
        args.games,
    )

    print("DQN_R Assassin...")
    dqnr_kills, dqnr_games, dqnr_wins = evaluate(
        DQNRandomBot,
        args.games,
    )

    print("DQN_ML Assassin...")
    dqnml_kills, dqnml_games, dqnml_wins = evaluate(
        DQNMLBot,
        args.games,
    )

    print("\nREZULTATI")
    print("-" * 40)

    print(
        f"Supervised: "
        f"{supervised_games}/{args.games} "
        f"{supervised_kills}/{supervised_games} "
        f"(kill/assassin_games:{supervised_kills / supervised_games:.2%})"
        f"(kill/wins:{supervised_kills / supervised_wins:.2%})"
    )

    print(
        f"DQN RANDOM: "
        f"{dqnr_games}/{args.games} "
        f"{dqnr_kills}/{dqnr_games} "
        f"(kill/assassin_games:{dqnr_kills / dqnr_games:.2%})"
        f"(kill/wins:{dqnr_kills / dqnr_wins:.2%})"
    )
    print(
        f"DQN ML: "
        f"{dqnml_games}/{args.games} "
        f"{dqnml_kills}/{dqnml_games} "
        f"(kill/assassin_games:{dqnml_kills / dqnml_games:.2%})"
        f"(kill/wins:{dqnml_kills / dqnml_wins:.2%})"
    )



if __name__ == "__main__":
    main()