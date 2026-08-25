from avalon.simulator import AvalonSimulator
from bots.ml_bot import MLBot


def main():
    simulator = AvalonSimulator(bot_cls=MLBot)
    simulator.play_game()


if __name__ == "__main__":
    main()
