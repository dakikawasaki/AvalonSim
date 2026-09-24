import torch

from bots.ml_bot import MLBot
from ml.features import build_assassination_state, MAX_PLAYERS
from ml.torch_models import AssassinDQN


class DQNAssassinBot(MLBot):

    def __init__(self, player_id: int, model_path: str = "models/assassin_MLdqn.pt"):
        super().__init__(player_id=player_id)

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
        )

        self.dqn_model = AssassinDQN(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_actions=checkpoint["num_actions"],
        )

        self.dqn_model.load_state_dict(checkpoint["state_dict"])
        self.dqn_model.eval()

    def assassinate(self, observation, players):
        valid_actions = [
            player_id
            for player_id in players
            if player_id not in observation["known_evil_ids"]
        ]

        state = build_assassination_state(observation)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.dqn_model(state_tensor)[0]

        masked_q_values = q_values.clone()

        for player_id in range(MAX_PLAYERS):
            if player_id not in valid_actions:
                masked_q_values[player_id] = float("-inf")

        return int(torch.argmax(masked_q_values).item())