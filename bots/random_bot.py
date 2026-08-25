import random
from typing import List, Dict, Any

from bots.base_bot import BaseBot


class RandomBot(BaseBot):
    def propose_team(self, observation: Dict[str, Any], team_size: int) -> List[int]:
        all_player_ids = observation["all_player_ids"]
        return random.sample(all_player_ids, team_size)

    def vote_team(self, observation: Dict[str, Any], proposed_team: List[int]) -> bool:
        return random.choice([True, False])

    def play_mission(self, observation: Dict[str, Any], proposed_team: List[int]) -> bool:
        role = observation["role"]
        if role in ["Merlin", "Percival", "Loyal Servant"]:
            return True  # good igrači uvek success
        return random.choice([True, False])  # evil bira random success/fail

    def assassinate(self, observation: Dict[str, Any], players: List[int]) -> int:
        possible_targets = [p for p in players if p != self.player_id]
        return random.choice(possible_targets)