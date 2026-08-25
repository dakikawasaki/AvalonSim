from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseBot(ABC):
    def __init__(self, player_id: int):
        self.player_id = player_id

    @abstractmethod
    def propose_team(self, observation: Dict[str, Any], team_size: int) -> List[int]:
        pass

    @abstractmethod
    def vote_team(self, observation: Dict[str, Any], proposed_team: List[int]) -> bool:
        pass

    @abstractmethod
    def play_mission(self, observation: Dict[str, Any], proposed_team: List[int]) -> bool:
        pass

    @abstractmethod
    def assassinate(self, observation: Dict[str, Any], players: List[int]) -> int:
        pass