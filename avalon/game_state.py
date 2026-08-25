from dataclasses import dataclass, field
from typing import List, Optional

from avalon.player import Player


MISSION_TEAM_SIZES = [2, 3, 3, 4, 4]


@dataclass
class MissionRecord:
    mission_number: int
    attempt_number: int
    leader_id: int
    proposed_team: List[int]
    votes: List[bool]
    team_approved: bool
    mission_result: Optional[bool] = None
    fail_count: int = 0


@dataclass
class GameState:
    players: List[Player]
    current_leader_index: int = 0
    mission_number: int = 1
    attempt_number: int = 1
    successful_missions: int = 0
    failed_missions: int = 0
    consecutive_rejections: int = 0
    current_team: List[int] = field(default_factory=list)
    history: List[MissionRecord] = field(default_factory=list)
    game_over: bool = False
    winner: Optional[str] = None

    def get_current_leader(self) -> Player:
        return self.players[self.current_leader_index]

    def get_team_size_for_current_mission(self) -> int:
        return MISSION_TEAM_SIZES[self.mission_number - 1]

    def get_required_fails_for_current_mission(self) -> int:
        if len(self.players) >= 7 and self.mission_number == 4:
            return 2
        return 1

    def rotate_leader(self) -> None:
        self.current_leader_index = (self.current_leader_index + 1) % len(self.players)

    def record_mission(self, record: MissionRecord) -> None:
        self.history.append(record)

    def set_current_team(self, team: List[int]) -> None:
        self.current_team = team

    def clear_current_team(self) -> None:
        self.current_team = []

    def mission_succeeded(self) -> None:
        self.successful_missions += 1
        self.consecutive_rejections = 0
        self.mission_number += 1
        self.attempt_number = 1
        self.clear_current_team()

    def mission_failed(self) -> None:
        self.failed_missions += 1
        self.consecutive_rejections = 0
        self.mission_number += 1
        self.attempt_number = 1
        self.clear_current_team()

    def team_rejected(self) -> None:
        self.consecutive_rejections += 1
        self.attempt_number += 1
        self.clear_current_team()

    def set_winner(self, winner: str) -> None:
        self.game_over = True
        self.winner = winner
