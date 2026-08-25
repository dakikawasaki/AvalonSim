from dataclasses import dataclass
from typing import Optional, Any

from avalon.roles import Role, Alignment, get_alignment


@dataclass
class Player:
    player_id: int
    role: Role
    bot: Optional[Any] = None

    def __post_init__(self):
        self.alignment: Alignment = get_alignment(self.role)

    def is_good(self) -> bool:
        return self.alignment == Alignment.GOOD

    def is_evil(self) -> bool:
        return self.alignment == Alignment.EVIL

    def __str__(self) -> str:
        return f"Player {self.player_id} ({self.role.value})"