from enum import Enum


class Role(Enum):
    MERLIN = "Merlin"
    PERCIVAL = "Percival"
    ASSASSIN = "Assassin"
    MORDRED = "Mordred"
    MORGANA = "Morgana"
    LOYAL_SERVANT = "Loyal Servant"


class Alignment(Enum):
    GOOD = "Good"
    EVIL = "Evil"


def get_alignment(role: Role) -> Alignment:
    if role in {Role.MERLIN, Role.PERCIVAL, Role.LOYAL_SERVANT}:
        return Alignment.GOOD
    elif role in {Role.ASSASSIN, Role.MORDRED, Role.MORGANA}:
        return Alignment.EVIL
    else:
        raise ValueError(f"Unknown role: {role}")
