import random
from typing import Callable, List, Dict, Optional, Type

from avalon.roles import Role
from avalon.player import Player
from avalon.game_state import GameState, MissionRecord
from bots.base_bot import BaseBot
from bots.random_bot import RandomBot


class AvalonSimulator:
    def __init__(
        self,
        bot_cls: Type[BaseBot] = RandomBot,
        verbose: bool = True,
        team_callback: Optional[Callable[[Dict, int, List[int]], None]] = None,
        vote_callback: Optional[Callable[[Dict, List[int], bool], None]] = None,
        mission_callback: Optional[Callable[[Dict, List[int], bool], None]] = None,
        assassinate_callback: Optional[Callable[[Dict, List[int], int], Optional[int]]] = None,
    ):
        self.bot_cls = bot_cls
        self.verbose = verbose
        self.team_callback = team_callback
        self.vote_callback = vote_callback
        self.mission_callback = mission_callback
        self.assassinate_callback = assassinate_callback
        self.roles = [
            Role.MERLIN,
            Role.PERCIVAL,
            Role.ASSASSIN,
            Role.MORDRED,
            Role.MORGANA,
            Role.LOYAL_SERVANT,
            Role.LOYAL_SERVANT,
        ]

    def log(self, message: str = "") -> None:
        if self.verbose:
            print(message)

    def setup_game(self) -> GameState:
        shuffled_roles = self.roles[:]
        random.shuffle(shuffled_roles)

        players = []
        for i, role in enumerate(shuffled_roles):
            bot = self.bot_cls(player_id=i)
            player = Player(player_id=i, role=role, bot=bot)
            players.append(player)

        state = GameState(players=players)
        return state

    def get_player_observation(self, state: GameState, player: Player) -> Dict:
        all_player_ids = [p.player_id for p in state.players]
        evil_ids = [p.player_id for p in state.players if p.is_evil()]
        visible_evil_ids = [
            p.player_id
            for p in state.players
            if p.is_evil() and p.role != Role.MORDRED
        ]
        merlin_morgana_candidates = [
            p.player_id
            for p in state.players
            if p.role in {Role.MERLIN, Role.MORGANA}
        ]

        known_evil_ids = []
        if player.is_evil():
            known_evil_ids = evil_ids
        elif player.role == Role.MERLIN:
            known_evil_ids = visible_evil_ids

        merlin_candidates = []
        if player.role == Role.PERCIVAL:
            merlin_candidates = merlin_morgana_candidates

        history = [
            {
                "mission_number": record.mission_number,
                "attempt_number": record.attempt_number,
                "leader_id": record.leader_id,
                "proposed_team": record.proposed_team,
                "votes": record.votes,
                "team_approved": record.team_approved,
                "mission_result": record.mission_result,
                "fail_count": record.fail_count,
            }
            for record in state.history
        ]

        return {
            "self_id": player.player_id,
            "role": player.role.value,
            "alignment": player.alignment.value,
            "all_player_ids": all_player_ids,
            "known_evil_ids": known_evil_ids,
            "merlin_candidates": merlin_candidates,
            "mission_number": state.mission_number,
            "attempt_number": state.attempt_number,
            "leader_id": state.get_current_leader().player_id,
            "history_length": len(state.history),
            "successful_missions": state.successful_missions,
            "failed_missions": state.failed_missions,
            "consecutive_rejections": state.consecutive_rejections,
            "history": history,
        }

    def print_roles(self, state: GameState) -> None:
        self.log("\n=== ROLES ===")
        for player in state.players:
            self.log(
                f"Player {player.player_id}: {player.role.value} ({player.alignment.value})"
            )
        self.log("=============\n")

    def propose_team(self, state: GameState) -> List[int]:
        leader = state.get_current_leader()
        observation = self.get_player_observation(state, leader)
        team_size = state.get_team_size_for_current_mission()

        proposed_team = leader.bot.propose_team(observation, team_size)
        if self.team_callback is not None:
            self.team_callback(observation, team_size, proposed_team)
        state.set_current_team(proposed_team)

        self.log(f"Leader is Player {leader.player_id}({leader.role.value})")
        self.log(f"Mission {state.mission_number}, Attempt {state.attempt_number}")
        self.log(f"Leader proposed team: {proposed_team}")

        return proposed_team

    def vote_on_team(self, state: GameState, proposed_team: List[int]) -> List[bool]:
        votes = []

        self.log("Voting phase:")
        for player in state.players:
            observation = self.get_player_observation(state, player)
            vote = player.bot.vote_team(observation, proposed_team)
            if self.vote_callback is not None:
                self.vote_callback(observation, proposed_team, vote)
            votes.append(vote)
            vote_text = "APPROVE" if vote else "REJECT"
            self.log(f"Player {player.player_id}({player.role.value}) voted: {vote_text}")

        return votes

    def is_team_approved(self, votes: List[bool]) -> bool:
        approve_count = sum(votes)
        return approve_count*2 >= len(votes)

    def run_mission(self, state: GameState, team: List[int]) -> (bool, int):
        mission_votes = []

        self.log("Mission phase:")
        self.log(f"Players on mission: {team}")

        for player_id in team:
            player = state.players[player_id]
            observation = self.get_player_observation(state, player)
            choice = player.bot.play_mission(observation, team)
            if self.mission_callback is not None:
                self.mission_callback(observation, team, choice)
            mission_votes.append(choice)

            if choice:
                self.log(f"Player {player.player_id}({player.role.value}) played: SUCCESS")
            else:
                self.log(f"Player {player.player_id}({player.role.value}) played: FAIL")

        fail_count = mission_votes.count(False)
        required_fails = state.get_required_fails_for_current_mission()
        mission_success = fail_count < required_fails

        self.log(f"Fail cards played: {fail_count}")
        if required_fails > 1:
            self.log(f"Fail cards required: {required_fails}")
        self.log(f"Mission result: {'SUCCESS' if mission_success else 'FAIL'}")

        return mission_success, fail_count

    def check_win_conditions(self, state: GameState) -> bool:
        if state.successful_missions >= 3:
            self.log("\nGood reached 3 successful missions.")
            return True

        if state.failed_missions >= 3:
            self.log("\nEvil reached 3 failed missions.")
            state.set_winner("Evil")
            return True

        if state.consecutive_rejections >= 5:
            self.log("\n5 teams were rejected in a row. Evil wins automatically.")
            state.set_winner("Evil")
            return True

        return False

    def run_assassination(self, state: GameState) -> None:
        assassins = [p for p in state.players if p.role == Role.ASSASSIN]
        evil_players = [p for p in state.players if p.is_evil()]
        assassin = assassins[0] if assassins else random.choice(evil_players)

        observation = self.get_player_observation(state, assassin)
        all_player_ids = [p.player_id for p in state.players]
        target_id = assassin.bot.assassinate(observation, all_player_ids)

        if self.assassinate_callback is not None:
            callback_target = self.assassinate_callback(
                observation,
                all_player_ids,
                target_id,
            )
            if callback_target is not None:
                target_id = callback_target

        target_player = state.players[target_id]

        self.log("\n=== ASSASSINATION PHASE ===")
        self.log(f"Assassin is Player {assassin.player_id} ({assassin.role.value})")
        self.log(f"Assassin targeted Player {target_id} ({target_player.role.value})")

        if target_player.role == Role.MERLIN:
            self.log("Merlin was killed. Evil wins!")
            state.set_winner("Evil")
        else:
            self.log("Merlin survived. Good wins!")
            state.set_winner("Good")

    def play_game(self) -> GameState:
        state = self.setup_game()
        self.print_roles(state)

        while not state.game_over:
            proposed_team = self.propose_team(state)
            votes = self.vote_on_team(state, proposed_team)
            team_approved = self.is_team_approved(votes)

            self.log("Team approved!" if team_approved else "Team rejected!")

            record = MissionRecord(
                mission_number=state.mission_number,
                attempt_number=state.attempt_number,
                leader_id=state.get_current_leader().player_id,
                proposed_team=proposed_team,
                votes=votes,
                team_approved=team_approved,
            )

            if team_approved:
                mission_success, fail_count = self.run_mission(state, proposed_team)
                record.mission_result = mission_success
                record.fail_count = fail_count
                state.record_mission(record)

                if mission_success:
                    state.mission_succeeded()
                else:
                    state.mission_failed()

                if self.check_win_conditions(state):
                    if state.successful_missions >= 3 and state.winner is None:
                        self.run_assassination(state)
                    break
            else:
                state.record_mission(record)
                state.team_rejected()

                if self.check_win_conditions(state):
                    break

            state.rotate_leader()
            self.log("\n-----------------------------\n")

        self.log("\n=== GAME OVER ===")
        self.log(f"Winner: {state.winner}")
        self.log("=================\n")

        return state
