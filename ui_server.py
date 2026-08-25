import json
import random
import socket
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from avalon.game_state import GameState, MissionRecord
from avalon.player import Player
from avalon.roles import Role
from bots.ml_bot import MLBot
from bots.random_bot import RandomBot
from bots.smart_bot import SmartBot


PLAYER_COUNT = 7
BOT_TYPES = {
    "random": RandomBot,
    "smart": SmartBot,
    "ml": MLBot,
}


@dataclass
class PlayerConfig:
    kind: str
    name: str


class HumanBot:
    def __init__(self, player_id: int):
        self.player_id = player_id


class UIGame:
    def __init__(self) -> None:
        self.state: Optional[GameState] = None
        self.player_configs: List[PlayerConfig] = []
        self.phase = "setup"
        self.pending_record: Optional[MissionRecord] = None
        self.pending_votes: Dict[int, bool] = {}
        self.pending_mission_choices: Dict[int, bool] = {}
        self.assassin_id: Optional[int] = None
        self.log: List[str] = []

    def start(self, configs: List[Dict[str, str]]) -> None:
        if len(configs) != PLAYER_COUNT:
            raise ValueError(f"Expected {PLAYER_COUNT} players.")

        self.player_configs = [
            PlayerConfig(
                kind=config.get("kind", "human"),
                name=(config.get("name") or f"Player {index}").strip(),
            )
            for index, config in enumerate(configs)
        ]

        roles = [
            Role.MERLIN,
            Role.PERCIVAL,
            Role.ASSASSIN,
            Role.MORDRED,
            Role.MORGANA,
            Role.LOYAL_SERVANT,
            Role.LOYAL_SERVANT,
        ]
        random.shuffle(roles)

        players = []
        for player_id, role in enumerate(roles):
            bot = self._make_bot(player_id, self.player_configs[player_id].kind)
            players.append(Player(player_id=player_id, role=role, bot=bot))

        self.state = GameState(players=players)
        self.phase = "propose"
        self.pending_record = None
        self.pending_votes = {}
        self.pending_mission_choices = {}
        self.assassin_id = None
        self.log = ["New game started.", "Roles were assigned."]
        self._advance_until_human_needed()

    def submit_team(self, player_id: int, team: List[int]) -> None:
        state = self._require_state()
        leader_id = state.get_current_leader().player_id
        if self.phase != "propose" or player_id != leader_id:
            raise ValueError("This player is not choosing a team right now.")

        self._validate_team(team, state.get_team_size_for_current_mission())
        self._set_team(team, "Human")
        self.phase = "vote"
        self._advance_until_human_needed()

    def submit_vote(self, player_id: int, vote: bool) -> None:
        self._require_state()
        if self.phase != "vote" or not self._is_human(player_id):
            raise ValueError("This player is not voting right now.")
        self.pending_votes[player_id] = bool(vote)
        self.log.append(f"{self._label(player_id)} voted {'APPROVE' if vote else 'REJECT'}.")
        self._advance_until_human_needed()

    def submit_mission(self, player_id: int, choice: bool) -> None:
        state = self._require_state()
        if self.phase != "mission" or player_id not in state.current_team:
            raise ValueError("This player is not on the mission right now.")
        if not self._is_human(player_id):
            raise ValueError("This player is controlled by a bot.")

        player = state.players[player_id]
        if player.is_good():
            choice = True

        self.pending_mission_choices[player_id] = bool(choice)
        self.log.append(f"{self._label(player_id)} submitted a mission card.")
        self._advance_until_human_needed()

    def submit_assassination(self, player_id: int, target_id: int) -> None:
        state = self._require_state()
        if self.phase != "assassinate" or player_id != self.assassin_id:
            raise ValueError("This player is not assassinating right now.")
        if target_id not in self._assassination_targets():
            raise ValueError("Invalid assassination target.")
        self._finish_assassination(target_id)

    def snapshot(self) -> Dict[str, Any]:
        if self.state is None:
            return {
                "phase": "setup",
                "players": [],
                "log": self.log,
                "pending": None,
            }

        state = self.state
        players = []
        for player in state.players:
            observation = self._observation(player)
            players.append(
                {
                    "id": player.player_id,
                    "name": self.player_configs[player.player_id].name,
                    "kind": self.player_configs[player.player_id].kind,
                    "role": player.role.value,
                    "alignment": player.alignment.value,
                    "known_evil_ids": observation["known_evil_ids"],
                    "known_evil_players": [
                        self._private_player_info(
                            player_id,
                            include_role=player.is_evil(),
                            hidden_role_hint="Evil",
                        )
                        for player_id in observation["known_evil_ids"]
                    ],
                    "merlin_candidates": observation["merlin_candidates"],
                    "merlin_candidate_players": [
                        self._private_player_info(
                            player_id,
                            include_role=False,
                            hidden_role_hint="Merlin or Morgana",
                        )
                        for player_id in observation["merlin_candidates"]
                    ],
                }
            )

        return {
            "phase": self.phase,
            "players": players,
            "score": {
                "successful_missions": state.successful_missions,
                "failed_missions": state.failed_missions,
                "mission_number": state.mission_number,
                "attempt_number": state.attempt_number,
                "consecutive_rejections": state.consecutive_rejections,
                "leader_id": state.get_current_leader().player_id
                if not state.game_over
                else None,
                "team_size": state.get_team_size_for_current_mission()
                if not state.game_over and state.mission_number <= 5
                else None,
                "winner": state.winner,
            },
            "current_team": state.current_team,
            "pending": self._pending_action(),
            "history": [self._record_to_dict(record) for record in state.history],
            "log": self.log[-80:],
        }

    def _advance_until_human_needed(self) -> None:
        while self.state is not None and not self.state.game_over:
            if self.phase == "propose":
                leader_id = self.state.get_current_leader().player_id
                if self._is_human(leader_id):
                    return
                observation = self._observation(self.state.players[leader_id])
                team_size = self.state.get_team_size_for_current_mission()
                team = self.state.players[leader_id].bot.propose_team(observation, team_size)
                self._validate_team(team, team_size)
                self._set_team(team, "Bot")
                self.phase = "vote"
                continue

            if self.phase == "vote":
                for player in self.state.players:
                    if player.player_id in self.pending_votes:
                        continue
                    if self._is_human(player.player_id):
                        return
                    observation = self._observation(player)
                    vote = player.bot.vote_team(observation, self.state.current_team)
                    self.pending_votes[player.player_id] = bool(vote)
                    self.log.append(
                        f"{self._label(player.player_id)} voted "
                        f"{'APPROVE' if vote else 'REJECT'}."
                    )

                self._resolve_vote()
                continue

            if self.phase == "mission":
                for player_id in self.state.current_team:
                    if player_id in self.pending_mission_choices:
                        continue
                    if self._is_human(player_id):
                        return
                    player = self.state.players[player_id]
                    observation = self._observation(player)
                    choice = player.bot.play_mission(observation, self.state.current_team)
                    self.pending_mission_choices[player_id] = bool(choice)
                    self.log.append(f"{self._label(player_id)} submitted a mission card.")

                self._resolve_mission()
                continue

            if self.phase == "assassinate":
                if self.assassin_id is None:
                    assassins = [
                        p.player_id
                        for p in self.state.players
                        if p.role == Role.ASSASSIN
                    ]
                    evil_ids = [p.player_id for p in self.state.players if p.is_evil()]
                    self.assassin_id = assassins[0] if assassins else random.choice(evil_ids)
                    self.log.append(f"{self._label(self.assassin_id)} is the assassin.")

                if self._is_human(self.assassin_id):
                    return

                assassin = self.state.players[self.assassin_id]
                target = assassin.bot.assassinate(
                    self._observation(assassin),
                    [p.player_id for p in self.state.players],
                )
                self._finish_assassination(target)
                continue

            return

    def _set_team(self, team: List[int], actor: str) -> None:
        state = self._require_state()
        state.set_current_team(team)
        self.pending_votes = {}
        self.pending_mission_choices = {}
        leader_id = state.get_current_leader().player_id
        self.log.append(
            f"{actor} leader {self._label(leader_id)} proposed team {self._team_label(team)}."
        )

    def _resolve_vote(self) -> None:
        state = self._require_state()
        votes = [self.pending_votes[index] for index in range(PLAYER_COUNT)]
        approved = sum(votes) > PLAYER_COUNT // 2
        self.pending_record = MissionRecord(
            mission_number=state.mission_number,
            attempt_number=state.attempt_number,
            leader_id=state.get_current_leader().player_id,
            proposed_team=list(state.current_team),
            votes=votes,
            team_approved=approved,
        )

        if approved:
            self.log.append("Team approved.")
            self.phase = "mission"
            self.pending_mission_choices = {}
            return

        self.log.append("Team rejected.")
        state.record_mission(self.pending_record)
        self.pending_record = None
        state.team_rejected()
        if self._evil_won_if_needed():
            return
        state.rotate_leader()
        self.phase = "propose"

    def _resolve_mission(self) -> None:
        state = self._require_state()
        if self.pending_record is None:
            raise RuntimeError("Missing pending mission record.")

        mission_votes = [
            self.pending_mission_choices[player_id] for player_id in state.current_team
        ]
        fail_count = mission_votes.count(False)
        required_fails = state.get_required_fails_for_current_mission()
        mission_success = fail_count < required_fails

        self.pending_record.mission_result = mission_success
        self.pending_record.fail_count = fail_count
        state.record_mission(self.pending_record)
        self.pending_record = None

        self.log.append(
            f"Mission {'succeeded' if mission_success else 'failed'} "
            f"with {fail_count} fail card(s)."
        )
        if required_fails > 1:
            self.log.append(f"This mission required {required_fails} fail cards to fail.")

        if mission_success:
            state.mission_succeeded()
            if state.successful_missions >= 3:
                self.log.append("Good completed 3 missions. Assassination begins.")
                self.phase = "assassinate"
                return
        else:
            state.mission_failed()
            if self._evil_won_if_needed():
                return

        state.rotate_leader()
        self.phase = "propose"

    def _finish_assassination(self, target_id: int) -> None:
        state = self._require_state()
        target = state.players[target_id]
        self.log.append(f"Assassin targeted {self._label(target_id)} ({target.role.value}).")
        if target.role == Role.MERLIN:
            state.set_winner("Evil")
            self.log.append("Merlin was killed. Evil wins.")
        else:
            state.set_winner("Good")
            self.log.append("Merlin survived. Good wins.")
        self.phase = "game_over"

    def _evil_won_if_needed(self) -> bool:
        state = self._require_state()
        if state.failed_missions >= 3:
            state.set_winner("Evil")
            self.phase = "game_over"
            self.log.append("Evil reached 3 failed missions. Evil wins.")
            return True
        if state.consecutive_rejections >= 5:
            state.set_winner("Evil")
            self.phase = "game_over"
            self.log.append("Five teams were rejected in a row. Evil wins.")
            return True
        return False

    def _pending_action(self) -> Optional[Dict[str, Any]]:
        if self.state is None or self.state.game_over:
            return None

        state = self.state
        if self.phase == "propose":
            leader_id = state.get_current_leader().player_id
            if self._is_human(leader_id):
                return {
                    "type": "propose",
                    "player_id": leader_id,
                    "team_size": state.get_team_size_for_current_mission(),
                }
        if self.phase == "vote":
            waiting = [
                player.player_id
                for player in state.players
                if self._is_human(player.player_id)
                and player.player_id not in self.pending_votes
            ]
            if waiting:
                return {"type": "vote", "player_ids": waiting, "team": state.current_team}
        if self.phase == "mission":
            waiting = [
                player_id
                for player_id in state.current_team
                if self._is_human(player_id)
                and player_id not in self.pending_mission_choices
            ]
            if waiting:
                return {"type": "mission", "player_ids": waiting, "team": state.current_team}
        if self.phase == "assassinate" and self.assassin_id is not None:
            if self._is_human(self.assassin_id):
                return {
                    "type": "assassinate",
                    "player_id": self.assassin_id,
                    "targets": self._assassination_targets(),
                }
        return None

    def _observation(self, player: Player) -> Dict[str, Any]:
        state = self._require_state()
        all_player_ids = [p.player_id for p in state.players]
        evil_ids = [p.player_id for p in state.players if p.is_evil()]
        visible_evil_ids = [
            p.player_id for p in state.players if p.is_evil() and p.role != Role.MORDRED
        ]
        merlin_morgana_candidates = [
            p.player_id for p in state.players if p.role in {Role.MERLIN, Role.MORGANA}
        ]

        known_evil_ids = []
        if player.is_evil():
            known_evil_ids = evil_ids
        elif player.role == Role.MERLIN:
            known_evil_ids = visible_evil_ids

        merlin_candidates = []
        if player.role == Role.PERCIVAL:
            merlin_candidates = merlin_morgana_candidates

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
            "history": [self._record_to_dict(record) for record in state.history],
        }

    def _record_to_dict(self, record: MissionRecord) -> Dict[str, Any]:
        return {
            "mission_number": record.mission_number,
            "attempt_number": record.attempt_number,
            "leader_id": record.leader_id,
            "proposed_team": record.proposed_team,
            "votes": record.votes,
            "team_approved": record.team_approved,
            "mission_result": record.mission_result,
            "fail_count": record.fail_count,
        }

    def _assassination_targets(self) -> List[int]:
        state = self._require_state()
        if self.assassin_id is None:
            return []
        known_evil = {
            player.player_id for player in state.players if player.is_evil()
        }
        return [player.player_id for player in state.players if player.player_id not in known_evil]

    def _make_bot(self, player_id: int, kind: str):
        if kind == "human":
            return HumanBot(player_id)
        if kind not in BOT_TYPES:
            raise ValueError(f"Unknown player type: {kind}")
        return BOT_TYPES[kind](player_id)

    def _validate_team(self, team: List[int], team_size: int) -> None:
        if len(team) != team_size:
            raise ValueError(f"Team must have {team_size} players.")
        if len(set(team)) != len(team):
            raise ValueError("Team cannot contain duplicate players.")
        if any(player_id < 0 or player_id >= PLAYER_COUNT for player_id in team):
            raise ValueError("Team contains an invalid player.")

    def _is_human(self, player_id: int) -> bool:
        return self.player_configs[player_id].kind == "human"

    def _label(self, player_id: int) -> str:
        return f"P{player_id} {self.player_configs[player_id].name}"

    def _team_label(self, team: List[int]) -> str:
        return "[" + ", ".join(self._label(player_id) for player_id in team) + "]"

    def _private_player_info(
        self,
        player_id: int,
        include_role: bool,
        hidden_role_hint: str,
    ) -> Dict[str, Any]:
        state = self._require_state()
        player = state.players[player_id]
        return {
            "id": player.player_id,
            "name": self.player_configs[player.player_id].name,
            "role": player.role.value if include_role else None,
            "hint": hidden_role_hint,
        }

    def _require_state(self) -> GameState:
        if self.state is None:
            raise ValueError("No game is running.")
        return self.state


GAME = UIGame()


class UIHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(INDEX_HTML)
            return
        if path == "/api/state":
            self._send_json(GAME.snapshot())
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/start":
                GAME.start(payload["players"])
            elif path == "/api/team":
                GAME.submit_team(int(payload["player_id"]), [int(x) for x in payload["team"]])
            elif path == "/api/vote":
                GAME.submit_vote(int(payload["player_id"]), bool(payload["vote"]))
            elif path == "/api/mission":
                GAME.submit_mission(int(payload["player_id"]), bool(payload["choice"]))
            elif path == "/api/assassinate":
                GAME.submit_assassination(int(payload["player_id"]), int(payload["target_id"]))
            else:
                self.send_error(404)
                return
            self._send_json(GAME.snapshot())
        except Exception as exc:
            self._send_json({"error": str(exc), "state": GAME.snapshot()}, status=400)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AvalonSim UI</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f1e8;
      --ink: #1d2430;
      --muted: #68707d;
      --panel: #fffdf8;
      --line: #d8cfbf;
      --good: #287a5f;
      --evil: #9b2f3f;
      --accent: #315c96;
      --accent-2: #8b5e24;
      --shadow: 0 14px 36px rgba(29, 36, 48, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      min-height: 132px;
      padding: 28px clamp(18px, 4vw, 48px);
      background:
        linear-gradient(rgba(18, 25, 37, 0.64), rgba(18, 25, 37, 0.58)),
        url("https://images.unsplash.com/photo-1518709268805-4e9042af2176?auto=format&fit=crop&w=1600&q=80");
      background-size: cover;
      background-position: center;
      color: white;
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 24px;
    }
    h1 { margin: 0; font-size: clamp(32px, 5vw, 58px); letter-spacing: 0; }
    .subtitle { margin: 8px 0 0; max-width: 680px; color: rgba(255,255,255,.86); }
    main { padding: 24px clamp(14px, 3vw, 36px) 36px; }
    .layout {
      display: grid;
      grid-template-columns: minmax(260px, 360px) 1fr;
      gap: 18px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px;
    }
    .panel h2 {
      margin: 0 0 12px;
      font-size: 18px;
      letter-spacing: 0;
    }
    .players { display: grid; gap: 10px; }
    .player-row {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fffaf0;
      display: grid;
      gap: 8px;
    }
    .row-top, .score-grid, .actions, .history-row {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    label { color: var(--muted); font-size: 13px; }
    input, select, button {
      font: inherit;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
    }
    input, select { min-height: 36px; padding: 7px 9px; width: 100%; }
    button {
      min-height: 38px;
      padding: 8px 12px;
      cursor: pointer;
      background: #fff;
    }
    button.primary {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
      font-weight: 700;
    }
    button.danger {
      background: var(--evil);
      color: white;
      border-color: var(--evil);
    }
    button.success {
      background: var(--good);
      color: white;
      border-color: var(--good);
    }
    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 3px 8px;
      border-radius: 999px;
      background: #ede3d1;
      color: var(--ink);
      font-size: 13px;
      font-weight: 700;
    }
    .good { color: var(--good); }
    .evil { color: var(--evil); }
    .board { display: grid; gap: 18px; }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      padding: 12px;
      min-height: 134px;
      display: grid;
      gap: 6px;
      align-content: start;
    }
    .card.good-card { border-top: 4px solid var(--good); }
    .card.evil-card { border-top: 4px solid var(--evil); }
    .small { color: var(--muted); font-size: 13px; line-height: 1.35; }
    .action-box {
      border: 1px solid var(--accent);
      border-radius: 8px;
      background: #f2f6fc;
      padding: 14px;
      display: grid;
      gap: 12px;
    }
    .check-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
      gap: 8px;
    }
    .check-grid label {
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      color: var(--ink);
      padding: 10px;
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .log {
      max-height: 300px;
      overflow: auto;
      display: grid;
      gap: 6px;
      padding-right: 4px;
    }
    .log div {
      padding: 8px 10px;
      border-left: 3px solid var(--accent-2);
      background: #fffaf0;
      border-radius: 4px;
      font-size: 14px;
    }
    .history { display: grid; gap: 8px; }
    .history-row {
      justify-content: space-between;
      border-bottom: 1px solid var(--line);
      padding: 8px 0;
    }
    .error { color: var(--evil); font-weight: 700; min-height: 20px; }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
      .cards, .check-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      header { align-items: start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>AvalonSim</h1>
      <p class="subtitle">Play a mixed human and bot game on top of the existing simulator.</p>
    </div>
    <button class="primary" onclick="startGame()">New Game</button>
  </header>
  <main>
    <div class="layout">
      <section class="panel">
        <h2>Players</h2>
        <div id="setup" class="players"></div>
        <div class="error" id="error"></div>
      </section>
      <section class="board">
        <div class="panel">
          <div class="score-grid" id="score"></div>
        </div>
        <div class="panel">
          <h2>Private Info</h2>
          <div class="cards" id="cards"></div>
        </div>
        <div class="panel">
          <h2>Current Action</h2>
          <div id="action"></div>
        </div>
        <div class="panel">
          <h2>History</h2>
          <div class="history" id="history"></div>
        </div>
        <div class="panel">
          <h2>Game Log</h2>
          <div class="log" id="log"></div>
        </div>
      </section>
    </div>
  </main>
  <script>
    const kinds = [
      ["human", "Human"],
      ["random", "RandomBot"],
      ["smart", "SmartBot"],
      ["ml", "MLBot"]
    ];
    let state = { phase: "setup", players: [], log: [] };
    const revealed = new Set();

    function initSetup() {
      const setup = document.getElementById("setup");
      setup.innerHTML = "";
      for (let i = 0; i < 7; i++) {
        const row = document.createElement("div");
        row.className = "player-row";
        row.innerHTML = `
          <div class="row-top"><span class="badge">P${i}</span><strong>Seat ${i}</strong></div>
          <label>Name<input id="name-${i}" value="Player ${i}"></label>
          <label>Control
            <select id="kind-${i}">
              ${kinds.map(([value, label]) => `<option value="${value}" ${i > 1 && value === "smart" ? "selected" : ""}>${label}</option>`).join("")}
            </select>
          </label>`;
        setup.appendChild(row);
      }
    }

    async function api(path, payload) {
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Request failed");
      }
      return data;
    }

    async function refresh() {
      const response = await fetch("/api/state");
      state = await response.json();
      render();
    }

    async function startGame() {
      clearError();
      const players = [];
      for (let i = 0; i < 7; i++) {
        players.push({
          name: document.getElementById(`name-${i}`).value,
          kind: document.getElementById(`kind-${i}`).value
        });
      }
      try {
        revealed.clear();
        state = await api("/api/start", { players });
        render();
      } catch (error) {
        showError(error.message);
      }
    }

    async function submitTeam(playerId, teamSize) {
      clearError();
      const checked = [...document.querySelectorAll("input[name='team']:checked")].map(input => Number(input.value));
      if (checked.length !== teamSize) {
        showError(`Choose exactly ${teamSize} players.`);
        return;
      }
      try {
        state = await api("/api/team", { player_id: playerId, team: checked });
        render();
      } catch (error) {
        showError(error.message);
      }
    }

    async function submitVote(playerId, vote) {
      clearError();
      try {
        state = await api("/api/vote", { player_id: playerId, vote });
        render();
      } catch (error) {
        showError(error.message);
      }
    }

    async function submitMission(playerId, choice) {
      clearError();
      try {
        state = await api("/api/mission", { player_id: playerId, choice });
        render();
      } catch (error) {
        showError(error.message);
      }
    }

    async function submitAssassination(playerId, targetId) {
      clearError();
      try {
        state = await api("/api/assassinate", { player_id: playerId, target_id: targetId });
        render();
      } catch (error) {
        showError(error.message);
      }
    }

    function render() {
      renderScore();
      renderCards();
      renderAction();
      renderHistory();
      renderLog();
    }

    function renderScore() {
      const score = document.getElementById("score");
      if (state.phase === "setup") {
        score.innerHTML = `<span class="badge">Setup</span><span class="small">Choose players and start.</span>`;
        return;
      }
      const s = state.score;
      score.innerHTML = `
        <span class="badge">Phase: ${state.phase}</span>
        <span class="badge good">Good missions: ${s.successful_missions}</span>
        <span class="badge evil">Failed missions: ${s.failed_missions}</span>
        <span class="badge">Mission: ${s.mission_number || "-"}</span>
        <span class="badge">Attempt: ${s.attempt_number || "-"}</span>
        <span class="badge">Leader: ${s.leader_id === null ? "-" : playerName(s.leader_id)}</span>
        ${s.winner ? `<span class="badge">Winner: ${s.winner}</span>` : ""}`;
    }

    function renderCards() {
      const cards = document.getElementById("cards");
      if (!state.players.length) {
        cards.innerHTML = `<div class="small">No active game.</div>`;
        return;
      }
      cards.innerHTML = state.players.map(player => {
        const isRevealed = revealed.has(player.id);
        const body = isRevealed ? `
          <div>${escapeHtml(player.role)}</div>
          <div class="${player.alignment === "Good" ? "good" : "evil"}">${player.alignment}</div>
          ${infoLine(player)}
          <button onclick="hidePlayer(${player.id})">Hide</button>` : `
          <div class="small">Role hidden.</div>
          <button class="primary" onclick="revealPlayer(${player.id})">Reveal</button>`;
        return `
        <div class="card ${isRevealed && player.alignment === "Good" ? "good-card" : ""}${isRevealed && player.alignment === "Evil" ? "evil-card" : ""}">
          <div class="row-top"><span class="badge">P${player.id}</span><strong>${escapeHtml(player.name)}</strong></div>
          <div class="small">Control: ${player.kind}</div>
          ${body}
        </div>`;
      }).join("");
    }

    function infoLine(player) {
      const known = player.known_evil_players.map(privatePlayerLabel).join(", ");
      const candidates = player.merlin_candidate_players.map(privatePlayerLabel).join(", ");
      if (known) return `<div class="small">Known evil: ${known}</div>`;
      if (candidates) return `<div class="small">Merlin/Morgana: ${candidates}</div>`;
      return `<div class="small">No private info.</div>`;
    }

    function renderAction() {
      const action = document.getElementById("action");
      if (state.phase === "setup") {
        action.innerHTML = `<div class="small">Start a game to begin.</div>`;
        return;
      }
      if (state.score.winner) {
        action.innerHTML = `<div class="action-box"><strong>${state.score.winner} wins.</strong></div>`;
        return;
      }
      const pending = state.pending;
      if (!pending) {
        action.innerHTML = `<div class="small">Bots are resolving the current step.</div>`;
        setTimeout(refresh, 250);
        return;
      }
      if (pending.type === "propose") {
        action.innerHTML = `
          <div class="action-box">
            <strong>${playerName(pending.player_id)} chooses a team of ${pending.team_size}.</strong>
            <div class="check-grid">${state.players.map(player => `
              <label><input type="checkbox" name="team" value="${player.id}"> ${playerName(player.id)}</label>`).join("")}</div>
            <button class="primary" onclick="submitTeam(${pending.player_id}, ${pending.team_size})">Propose Team</button>
          </div>`;
      } else if (pending.type === "vote") {
        action.innerHTML = `
          <div class="action-box">
            <strong>Vote on team: ${pending.team.map(playerName).join(", ")}</strong>
            ${pending.player_ids.map(id => `
              <div class="actions">
                <span class="badge">${playerName(id)}</span>
                <button class="success" onclick="submitVote(${id}, true)">Approve</button>
                <button class="danger" onclick="submitVote(${id}, false)">Reject</button>
              </div>`).join("")}
          </div>`;
      } else if (pending.type === "mission") {
        action.innerHTML = `
          <div class="action-box">
            <strong>Mission team: ${pending.team.map(playerName).join(", ")}</strong>
            ${pending.player_ids.map(id => `
              <div class="actions">
                <span class="badge">${playerName(id)}</span>
                <button class="success" onclick="submitMission(${id}, true)">Success</button>
                <button class="danger" onclick="submitMission(${id}, false)">Fail</button>
              </div>`).join("")}
          </div>`;
      } else if (pending.type === "assassinate") {
        action.innerHTML = `
          <div class="action-box">
            <strong>${playerName(pending.player_id)} chooses who to assassinate.</strong>
            <div class="actions">${pending.targets.map(id => `
              <button class="danger" onclick="submitAssassination(${pending.player_id}, ${id})">${playerName(id)}</button>`).join("")}</div>
          </div>`;
      }
    }

    function renderHistory() {
      const history = document.getElementById("history");
      if (!state.history || !state.history.length) {
        history.innerHTML = `<div class="small">No completed rounds yet.</div>`;
        return;
      }
      history.innerHTML = state.history.slice().reverse().map(record => `
        <div class="history-row">
          <span>M${record.mission_number}.${record.attempt_number} by ${playerName(record.leader_id)}</span>
          <span class="small">${record.proposed_team.map(playerName).join(", ")}</span>
          <span class="badge">${record.team_approved ? "Approved" : "Rejected"}</span>
          <span class="badge">${record.mission_result === null ? "-" : record.mission_result ? "Success" : `Fail x${record.fail_count}`}</span>
        </div>`).join("");
    }

    function renderLog() {
      const log = document.getElementById("log");
      log.innerHTML = (state.log || []).slice().reverse().map(line => `<div>${escapeHtml(line)}</div>`).join("");
    }

    function revealPlayer(id) {
      revealed.add(id);
      renderCards();
    }

    function hidePlayer(id) {
      revealed.delete(id);
      renderCards();
    }

    function playerName(id) {
      const player = state.players.find(p => p.id === id);
      return player ? `P${id} ${player.name}` : `P${id}`;
    }

    function privatePlayerLabel(player) {
      return `P${player.id} ${player.name} (${player.role || player.hint})`;
    }

    function showError(message) {
      document.getElementById("error").textContent = message;
    }

    function clearError() {
      document.getElementById("error").textContent = "";
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
      }[char]));
    }

    initSetup();
    refresh();
  </script>
</body>
</html>
"""


def main() -> None:
    host = "127.0.0.1"
    port = _find_open_port(host, 8000)
    server = ThreadingHTTPServer((host, port), UIHandler)
    print(f"AvalonSim UI running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


def _find_open_port(host: str, start_port: int) -> int:
    for port in range(start_port, start_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(f"No free port found from {start_port} to {start_port + 19}.")


if __name__ == "__main__":
    main()
