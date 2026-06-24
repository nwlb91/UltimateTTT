"""In-memory match/room management for the web server.

A :class:`Match` holds one :class:`~uttt.game.Game` plus its two seats. Seats
are identified by player value (+1 = X, -1 = O). Each seat gets a secret token
so a browser can reconnect to the same seat after a refresh.

Everything lives in memory. On the Render free tier the service may sleep and
restart, which drops in-progress matches -- acceptable for casual play. The
:class:`MatchManager` reaps idle matches so memory does not grow unbounded.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from uttt.game import Game

X, O = 1, -1
MATCH_TTL_SECONDS = 60 * 60  # forget matches idle for an hour


def _new_id(n: int = 6) -> str:
    # URL-friendly, unambiguous-ish short id.
    alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(n))


@dataclass
class Seat:
    player: int
    token: str = field(default_factory=lambda: secrets.token_urlsafe(12))
    connection: Optional[object] = None  # the WebSocket, or None when absent
    is_ai: bool = False

    @property
    def connected(self) -> bool:
        return self.connection is not None or self.is_ai


@dataclass
class Match:
    id: str
    mode: str                      # "pvp" or "ai"
    game: Game
    seats: Dict[int, Seat]
    ai_level: int = 5
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_active = time.time()

    @property
    def ai_seat(self) -> Optional[Seat]:
        for seat in self.seats.values():
            if seat.is_ai:
                return seat
        return None

    @property
    def status(self) -> str:
        if self.game.is_terminal():
            return "finished"
        if all(s.connected for s in self.seats.values()):
            return "playing"
        return "waiting"

    def seat_for_token(self, token: Optional[str]) -> Optional[Seat]:
        if not token:
            return None
        for seat in self.seats.values():
            if seat.token == token:
                return seat
        return None

    def open_human_seat(self) -> Optional[Seat]:
        """First seat with no human attached (used when claiming a seat)."""
        for seat in self.seats.values():
            if not seat.is_ai and seat.connection is None:
                return seat
        return None

    def public_state(self) -> dict:
        state = self.game.to_dict()
        state.update(
            {
                "type": "state",
                "matchId": self.id,
                "mode": self.mode,
                "status": self.status,
                "aiLevel": self.ai_level if self.mode == "ai" else None,
                "seats": {
                    "X": {
                        "connected": self.seats[X].connected,
                        "isAI": self.seats[X].is_ai,
                    },
                    "O": {
                        "connected": self.seats[O].connected,
                        "isAI": self.seats[O].is_ai,
                    },
                },
            }
        )
        return state

    def reset(self) -> None:
        """Start a fresh game in the same room (for rematches)."""
        self.game = Game()
        self.touch()


class MatchManager:
    def __init__(self) -> None:
        self._matches: Dict[str, Match] = {}

    def get(self, match_id: str) -> Optional[Match]:
        return self._matches.get(match_id)

    def create_pvp(self) -> Match:
        match_id = self._unique_id()
        match = Match(
            id=match_id,
            mode="pvp",
            game=Game(),
            seats={X: Seat(player=X), O: Seat(player=O)},
        )
        self._matches[match_id] = match
        return match

    def create_ai(self, level: int, human_plays: str) -> Match:
        """Create a human-vs-AI match.

        ``human_plays`` is "X", "O", or "random".
        """
        match_id = self._unique_id()
        if human_plays == "random":
            human_plays = secrets.choice(["X", "O"])
        human = X if human_plays.upper() == "X" else O
        ai = -human
        seats = {
            human: Seat(player=human),
            ai: Seat(player=ai, is_ai=True),
        }
        match = Match(
            id=match_id,
            mode="ai",
            game=Game(),
            seats=seats,
            ai_level=max(0, min(10, int(level))),
        )
        self._matches[match_id] = match
        return match

    def reap(self) -> int:
        """Drop idle matches. Returns the number removed."""
        now = time.time()
        stale = [
            mid
            for mid, m in self._matches.items()
            if now - m.last_active > MATCH_TTL_SECONDS
        ]
        for mid in stale:
            del self._matches[mid]
        return len(stale)

    def _unique_id(self) -> str:
        for _ in range(1000):
            mid = _new_id()
            if mid not in self._matches:
                return mid
        return _new_id(8)  # extremely unlikely fallback
