"""Ultimate Tic-Tac-Toe game engine.

Pure-Python, dependency-free, and fast (bitboards). Shared by the web app,
the MCTS search, and the AlphaZero trainer so there is a single source of
truth for the rules.

Board layout
------------
There are 9 small boards arranged in a 3x3 grid (the "big board"). Each small
board has 9 cells. A *move* is an integer 0..80 == small_board * 9 + cell,
where both ``small_board`` and ``cell`` use this index within their 3x3 grid::

    0 1 2
    3 4 5
    6 7 8

Rules implemented (the common / canonical ruleset)
--------------------------------------------------
* Players alternate. X (player ``+1``) moves first.
* The cell you play in dictates which small board your opponent must play in
  next (e.g. playing in cell 4 sends them to small board 4).
* If you are sent to a small board that is already won or full, you may play
  in *any* open small board (a "free move").
* Winning a small board (3 in a row) claims it on the big board and closes it.
* Winning 3 small boards in a row on the big board wins the game.
* If every small board is decided with no 3-in-a-row, the game is a draw.
"""

from __future__ import annotations

from typing import List, Optional

FULL = 0x1FF  # 9 bits set -> a full small board

# The 8 winning line bitmasks for a 3x3 board.
LINES = (
    0b000000111, 0b000111000, 0b111000000,  # rows
    0b001001001, 0b010010010, 0b100100100,  # columns
    0b100010001, 0b001010100,               # diagonals
)


def _build_won_table() -> List[bool]:
    table = [False] * 512
    for bits in range(512):
        table[bits] = any((bits & line) == line for line in LINES)
    return table


# WON[bits] is True iff that 9-bit configuration contains a 3-in-a-row.
WON = _build_won_table()


class Game:
    """A single Ultimate Tic-Tac-Toe position.

    State is mutated in place by :meth:`play`; use :meth:`clone` to branch
    (the search relies on cheap clones).
    """

    __slots__ = (
        "x", "o", "big_x", "big_o", "big_drawn",
        "player", "forced", "winner", "last_move", "move_count",
    )

    def __init__(self) -> None:
        self.x = [0] * 9          # per-small-board X bitboards
        self.o = [0] * 9          # per-small-board O bitboards
        self.big_x = 0            # small boards won by X (9-bit)
        self.big_o = 0            # small boards won by O (9-bit)
        self.big_drawn = 0        # small boards full with no winner (9-bit)
        self.player = 1           # +1 = X to move, -1 = O to move
        self.forced = -1          # forced small board, or -1 for a free move
        self.winner: Optional[int] = None  # None=ongoing, +1/-1=winner, 0=draw
        self.last_move: Optional[int] = None
        self.move_count = 0

    # -- core API ---------------------------------------------------------

    @property
    def current_player(self) -> int:
        return self.player

    def is_terminal(self) -> bool:
        return self.winner is not None

    def legal_moves(self) -> List[int]:
        if self.winner is not None:
            return []
        closed = self.big_x | self.big_o | self.big_drawn
        if self.forced != -1 and not (closed >> self.forced) & 1:
            boards = (self.forced,)
        else:
            # Free move: either no constraint, or sent to a closed board.
            boards = tuple(b for b in range(9) if not (closed >> b) & 1)
        moves: List[int] = []
        for b in boards:
            free = (~(self.x[b] | self.o[b])) & FULL
            base = b * 9
            for pos in range(9):
                if (free >> pos) & 1:
                    moves.append(base + pos)
        return moves

    def play(self, move: int) -> "Game":
        """Apply ``move`` (0..80) for the current player and switch turns."""
        b, pos = divmod(move, 9)
        bit = 1 << pos
        player = self.player

        if player == 1:
            self.x[b] |= bit
            if WON[self.x[b]]:
                self.big_x |= 1 << b
            elif (self.x[b] | self.o[b]) == FULL:
                self.big_drawn |= 1 << b
        else:
            self.o[b] |= bit
            if WON[self.o[b]]:
                self.big_o |= 1 << b
            elif (self.x[b] | self.o[b]) == FULL:
                self.big_drawn |= 1 << b

        # Where must the opponent play next?
        closed = self.big_x | self.big_o | self.big_drawn
        self.forced = pos if not (closed >> pos) & 1 else -1

        # Did this move end the game?
        if player == 1 and WON[self.big_x]:
            self.winner = 1
        elif player == -1 and WON[self.big_o]:
            self.winner = -1
        elif closed == FULL:
            self.winner = 0  # every board decided, nobody has 3-in-a-row

        self.last_move = move
        self.move_count += 1
        self.player = -player
        return self

    def clone(self) -> "Game":
        g = Game.__new__(Game)
        g.x = self.x[:]
        g.o = self.o[:]
        g.big_x = self.big_x
        g.big_o = self.big_o
        g.big_drawn = self.big_drawn
        g.player = self.player
        g.forced = self.forced
        g.winner = self.winner
        g.last_move = self.last_move
        g.move_count = self.move_count
        return g

    # -- helpers ----------------------------------------------------------

    def result_for(self, player: int) -> Optional[float]:
        """Game result from ``player``'s view: +1 win, -1 loss, 0 draw, None ongoing."""
        if self.winner is None:
            return None
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def cell(self, index: int) -> int:
        """Value of cell 0..80: 0 empty, 1 X, 2 O."""
        b, pos = divmod(index, 9)
        if (self.x[b] >> pos) & 1:
            return 1
        if (self.o[b] >> pos) & 1:
            return 2
        return 0

    def board_status(self, b: int) -> int:
        """Status of small board ``b``: 0 open, 1 X, 2 O, 3 draw."""
        if (self.big_x >> b) & 1:
            return 1
        if (self.big_o >> b) & 1:
            return 2
        if (self.big_drawn >> b) & 1:
            return 3
        return 0

    def to_dict(self) -> dict:
        """JSON-friendly snapshot for the web client (X=1, O=2)."""
        winner = None
        if self.winner == 1:
            winner = "X"
        elif self.winner == -1:
            winner = "O"
        elif self.winner == 0:
            winner = "draw"
        return {
            "cells": [self.cell(i) for i in range(81)],
            "boardStatus": [self.board_status(b) for b in range(9)],
            "currentPlayer": 1 if self.player == 1 else 2,
            "forcedBoard": self.forced if self.forced != -1 else None,
            "legalMoves": self.legal_moves(),
            "lastMove": self.last_move,
            "winner": winner,
            "moveCount": self.move_count,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        rows = []
        for big_r in range(3):
            for sub_r in range(3):
                cells = []
                for big_c in range(3):
                    for sub_c in range(3):
                        b = big_r * 3 + big_c
                        pos = sub_r * 3 + sub_c
                        v = self.cell(b * 9 + pos)
                        cells.append(".XO"[v])
                    cells.append(" ")
                rows.append(" ".join(cells))
            rows.append("")
        return "\n".join(rows)
