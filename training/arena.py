"""Pit two evaluators against each other to measure progress.

Useful for AlphaGo-Zero-style gating ("is the new net actually better?") or
just to sanity-check that training is improving strength.
"""

from __future__ import annotations

import numpy as np

from uttt.game import Game
from uttt.mcts import MCTS


def play_game(eval_x, eval_o, sims: int = 100, c_puct: float = 1.5, rng=None) -> int:
    """Play one game; ``eval_x`` is X, ``eval_o`` is O. Returns +1/-1/0."""
    g = Game()
    while not g.is_terminal():
        ev = eval_x if g.current_player == 1 else eval_o
        mcts = MCTS(ev, c_puct=c_puct, rng=rng)
        counts = mcts.run(g, sims, add_noise=False)
        if counts.sum() == 0:
            break
        g.play(int(np.argmax(counts)))
    return g.winner if g.winner is not None else 0


def arena(eval_a, eval_b, games: int = 20, sims: int = 100, rng=None) -> dict:
    """Play ``games`` games, alternating colors. Returns A's results."""
    rng = rng or np.random.default_rng()
    wins = losses = draws = 0
    for i in range(games):
        if i % 2 == 0:
            result = play_game(eval_a, eval_b, sims=sims, rng=rng)  # A is X
            a_score = result
        else:
            result = play_game(eval_b, eval_a, sims=sims, rng=rng)  # A is O
            a_score = -result
        if a_score > 0:
            wins += 1
        elif a_score < 0:
            losses += 1
        else:
            draws += 1
    win_rate = (wins + 0.5 * draws) / games
    return {"wins": wins, "losses": losses, "draws": draws, "win_rate": win_rate}
