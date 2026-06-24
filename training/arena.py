"""Pit two evaluators against each other to measure progress.

Useful for AlphaGo-Zero-style gating ("is the new net actually better?") or
just to sanity-check that training is improving strength.

Games use a little opening randomness (temperature sampling for the first few
plies) so repeated games between the *same* two networks aren't identical --
otherwise deterministic argmax play would replay one game forever and the win
rate would be meaningless.
"""

from __future__ import annotations

import numpy as np

from uttt.game import Game
from uttt.mcts import MCTS, visit_counts_to_policy


def play_game(
    eval_x,
    eval_o,
    sims: int = 100,
    c_puct: float = 1.5,
    rng=None,
    temperature: float = 0.6,
    temp_moves: int = 10,
) -> int:
    """Play one game; ``eval_x`` is X, ``eval_o`` is O. Returns +1/-1/0.

    The first ``temp_moves`` plies are sampled with ``temperature`` to diversify
    openings; the rest are greedy (best move).
    """
    rng = rng or np.random.default_rng()
    g = Game()
    while not g.is_terminal():
        ev = eval_x if g.current_player == 1 else eval_o
        mcts = MCTS(ev, c_puct=c_puct, rng=rng)
        counts = mcts.run(g, sims, add_noise=False)
        if counts.sum() == 0:
            break
        if g.move_count < temp_moves and temperature > 0:
            pi = visit_counts_to_policy(counts, temperature)
            move = int(rng.choice(len(pi), p=pi))
        else:
            move = int(np.argmax(counts))
        g.play(move)
    return g.winner if g.winner is not None else 0


def arena(
    eval_a,
    eval_b,
    games: int = 20,
    sims: int = 100,
    rng=None,
    temperature: float = 0.6,
    temp_moves: int = 10,
) -> dict:
    """Play ``games`` games, alternating colors. Returns A's results.

    ``win_rate`` counts draws as half a win. ~0.5 means the two are evenly
    matched; markedly above 0.5 means A is stronger.
    """
    rng = rng or np.random.default_rng()
    wins = losses = draws = 0
    for i in range(games):
        if i % 2 == 0:
            result = play_game(eval_a, eval_b, sims=sims, rng=rng,
                               temperature=temperature, temp_moves=temp_moves)
            a_score = result            # A is X
        else:
            result = play_game(eval_b, eval_a, sims=sims, rng=rng,
                               temperature=temperature, temp_moves=temp_moves)
            a_score = -result           # A is O
        if a_score > 0:
            wins += 1
        elif a_score < 0:
            losses += 1
        else:
            draws += 1
    win_rate = (wins + 0.5 * draws) / games
    return {"wins": wins, "losses": losses, "draws": draws, "win_rate": win_rate}
