"""Tests for MCTS search and the rollout-based AI."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uttt.ai import AIPlayer, RolloutEvaluator, level_params  # noqa: E402
from uttt.game import Game  # noqa: E402


def test_level_params_monotonic():
    sims_prev = -1
    for level in range(1, 11):
        sims, temp = level_params(level)
        assert sims > sims_prev  # more search at higher levels
        sims_prev = sims
        assert 0.1 <= temp <= 1.1
    assert level_params(0)[0] == 0  # level 0 = random


def test_ai_returns_legal_moves():
    ai = AIPlayer(RolloutEvaluator())
    g = Game()
    for _ in range(6):
        move = ai.choose_move(g, level=3)
        assert move in g.legal_moves()
        g.play(move)


def test_mcts_finds_immediate_winning_move():
    # X owns boards 0 and 4 (big diagonal). Completing board 8 wins the game.
    g = Game()
    g.big_x = (1 << 0) | (1 << 4)
    g.player = 1
    g.forced = 8
    g.x[8] = 0b000000011  # X has cells 0,1 of board 8
    winning_move = 8 * 9 + 2  # complete the small board -> big line 0,4,8

    ai = AIPlayer(RolloutEvaluator())
    # Greedy, decent search budget -> should find the forced win.
    move = ai.choose_move(g, level=6)
    assert move == winning_move


def test_mcts_blocks_or_wins_not_random():
    # Sanity: with a real search budget the move should be deterministic-ish
    # toward the winning line across repeated runs.
    g = Game()
    g.big_x = (1 << 0) | (1 << 4)
    g.player = 1
    g.forced = 8
    g.x[8] = 0b000000011
    ai = AIPlayer(RolloutEvaluator())
    wins = sum(ai.choose_move(g, level=8) == 8 * 9 + 2 for _ in range(5))
    assert wins == 5
