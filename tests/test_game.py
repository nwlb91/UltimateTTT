"""Rules tests for the Ultimate Tic-Tac-Toe engine."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uttt.encoding import NUM_PLANES, encode, symmetries  # noqa: E402
from uttt.game import WON, Game  # noqa: E402


def test_initial_state():
    g = Game()
    assert g.current_player == 1
    assert g.forced == -1
    assert len(g.legal_moves()) == 81
    assert not g.is_terminal()


def test_won_table():
    assert WON[0b111000000]
    assert WON[0b001010100]  # anti-diagonal
    assert WON[0b100010001]  # main diagonal
    assert not WON[0b110000000]
    assert not WON[0]


def test_forced_board_follows_cell():
    g = Game()
    g.play(0 * 9 + 4)  # X plays centre cell of board 0 -> sends O to board 4
    assert g.forced == 4
    assert all(m // 9 == 4 for m in g.legal_moves())


def test_free_move_when_sent_to_won_board():
    # Being forced into a board that is already decided yields a free move.
    g = Game()
    g.big_x = 1     # board 0 won by X (closed)
    g.forced = 0    # current player is "sent" to the closed board 0
    moves = g.legal_moves()
    assert len(moves) == 9 * 8  # every board except the closed board 0 is open
    assert all(m // 9 != 0 for m in moves)


def test_small_board_win_claims_big_cell():
    g = Game()
    g.play(0)         # X b0c0
    g.play(0 * 9 + 3)  # O
    g.play(3 * 9 + 1)  # X -> O to b1
    g.play(1 * 9 + 3)  # O
    g.play(3 * 9 + 2)  # X -> O to b2
    g.play(2 * 9 + 1)  # O
    g.play(1 * 9 + 1)  # this is X? track parity below
    # Simpler: directly drive X to win board 0 with cells 0,1,2.
    g = Game()
    g.x[0] = 0b011  # cells 0,1 by X
    g.o[3] = 0
    g.player = 1
    g.forced = 0
    g.play(0 * 9 + 2)  # X completes 0,1,2
    assert g.board_status(0) == 1
    assert (g.big_x >> 0) & 1 == 1


def test_full_game_x_wins_big_diagonal():
    # X wins boards 0, 4, 8 (a big diagonal) -> wins the game.
    g = Game()
    g.big_x = (1 << 0) | (1 << 4)
    g.player = 1
    g.forced = 8
    g.x[8] = 0b011
    g.play(8 * 9 + 2)  # X completes board 8 -> big diagonal 0,4,8
    assert g.winner == 1
    assert g.is_terminal()
    assert g.result_for(1) == 1.0
    assert g.result_for(-1) == -1.0


def test_draw_detection():
    # Big board is a cat's game (no 3-in-a-row) once all 9 boards are decided.
    #   X O X
    #   X O O   -> X: {0,2,3,7,8}  O: {1,4,5,6}  (no line for either)
    #   O X X
    g = Game()
    g.big_x = (1 << 0) | (1 << 2) | (1 << 3) | (1 << 7)  # board 8 left to X
    g.big_o = (1 << 1) | (1 << 4) | (1 << 5) | (1 << 6)
    g.player = 1
    g.forced = 8
    g.x[8] = 0b000000011  # X has cells 0,1 of board 8
    g.play(8 * 9 + 2)     # X completes board 8 -> all boards decided, no big line
    assert g.is_terminal()
    assert g.winner == 0  # draw
    assert g.result_for(1) == 0.0
    assert g.result_for(-1) == 0.0


def test_clone_is_independent():
    g = Game()
    g.play(40)
    h = g.clone()
    h.play(h.legal_moves()[0])
    assert g.move_count == 1
    assert h.move_count == 2


def test_encoding_shape_and_perspective():
    g = Game()
    g.play(0)  # X at cell 0; now O to move
    planes = encode(g)
    assert planes.shape == (NUM_PLANES, 9, 9)
    # Plane 0 = current player's (O's) stones -> none yet. Plane 1 = opp (X) has 1.
    assert planes[0].sum() == 0
    assert planes[1].sum() == 1
    # Legal-move plane matches legal moves count.
    assert planes[5].sum() == len(g.legal_moves())


def test_symmetries_preserve_mass_and_count():
    g = Game()
    for m in [40, 36, 0, 4, 37]:
        if m in g.legal_moves():
            g.play(m)
    planes = encode(g)
    pi = np.zeros(81)
    for m in g.legal_moves():
        pi[m] = 1.0
    pi /= pi.sum()
    variants = list(symmetries(planes, pi))
    assert len(variants) == 8
    for p2, pi2 in variants:
        assert p2.shape == planes.shape
        assert abs(pi2.sum() - 1.0) < 1e-9
        # stone count preserved
        assert p2[0].sum() == planes[0].sum()
        assert p2[1].sum() == planes[1].sum()


def test_identity_symmetry_is_first():
    g = Game()
    g.play(40)
    planes = encode(g)
    pi = np.zeros(81)
    pi[g.legal_moves()[0]] = 1.0
    first_planes, first_pi = next(iter(symmetries(planes, pi)))
    assert np.array_equal(first_planes, planes)
    assert np.array_equal(first_pi, pi)
