"""Neural-network encoding of a position + board symmetries.

The network sees the position as a stack of 9x9 planes, always from the
*current player's* perspective (canonical form). This means the value head
predicts "is it good for the side to move", which halves what the net must
learn. Cell/move indices are absolute (independent of perspective), so policy
targets never need flipping when we canonicalise.

Plane layout (channel, 9x9):
    0  my stones
    1  opponent stones
    2  small boards I have won            (broadcast over the 3x3 block)
    3  small boards opponent has won      (broadcast)
    4  small boards drawn                 (broadcast)
    5  legal-move mask
    6  constant ones (helps conv edge awareness)
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np

from .game import Game

NUM_PLANES = 7
BOARD = 9
NUM_ACTIONS = 81


def _cell_to_yx(index: int) -> Tuple[int, int]:
    b, pos = divmod(index, 9)
    y = 3 * (b // 3) + pos // 3
    x = 3 * (b % 3) + pos % 3
    return y, x


def _yx_to_cell(y: int, x: int) -> int:
    b = (y // 3) * 3 + (x // 3)
    pos = (y % 3) * 3 + (x % 3)
    return b * 9 + pos


# Precompute cell <-> (y, x) maps.
_CELL_YX = np.array([_cell_to_yx(i) for i in range(81)], dtype=np.int64)
# For each small board, the (y, x) of its top-left corner, to broadcast big-board planes.
_BLOCK_YX = [(3 * (b // 3), 3 * (b % 3)) for b in range(9)]


def encode(game: Game) -> np.ndarray:
    """Return a ``(7, 9, 9)`` float32 tensor for ``game`` in canonical form."""
    planes = np.zeros((NUM_PLANES, BOARD, BOARD), dtype=np.float32)
    me, opp = (game.x, game.o) if game.player == 1 else (game.o, game.x)
    big_me, big_opp = (
        (game.big_x, game.big_o) if game.player == 1 else (game.big_o, game.big_x)
    )

    for b in range(9):
        ys, xs = 3 * (b // 3), 3 * (b % 3)
        mine, theirs = me[b], opp[b]
        for pos in range(9):
            y = ys + pos // 3
            x = xs + pos % 3
            if (mine >> pos) & 1:
                planes[0, y, x] = 1.0
            elif (theirs >> pos) & 1:
                planes[1, y, x] = 1.0

    for b in range(9):
        ys, xs = _BLOCK_YX[b]
        if (big_me >> b) & 1:
            planes[2, ys:ys + 3, xs:xs + 3] = 1.0
        if (big_opp >> b) & 1:
            planes[3, ys:ys + 3, xs:xs + 3] = 1.0
        if (game.big_drawn >> b) & 1:
            planes[4, ys:ys + 3, xs:xs + 3] = 1.0

    for move in game.legal_moves():
        y, x = _CELL_YX[move]
        planes[5, y, x] = 1.0

    planes[6, :, :] = 1.0
    return planes


# -- symmetries (dihedral group D4: 8 transforms) -------------------------

def _build_symmetries() -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (raster_perm, cell_perm) for the 8 board symmetries.

    ``raster_perm`` permutes a flattened 9x9 (row-major y*9+x) plane.
    ``cell_perm`` permutes a length-81 policy vector in move-index space.
    Both come from the same geometric (y, x) -> (y', x') map, so a plane and
    its policy stay aligned.
    """
    n = BOARD
    transforms = [
        lambda y, x: (y, x),                  # identity
        lambda y, x: (x, n - 1 - y),          # rot 90
        lambda y, x: (n - 1 - y, n - 1 - x),  # rot 180
        lambda y, x: (n - 1 - x, y),          # rot 270
        lambda y, x: (y, n - 1 - x),          # flip horizontal
        lambda y, x: (n - 1 - y, x),          # flip vertical
        lambda y, x: (x, y),                  # transpose
        lambda y, x: (n - 1 - x, n - 1 - y),  # anti-transpose
    ]
    syms = []
    for t in transforms:
        raster = np.empty(n * n, dtype=np.int64)
        cell = np.empty(81, dtype=np.int64)
        for y in range(n):
            for x in range(n):
                y2, x2 = t(y, x)
                raster[y * n + x] = y2 * n + x2
                cell[_yx_to_cell(y, x)] = _yx_to_cell(y2, x2)
        syms.append((raster, cell))
    return syms


SYMMETRIES = _build_symmetries()


def symmetries(planes: np.ndarray, pi: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield all 8 (planes, pi) symmetric variants for data augmentation."""
    c = planes.shape[0]
    flat = planes.reshape(c, -1)
    for raster, cell in SYMMETRIES:
        new_planes = flat[:, raster].reshape(c, BOARD, BOARD).copy()
        new_pi = np.zeros_like(pi)
        new_pi[cell] = pi
        yield new_planes, new_pi
