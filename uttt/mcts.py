"""Monte-Carlo Tree Search (AlphaZero-style PUCT).

The search is decoupled from *how* positions are evaluated via an
``Evaluator`` protocol: ``evaluate(game) -> (policy, value)`` where

* ``policy`` is a length-81 numpy array of priors (will be renormalised over
  legal moves), and
* ``value`` is a scalar in ``[-1, 1]`` from the perspective of the player to
  move in ``game``.

This lets one search drive both a neural-net evaluator (AlphaZero) and a
random-rollout evaluator (classic UCT, used when no model is loaded). PUCT is
the AlphaZero generalisation of the UCB1/UCT selection rule.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Protocol, Tuple

import numpy as np

from .game import Game

NUM_ACTIONS = 81


class Evaluator(Protocol):
    def evaluate(self, game: Game) -> Tuple[np.ndarray, float]:
        ...


class _Node:
    __slots__ = ("N", "W", "P", "children", "expanded")

    def __init__(self, prior: float) -> None:
        self.N = 0          # visit count
        self.W = 0.0        # total value (from the perspective of the player
                            #   to move at this node's PARENT)
        self.P = prior      # prior probability of the edge leading here
        self.children: Dict[int, "_Node"] = {}
        self.expanded = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N else 0.0


class MCTS:
    def __init__(
        self,
        evaluator: Evaluator,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.rng = rng or np.random.default_rng()

    def run(
        self, root_game: Game, num_simulations: int, add_noise: bool = False
    ) -> np.ndarray:
        """Search and return visit counts (length-81 numpy array)."""
        root = _Node(prior=1.0)
        self._expand(root, root_game)
        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(num_simulations):
            self._simulate(root, root_game.clone())

        counts = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for move, child in root.children.items():
            counts[move] = child.N
        return counts

    # -- internals --------------------------------------------------------

    def _simulate(self, node: _Node, game: Game) -> float:
        """Run one simulation from ``node`` over a private ``game`` clone.

        Returns the value from the perspective of the player to move at
        ``node`` (so the caller negates it for the parent).
        """
        if game.is_terminal():
            r = game.result_for(game.current_player)
            return r if r is not None else 0.0

        if not node.expanded:
            return self._expand(node, game)

        move, child = self._select(node)
        game.play(move)
        value = -self._simulate(child, game)
        child.N += 1
        child.W += value
        return value

    def _expand(self, node: _Node, game: Game) -> float:
        policy, value = self.evaluator.evaluate(game)
        legal = game.legal_moves()
        if legal:
            priors = np.array([policy[m] for m in legal], dtype=np.float64)
            total = priors.sum()
            if total <= 0 or not np.isfinite(total):
                priors = np.ones(len(legal)) / len(legal)
            else:
                priors = priors / total
            for move, p in zip(legal, priors):
                node.children[move] = _Node(prior=float(p))
        node.expanded = True
        return value

    def _select(self, node: _Node) -> Tuple[int, _Node]:
        sqrt_parent = math.sqrt(max(1, node.N))
        best_score = -float("inf")
        best_move = -1
        best_child: Optional[_Node] = None
        c = self.c_puct
        for move, child in node.children.items():
            u = c * child.P * sqrt_parent / (1 + child.N)
            score = child.Q + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        # Ensure the chosen node's visit accounting stays correct.
        node.N += 1
        assert best_child is not None
        return best_move, best_child

    def _add_dirichlet_noise(self, root: _Node) -> None:
        moves = list(root.children.keys())
        if not moves:
            return
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(moves))
        eps = self.dirichlet_eps
        for move, n in zip(moves, noise):
            child = root.children[move]
            child.P = (1 - eps) * child.P + eps * n


def visit_counts_to_policy(counts: np.ndarray, temperature: float) -> np.ndarray:
    """Convert visit counts to a move distribution given a temperature.

    ``temperature`` -> 0 gives (near) argmax; 1.0 is proportional to visits.
    """
    if counts.sum() == 0:
        return counts
    if temperature <= 1e-3:
        pi = np.zeros_like(counts)
        pi[int(np.argmax(counts))] = 1.0
        return pi
    logits = np.log(np.maximum(counts, 1e-12)) / temperature
    logits -= logits.max()
    probs = np.exp(logits)
    probs[counts == 0] = 0.0
    return probs / probs.sum()
