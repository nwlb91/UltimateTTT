"""AI move selection for the web app.

Two evaluators plug into the same MCTS:

* :class:`NeuralEvaluator` -- AlphaZero net via ONNX Runtime (used when a
  trained model is present).
* :class:`RolloutEvaluator` -- uniform priors + random playouts (classic UCT),
  so the app has a real opponent even before any training has happened.

Difficulty (0-10) modulates the MCTS search budget and the move-sampling
temperature: low levels search shallowly and play loosely; high levels search
deeply and play (near) greedily.
"""

from __future__ import annotations

import os
import random
from typing import Optional, Tuple

import numpy as np

from .encoding import encode
from .game import Game
from .mcts import MCTS, visit_counts_to_policy

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "uttt.onnx"
)


def default_model_path() -> str:
    """Resolve the model path at call time (so UTTT_MODEL_PATH is honoured)."""
    return os.environ.get("UTTT_MODEL_PATH", _DEFAULT_MODEL_PATH)

# Per-level MCTS simulation budget. Level 0 is "random" (handled separately).
_SIMS = {1: 24, 2: 48, 3: 96, 4: 160, 5: 256, 6: 400, 7: 600, 8: 900, 9: 1300, 10: 1800}


def level_params(level: int) -> Tuple[int, float]:
    """Map difficulty 0-10 to (num_simulations, temperature)."""
    level = max(0, min(10, int(level)))
    if level == 0:
        return 0, 1.0
    sims = _SIMS[level]
    temperature = max(0.1, 1.1 - 0.1 * level)
    return sims, temperature


class RolloutEvaluator:
    """Uniform policy + random-rollout value. No model required."""

    uses_model = False

    def __init__(self) -> None:
        self._uniform = np.ones(81, dtype=np.float64)

    def evaluate(self, game: Game) -> Tuple[np.ndarray, float]:
        return self._uniform, self._rollout(game.clone(), game.current_player)

    @staticmethod
    def _rollout(game: Game, perspective: int) -> float:
        while not game.is_terminal():
            moves = game.legal_moves()
            if not moves:
                break
            game.play(random.choice(moves))
        result = game.result_for(perspective)
        return result if result is not None else 0.0


class NeuralEvaluator:
    """AlphaZero policy/value network served through ONNX Runtime."""

    uses_model = True

    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort  # imported lazily; optional at runtime

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("UTTT_ORT_THREADS", "1"))
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.model_path = model_path

    def evaluate(self, game: Game) -> Tuple[np.ndarray, float]:
        planes = encode(game)[None].astype(np.float32)
        policy_logits, value = self.session.run(None, {self.input_name: planes})
        logits = policy_logits[0].astype(np.float64)
        logits -= logits.max()
        policy = np.exp(logits)
        return policy, float(value.reshape(-1)[0])


class AIPlayer:
    def __init__(self, evaluator, **mcts_kwargs) -> None:
        self.evaluator = evaluator
        self.mcts_kwargs = mcts_kwargs

    @property
    def uses_model(self) -> bool:
        return getattr(self.evaluator, "uses_model", False)

    def choose_move(self, game: Game, level: int = 5) -> int:
        legal = game.legal_moves()
        if not legal:
            raise ValueError("no legal moves")
        if len(legal) == 1:
            return legal[0]

        sims, temperature = level_params(level)
        rng = np.random.default_rng()
        if sims <= 0:  # level 0: pick at random
            return int(rng.choice(legal))

        mcts = MCTS(self.evaluator, rng=rng, **self.mcts_kwargs)
        counts = mcts.run(game, sims, add_noise=False)
        if counts.sum() == 0:
            return int(rng.choice(legal))

        pi = visit_counts_to_policy(counts, temperature)
        if temperature <= 1e-3:
            return int(np.argmax(pi))
        return int(rng.choice(len(pi), p=pi))


def load_ai(model_path: Optional[str] = None) -> AIPlayer:
    """Build an AIPlayer, preferring a trained ONNX model if one is available."""
    path = model_path or default_model_path()
    if os.path.exists(path):
        try:
            return AIPlayer(NeuralEvaluator(path))
        except Exception as exc:  # pragma: no cover - fall back gracefully
            print(f"[uttt.ai] Failed to load model at {path}: {exc!r}; using rollouts.")
    return AIPlayer(RolloutEvaluator())
