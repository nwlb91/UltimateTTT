"""Bridge a PyTorch ``UTTTNet`` to the MCTS ``Evaluator`` protocol.

Used during self-play and arena evaluation. (The deployed app uses the ONNX
evaluator in ``uttt.ai`` instead, so it doesn't need PyTorch.)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from uttt.encoding import encode
from uttt.game import Game


class TorchEvaluator:
    uses_model = True

    def __init__(self, net, device: str = "cpu") -> None:
        self.net = net
        self.device = device
        self.net.eval()

    @torch.no_grad()
    def evaluate(self, game: Game) -> Tuple[np.ndarray, float]:
        planes = encode(game)
        t = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        logits, value = self.net(t)
        logits = logits[0].detach().cpu().numpy().astype(np.float64)
        logits -= logits.max()
        policy = np.exp(logits)
        return policy, float(value.item())
