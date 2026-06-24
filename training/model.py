"""AlphaZero policy/value network for Ultimate Tic-Tac-Toe.

A small ConvNet over the 9x9 board: a conv stem, a stack of residual blocks,
then two heads:

* policy -> 81 logits (one per cell)
* value  -> scalar in [-1, 1] (expected result for the side to move)

It's intentionally small so it trains quickly on a free Colab GPU. Scale
``channels`` / ``blocks`` up for a stronger (slower) net.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_PLANES = 7
NUM_ACTIONS = 81


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class UTTTNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 5) -> None:
        super().__init__()
        self.channels = channels
        self.blocks = blocks

        self.stem = nn.Sequential(
            nn.Conv2d(IN_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        # Policy head
        self.p_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * 9 * 9, NUM_ACTIONS)

        # Value head
        self.v_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(9 * 9, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.res(x)

        p = F.relu(self.p_bn(self.p_conv(x)))
        p = self.p_fc(p.flatten(1))  # logits

        v = F.relu(self.v_bn(self.v_conv(x)))
        v = F.relu(self.v_fc1(v.flatten(1)))
        v = torch.tanh(self.v_fc2(v))  # (N, 1) in [-1, 1]
        return p, v


def save_checkpoint(path, net, optimizer=None, iteration=0, config=None):
    torch.save(
        {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "iteration": iteration,
            "channels": net.channels,
            "blocks": net.blocks,
            "config": config,
        },
        path,
    )


def load_net(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = UTTTNet(channels=ckpt.get("channels", 64), blocks=ckpt.get("blocks", 5))
    net.load_state_dict(ckpt["model"])
    net.to(device)
    return net, ckpt
