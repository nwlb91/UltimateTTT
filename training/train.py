"""AlphaZero self-play training loop for Ultimate Tic-Tac-Toe.

One iteration = generate self-play games with the current net (MCTS + Dirichlet
exploration) -> add (state, search-policy, outcome) samples to a replay buffer
-> train the net on minibatches -> checkpoint -> (periodically) export ONNX.

Runs on CPU or a single GPU. Designed to be **resumable**: re-running picks up
from ``checkpoints/latest.pt``, which is what makes spare-time / Colab training
practical (Colab disconnects; just run the cell again).

Run from the repo root::

    python -m training.train --iterations 50 --games-per-iter 40 --sims 100

See ``Config`` for all knobs, or the Colab notebook for a guided run.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from uttt.encoding import encode, symmetries
from uttt.game import Game
from uttt.mcts import MCTS

from .export_onnx import export
from .model import UTTTNet, load_net, save_checkpoint
from .torch_evaluator import TorchEvaluator


@dataclass
class Config:
    # network
    channels: int = 64
    blocks: int = 5
    # search
    sims: int = 100             # MCTS simulations per move during self-play
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    temp_moves: int = 12        # plies to sample (temp=1) before going greedy
    # self-play / data
    games_per_iter: int = 40
    augment: bool = True        # 8x dihedral symmetry augmentation
    buffer_size: int = 60000
    min_buffer: int = 2000      # wait until the buffer has this many before training
    # optimization
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    train_steps: int = 400      # gradient steps per iteration
    # loop / io
    iterations: int = 100
    ckpt_dir: str = "training/checkpoints"
    onnx_out: str = "models/uttt.onnx"
    export_every: int = 1       # export ONNX every N iterations
    save_every: int = 5         # also keep a numbered checkpoint every N iters
    seed: int = 0


def self_play_game(net, device, cfg: Config, rng: np.random.Generator):
    """Play one game by self-play; return (samples_without_z, winner).

    Each sample is ``(planes, pi, player)`` where ``pi`` is the normalised MCTS
    visit distribution (the training target for the policy head).
    """
    g = Game()
    evaluator = TorchEvaluator(net, device)
    mcts = MCTS(
        evaluator,
        c_puct=cfg.c_puct,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_eps=cfg.dirichlet_eps,
        rng=rng,
    )
    history = []
    while not g.is_terminal():
        counts = mcts.run(g, cfg.sims, add_noise=True)
        total = counts.sum()
        if total == 0:
            break
        pi = counts / total
        history.append((encode(g), pi, g.current_player))
        if g.move_count < cfg.temp_moves:
            move = int(rng.choice(81, p=pi))      # explore early
        else:
            move = int(np.argmax(counts))         # exploit later
        g.play(move)
    return history, (g.winner if g.winner is not None else 0)


def expand_samples(history, winner, cfg: Config):
    """Attach value targets and (optionally) symmetry-augment."""
    out = []
    for planes, pi, player in history:
        if winner == 0:
            z = 0.0
        else:
            z = 1.0 if winner == player else -1.0
        if cfg.augment:
            for p2, pi2 in symmetries(planes, pi):
                out.append((p2, pi2.astype(np.float32), np.float32(z)))
        else:
            out.append((planes, pi.astype(np.float32), np.float32(z)))
    return out


def learn(net, optimizer, buffer, cfg: Config, device):
    net.train()
    p_losses, v_losses = [], []
    steps = min(cfg.train_steps, max(1, len(buffer) // cfg.batch_size))
    for _ in range(steps):
        batch = random.sample(buffer, min(cfg.batch_size, len(buffer)))
        planes = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
        target_pi = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
        target_z = torch.from_numpy(np.array([b[2] for b in batch], dtype=np.float32)).to(device)

        logits, value = net(planes)
        logp = F.log_softmax(logits, dim=1)
        policy_loss = -(target_pi * logp).sum(dim=1).mean()
        value_loss = F.mse_loss(value.squeeze(1), target_z)
        loss = value_loss + policy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        p_losses.append(policy_loss.item())
        v_losses.append(value_loss.item())
    return float(np.mean(p_losses)), float(np.mean(v_losses))


def train(cfg: Config, resume: bool = True):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    latest = os.path.join(cfg.ckpt_dir, "latest.pt")

    if resume and os.path.exists(latest):
        net, ckpt = load_net(latest, device)
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt.get("iteration", 0)
        print(f"Resumed from {latest} at iteration {start_iter} (device={device})")
    else:
        net = UTTTNet(channels=cfg.channels, blocks=cfg.blocks).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        start_iter = 0
        print(f"Fresh net: channels={cfg.channels} blocks={cfg.blocks} (device={device})")

    buffer = deque(maxlen=cfg.buffer_size)

    for it in range(start_iter, cfg.iterations):
        t0 = time.time()
        net.eval()
        new = 0
        for _ in range(cfg.games_per_iter):
            history, winner = self_play_game(net, device, cfg, rng)
            samples = expand_samples(history, winner, cfg)
            buffer.extend(samples)
            new += len(samples)

        if len(buffer) >= cfg.min_buffer:
            p_loss, v_loss = learn(net, optimizer, buffer, cfg, device)
            loss_str = f"policy={p_loss:.3f} value={v_loss:.3f}"
        else:
            loss_str = "(warming up buffer)"

        save_checkpoint(latest, net, optimizer, iteration=it + 1, config=asdict(cfg))
        if (it + 1) % cfg.save_every == 0:
            save_checkpoint(os.path.join(cfg.ckpt_dir, f"iter_{it + 1:04d}.pt"),
                            net, optimizer, iteration=it + 1, config=asdict(cfg))
        if (it + 1) % cfg.export_every == 0:
            export(net, cfg.onnx_out, device="cpu")

        dt = time.time() - t0
        print(f"iter {it + 1:4d}/{cfg.iterations} | +{new:5d} samples | "
              f"buffer {len(buffer):6d} | {loss_str} | {dt:5.1f}s", flush=True)

    export(net, cfg.onnx_out, device="cpu")
    print(f"Done. Final model exported to {cfg.onnx_out}")
    return net


def _parse_args() -> Config:
    cfg = Config()
    ap = argparse.ArgumentParser(description="AlphaZero trainer for Ultimate Tic-Tac-Toe")
    for field, value in asdict(cfg).items():
        flag = "--" + field.replace("_", "-")
        if isinstance(value, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=value)
        else:
            ap.add_argument(flag, type=type(value), default=value)
    ap.add_argument("--no-resume", action="store_true", help="ignore existing checkpoint")
    args = ap.parse_args()
    kwargs = {f: getattr(args, f) for f in asdict(cfg)}
    return Config(**kwargs), (not args.no_resume)


if __name__ == "__main__":
    config, do_resume = _parse_args()
    train(config, resume=do_resume)
