"""Export a trained checkpoint to ONNX for the web app.

The app's :class:`uttt.ai.NeuralEvaluator` expects an ONNX graph with input
``board`` of shape ``(N, 7, 9, 9)`` and outputs ``policy`` (N, 81 logits) and
``value`` (N, 1). This script produces exactly that.

Usage::

    python -m training.export_onnx --ckpt training/checkpoints/latest.pt \
                                   --out models/uttt.onnx
"""

from __future__ import annotations

import argparse
import os

import torch

from .model import load_net


def export(net, path: str, device: str = "cpu") -> str:
    net.eval().to(device)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    dummy = torch.zeros(1, 7, 9, 9, device=device)
    kwargs = dict(
        input_names=["board"],
        output_names=["policy", "value"],
        dynamic_axes={
            "board": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
    )
    # Prefer the stable TorchScript exporter (predictable I/O names, no extra
    # deps). Newer torch defaults to the dynamo exporter; dynamo=False selects
    # the legacy path. Older torch lacks that kwarg -> fall back.
    try:
        torch.onnx.export(net, dummy, path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(net, dummy, path, **kwargs)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="training/checkpoints/latest.pt")
    ap.add_argument("--out", default="models/uttt.onnx")
    args = ap.parse_args()

    net, _ = load_net(args.ckpt, device="cpu")
    path = export(net, args.out)
    size_kb = os.path.getsize(path) / 1024
    print(f"Exported {args.ckpt} -> {path}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
