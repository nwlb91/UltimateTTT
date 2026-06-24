# Models

The web app looks for a trained network at **`models/uttt.onnx`** (override with
the `UTTT_MODEL_PATH` env var).

- **No file here?** The app still works — the AI falls back to Monte-Carlo Tree
  Search with random rollouts (classic UCT). It's a genuine opponent, just not
  as sharp as a trained net.
- **After training** (see [`../training/`](../training/)), drop your exported
  `uttt.onnx` in this folder and commit it. Render redeploys automatically and
  the AI starts using the network.

The exported network is small (a few MB), so it's fine to commit to git.
