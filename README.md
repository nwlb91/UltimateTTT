# 🎯 Ultimate Tic-Tac-Toe

A web app for [Ultimate Tic-Tac-Toe](https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe)
with two ways to play:

- **Human vs human** — create a match, get a **shareable link + QR code**; whoever
  opens it joins your game in real time.
- **Human vs AI** — play an **AlphaZero-style** opponent (MCTS + a convolutional
  neural network) with an adjustable **difficulty slider**.

The whole thing is one small Python service (FastAPI) that serves the UI, runs
the real-time multiplayer, and hosts the AI. The AI is trained separately with a
self-play pipeline you can run **for free in Google Colab — entirely in your browser**.

> **On a locked-down work computer?** You can do everything without a terminal:
> deploy via Render's web dashboard, train via Colab, and update the model by
> uploading a file through GitHub's website. See the guides below.

---

## Table of contents
- [How to play](#how-to-play)
- [Deploy it (Render, no terminal)](#deploy-it-render-no-terminal)
- [Train the AI (Colab, no terminal)](#train-the-ai-colab-no-terminal)
- [How the difficulty slider works](#how-the-difficulty-slider-works)
- [Run locally (optional)](#run-locally-optional)
- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [FAQ / notes](#faq--notes)

---

## How to play

The rules, in four lines:

1. The big board is a 3×3 grid of nine small Tic-Tac-Toe boards.
2. **The square you play in sends your opponent to the matching board.** Play the
   top-right square → they must play in the top-right board next.
3. Win a small board (three in a row) to **claim it** on the big board.
4. **Win three small boards in a row to win the game.** If you're sent to a board
   that's already won or full, you may play anywhere.

In the app, your legal squares are highlighted and the board you must play in
glows.

### Play a friend
Tap **Play a Friend** → you get a link and a QR code. Send the link or have them
scan the code. The game starts the moment they join. Refreshing reconnects you to
your seat.

### Play the computer
Tap **Play the Computer** → pick a **difficulty (0–10)** and whether you're X or O
→ **Start**. The AI replies on the server.

---

## Deploy it (Render, no terminal)

This repo includes a [`render.yaml`](render.yaml) blueprint, so Render sets
everything up for you.

1. Make sure this code is on GitHub (it is — that's where you're reading this).
2. Go to **[render.com](https://render.com)** and sign in with GitHub (free).
3. Click **New + → Blueprint**.
4. Select this repository. Render reads `render.yaml` and shows a service named
   **`ultimate-ttt`**. Click **Apply**.
5. Wait for the first build/deploy (a couple of minutes). You'll get a public URL
   like `https://ultimate-ttt.onrender.com`.
6. Open it on your phone and laptop — share a "Play a Friend" link between them to
   see multiplayer working. 🎉

Visit **`/healthz`** to check status; it reports whether a trained model is loaded
(`"model": true`).

**Free-tier notes:** the free instance sleeps after ~15 minutes idle, so the
first request after a nap takes ~30–60s to wake. Matches are kept in memory, so a
sleep/restart ends any in-progress game (fine for casual play; upgrade the plan or
add a datastore if you want persistence).

> Prefer a different host? There's a [`Dockerfile`](Dockerfile) for Fly.io,
> Railway, a VPS, etc. The start command is
> `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

---

## Train the AI (Colab, no terminal)

The app is **playable immediately** — with no trained model it uses Monte-Carlo
Tree Search with random rollouts (a real opponent, just not razor-sharp). Training
a network makes it much stronger.

Everything runs in your browser via the included Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nwlb91/ultimatettt/blob/main/training/UltimateTTT_AlphaZero.ipynb)

> The badge points at the `main` branch. If the code is on another branch, open
> `training/UltimateTTT_AlphaZero.ipynb` in Colab and set `BRANCH` in Step 3.

**What you do:**
1. Open the notebook (badge above, or **colab.research.google.com → GitHub →** this repo).
2. **Runtime → Change runtime type → T4 GPU.**
3. Run the cells top to bottom. They mount your Google Drive, pull the code,
   install deps, and start self-play training. Checkpoints save to **Drive**, so
   if Colab disconnects you just **re-run the train cell and it resumes**.
4. When you're happy with it, the last cells **export `uttt.onnx`** and let you
   download it.

**Get the model into the app (no terminal):** in your GitHub repo, open the
**`models/`** folder → **Add file → Upload files** → drop in `uttt.onnx` (named
exactly that) → commit. Render redeploys automatically and the AI starts using
your network. (The notebook also shows an optional "push from Colab" path.)

Training is resumable and open-ended — run it for an hour or spread it across
days. Longer training = stronger AI.

---

## How the difficulty slider works

Difficulty (0–10) modulates the AI's **search budget** and how **greedy** it is —
exactly the "look-up depth" idea, implemented as MCTS simulation count plus a
sampling temperature:

| Level | MCTS simulations | Behaviour |
|------:|-----------------:|-----------|
| 0     | 0 (random)       | Plays a random legal move — for absolute beginners |
| 1–3   | 24–96            | Shallow search, samples loosely → makes mistakes |
| 4–6   | 160–400          | Medium search, mostly sensible |
| 7–8   | 600–900          | Deep search, sharp play |
| 9–10  | 1300–1800        | Deepest search, near-greedy → strongest |

Low levels also use a higher **temperature**, so they sometimes pick a good-but-not
-best move; high levels play the most-visited (best) move. The same slider works
whether or not a neural network is loaded — the network just makes each simulation
much smarter. See [`uttt/ai.py`](uttt/ai.py) (`level_params`).

---

## Run locally (optional)

If you can use a terminal (e.g. at home):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

To play a friend on your network, share `http://<your-LAN-ip>:8000`. For a link
that works anywhere, deploy to Render (above) or use a tunnel like `ngrok`.

Run the tests with `pip install pytest && pytest`.

---

## How it works

```
            Browser (vanilla JS)
        ┌─────────────────────────┐
        │ board UI · QR · slider   │
        └───────────┬─────────────┘
            REST (create match)  +  WebSocket (live moves)
        ┌───────────▼─────────────┐
        │   FastAPI server (app/)  │
        │  • matchmaking + seats   │
        │  • authoritative game    │
        │  • AI moves off-thread   │
        └───────────┬─────────────┘
                    │ imports
        ┌───────────▼─────────────┐        ┌────────────────────────┐
        │  uttt/  (shared core)    │        │  training/ (PyTorch)   │
        │  game · encoding · mcts  │◄───────┤  self-play → uttt.onnx │
        │  ai (ONNX or rollouts)   │  same  │  (run in Colab)        │
        └──────────────────────────┘ rules └────────────────────────┘
```

- **`uttt/`** is a dependency-light core (rules as bitboards, NN encoding, and a
  generic PUCT/MCTS) shared by *both* the server and the trainer, so the rules can
  never drift between training and play.
- **MCTS** is decoupled from evaluation: it calls an `evaluate(game) → (policy,
  value)` function. The app uses an **ONNX** network when `models/uttt.onnx`
  exists, and falls back to **random rollouts** (classic UCT) otherwise. PUCT is
  the AlphaZero generalisation of the UCB/UCT selection rule.
- **Training** is standard AlphaZero: self-play with Dirichlet-noised MCTS produces
  `(state, search-policy, outcome)` samples (with 8× board-symmetry augmentation);
  a small ConvNet learns from them; repeat. The net is exported to ONNX for the app.
- AI moves run in a worker thread (`asyncio.to_thread`) so one slow search never
  blocks other matches.

---

## Project layout

```
uttt/                 Shared core (no heavy deps)
  game.py             Rules engine (bitboards)
  encoding.py         Board → NN planes + 8 symmetries
  mcts.py             PUCT / MCTS (evaluator-agnostic)
  ai.py               App AI: ONNX or rollout evaluator + difficulty mapping
app/                  FastAPI web server
  main.py             Routes, WebSocket game loop, AI dispatch
  matchmaking.py      Matches, seats, reconnect tokens
web/                  Frontend (no build step)
  index.html · style.css · app.js · vendor/qrcode.js
training/             AlphaZero pipeline (PyTorch; not needed to run the app)
  model.py · train.py · torch_evaluator.py · export_onnx.py · arena.py
  UltimateTTT_AlphaZero.ipynb     ← the Colab notebook
models/               Put trained uttt.onnx here (app auto-loads it)
tests/                Engine + AI tests
render.yaml           Render blueprint     Dockerfile   Other hosts
```

---

## FAQ / notes

- **Do I need the model to launch?** No. The app runs with rollout-MCTS out of the
  box. Add `models/uttt.onnx` later to upgrade the AI.
- **Where does the AI run?** On the server (Python). The browser only renders.
- **Is multiplayer state saved?** It's in memory (simple + free-tier friendly).
  Refresh reconnects within the session; a server restart clears matches.
- **Custom model path:** set the `UTTT_MODEL_PATH` env var to point elsewhere.
- **Make AI moves faster on a paid plan:** raise `UTTT_ORT_THREADS`.
