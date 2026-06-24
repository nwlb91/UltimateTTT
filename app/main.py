"""FastAPI server for Ultimate Tic-Tac-Toe.

Responsibilities
----------------
* Serve the single-page web client.
* Create matches (human-vs-human or human-vs-AI) over a small REST API.
* Run each match over a WebSocket: validate moves, keep the authoritative
  game state, broadcast updates, and reconnect players by token.
* Compute AI moves off the event loop (``asyncio.to_thread``) so one slow
  search never blocks other matches.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections import defaultdict
from typing import Dict, Optional, Set

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from uttt.ai import load_ai

from .matchmaking import MatchManager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(ROOT, "web")
INDEX = os.path.join(WEB_DIR, "index.html")

manager = MatchManager()
AI = load_ai()

# WebSocket bookkeeping kept out of the (sync) matchmaking layer.
connections: Dict[str, Set[WebSocket]] = defaultdict(set)
_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    reaper = asyncio.create_task(_reaper())
    try:
        yield
    finally:
        reaper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reaper


app = FastAPI(title="Ultimate Tic-Tac-Toe", lifespan=lifespan)


async def _reaper() -> None:
    while True:
        await asyncio.sleep(300)
        removed = manager.reap()
        for mid in list(connections):
            if manager.get(mid) is None and not connections[mid]:
                connections.pop(mid, None)
                _locks.pop(mid, None)
        if removed:
            print(f"[uttt] reaped {removed} idle match(es)")


# -- REST -----------------------------------------------------------------

class CreateMatch(BaseModel):
    mode: str = "pvp"          # "pvp" | "ai"
    level: int = 5             # AI difficulty 0-10
    side: str = "random"       # "X" | "O" | "random" (human's side vs AI)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "model": AI.uses_model}


@app.post("/api/match")
async def create_match(req: CreateMatch):
    if req.mode == "ai":
        match = manager.create_ai(level=req.level, human_plays=req.side)
    else:
        match = manager.create_pvp()
    return {
        "matchId": match.id,
        "mode": match.mode,
        "joinPath": f"/m/{match.id}",
        "aiLevel": match.ai_level if match.mode == "ai" else None,
    }


@app.get("/")
async def index():
    return FileResponse(INDEX)


@app.get("/m/{match_id}")
async def match_page(match_id: str):
    return FileResponse(INDEX)


# -- WebSocket ------------------------------------------------------------

@app.websocket("/ws/{match_id}")
async def ws_endpoint(websocket: WebSocket, match_id: str):
    await websocket.accept()
    match = manager.get(match_id)
    if match is None:
        await websocket.send_json({"type": "error", "code": "no_match",
                                   "message": "This match no longer exists."})
        await websocket.close()
        return

    token = websocket.query_params.get("token")
    seat = match.seat_for_token(token)
    if seat is None:
        seat = match.open_human_seat()  # claim an empty human seat
    if seat is not None:
        seat.connection = websocket
    connections[match_id].add(websocket)
    match.touch()

    seat_label = None if seat is None else ("X" if seat.player == 1 else "O")
    await websocket.send_json({
        "type": "joined",
        "matchId": match_id,
        "mode": match.mode,
        "seat": seat_label,
        "token": seat.token if seat else None,
        "role": "player" if seat else "spectator",
    })
    await broadcast(match)
    await maybe_ai_move(match)  # AI may move first (e.g. human chose O)

    try:
        while True:
            msg = await websocket.receive_json()
            await handle_message(match, seat, msg)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[uttt] ws error in {match_id}: {exc!r}")
    finally:
        connections[match_id].discard(websocket)
        if seat is not None and seat.connection is websocket:
            seat.connection = None
        match.touch()
        await broadcast(match)


async def handle_message(match, seat, msg: dict) -> None:
    mtype = msg.get("type")
    if mtype == "move":
        await handle_move(match, seat, msg)
    elif mtype == "rematch":
        match.reset()
        await broadcast(match)
        await maybe_ai_move(match)
    elif mtype == "ping":
        pass  # keep-alive; state is pushed on real changes


async def handle_move(match, seat, msg: dict) -> None:
    if seat is None:
        return
    move = msg.get("move")
    async with _locks[match.id]:
        game = match.game
        if game.is_terminal():
            return
        if game.current_player != seat.player:
            return  # not your turn
        if not isinstance(move, int) or move not in game.legal_moves():
            return
        game.play(move)
        match.touch()
    await broadcast(match)
    await maybe_ai_move(match)


async def maybe_ai_move(match) -> None:
    """If it is the AI's turn, compute and apply its move."""
    if match.mode != "ai":
        return
    ai_seat = match.ai_seat
    if ai_seat is None:
        return
    while not match.game.is_terminal() and match.game.current_player == ai_seat.player:
        await broadcast(match, extra={"type": "thinking"})
        snapshot = match.game.clone()
        try:
            move = await asyncio.to_thread(AI.choose_move, snapshot, match.ai_level)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[uttt] AI error: {exc!r}")
            return
        async with _locks[match.id]:
            game = match.game
            if game.is_terminal() or game.current_player != ai_seat.player:
                return
            legal = game.legal_moves()
            if move not in legal:
                if not legal:
                    return
                move = legal[0]
            game.play(move)
            match.touch()
        await broadcast(match)


# -- broadcasting ---------------------------------------------------------

async def broadcast(match, extra: Optional[dict] = None) -> None:
    payload = match.public_state()
    if extra:
        payload = {**payload, **extra}
    dead = []
    for ws in list(connections.get(match.id, ())):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connections[match.id].discard(ws)


# Static assets (css/js/vendor). Mounted last so it doesn't shadow routes.
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.exception_handler(404)
async def not_found(request: Request, exc):  # pragma: no cover
    return JSONResponse({"error": "not found"}, status_code=404)
