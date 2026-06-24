"use strict";

// ---- state --------------------------------------------------------------
const S = {
  matchId: null,
  ws: null,
  seat: null,      // "X" | "O" | null (spectator)
  role: null,      // "player" | "spectator"
  mode: null,      // "pvp" | "ai"
  last: null,      // last state payload from server
  thinking: false,
  aiLevel: 5,
  side: "X",
  reconnectAttempts: 0,
  boardBuilt: false,
};

const $ = (sel) => document.querySelector(sel);
const seatValue = (s) => (s === "X" ? 1 : s === "O" ? 2 : 0);

// ---- views --------------------------------------------------------------
function showView(id) {
  document.querySelectorAll(".view").forEach((v) => v.classList.add("hidden"));
  $("#" + id).classList.remove("hidden");
}

function goHome() {
  closeSocket();
  S.matchId = S.seat = S.role = S.mode = S.last = null;
  S.thinking = false;
  history.pushState({}, "", "/");
  $("#share-overlay").classList.add("hidden");
  showView("view-home");
}

// ---- difficulty labels --------------------------------------------------
function diffName(v) {
  return ["Random", "Very Easy", "Very Easy", "Easy", "Easy", "Medium",
          "Medium", "Hard", "Hard", "Expert", "Maximum"][v];
}

// ---- networking ---------------------------------------------------------
async function createMatch(body) {
  const res = await fetch("/api/match", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error("create failed");
  return res.json();
}

function wsUrl(matchId) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const token = localStorage.getItem("uttt_token_" + matchId);
  const q = token ? "?token=" + encodeURIComponent(token) : "";
  return `${proto}://${location.host}/ws/${matchId}${q}`;
}

function connect(matchId) {
  S.matchId = matchId;
  closeSocket();
  showView("view-game");
  setConn(false);
  const ws = new WebSocket(wsUrl(matchId));
  S.ws = ws;
  ws.onopen = () => { S.reconnectAttempts = 0; setConn(true); };
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  ws.onclose = () => {
    setConn(false);
    if (S.matchId === matchId && !(S.last && S.last.winner)) scheduleReconnect(matchId);
  };
  ws.onerror = () => {};
}

function scheduleReconnect(matchId) {
  if (S.reconnectAttempts >= 6) { toast("Connection lost"); return; }
  S.reconnectAttempts++;
  setTimeout(() => { if (S.matchId === matchId) connect(matchId); }, 1200);
}

function closeSocket() {
  if (S.ws) { try { S.ws.onclose = null; S.ws.close(); } catch (_) {} S.ws = null; }
}

function send(obj) {
  if (S.ws && S.ws.readyState === WebSocket.OPEN) S.ws.send(JSON.stringify(obj));
}

function onMessage(msg) {
  if (msg.type === "joined") {
    S.seat = msg.seat; S.role = msg.role; S.mode = msg.mode;
    if (msg.token) localStorage.setItem("uttt_token_" + S.matchId, msg.token);
    return;
  }
  if (msg.type === "error") { toast(msg.message || "Error"); if (msg.code === "no_match") setTimeout(goHome, 1500); return; }
  if (msg.cells) { S.last = msg; S.thinking = msg.type === "thinking"; render(); }
}

function setConn(online) {
  const dot = $("#conn-dot");
  dot.classList.toggle("online", online);
  dot.classList.toggle("offline", !online);
}

// ---- board --------------------------------------------------------------
function buildBoard() {
  const board = $("#board");
  board.innerHTML = "";
  for (let b = 0; b < 9; b++) {
    const sub = document.createElement("div");
    sub.className = "subboard";
    sub.dataset.board = b;
    for (let pos = 0; pos < 9; pos++) {
      const cell = document.createElement("button");
      cell.className = "cell";
      cell.dataset.cell = b * 9 + pos;
      cell.addEventListener("click", onCellClick);
      sub.appendChild(cell);
    }
    const mark = document.createElement("span");
    mark.className = "big-mark";
    sub.appendChild(mark);
    board.appendChild(sub);
  }
  S.boardBuilt = true;
}

function canMove() {
  const st = S.last;
  return st && S.role === "player" && st.status === "playing" &&
         !S.thinking && st.currentPlayer === seatValue(S.seat);
}

function onCellClick(e) {
  if (!canMove()) return;
  const i = +e.currentTarget.dataset.cell;
  if (!S.last.legalMoves.includes(i)) return;
  send({ type: "move", move: i });
}

function render() {
  if (!S.boardBuilt) buildBoard();
  const st = S.last;
  const legal = new Set(canMove() ? st.legalMoves : []);
  const openBoards = [];
  for (let b = 0; b < 9; b++) if (st.boardStatus[b] === 0) openBoards.push(b);
  const activeBoards = new Set(st.forcedBoard != null ? [st.forcedBoard] : openBoards);
  const showActive = st.status === "playing";

  // subboards
  document.querySelectorAll(".subboard").forEach((sub) => {
    const b = +sub.dataset.board;
    const status = st.boardStatus[b];
    sub.classList.toggle("won-x", status === 1);
    sub.classList.toggle("won-o", status === 2);
    sub.classList.toggle("drawn", status === 3);
    sub.classList.toggle("active", showActive && activeBoards.has(b));
    sub.querySelector(".big-mark").textContent = status === 1 ? "X" : status === 2 ? "O" : "";
  });

  // cells
  document.querySelectorAll(".cell").forEach((cell) => {
    const i = +cell.dataset.cell;
    const v = st.cells[i];
    cell.textContent = v === 1 ? "X" : v === 2 ? "O" : "";
    cell.classList.toggle("x", v === 1);
    cell.classList.toggle("o", v === 2);
    cell.classList.toggle("legal", legal.has(i));
    cell.classList.toggle("lastmove", st.lastMove === i);
  });

  renderStatus();
  renderResult();
  renderShareOverlay();
}

function renderStatus() {
  const st = S.last;
  const pill = $("#turn-pill");
  pill.classList.remove("my-turn", "thinking");
  let text;
  if (st.status === "waiting") {
    text = "Waiting for opponent…";
  } else if (st.status === "finished") {
    text = "Game over";
  } else if (S.role === "spectator") {
    text = (st.currentPlayer === 1 ? "X" : "O") + " to move";
  } else if (canMove()) {
    text = "Your move"; pill.classList.add("my-turn");
  } else if (S.thinking) {
    text = "Computer is thinking…"; pill.classList.add("thinking");
  } else {
    text = (S.mode === "ai" ? "Computer's move" : "Opponent's move");
  }
  pill.textContent = text;

  const youAre = $("#you-are");
  const opp = S.mode === "ai" ? `Computer · ${diffName(st.aiLevel ?? S.aiLevel)}` : "Friend";
  if (S.role === "player") {
    const mk = `<span class="mk" style="color:var(--${S.seat === "X" ? "x" : "o"})">${S.seat}</span>`;
    youAre.innerHTML = `You are ${mk} &nbsp;·&nbsp; vs ${opp}`;
  } else {
    youAre.textContent = "Spectating";
  }
}

function renderResult() {
  const st = S.last;
  const banner = $("#result-banner");
  banner.classList.remove("win", "lose", "draw");
  if (!st.winner) { banner.classList.add("hidden"); return; }
  banner.classList.remove("hidden");
  let text, cls;
  if (st.winner === "draw") { text = "It's a draw."; cls = "draw"; }
  else if (S.role === "player") {
    if (st.winner === S.seat) { text = "You win! 🎉"; cls = "win"; }
    else { text = "You lost."; cls = "lose"; }
  } else { text = st.winner + " wins"; cls = "win"; }
  banner.classList.add(cls);
  $("#result-text").textContent = text;
}

// ---- share overlay / QR -------------------------------------------------
let qrRenderedFor = null;
function renderShareOverlay() {
  const st = S.last;
  const overlay = $("#share-overlay");
  const oppSeat = S.seat === "X" ? "O" : "X";
  const waitingForOpp = S.mode === "pvp" && S.role === "player" &&
    st.status === "waiting" && st.seats && !st.seats[oppSeat].connected;
  if (!waitingForOpp) { overlay.classList.add("hidden"); return; }

  overlay.classList.remove("hidden");
  const url = location.origin + "/m/" + S.matchId;
  $("#share-input").value = url;
  if (qrRenderedFor !== url && window.qrcode) {
    try {
      const qr = qrcode(0, "M");
      qr.addData(url);
      qr.make();
      $("#qr").innerHTML = qr.createSvgTag({ scalable: true, margin: 0 });
      qrRenderedFor = url;
    } catch (_) { $("#qr").textContent = "(QR unavailable)"; }
  }
}

function copyLink() {
  const url = location.origin + "/m/" + S.matchId;
  navigator.clipboard?.writeText(url).then(() => toast("Link copied!"),
    () => toast(url));
}

// ---- toast --------------------------------------------------------------
let toastTimer = null;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), 2200);
}

// ---- actions / wiring ---------------------------------------------------
async function actPlayFriend() {
  try {
    const r = await createMatch({ mode: "pvp" });
    history.pushState({}, "", r.joinPath);
    connect(r.matchId);
  } catch (_) { toast("Could not create match"); }
}

async function actStartAI() {
  try {
    const r = await createMatch({ mode: "ai", level: S.aiLevel, side: S.side });
    history.pushState({}, "", r.joinPath);
    connect(r.matchId);
  } catch (_) { toast("Could not start game"); }
}

function wire() {
  document.body.addEventListener("click", (e) => {
    const el = e.target.closest("[data-action]");
    if (!el) return;
    const a = el.dataset.action;
    if (a === "play-friend") actPlayFriend();
    else if (a === "play-ai") { showView("view-ai-setup"); updateModelNote(); }
    else if (a === "start-ai") actStartAI();
    else if (a === "back-home") goHome();
    else if (a === "show-rules") showView("view-rules");
    else if (a === "rematch") send({ type: "rematch" });
    else if (a === "copy-link") copyLink();
  });

  $("#home-btn").addEventListener("click", goHome);
  $("#copy-btn").addEventListener("click", copyLink);

  const slider = $("#diff-slider");
  slider.addEventListener("input", () => {
    S.aiLevel = +slider.value;
    $("#diff-value").textContent = S.aiLevel;
    $("#diff-name").textContent = "— " + diffName(S.aiLevel);
  });

  $("#side-seg").addEventListener("click", (e) => {
    const btn = e.target.closest(".seg-btn");
    if (!btn) return;
    document.querySelectorAll("#side-seg .seg-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    S.side = btn.dataset.side;
  });

  window.addEventListener("popstate", route);
}

async function updateModelNote() {
  try {
    const r = await (await fetch("/healthz")).json();
    $("#model-note").textContent = r.model
      ? "✓ A trained neural network is loaded."
      : "No trained model yet — the AI uses Monte-Carlo search (still a real opponent!).";
  } catch (_) {}
}

// ---- routing ------------------------------------------------------------
function route() {
  const m = location.pathname.match(/^\/m\/([A-Za-z0-9]+)/);
  if (m) {
    if (S.matchId !== m[1]) connect(m[1]);
    else showView("view-game");
  } else {
    closeSocket();
    S.matchId = S.seat = S.role = S.mode = S.last = null;
    showView("view-home");
  }
}

wire();
route();
