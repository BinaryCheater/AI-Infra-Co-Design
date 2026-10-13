/* Wiring: toasts, run switching, connect. */
(() => {
  const S = Session;
  const toasts = document.getElementById("toasts");
  S.on("toast", (msg, bad) => { const t = document.createElement("div"); t.className = "toast" + (bad ? " bad" : ""); t.textContent = msg; toasts.append(t); setTimeout(() => t.remove(), 4000); });
  S.on("status", s => { if (s === "disconnected") S.emit("toast", "disconnected from the server; retrying", true); });
  S.on("state-reload", async s => { S.state.run = s.run; S.state.runs = s.runs; S.state.controller = s.controller; S.state.summary = s.summary; await S.loadStory(); S.state.live = !!s.controller.live; S.state.detached = false; if (S.state.live) S.backToLive(); else S.seekEnd(); S.emit("state"); S.emit("controller"); });
  S.connect();
})();
