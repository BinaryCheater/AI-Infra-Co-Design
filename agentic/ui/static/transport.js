/* Transport: the buttons, speed and breakpoint toggles. In a recorded run they drive the play clock;
   in a live run they also drive the Controller (Run = observe, Step iteration = step, Pause = pause). */
window.Transport = (() => {
  const S = Session;
  const $ = id => document.getElementById(id);
  const BPS = ["refuted", "regime_flip", "proposed", "solved", "admitted"];
  let bps = new Set(["refuted", "regime_flip"]);

  function liveControl() { return S.state.live && !S.state.detached; }
  $("tp-first").onclick = () => { if (liveControl()) S.detach(); S.pause(); S.setClock(S.state.story.length ? S.state.story[0].iteration : 1, 0); };
  $("tp-prev").onclick = () => { if (liveControl()) S.detach(); S.pause(); S.stepIteration(-1); };
  $("tp-next").onclick = () => { S.pause(); S.stepIteration(1); };
  $("tp-last").onclick = () => { S.pause(); if (S.state.live) S.backToLive(); else S.seekEnd(); };
  $("tp-run").onclick = () => {
    if (liveControl()) { S.send({ op: "mode", mode: "observe" }); S.send({ op: "resume" }); }
    else S.play();
  };
  $("tp-pause").onclick = () => { if (liveControl()) S.send({ op: "pause" }); S.pause(); };
  $("tp-step-it").onclick = () => {
    if (liveControl()) { S.send({ op: "mode", mode: "step" }); S.send({ op: "resume" }); }
    else { S.pause(); S.stepIteration(1); }
  };
  $("tp-step-st").onclick = () => {
    if (liveControl()) { S.send({ op: "mode", mode: "drive" }); S.send({ op: "resume" }); }
    else { S.pause(); S.stepStation(1); }
  };
  $("tp-continue").onclick = () => { if (liveControl()) S.send({ op: "resume" }); else { S.pause(); S.stepIteration(1); } };
  for (const b of $("speed-seg").children) b.onclick = () => S.setSpeed(b.dataset.speed === "inf" ? "inf" : +b.dataset.speed);
  for (const b of $("breakpoints").children) b.onclick = () => { const n = b.dataset.bp; if (bps.has(n)) bps.delete(n); else bps.add(n); S.send({ op: "breakpoints", names: [...bps] }); render(); };
  $("btn-new-run").onclick = () => {
    if (window.Workload && !Workload.chosen()) { S.emit("toast", "Choose a workload first (Step 1).", true); return; }
    const name = `${Workload.chosen().replace(/_.*/, "")}-${String((S.state.runs || []).length + 1).padStart(3, "0")}`;   // e.g. square-001
    S.send({ op: "breakpoints", names: [...bps] });
    // stop after every station, so each step (propose, prove, solve, measure) waits for "Step station" or "Continue"
    S.send({ op: "start", run: name, mode: "drive" });
  };
  $("btn-back-live").onclick = () => S.backToLive();
  $("run-select").onchange = e => { S.send({ op: "select", run: e.target.value }); setTimeout(() => fetch("/api/state").then(r => r.json()).then(s => S.emit("state-reload", s)), 100); };

  function render() {
    const c = S.state.controller, live = S.state.live;
    for (const b of $("speed-seg").children) b.classList.toggle("on", String(S.state.speed) === b.dataset.speed);
    for (const b of $("breakpoints").children) b.classList.toggle("on", bps.has(b.dataset.bp));
    const ls = $("live-state");
    ls.className = "live-state" + (live ? (c.waiting ? " paused" : " live") : "");
    ls.textContent = live ? (c.waiting ? `● live · paused at “${c.waiting.checkpoint}” (iteration ${c.waiting.iteration})` : `● live · ${c.mode}`) : (c.finished ? "recorded · run finished" : "recorded");
    $("btn-back-live").hidden = !(live && S.state.detached);
    $("tp-status").textContent = S.state.playing ? `playing at ${S.state.speed}×` : (live && !S.state.detached ? (c.waiting ? "waiting for you" : "running") : `iteration ${S.state.clock.it}, station ${S.state.clock.v}`);
    $("tp-status").className = "status" + (live && c.waiting ? " paused" : "");
    $("btn-new-run").disabled = !!live;
    $("tp-continue").disabled = !(live && c.waiting);
    const sel = $("run-select"); sel.innerHTML = "";
    for (const r of S.state.runs || []) { const o = document.createElement("option"); o.value = r; o.textContent = r; o.selected = r === S.state.run; sel.append(o); }
  }
  // the local toggles are the source of truth; take the server's set only when a run is (re)loaded
  S.on("state", () => { const c = S.state.controller; if (c && c.breakpoints && S.state.live) bps = new Set(c.breakpoints); render(); });
  S.on("controller", render); S.on("clock", render);
  return { render, breakpoints: () => bps };
})();
