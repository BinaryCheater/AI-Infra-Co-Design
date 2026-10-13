/* Session: WebSocket, /api fetches, live vs recorded mode, the play clock, and detach / back-to-live.
   The play clock (iteration, visit) is the single source of "where are we"; every component renders
   from it. In recorded mode a timer advances it; in live mode it follows the ledger as events arrive. */
window.Session = (() => {
  const state = {
    run: null, runs: [], controller: {}, summary: {}, story: [], storyByIt: {}, timeline: [],
    clock: { it: 1, v: 0 }, speed: 1, playing: false, live: false, detached: false, selected: null,
  };
  const listeners = {};
  const on = (ev, fn) => ((listeners[ev] = listeners[ev] || []).push(fn), fn);
  const emit = (ev, ...a) => (listeners[ev] || []).forEach(fn => fn(...a));
  let ws, timer = null;

  // ------------------------------------------------------------- transport
  function connect() {
    ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
    ws.onopen = () => emit("status", "connected");
    ws.onclose = () => { emit("status", "disconnected"); setTimeout(connect, 1500); };
    ws.onmessage = m => {
      const msg = JSON.parse(m.data);
      if (msg.type === "hello") applyState(msg.state, true);
      else if (msg.type === "controller") { if (msg.run && msg.run !== state.run) reloadState(); else { state.controller = msg.controller; onController(); } }
      else if (msg.type === "story") { if (msg.run === state.run) onStory(msg.entry); }
      else if (msg.type === "event") {
        const ev = msg.event;
        if (ev.run !== state.run) return;
        if (ev.kind === "run_finished" || ev.kind === "admitted") { refreshSummary(); loadDescribe(); }
        if (["proposed", "family_check", "solved", "instance_check", "measured", "diagnosis"].includes(ev.kind)) loadDescribe();
        if (ev.kind === "human_refused") emit("toast", "Not allowed: " + ev.payload.rule, true);
        if (ev.kind === "human_edit" || ev.kind === "human_inject") refreshSummary();
      }
      else if (msg.type === "error") emit("toast", "error: " + msg.error, true);
      else if (msg.type === "ok") { if (msg.run && msg.run !== state.run) reloadState(); else { state.controller = msg.controller; onController(); } }
    };
  }
  const send = msg => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(msg)); else emit("toast", "not connected", true); };
  const api = async path => (await fetch("/api/" + path + (path.includes("?") ? "&" : "?") + "run=" + encodeURIComponent(state.run || ""))).json();

  async function applyState(s, initial) {
    const runChanged = s.run !== state.run;
    Object.assign(state, { run: s.run, runs: s.runs, controller: s.controller, summary: s.summary, timeline: s.timeline });
    if (runChanged || initial) await loadStory();
    state.live = !!s.controller.live;
    if (runChanged || initial) { state.detached = false; state.playing = false; stopTimer(); if (!state.live) seekEnd(); else followLive(); }
    emit("state"); onController();
  }
  async function reloadState() { const s = await (await fetch("/api/state")).json(); await applyState(s, false); }
  async function loadStory() {
    state.story = state.run ? await api("story") : [];
    state.storyByIt = Object.fromEntries(state.story.map(e => [e.iteration, e]));
    emit("story");
    loadDescribe();
  }
  async function loadDescribe() {
    state.describe = state.run ? await api("describe_all") : {};
    emit("describe");
  }
  async function refreshSummary() { const s = await api("state"); state.summary = s.summary; state.timeline = s.timeline; emit("summary"); }
  function onStory(entry) {
    state.storyByIt[entry.iteration] = entry;
    const i = state.story.findIndex(e => e.iteration === entry.iteration);
    if (i >= 0) state.story[i] = entry; else { state.story.push(entry); state.story.sort((a, b) => a.iteration - b.iteration); }
    emit("story");
    if (state.live && !state.detached) followLive();
  }
  function onController() {
    const c = state.controller;
    const wasLive = state.live;
    state.live = !!c.live;
    if (wasLive && !state.live) { state.detached = false; state.playing = false; stopTimer(); refreshSummary().then(() => { seekEnd(); emit("clock"); }); }
    emit("controller");
  }

  // ------------------------------------------------------------- the play clock
  const last = () => state.story.length ? state.story[state.story.length - 1] : null;
  const visitsOf = it => (state.storyByIt[it] || { visits: [] }).visits;
  function setClock(it, v) {
    if (!state.story.length) { state.clock = { it: 1, v: 0 }; emit("clock"); return; }
    const first = state.story[0].iteration, lastIt = last().iteration;
    it = Math.max(first, Math.min(lastIt, it));
    v = Math.max(0, Math.min(visitsOf(it).length, v));
    state.clock = { it, v };
    emit("clock");
  }
  function seekEnd() { const l = last(); if (l) setClock(l.iteration, l.visits.length); }
  function followLive() { const l = last(); if (l) setClock(l.iteration, l.visits.length); }
  function stepStation(dir = 1) {
    const { it, v } = state.clock;
    if (dir > 0) {
      if (v < visitsOf(it).length) setClock(it, v + 1);
      else if (state.storyByIt[it + 1]) setClock(it + 1, Math.min(1, visitsOf(it + 1).length));
      else return false;
    } else {
      if (v > 0) setClock(it, v - 1);
      else if (state.storyByIt[it - 1]) setClock(it - 1, visitsOf(it - 1).length);
    }
    return true;
  }
  function stepIteration(dir = 1) {
    const { it } = state.clock;
    const target = it + dir;
    if (state.storyByIt[target]) setClock(target, visitsOf(target).length);
    else if (dir > 0) seekEnd(); else setClock(it, 0);
  }
  function play() {
    if (state.speed === "inf") { seekEnd(); state.playing = false; emit("controller"); return; }
    state.playing = true; emit("controller");
    const period = 1500 / state.speed;
    stopTimer();
    timer = setInterval(() => { if (!stepStation(1)) { state.playing = false; stopTimer(); emit("controller"); } }, period);
  }
  function stopTimer() { if (timer) clearInterval(timer); timer = null; }
  function pause() { state.playing = false; stopTimer(); emit("controller"); }
  function setSpeed(s) { state.speed = s; if (state.playing) play(); emit("controller"); }
  function detach() { if (state.live) { state.detached = true; emit("controller"); } }
  function backToLive() { state.detached = false; followLive(); emit("controller"); }
  function atLiveEdge() { const l = last(); return !!l && state.clock.it === l.iteration && state.clock.v === l.visits.length; }

  return { state, on, emit, connect, send, api, setClock, seekEnd, stepStation, stepIteration, play, pause, setSpeed,
           detach, backToLive, atLiveEdge, loadStory, loadDescribe, refreshSummary };
})();
