/* Scrubber: one tick per iteration, coloured by outcome; drag or click to seek. */
window.Scrubber = (() => {
  const S = Session;
  const ticks = document.getElementById("ticks"), cursor = document.getElementById("cursor"), track = ticks.parentElement;
  let dragging = false;

  const pos = (i, n) => n <= 1 ? 50 : (i / (n - 1)) * 100;
  function render() {
    const st = S.state.story;
    ticks.innerHTML = "";
    st.forEach((e, i) => {
      const t = document.createElement("div");
      t.className = "tick " + (e.outcome === "rejected" ? "rejected" : e.flip ? "flip" : e.outcome === "accepted" ? "accepted" : "");
      if (e.iteration === S.state.clock.it) t.classList.add("current");
      t.style.left = pos(i, st.length) + "%";
      t.title = e.title + (e.flip ? " · bottleneck moved" : "");
      const n = document.createElement("span"); n.className = "n"; n.textContent = e.iteration; t.append(n);
      if (e.human && e.human.length) { const p = document.createElement("span"); p.className = "person"; p.textContent = "☺"; p.title = e.human.map(h => h.kind).join(", "); t.append(p); }
      t.onclick = () => seekTo(e.iteration);
      ticks.append(t);
    });
    const i = st.findIndex(e => e.iteration === S.state.clock.it);
    cursor.style.left = (i >= 0 ? pos(i, st.length) : 0) + "%";
  }
  function seekTo(it) {
    const e = S.state.storyByIt[it];
    if (!e) return;
    if (S.state.live && !S.atLiveEdgeAfter(it)) S.detach();
    S.pause();
    S.setClock(it, e.visits.length);
  }
  S.atLiveEdgeAfter = it => { const l = S.state.story[S.state.story.length - 1]; return !!l && it === l.iteration; };
  function fromX(x) {
    const r = track.getBoundingClientRect(), st = S.state.story;
    if (!st.length) return null;
    const f = Math.max(0, Math.min(1, (x - r.left) / r.width));
    return st[Math.round(f * (st.length - 1))].iteration;
  }
  track.addEventListener("pointerdown", ev => { dragging = true; track.setPointerCapture(ev.pointerId); const it = fromX(ev.clientX); if (it) seekTo(it); });
  track.addEventListener("pointermove", ev => { if (!dragging) return; const it = fromX(ev.clientX); if (it && it !== S.state.clock.it) seekTo(it); });
  track.addEventListener("pointerup", () => { dragging = false; });
  S.on("clock", render); S.on("story", render);
  return { render, seekTo };
})();
