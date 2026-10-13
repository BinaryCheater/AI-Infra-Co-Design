/* The map: five fixed stations (the gate drawn as the verifier's second door), the level ladder, the
   template token that moves, and result badges. Renders story[it].visits[0..v) from the play clock. */
window.MapView = (() => {
  const S = Session;
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.getElementById("map");
  const STATIONS = {
    formulator:   { x: 190, y: 40,  w: 170, h: 64, name: "FORMULATOR",   role: "propose" },
    verifier:     { x: 520, y: 40,  w: 170, h: 64, name: "VERIFIER",     role: "prove safe" },
    solver:       { x: 520, y: 170, w: 170, h: 64, name: "SOLVER",       role: "optimize" },
    gate:         { x: 535, y: 262, w: 140, h: 36, name: "GATE",         role: "" },
    evaluator:    { x: 520, y: 330, w: 170, h: 64, name: "EVALUATOR",    role: "measure" },
    orchestrator: { x: 190, y: 330, w: 170, h: 64, name: "ORCHESTRATOR", role: "decide" },
  };
  const ORDER = ["formulator", "verifier", "solver", "gate", "evaluator", "orchestrator"];
  const LEVELS = ["workload", "graph", "operators", "tensor map", "instructions", "architecture", "microarch."];
  const c = (s) => ({ cx: s.x + s.w / 2, cy: s.y + s.h / 2 });
  const mk = (tag, attrs = {}, parent = svg) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); parent.append(n); return n; };
  const text = (parent, x, y, str, cls, anchor = "middle") => { const t = mk("text", { x, y, class: cls || "", "text-anchor": anchor }, parent); t.textContent = str; return t; };
  const nodes = {}, badges = {}, ladder = {};
  let token, tokenShort, tokenWl, note;

  function build() {
    svg.innerHTML = "";
    const defs = mk("defs");
    const m = mk("marker", { id: "arrowhead", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
    mk("path", { d: "M0,0 L10,5 L0,10 z", fill: "#183039" }, m);
    // ladder
    const lg = mk("g", { class: "ladder" });
    text(lg, 20, 28, "levels", "", "start");
    LEVELS.forEach((name, i) => {
      const g = mk("g", { class: "level" }, lg); ladder[i + 1] = g;
      mk("rect", { x: 14, y: 40 + i * 50, width: 130, height: 36, rx: 5, fill: "transparent" }, g);
      mk("line", { x1: 14, y1: 40 + i * 50 + 36, x2: 144, y2: 40 + i * 50 + 36, class: "rung" }, g);
      text(g, 24, 40 + i * 50 + 24, `${i + 1}  ${name}`, "", "start");
    });
    // arrows: F -> V, V -> S, S -> G, G -> E, E -> O, O -> F
    const f = STATIONS.formulator, v = STATIONS.verifier, s = STATIONS.solver, g = STATIONS.gate, e = STATIONS.evaluator, o = STATIONS.orchestrator;
    const path = d => mk("path", { d, class: "arrow" });
    path(`M${f.x + f.w},${c(f).cy} L${v.x},${c(v).cy}`);
    path(`M${c(v).cx},${v.y + v.h} L${c(s).cx},${s.y}`);
    path(`M${c(s).cx},${s.y + s.h} L${c(g).cx},${g.y}`);
    path(`M${c(g).cx},${g.y + g.h} L${c(e).cx},${e.y}`);
    path(`M${e.x},${c(e).cy} L${o.x + o.w},${c(o).cy}`);
    path(`M${c(o).cx},${o.y} L${c(f).cx},${f.y + f.h}`);
    text(svg, g.x + g.w + 8, c(g).cy + 4, "the verifier's second door", "ladder", "start").setAttribute("fill", "#6E7E84");
    // stations + badges
    for (const [key, st] of Object.entries(STATIONS)) {
      const gr = mk("g", { class: "station " + key }); nodes[key] = gr;
      mk("rect", { x: st.x, y: st.y, width: st.w, height: st.h }, gr);
      if (key === "gate") text(gr, c(st).cx, st.y + 22, "GATE · checked", "name");
      else { text(gr, c(st).cx, st.y + 26, st.name, "name"); text(gr, c(st).cx, st.y + 46, st.role, "role"); }
      const bg = mk("g", { class: "badge" }); badges[key] = bg;
      const bx = key === "gate" ? st.x - 180 : st.x;
      const by = key === "gate" ? st.y + 7 : st.y + st.h + 6;
      mk("rect", { x: bx, y: by, width: 170, height: 22 }, bg);
      text(bg, bx + 8, by + 15, "", "", "start");
      bg.style.display = "none";
    }
    note = text(svg, c(o).cx, o.y + o.h + 48, "", "note");
    // token
    token = mk("g", { class: "token" });
    mk("rect", { x: -34, y: -17, width: 68, height: 34 }, token);
    tokenShort = text(token, 0, -2, "", "");
    tokenWl = text(token, 0, 12, "", "wl");
    placeToken("orchestrator", true);
  }

  function placeToken(station, instant) {
    // the token sits on the station's top-right corner so the station name and badge stay readable
    const st = STATIONS[station] || STATIONS.orchestrator;
    const x = st.x + st.w - 26, y = st.y - 2;
    token.classList.toggle("instant", !!instant || S.state.speed === "inf");
    token.setAttribute("transform", `translate(${x},${y})`);
  }

  function render() {
    const { it, v } = S.state.clock;
    const e = S.state.storyByIt[it];
    for (const k of ORDER) { nodes[k].classList.remove("visited", "bad", "pulse"); badges[k].style.display = "none"; }
    for (const g of Object.values(ladder)) g.classList.remove("current");
    note.textContent = "";
    if (!e) { placeToken("orchestrator", true); tokenShort.textContent = ""; tokenWl.textContent = ""; return; }
    if (e.level && ladder[e.level]) ladder[e.level].classList.add("current");
    const visits = e.visits.slice(0, v);
    for (const vis of visits) {
      const n = nodes[vis.station], b = badges[vis.station];
      if (!n) continue;
      n.classList.add("visited"); if (!vis.ok) n.classList.add("bad");
      b.style.display = ""; b.classList.toggle("bad", !vis.ok);
      b.querySelector("text").textContent = vis.verdict_text;
    }
    const cur = visits.length ? visits[visits.length - 1] : null;
    token.classList.toggle("rejected", !!cur && !cur.ok);
    tokenShort.textContent = e.template ? e.template.short : "";
    tokenWl.textContent = e.template ? e.template.workload.replace(/_.*/, "") : "";
    placeToken(cur ? cur.station : (v === 0 ? "orchestrator" : "formulator"), false);
    if (e.flip && visits.some(x => x.station === "evaluator") && !e.descended && visits.some(x => x.station === "orchestrator"))
      note.textContent = "bottleneck moved · next level not yet available (single backend) · stayed";
    // live: pulse the station where the loop waits
    const w = S.state.controller.waiting;
    if (S.state.live && w && w.iteration === it) {
      const at = { proposed: "formulator", solved: "solver", regime_flip: "evaluator", admitted: "orchestrator",
                   refuted: (cur && !cur.ok) ? cur.station : "verifier" }[w.checkpoint];
      if (at && nodes[at]) nodes[at].classList.add("pulse");
    }
  }
  build();
  S.on("clock", render); S.on("story", render); S.on("controller", render);
  return { render, STATIONS };
})();
