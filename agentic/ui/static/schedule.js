/* The schedule view (round 4): the grid and waves, the block pipeline (memory lane, one lane per
   slot, sync, compute), the resource strip, the proved overlay, the counterexample overlay and the
   parent ghost. Renders /api/schedule/<id>; one color per slot, compute in navy, barriers as thin
   red ticks, proved windows in teal, the counterexample pair in red with the rest dimmed. */
window.ScheduleView = (() => {
  const S = Session;
  const NS = "http://www.w3.org/2000/svg";
  const SLOT_COLORS = ["#5A7D78", "#B8860B", "#7A5C99", "#3F7D5A", "#C54A38", "#2D6FB5"];
  const NAVY = "#183039", RED = "#C54A38", TEAL = "#5A7D78", LIGHT = "#F1EFEB", LINE = "#D8D4CC", MUTED = "#6E7E84";
  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
  const mk = (tag, attrs, parent) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs || {})) if (v != null) n.setAttribute(k, v); if (parent) parent.append(n); return n; };
  const txt = (parent, x, y, s, attrs) => { const t = mk("text", { x, y, "font-size": 11, fill: MUTED, ...(attrs || {}) }, parent); t.textContent = s; return t; };
  const us = s => (s * 1e6).toFixed(2) + " µs";
  const state = { proved: false, cex: false, parent: false, all: false };

  async function render(container, id, opts) {
    const sch = await S.api("schedule/" + id);
    let parent = null;
    container.innerHTML = "";
    const box = el("div", "schedule");
    container.append(box);
    if (!sch.block) {
      box.append(el("div", "muted", sch.note || "no schedule"));
      if (sch.counterexample) { box.append(el("h3", "", "what the verifier found")); box.append(counterexamplePanel(sch)); }
      return sch;
    }
    const toggles = el("div", "toggles");
    const mkToggle = (key, label, onChange) => { const b = el("button", state[key] ? "on" : "", label); b.onclick = async () => { state[key] = !state[key]; if (onChange) await onChange(); draw(); }; toggles.append(b); return b; };
    mkToggle("proved", "show what was proved");
    if (sch.counterexample) mkToggle("cex", "show the counterexample");
    if (sch.parent) mkToggle("parent", "compare with parent", async () => { if (state.parent && !parent) parent = await S.api("schedule/" + sch.parent); });
    mkToggle("all", "all phases");
    box.append(toggles);
    const gridHost = el("div"), pipeHost = el("div", "pipe-host"), stripHost = el("div"), cexHost = el("div"), captionHost = el("div", "muted caption");
    box.append(el("h3", "", "the grid and its waves"), gridHost, el("h3", "", "the block pipeline"), pipeHost, el("h3", "", "resources"), stripHost, cexHost, captionHost);
    if (state.parent && sch.parent && !parent) parent = await S.api("schedule/" + sch.parent);
    function draw() {
      for (const b of toggles.children) b.classList.toggle("on", !!state[{ "show what was proved": "proved", "show the counterexample": "cex", "compare with parent": "parent", "all phases": "all" }[b.textContent]]);
      gridHost.innerHTML = ""; gridHost.append(gridPanel(sch));
      pipeHost.innerHTML = ""; pipeHost.append(pipeline(sch, state.parent ? parent : null));
      stripHost.innerHTML = ""; stripHost.append(strip(sch));
      cexHost.innerHTML = ""; if (state.cex && sch.counterexample) { cexHost.append(el("h3", "", "the counterexample, in the same lanes")); cexHost.append(counterexamplePanel(sch)); }
      captionHost.textContent = sch.axis.caption + (state.parent && parent ? ` · ${parent.short}: ${(parent.objective_us).toFixed(1)} µs → ${sch.short}: ${sch.objective_us.toFixed(1)} µs (${((sch.objective_us / parent.objective_us - 1) * 100).toFixed(1)} %)` : "");
      if (window.Glossary) Glossary.apply(captionHost);
    }
    draw();
    return sch;
  }

  // ------------------------------------------------------------------ grid and waves
  function gridPanel(sch) {
    const g = sch.grid;
    const W = 420, cell = Math.max(3, Math.min(14, Math.floor(W / Math.max(g.gn, 1))));
    const H = Math.min(260, cell * g.gm + 40);
    const svg = mk("svg", { viewBox: `0 0 ${W + 160} ${H + 10}`, class: "sched-svg" });
    const rowsShown = Math.min(g.gm, Math.floor((H - 30) / cell));
    const partial = new Set((g.partial || []).map(p => p.i + "," + p.j));
    for (let i = 0; i < rowsShown; i++) for (let j = 0; j < g.gn; j++) {
      const wave = Math.floor((i * g.gn + j) / g.sms);
      const r = mk("rect", { x: j * cell, y: 20 + i * cell, width: cell - 1, height: cell - 1, fill: wave % 2 ? "#CFDAD8" : "#B7C8C5", stroke: partial.has(i + "," + j) ? NAVY : "none", "stroke-dasharray": partial.has(i + "," + j) ? "2 1" : null }, svg);
      const rows = Math.min(g.tm, g.m - i * g.tm), cols = Math.min(g.tn, g.n - j * g.tn);
      mk("title", {}, r).textContent = `block (${i}, ${j}) · rows ${i * g.tm}–${i * g.tm + rows - 1}, cols ${j * g.tn}–${j * g.tn + cols - 1} · wave ${wave + 1}` + (partial.has(i + "," + j) ? " · partial tile" : "");
    }
    if (rowsShown < g.gm) txt(svg, 0, 20 + rowsShown * cell + 12, `… ${g.gm - rowsShown} more block rows`);
    for (const u of g.uncovered || []) {
      const r = u.dim === "M" ? mk("rect", { x: 0, y: 20 + Math.min(rowsShown, g.gm) * cell, width: g.gn * cell, height: 6, fill: RED }, svg)
                              : mk("rect", { x: g.gn * cell, y: 20, width: 6, height: Math.min(rowsShown, g.gm) * cell, fill: RED }, svg);
      mk("title", {}, r).textContent = `uncovered: ${u.dim} indices ${u.from}–${u.to - 1} (${u.count}) are never computed`;
    }
    txt(svg, 0, 12, `${g.gm} × ${g.gn} blocks of ${g.tm} × ${g.tn} outputs · ${g.waves} wave${g.waves !== 1 ? "s" : ""} of ${g.sms} SMs`, { fill: NAVY, "font-size": 12 });
    const lx = Math.min(W, g.gn * cell) + 14;
    txt(svg, lx, 40, "shaded by wave"); mk("rect", { x: lx, y: 46, width: 12, height: 10, fill: "#B7C8C5" }, svg); mk("rect", { x: lx + 14, y: 46, width: 12, height: 10, fill: "#CFDAD8" }, svg);
    txt(svg, lx, 74, "dashed = partial tile"); if (g.uncovered && g.uncovered.length) txt(svg, lx, 92, "red = uncovered tail", { fill: RED });
    return svg;
  }

  // ------------------------------------------------------------------ the pipeline
  function pipeline(sch, parent) {
    const b = sch.block, Sn = sch.design.S;
    const shown = state.all ? [...Array(sch.axis.phases).keys()] : sch.axis.shown;
    const laneNames = ["memory: DRAM → shared", ...Array.from({ length: Sn }, (_, i) => `slot ${i}`), "sync", "compute: MMA"];
    const laneH = 22, left = 130, top = 26, rowGap = 4;
    const H = top + laneNames.length * (laneH + rowGap) + 30;
    // time window: from the first shown phase's earliest event to the last shown phase's end; abridged view breaks the axis before the last phase
    const phaseEvents = p => ({ loads: b.loads.filter(l => l.issue_phase === p || (p === 0 && l.issue_phase === -1)), comp: b.computes[p], bars: b.barriers.filter(x => x.phase === p) });
    const segs = [];
    const contiguous = state.all ? shown : shown.filter((p, i) => i === 0 || shown[i - 1] === p - 1);
    const tail = state.all ? [] : shown.filter(p => !contiguous.includes(p));
    const t0 = 0, tEnd = Math.max(...contiguous.map(p => b.computes[p].t1), ...contiguous.flatMap(p => phaseEvents(p).loads.map(l => l.t1)));
    const W = 760, axisW = W - left - 20, scale = tail.length ? 0.82 : 1.0;
    const X = t => left + (t - t0) / (tEnd - t0) * axisW * scale;
    const svg = mk("svg", { viewBox: `0 0 ${W} ${H}`, class: "sched-svg" });
    laneNames.forEach((n, i) => { const y = top + i * (laneH + rowGap); mk("rect", { x: left, y, width: axisW, height: laneH, fill: i === 0 || i === laneNames.length - 1 ? "#FAFAF8" : LIGHT, stroke: LINE }, svg); txt(svg, 4, y + 15, n, { fill: NAVY, "font-size": 12 }); });
    const laneY = i => top + i * (laneH + rowGap);
    const ghostFill = "rgba(24,48,57,0.18)";
    const drawSet = (blk, ghost) => {
      const Xg = ghost ? (t => left + (t - t0) / (tEnd - t0) * axisW * scale) : X;
      // loads
      for (const l of blk.loads) { if (!(l.t1 >= t0 && l.t0 <= tEnd)) continue; const r = mk("rect", { x: Xg(l.t0), y: laneY(0) + 3, width: Math.max(1.5, Xg(l.t1) - Xg(l.t0)), height: laneH - 6, fill: ghost ? ghostFill : SLOT_COLORS[l.slot % SLOT_COLORS.length], opacity: ghost ? 1 : 0.9, rx: 2 }, svg); mk("title", {}, r).textContent = (ghost ? "parent: " : "") + `load k-tile ${l.tile} → slot ${l.slot} · ${us(l.t0)}–${us(l.t1)} · ${l.bytes} bytes`; if (!ghost && Xg(l.t1) - Xg(l.t0) > 26) txt(svg, Xg(l.t0) + 3, laneY(0) + 15, `k${l.tile}`, { fill: "#fff" }); }
      // slot occupancy
      for (const s of blk.slots) { if (s.slot >= Sn) continue; if (!(s.t_free >= t0 && s.t_ready <= tEnd)) continue; const r = mk("rect", { x: Xg(s.t_ready), y: laneY(1 + s.slot) + 3, width: Math.max(1.5, Xg(s.t_last_read) - Xg(s.t_ready)), height: laneH - 6, fill: ghost ? ghostFill : SLOT_COLORS[s.slot % SLOT_COLORS.length], opacity: ghost ? 1 : 0.55, rx: 2 }, svg); mk("title", {}, r).textContent = (ghost ? "parent: " : "") + `slot ${s.slot} holds k-tile ${s.tile} · ready ${us(s.t_ready)} · last read ${us(s.t_last_read)} (lifetime)`; if (!ghost && Xg(s.t_last_read) - Xg(s.t_ready) > 26) txt(svg, Xg(s.t_ready) + 3, laneY(1 + s.slot) + 15, `k${s.tile}`, { fill: NAVY }); }
      // barriers
      for (const x of blk.barriers) { if (!(x.t >= t0 && x.t <= tEnd)) continue; const l = mk("line", { x1: Xg(x.t), x2: Xg(x.t), y1: laneY(1 + Sn) + 2, y2: laneY(1 + Sn) + laneH - 2, stroke: ghost ? "rgba(197,74,56,0.35)" : RED, "stroke-width": x.kind === "trailing" ? 2.5 : 1.5, "stroke-dasharray": x.kind === "trailing" ? "3 2" : null }, svg); mk("title", {}, l).textContent = (ghost ? "parent: " : "") + `${x.kind} barrier, phase ${x.phase} · ${us(x.t)}`; }
      // computes
      for (const c of blk.computes) { if (!(c.t1 >= t0 && c.t0 <= tEnd)) continue; const r = mk("rect", { x: Xg(c.t0), y: laneY(laneNames.length - 1) + 3, width: Math.max(1.5, Xg(c.t1) - Xg(c.t0)), height: laneH - 6, fill: ghost ? ghostFill : NAVY, rx: 2, "data-tile": ghost ? null : c.tile }, svg); mk("title", {}, r).textContent = (ghost ? "parent: " : "") + `compute k-tile ${c.tile} from slot ${c.slot} · ${us(c.t0)}–${us(c.t1)} · ${(c.flops / 1e6).toFixed(2)} MFLOP`; if (!ghost && Xg(c.t1) - Xg(c.t0) > 26) txt(svg, Xg(c.t0) + 3, laneY(laneNames.length - 1) + 15, `k${c.tile}`, { fill: "#fff" }); }
    };
    if (parent && parent.block) drawSet(parent.block, true);
    drawSet(b, false);
    // proved overlay
    if (state.proved && sch.proved) {
      const ok = sch.proved.status;
      const bracket = (x1, x2, y, color, title) => { const g = mk("g", {}, svg); mk("path", { d: `M${x1},${y + 6} v-5 H${x2} v5`, fill: "none", stroke: color, "stroke-width": 1.5 }, g); mk("title", {}, g).textContent = title; };
      for (const w of sch.proved.readiness) { if (!(w.t1 >= t0 && w.t0 <= tEnd)) continue; bracket(X(w.t0), X(w.t1), laneY(laneNames.length - 1) - 3, ok.readiness === "proved" ? TEAL : RED, `readiness (${ok.readiness}): load of k-tile ${w.tile} finished at ${us(w.t0)}, read begins ${us(w.t1)}`); }
      for (const w of sch.proved.lifetime) { if (w.slot >= Sn || !(w.t1 >= t0 && w.t0 <= tEnd)) continue; bracket(X(w.t0), X(w.t1), laneY(1 + w.slot) - 3, ok.lifetime === "proved" ? TEAL : RED, `lifetime (${ok.lifetime}): slot ${w.slot} kept k-tile ${w.tile} from its last read ${us(w.t0)} until ${w.next_tile != null ? "the load of k-tile " + w.next_tile + " at " : ""}${us(w.t1)}`); }
      txt(svg, left, H - 6, `proved on this design: readiness ${ok.readiness}, lifetime ${ok.lifetime}, coverage ${ok.coverage}, exact-once ${ok.exact_once}, no-phantom ${ok.no_phantom}`, { fill: TEAL });
    }
    // axis
    const axY = top + laneNames.length * (laneH + rowGap) + 4;
    mk("line", { x1: left, x2: left + axisW * scale, y1: axY, y2: axY, stroke: LINE }, svg);
    for (const p of contiguous) { const x = X(b.computes[p].t0); txt(svg, x, axY + 12, `k${p}`, { "text-anchor": "middle" }); }
    txt(svg, left, axY + 24, `0 → ${us(tEnd)}` + (state.all ? "" : ` · phases ${contiguous[0]}–${contiguous[contiguous.length - 1]} of ${sch.axis.phases}`), {});
    if (tail.length) { const x = left + axisW * scale + 8; txt(svg, x, axY + 12, "…", { "font-size": 16 }); const p = tail[tail.length - 1], c = b.computes[p]; const r = mk("rect", { x: x + 16, y: laneY(laneNames.length - 1) + 3, width: Math.max(8, (c.t1 - c.t0) / (tEnd - t0) * axisW * scale), height: laneH - 6, fill: NAVY, rx: 2 }, svg); mk("title", {}, r).textContent = `compute k-tile ${c.tile} (last phase) · ${us(c.t0)}–${us(c.t1)}`; txt(svg, x + 16, axY + 12, `k${p}`); txt(svg, x + 16, axY + 24, `ends ${us(b.block_time)}`); }
    return svg;
  }

  // ------------------------------------------------------------------ the resource strip
  function strip(sch) {
    const r = sch.resources, W = 760, H = 86, left = 130, barW = W - left - 150;
    const svg = mk("svg", { viewBox: `0 0 ${W} ${H}`, class: "sched-svg" });
    const rows = [["DRAM bandwidth", r.dram.util, `${(r.dram.bytes / 1e6).toFixed(1)} MB · ${us(r.dram.s)}`, r.dominant === "DRAM"], ["compute", r.compute.util, us(r.compute.s), r.dominant === "compute"], ["shared memory", r.smem.util, `${(r.smem.bytes / 1024).toFixed(0)} KiB of ${(r.smem.cap / 1024).toFixed(0)} KiB`, false]];
    rows.forEach(([name, u, label, dom], i) => { const y = 6 + i * 26; txt(svg, 4, y + 13, name + (dom ? " · dominant" : ""), { fill: dom ? NAVY : MUTED, "font-weight": dom ? 700 : 400, "font-size": 12 }); mk("rect", { x: left, y, width: barW, height: 18, fill: LIGHT, stroke: LINE }, svg); const f = mk("rect", { x: left, y, width: Math.max(0, Math.min(1, u)) * barW, height: 18, fill: dom ? NAVY : TEAL, opacity: dom ? 1 : 0.75 }, svg); mk("title", {}, f).textContent = `${name}: ${(u * 100).toFixed(0)} %`; txt(svg, left + barW + 8, y + 13, `${(u * 100).toFixed(0)} % · ${label}`, { "font-size": 11 }); });
    return svg;
  }

  // ------------------------------------------------------------------ the counterexample
  function counterexamplePanel(sch) {
    const c = sch.counterexample;
    const box = el("div");
    if (c.kind === "index") {
      box.append(el("div", "bad", `${c.violation.obligation}: ${c.words}`));
      if (c.replay && c.replay.shape) box.append(el("div", "muted", `Replayed on shape ${JSON.stringify(c.replay.shape)}: ${c.replay.differences_total} output terms wrong. The red tail in the grid panel is the uncovered part.`));
      return box;
    }
    const Sn = c.event_model.S, steps = c.steps, n = steps.length;
    const lanes = ["memory (loads)", ...Array.from({ length: Sn }, (_, i) => `slot ${i}`), "sync", "compute (reads)"];
    const left = 130, colW = Math.max(26, Math.min(60, Math.floor(600 / Math.max(n, 1)))), laneH = 22, gap = 4, top = 22;
    const W = left + colW * n + 20, H = top + lanes.length * (laneH + gap) + 34;
    const svg = mk("svg", { viewBox: `0 0 ${W} ${H}`, class: "sched-svg" });
    const laneY = i => top + i * (laneH + gap);
    lanes.forEach((nm, i) => { mk("rect", { x: left, y: laneY(i), width: colW * n, height: laneH, fill: LIGHT, stroke: LINE }, svg); txt(svg, 4, laneY(i) + 15, nm, { fill: NAVY, "font-size": 12 }); });
    const pair = new Set((c.violation.pair || []).map(e => e.step));
    const X = s => left + s * colW;
    // slot occupancy from loads to the last read
    const lastRead = {}; for (const e of steps) if (e.kind === "compute") lastRead[e.slot + ":" + e.tile] = e.step;
    for (const e of steps) {
      const dim = pair.size && !pair.has(e.step);
      if (e.kind === "barrier") { mk("line", { x1: X(e.step) + colW / 2, x2: X(e.step) + colW / 2, y1: laneY(1 + Sn) + 2, y2: laneY(1 + Sn) + laneH - 2, stroke: RED, opacity: dim ? 0.35 : 1 }, svg); txt(svg, X(e.step) + colW / 2, laneY(1 + Sn) + 15, e.name, { "text-anchor": "middle", "font-size": 9, opacity: dim ? 0.5 : 1 }); continue; }
      const color = pair.has(e.step) ? RED : SLOT_COLORS[e.slot % SLOT_COLORS.length];
      if (e.kind === "load") {
        const r = mk("rect", { x: X(e.step) + 2, y: laneY(0) + 3, width: colW - 4, height: laneH - 6, fill: color, opacity: dim ? 0.3 : 0.95, rx: 2 }, svg);
        mk("title", {}, r).textContent = `warp ${e.warp}: LOAD k-tile ${e.tile} → slot ${e.slot}` + (pair.has(e.step) ? " ← the write that broke the read" : "");
        txt(svg, X(e.step) + 4, laneY(0) + 15, `w${e.warp} k${e.tile}`, { fill: "#fff", "font-size": 10 });
        const end = lastRead[e.slot + ":" + e.tile]; const x2 = end != null ? X(end) + colW - 2 : X(e.step) + colW - 2;
        const occ = mk("rect", { x: X(e.step) + 2, y: laneY(1 + e.slot) + 3, width: Math.max(4, x2 - X(e.step) - 2), height: laneH - 6, fill: color, opacity: dim ? 0.2 : 0.45, rx: 2 }, svg);
        mk("title", {}, occ).textContent = `slot ${e.slot} holds k-tile ${e.tile} (warp ${e.warp}'s part)`;
      } else if (e.kind === "compute") {
        const r = mk("rect", { x: X(e.step) + 2, y: laneY(lanes.length - 1) + 3, width: colW - 4, height: laneH - 6, fill: pair.has(e.step) ? RED : NAVY, opacity: dim ? 0.3 : 1, rx: 2 }, svg);
        mk("title", {}, r).textContent = `warp ${e.warp}: COMPUTE k-tile ${e.tile} reads slot ${e.slot}` + (e.stale ? " ← reads a slot already overwritten" : "");
        txt(svg, X(e.step) + 4, laneY(lanes.length - 1) + 15, `w${e.warp} k${e.tile}`, { fill: "#fff", "font-size": 10 });
      }
    }
    // the broken lifetime bracket: from the clobbered tile's load to its (stale) read, in the slot lane
    if (c.violation.pair && c.violation.pair.length === 2) {
      const [wr, rd] = c.violation.pair;
      const defn = steps.find(e => e.kind === "load" && e.slot === rd.slot && e.tile === rd.tile);
      if (defn) { const y = laneY(1 + rd.slot) - 2; mk("path", { d: `M${X(defn.step) + 2},${y + 5} v-5 H${X(rd.step) + colW - 2} v5`, fill: "none", stroke: RED, "stroke-width": 1.5, "stroke-dasharray": "4 2" }, svg); txt(svg, X(wr.step) + colW / 2, y - 4, "✕ lifetime broken here", { fill: RED, "text-anchor": "middle", "font-size": 10, "font-weight": 700 }); }
    }
    txt(svg, left, H - 8, `event model: ${c.event_model.warps} warps, ${c.event_model.phases} k-phases, S = ${Sn}; columns are the interleaving the checker found (${c.source})`, {});
    const pairText = c.violation.pair && c.violation.pair.length === 2 ? `warp ${c.violation.pair[0].warp}'s LOAD k-tile ${c.violation.pair[0].tile} → slot ${c.violation.pair[0].slot} lands inside warp ${c.violation.pair[1].warp}'s COMPUTE k-tile ${c.violation.pair[1].tile} read of slot ${c.violation.pair[1].slot}.` : "";
    box.append(svg, el("div", "bad", `${c.violation.obligation}: ${c.words}. ${pairText}`));
    return box;
  }
  return { render, state };
})();
