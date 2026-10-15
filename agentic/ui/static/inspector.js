/* The inspector (round 3): the shared headline, then three lenses — Systems · Optimization · Proof —
   over the describer's output (/api/describe/<id>). The default lens follows the template's status
   (Systems when accepted, Proof when rejected: the question each status raises first); a lens the
   user picks overrides that for the rest of the browser session. */
window.Inspector = (() => {
  const S = Session;
  const pane = document.getElementById("inspector"), body = document.getElementById("insp-body"), actions = document.getElementById("insp-actions");
  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
  const fmt = (x, d = 2) => x == null ? "–" : Number(x).toFixed(d);
  const t = s => s == null ? "?" : s < 1e-3 ? (s * 1e6).toFixed(0) + " µs" : s < 1 ? (s * 1e3).toFixed(0) + " ms" : s.toFixed(2) + " s";
  const LENSES = [["systems", "Systems"], ["optimization", "Optimization"], ["proof", "Proof"], ["schedule", "Schedule"]];
  let current = null, lens = null, view = null, desc = null;
  const stored = () => { try { return sessionStorage.getItem("agentic.lens"); } catch (e) { return null; } };
  const remember = l => { try { sessionStorage.setItem("agentic.lens", l); } catch (e) { /* ignore */ } };

  async function open(id, iteration, keepLens, wantLens) {
    current = { id, iteration };
    S.state.selected = id;
    [view, desc] = await Promise.all([S.api("template/" + id), S.api("describe/" + id)]);
    if (wantLens) lens = wantLens;                 // e.g. from a Solver or Gate badge on the map
    else if (!keepLens) lens = stored() || (desc.status === "rejected" ? "proof" : "systems");
    render();
    pane.hidden = false;
  }
  function close() { pane.hidden = true; current = null; }
  document.getElementById("insp-close").onclick = close;

  function dl(pairs) { const d = el("dl"); for (const [k, val] of pairs) { d.append(el("dt", "", k)); const dd = el("dd"); if (val && val.nodeType) dd.append(val); else dd.textContent = val; d.append(dd); } return d; }
  function paras(text, cls) { const box = el("div", cls || "lens-text"); for (const p of (text || "").split("\n\n")) box.append(el("p", "", p)); return box; }
  function table(cols, rows) { const tb = el("table", "grid"); const hr = el("tr"); for (const c of cols) hr.append(el("th", "", c)); tb.append(hr); for (const r of rows) { const tr = el("tr"); for (const c of r) { const td = el("td"); if (c && c.nodeType) td.append(c); else td.textContent = c == null ? "–" : c; tr.append(td); } tb.append(tr); } return tb; }

  const HELP = {
    systems: "What the template does, what was chosen, and how fast it runs",
    optimization: "The choices, the goal and the rules the optimizer works with",
    proof: "Is it correct, and how we know",
    schedule: "How one block runs over time",
  };

  function render() {
    const T = view.contract;
    document.getElementById("insp-short").textContent = desc.short;
    document.getElementById("insp-id").textContent = "";
    body.innerHTML = "";
    body.append(el("div", "headline-big", desc.headline));
    body.append(el("div", "delta", desc.delta_from_parent));
    const tabs = el("div", "tabs");
    for (const [key, label] of LENSES) { const b = el("button", key === lens ? "on" : "", label); b.onclick = () => { lens = key; remember(key); render(); }; tabs.append(b); }
    body.append(tabs);
    body.append(el("div", "lens-help", HELP[lens]));
    if (lens === "schedule") {
      const host = el("div"); body.append(host);
      ScheduleView.render(host, T.id).then(() => { if (window.Glossary) Glossary.apply(host); });
    } else {
      const L = desc[lens];
      for (const sec of L.sections || []) {
        body.append(renderSection(sec));
        if (lens === "proof" && L.counterexample && sec.title === "What the checker did") renderBug(L.counterexample);   // the bug right after the verdict
      }
    }
    if (desc.rationale) { const q = el("details", "quote"); q.append(el("summary", "muted", "Why the proposer suggested it, in its own words")); q.append(el("blockquote", "", desc.rationale)); body.append(q); }
    const raw = el("details", "raw"); raw.append(el("summary", "", `The template as stored (internal id ${T.id})`)); raw.append(el("pre", "", JSON.stringify(T, null, 2))); body.append(raw);
    renderActions(T);
    if (window.Glossary) Glossary.apply(body);
  }

  function renderSection(sec) {
    const box = el("section", "lens-sec");
    if (sec.title) box.append(el("h3", "", sec.title));
    for (const p of sec.paras || []) box.append(el("p", "", p));
    if (sec.items && sec.items.length) { const ul = el("ul"); for (const it of sec.items) ul.append(el("li", /^✗/.test(it) ? "bad" : (/^✓/.test(it) ? "ok-line" : ""), it)); box.append(ul); }
    if (sec.table) box.append(table(sec.table.cols, sec.table.rows));
    if (sec.bars && sec.bars.length) {
      for (const b of sec.bars) {
        box.append(el("div", "bar-label" + (b.main ? " main" : ""), `${b.label}: busy ${(b.value * 100).toFixed(0)} % of the time` + (b.main ? " (the limit)" : "")));
        const bar = el("div", "bar" + (b.main ? " dom" : "")); const i = el("i"); i.style.width = Math.min(100, b.value * 100) + "%"; bar.append(i); box.append(bar);
      }
    }
    if (sec.details && sec.details.items && sec.details.items.length) {
      const d = el("details", "spec"); d.append(el("summary", "muted", sec.details.summary));
      const ul = el("ul"); for (const it of sec.details.items) ul.append(el("li", "mono", it)); d.append(ul); box.append(d);
    }
    return box;
  }

  function renderBug(cex) {
    const box = el("section", "lens-sec");
    box.append(el("h3", "bad", "The bug, step by step"));
    const rp = cex.replay || {};
    if (rp.trace) box.append(raceTimeline(rp.trace));
    else if (rp.differences_sample) {
      box.append(el("p", "", `The checker ran the design on a small ${(rp.shape || []).join(" × ")} case and compared it with A × B. ${rp.differences_total} values of C came out wrong. The first few:`));
      const ul = el("ul", "defects");
      for (const x of rp.differences_sample) ul.append(el("li", "", x.phantom_writes ? `${x.phantom_writes} values were written outside C (${x.reason}).` : `C[${x.output.join(", ")}] is wrong (${x.reason}): it should include A[${x.token[0]}, ${x.token[1]}] × B[${x.token[2]}, ${x.token[3]}]; expected ${x.expected}, got ${x.actual}.`));
      box.append(ul);
    }
    body.append(box);
  }

  function raceTimeline(trace) {
    const steps = trace.map(line => {
      let m;
      if ((m = /^LOAD warp(\d+) k-tile(\d+) -> slot(\d+)/.exec(line))) return { kind: "load", warp: +m[1], text: `loads tile ${m[2]} into buffer ${m[3]}` };
      if ((m = /^COMP warp(\d+) k-tile(\d+) reads slot(\d+)/.exec(line))) return { kind: "comp", warp: +m[1], text: `uses tile ${m[2]} from buffer ${m[3]}`, stale: /STALE/.test(line) };
      if ((m = /^BARRIER (\w+)/.exec(line))) return { kind: "bar", text: "all wait (" + m[1] + ")" };
      return { kind: "other", warp: 0, text: line };
    });
    const warps = [...new Set(steps.filter(s => s.warp != null).map(s => s.warp))].sort();
    const tbl = el("table", "race");
    const hdr = el("tr"); hdr.append(el("th", "", "step")); steps.forEach((_, i) => hdr.append(el("th", "", i + 1))); tbl.append(hdr);
    warps.forEach((w, wi) => {
      const tr = el("tr"); tr.append(el("td", "rowname", "thread group " + w));
      steps.forEach(s => {
        if (s.kind === "bar") { if (wi === 0) { const td = el("td", "bar", s.text); td.rowSpan = warps.length; tr.append(td); } return; }
        tr.append(el("td", s.warp === w ? (s.stale ? "stale" : s.kind) : "", s.warp === w ? s.text + (s.stale ? " ⚠ but that tile was already overwritten" : "") : ""));
      });
      tbl.append(tr);
    });
    const wrap = el("div"); wrap.style.overflowX = "auto"; wrap.append(tbl);
    const box = el("div"); box.append(el("p", "", "Read left to right. Each column is one step, in the order the checker found. Two groups of threads take turns; the red cell is the moment one of them uses a buffer that the other has already filled with the next tile. That is the bug."), wrap); return box;
  }

  function renderActions(T) {
    actions.innerHTML = "";
    const c = S.state.controller, w = c.waiting;
    const here = S.state.live && w && w.pending === T.id;
    if (here && w.checkpoint === "proposed") {
      const edit = el("button", "primary", "Edit the proposal"), ins = el("button", "", "Use one from the pool instead");
      edit.onclick = () => editor(T); ins.onclick = () => insertFromPool(T);
      actions.append(edit, ins, el("div", "rule", "You can change this proposal, or swap in one from the pool, before the checker sees it. You cannot overrule the checker: if it finds a bug, the fix is to change the template."));
    } else if (here && w.checkpoint === "regime_flip") {
      const stay = el("button", "primary", "stay"), deeper = el("button", "", "go deeper");
      stay.onclick = () => { S.send({ op: "override", target: "regime", decision: "keep", reason: "architect: stay" }); S.send({ op: "resume" }); toast("decided: stay"); };
      deeper.onclick = () => { S.send({ op: "override", target: "regime", decision: "descend", reason: "architect: go deeper" }); S.send({ op: "resume" }); toast("decided: go deeper (next level not yet available in v1)"); };
      actions.append(el("div", "", "What limits the design is not what the template assumed. Stay at this level, or go one level deeper?"), stay, deeper);
    } else if (S.state.live && w) actions.append(el("div", "muted", `The run is paused ${({ proposed: "after the proposal", solved: "after the solver", admitted: "after the template was accepted", refuted: "after the checker found a bug", regime_flip: "because the bottleneck moved" })[w.checkpoint] || "at “" + w.checkpoint + "”"} in step ${w.iteration}.`));
  }
  function editor(T) {
    actions.innerHTML = "";
    const ta = el("textarea"); ta.value = JSON.stringify(T, null, 2);
    const reason = document.createElement("input"); reason.placeholder = "why (recorded in the ledger)"; reason.style.width = "60%";
    const ok = el("button", "primary", "submit edit"), cancel = el("button", "", "cancel");
    ok.onclick = () => { try { const contract = JSON.parse(ta.value); S.send({ op: "edit", contract, reason: reason.value }); toast("edit sent; press Continue when ready"); open(T.id, current.iteration, true); } catch (e) { toast("invalid JSON: " + e.message, true); } };
    cancel.onclick = () => open(T.id, current.iteration, true);
    actions.append(ta, reason, ok, cancel);
  }
  function insertFromPool(T) {
    const pop = (S.state.summary.population || []);
    if (!pop.length) return toast("the pool has no accepted templates yet", true);
    actions.innerHTML = "";
    const sel = document.createElement("select");
    for (const p of pop) { const o = document.createElement("option"); o.value = p.id; const d = (S.state.describe || {})[p.id]; o.textContent = d ? d.headline : p.id; sel.append(o); }
    const ok = el("button", "primary", "insert"), cancel = el("button", "", "cancel");
    ok.onclick = () => { S.send({ op: "inject", template: sel.value, reason: "architect: insert from pool" }); toast("inserted; press Continue when ready"); open(T.id, current.iteration, true); };
    cancel.onclick = () => open(T.id, current.iteration, true);
    actions.append(sel, ok, cancel);
  }
  function toast(msg, bad) { S.emit("toast", msg, bad); }
  S.on("controller", () => { if (current && !pane.hidden) open(current.id, current.iteration, true); });
  return { open, close, lens: () => lens };
})();
