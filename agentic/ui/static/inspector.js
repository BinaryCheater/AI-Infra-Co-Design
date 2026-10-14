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

  function render() {
    const T = view.contract;
    document.getElementById("insp-short").textContent = desc.short;
    document.getElementById("insp-id").textContent = T.id;
    body.innerHTML = "";
    body.append(el("div", "headline-big", desc.headline));
    body.append(el("div", "delta", desc.delta_from_parent));
    const tabs = el("div", "tabs");
    for (const [key, label] of LENSES) { const b = el("button", key === lens ? "on" : "", label); b.onclick = () => { lens = key; remember(key); render(); }; tabs.append(b); }
    const help = el("span", "muted help", lens === "systems" ? "for the architect: the decision and its cost" : lens === "optimization" ? "for the mathematician: the problem the solver saw" : lens === "proof" ? "for the verifier: what was proved, in what scope, and what failed" : "how the design runs: loads, slot lifetimes, barriers and computes, as the solver chose and the verifier certified");
    tabs.append(help);
    body.append(tabs);
    if (lens === "schedule") {
      const host = el("div"); body.append(host);
      ScheduleView.render(host, T.id).then(() => { if (window.Glossary) Glossary.apply(host); });
    } else {
      const L = desc[lens];
      body.append(paras(L.text));
      if (lens === "systems") renderSystems(L);
      else if (lens === "optimization") renderOptimization(L);
      else renderProof(L);
    }
    if (desc.rationale) { const q = el("details", "quote"); q.append(el("summary", "muted", "the agent's own words")); q.append(el("blockquote", "", desc.rationale)); body.append(q); }
    const raw = el("details", "raw"); raw.append(el("summary", "", "contract JSON")); raw.append(el("pre", "", JSON.stringify(T, null, 2))); body.append(raw);
    renderActions(T);
    if (window.Glossary) Glossary.apply(body);
  }

  function renderSystems(L) {
    const rows = L.design_table.map(r => [r.name, r.meaning, r.value == null ? "–" : (r.name === "beta" ? (r.value ? "kept" : "dropped") : r.value + (r.unit ? " " + r.unit : "")),
      r.parent_value == null ? "–" : (r.name === "beta" ? (r.parent_value ? "kept" : "dropped") : r.parent_value + (r.unit ? " " + r.unit : ""))]);
    body.append(el("h3", "", "the design"), table(["decision", "meaning", "this template", L.parent_short ? "parent " + L.parent_short : "parent"], rows));
    if (L.measurement) {
      body.append(el("h3", "", "cost"));
      body.append(dl([["predicted", fmt(L.measurement.predicted) + " µs"], ["measured", fmt(L.measurement.measured) + " µs"], ["bottleneck", { compute: "compute-bound", DRAM: "memory-bound" }[L.measurement.dominant] || L.measurement.dominant]]));
      for (const [k, u] of Object.entries(L.measurement.utilizations)) { body.append(el("div", "muted", `${k} ${(u * 100).toFixed(0)}%`)); const bar = el("div", "bar" + (k === L.measurement.dominant ? " dom" : "")); const i = el("i"); i.style.width = Math.min(100, u * 100) + "%"; bar.append(i); body.append(bar); }
    }
    body.append(el("div", "muted", "chip: " + L.chip + " · structural choices: " + Object.entries(L.structural).map(([k, v]) => `${k}=${v}`).join(", ")));
  }

  function renderOptimization(L) {
    body.append(el("h3", "", "variables"), table(["name", "domain", "meaning"], L.variables.map(v => [v.name, "{" + v.domain.join(", ") + "}", v.meaning])));
    body.append(el("h3", "", "objective"), el("div", "mono", "minimize " + (L.objective.objective || "T") + " with " + L.objective.model));
    if (L.objective.terms) body.append(dl(Object.entries(L.objective.terms)));
    body.append(el("h3", "", "constraints"), table(["id", "expression", "meaning"], L.constraints.map(c => [c.id, el("span", "mono", c.expr), c.meaning]).concat([["Pre", el("span", "mono", L.precondition.expr), "lifetime precondition (" + L.precondition.mode + "): " + L.precondition.meaning]])));
    if (L.size) body.append(el("h3", "", "solver"), dl([["size", `${L.size.variables} variables (${L.size.binary} binary), ${L.size.constraints} constraints`], ["backend", L.solver.backend], ["status", L.solver.status], ["objective", fmt(L.solver.objective) + " µs"], ["dual bound", fmt(L.solver.bound) + " µs"], ["gap", (L.solver.gap * 100).toFixed(0) + " %"], ["time", t(L.solver.time)], ["nodes", L.solver.nodes], ["solution", Object.entries(L.values || {}).map(([k, y]) => `${k}=${y}`).join("  ")]]));
    else body.append(el("div", "muted", "The solver did not run."));
  }

  function renderProof(L) {
    const rows = [];
    if (L.family) rows.push(["family (K1–K3, all designs)", el("span", L.family.verdict === "proved" ? "ok" : "bad", L.family.verdict === "proved" ? (L.family.reused_from ? "proved · certificate reused from " + L.family.reused_from : "proved") : "refuted"), L.family.weakest == null ? "" : (L.family.weakest ? "weakest precondition" : "sound, stronger than needed"), t(L.family.time)]);
    if (L.instance) rows.push(["instance (this design)", el("span", L.instance.verdict === "proved" ? "ok" : "bad", L.instance.verdict), `${L.instance.proved}/${L.instance.n} obligations`, t(L.instance.time)]);
    if (rows.length) body.append(el("h3", "", "verdicts"), table(["check", "verdict", "scope note", "time"], rows));
    if (L.counterexample) {
      body.append(el("h3", "bad", "the bug, step by step · " + L.counterexample.obligation.split("[")[0]));
      const d = L.counterexample.design;
      body.append(el("div", "muted", `failing design: Tm=${d.tm} Tn=${d.tn} Tk=${d.tk} S=${d.stages} P=${d.prefetch} β=${d.trailing_barrier ? 1 : 0} grid=${d.grid_rounding}`));
      const rp = L.counterexample.replay || {};
      if (rp.trace) body.append(raceTimeline(rp.trace));
      else if (rp.differences_sample) {
        body.append(el("div", "", `Replayed on shape ${JSON.stringify(rp.shape)}: ${rp.differences_total} output terms wrong. First few:`));
        const ul = el("ul", "defects");
        for (const x of rp.differences_sample) ul.append(el("li", "", x.phantom_writes ? `${x.phantom_writes} phantom writes outside C (${x.reason})` : `C[${x.output.join(",")}] ${x.reason}: term A[${x.token[0]},${x.token[1]}]·B[${x.token[2]},${x.token[3]}] expected ${x.expected}, got ${x.actual}`));
        body.append(ul);
      }
    }
    if (L.diagnosis) body.append(el("h3", "", "what the orchestrator concluded"), dl([["symptom", L.diagnosis.symptom], ["cause", L.diagnosis.source], ["response", L.diagnosis.response]]));
    const obl = el("details"); obl.append(el("summary", "muted", `all obligations (${L.obligations.length})`));
    obl.append(table(["obligation", "kind", "scope", "status", "time", "reused"], L.obligations.map(o => [(o.point ? `S=${o.point.S} P=${o.point.P} β=${o.point.beta} ` : "") + o.name, o.kind, o.scope, el("span", /^(proved|agrees)/.test(o.status) ? "ok" : "bad", o.status), o.time == null ? "" : t(o.time), o.reused ? "yes" : ""])));
    body.append(obl);
    body.append(el("div", "muted", "relation: " + L.relation + " · rank: " + L.rank));
  }

  function raceTimeline(trace) {
    const steps = trace.map(line => {
      let m;
      if ((m = /^LOAD warp(\d+) k-tile(\d+) -> slot(\d+)/.exec(line))) return { kind: "load", warp: +m[1], text: `load k${m[2]}→slot${m[3]}` };
      if ((m = /^COMP warp(\d+) k-tile(\d+) reads slot(\d+)/.exec(line))) return { kind: "comp", warp: +m[1], text: `compute k${m[2]} ← slot${m[3]}`, stale: /STALE/.test(line) };
      if ((m = /^BARRIER (\w+)/.exec(line))) return { kind: "bar", text: "barrier " + m[1] };
      return { kind: "other", warp: 0, text: line };
    });
    const warps = [...new Set(steps.filter(s => s.warp != null).map(s => s.warp))].sort();
    const tbl = el("table", "race");
    const hdr = el("tr"); hdr.append(el("th", "", "step")); steps.forEach((_, i) => hdr.append(el("th", "", i + 1))); tbl.append(hdr);
    warps.forEach((w, wi) => {
      const tr = el("tr"); tr.append(el("td", "rowname", "warp " + w));
      steps.forEach(s => {
        if (s.kind === "bar") { if (wi === 0) { const td = el("td", "bar", s.text); td.rowSpan = warps.length; tr.append(td); } return; }
        tr.append(el("td", s.warp === w ? (s.stale ? "stale" : s.kind) : "", s.warp === w ? s.text + (s.stale ? " ⚠ stale" : "") : ""));
      });
      tbl.append(tr);
    });
    const wrap = el("div"); wrap.style.overflowX = "auto"; wrap.append(tbl);
    const box = el("div"); box.append(el("div", "muted", "Read left to right: each column is one step of the interleaving the checker found; a red cell is a read of a slot that was already overwritten (the race)."), wrap); return box;
  }

  function renderActions(T) {
    actions.innerHTML = "";
    const c = S.state.controller, w = c.waiting;
    const here = S.state.live && w && w.pending === T.id;
    if (here && w.checkpoint === "proposed") {
      const edit = el("button", "primary", "edit the proposal"), ins = el("button", "", "insert from pool");
      edit.onclick = () => editor(T); ins.onclick = () => insertFromPool(T);
      actions.append(edit, ins, el("span", "muted", "the verifier still has to certify it"));
    } else if (here && w.checkpoint === "regime_flip") {
      const stay = el("button", "primary", "stay"), deeper = el("button", "", "go deeper");
      stay.onclick = () => { S.send({ op: "override", target: "regime", decision: "keep", reason: "architect: stay" }); S.send({ op: "resume" }); toast("decided: stay"); };
      deeper.onclick = () => { S.send({ op: "override", target: "regime", decision: "descend", reason: "architect: go deeper" }); S.send({ op: "resume" }); toast("decided: go deeper (next level not yet available in v1)"); };
      actions.append(el("span", "", "The bottleneck moved. "), stay, deeper);
    } else if (S.state.live && w) actions.append(el("span", "muted", `the loop is paused at “${w.checkpoint}” of iteration ${w.iteration}; actions apply to the paused template`));
    actions.append(el("div", "rule", "Verdicts cannot be changed: correctness obligations gate, for humans too. Change the template instead."));
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
