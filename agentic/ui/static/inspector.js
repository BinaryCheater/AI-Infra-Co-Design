/* The inspector: six sections over template_view(), in the order of the architect's questions, plus
   the actions available while the loop is paused on this template. */
window.Inspector = (() => {
  const S = Session;
  const pane = document.getElementById("inspector"), body = document.getElementById("insp-body"), actions = document.getElementById("insp-actions");
  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
  const fmt = (x, d = 2) => x == null ? "–" : Number(x).toFixed(d);
  const t = s => s == null ? "?" : s < 1e-3 ? (s * 1e6).toFixed(0) + " µs" : s < 1 ? (s * 1e3).toFixed(0) + " ms" : s.toFixed(2) + " s";
  const REGIME = { compute: "compute-bound", DRAM: "memory-bound" };
  let current = null, catalog = null;

  async function open(id, iteration) {
    current = { id, iteration };
    S.state.selected = id;
    const v = await S.api("template/" + id);
    if (!catalog) catalog = await (await fetch("/api/catalog")).json();
    render(v);
    pane.hidden = false;
  }
  function close() { pane.hidden = true; current = null; }
  document.getElementById("insp-close").onclick = close;

  function dl(pairs) { const d = el("dl"); for (const [k, val] of pairs) { d.append(el("dt", "", k)); const dd = el("dd"); if (val && val.nodeType) dd.append(val); else dd.textContent = val; d.append(dd); } return d; }
  const entryFor = id => S.state.story.find(e => e.template && e.template.id === id) || S.state.storyByIt[current && current.iteration];

  function render(v) {
    const T = v.contract, s = T.structure, e = entryFor(T.id);
    document.getElementById("insp-short").textContent = (e && e.template ? e.template.short : "T?") + " · " + (e && e.template ? e.template.workload : T.scope.workload.name);
    document.getElementById("insp-id").textContent = T.id;
    body.innerHTML = "";
    // 1. why
    body.append(el("h3", "", "1 · Why"));
    const cat = catalog.find(c => c.name === s.name);
    const strength = (v.admitted && v.admitted.strength) || s.strength;
    const why = el("div");
    why.append(el("div", "", `Assumed ${s.regime.split(":")[0].trim()}. Move: ${s.move}. Obligation: ${s.obligation}. Strength: ${strength === "proved" ? "proven (blocks: may prune)" : "assumed (guides: may reorder, not prune)"}.`));
    if (cat) { const d = el("details"); d.append(el("summary", "muted", `structure “${s.name}” (${s.bucket}) — what the catalog says`)); d.append(el("div", "", `Detect: ${cat.detect}. Move: ${cat.move}. Obligation: ${cat.obligation}. Usually ${cat.default_strength}; verification ${cat.verification_cost}.`)); why.append(d); }
    if (T.rationale) why.append(el("div", "muted", "The agent's own words: " + T.rationale));
    body.append(why);
    // 2. decided
    body.append(el("h3", "", "2 · What was decided"));
    const x = v.assignments[v.assignments.length - 1];
    if (!x) body.append(el("div", "muted", "Nothing: the solver never ran (the template was rejected before it)."));
    else {
      const inc = Object.values(S.state.summary.incumbents || {}).find(i => i.key.startsWith(T.scope.workload.name + "|"));
      body.append(dl([["design", Object.entries(x.values).map(([k, y]) => `${k}=${y}`).join("  ")], ["cost", fmt(x.objective) + " µs"], ["bound", fmt(x.bound) + " µs"], ["gap", (x.gap * 100).toFixed(0) + "%"], ["solver time", t(x.solver_time)],
        ["best so far", inc ? `${fmt(inc.cost)} µs` + (inc.contract === T.id ? " (this template)" : ` (${(x.objective / inc.cost * 100 - 100).toFixed(2)}% vs this)`) : "–"]]));
    }
    // 3. safe
    body.append(el("h3", "", "3 · Was it safe"));
    const fam = v.verdicts.filter(w => w.check === "family").pop(), inst = v.verdicts.filter(w => w.check === "instance").pop();
    const vt = w => { const n = el("span", w.verdict === "proved" ? "ok" : "bad"); n.textContent = w.verdict === "proved" ? (w.provenance.reused_from ? "proof reused" : "proved safe for all designs") : w.verdict === "open" ? "could not decide" : "rejected: bug found"; return n; };
    const rows = [];
    if (fam) { const n = el("span"); n.append(vt(fam), ` · ${t(fam.checker_time)}`); if (fam.provenance.precondition_is_weakest != null) n.append(fam.provenance.precondition_is_weakest ? " · precondition is the weakest on the family" : " · precondition is safe but stronger than needed"); rows.push(["for all designs", n]); }
    if (inst) { const n = el("span"); n.append(vt(inst).textContent === "proved safe for all designs" ? el("span", "ok", "this design checked") : vt(inst), ` · ${t(inst.checker_time)}`); rows.push(["this design", n]); }
    if (!rows.length) rows.push(["verifier", "not reached"]);
    body.append(dl(rows));
    const cex = (inst && inst.counterexample) || (fam && fam.counterexample);
    if (cex) {
      body.append(el("div", "bad", "The bug, step by step · " + cex.obligation.split("[")[0]));
      body.append(el("div", "muted", "failing design: " + Object.entries(cex.design).filter(([k]) => ["tm", "tn", "tk", "stages", "prefetch", "trailing_barrier", "grid_rounding"].includes(k)).map(([k, y]) => `${k}=${y}`).join(" ")));
      const rp = cex.replay || {};
      if (rp.trace) body.append(raceTimeline(rp.trace));
      else if (rp.differences_sample) {
        body.append(el("div", "", `Replayed on shape ${JSON.stringify(rp.shape)}: ${rp.differences_total} output terms wrong. First few:`));
        const ul = el("ul", "defects");
        for (const d of rp.differences_sample) ul.append(el("li", "", d.phantom_writes ? `${d.phantom_writes} phantom writes outside C (${d.reason})` : `C[${d.output.join(",")}] ${d.reason}: term A[${d.token[0]},${d.token[1]}]·B[${d.token[2]},${d.token[3]}] expected ${d.expected}, got ${d.actual}`));
        body.append(ul);
      }
    }
    // 4. measured
    body.append(el("h3", "", "4 · What was measured"));
    const m = v.measurements[v.measurements.length - 1];
    if (!m) body.append(el("div", "muted", "Not measured."));
    else {
      body.append(dl([["predicted", fmt(m.predicted_cost) + " µs"], ["measured", fmt(m.measured_cost) + " µs"], ["bottleneck", REGIME[m.dominant_resource] || m.dominant_resource]]));
      for (const [k, u] of Object.entries(m.utilizations)) { body.append(el("div", "muted", `${k} ${(u * 100).toFixed(0)}%`)); const bar = el("div", "bar" + (k === m.dominant_resource ? " dom" : "")); const i = el("i"); i.style.width = Math.min(100, u * 100) + "%"; bar.append(i); body.append(bar); }
      body.append(el("div", "muted", m.simulator));
    }
    // 5. concluded
    body.append(el("h3", "", "5 · What the orchestrator concluded"));
    if (!v.diagnoses.length) body.append(el("div", "muted", v.admitted ? "Accepted into the pool." : "No diagnosis."));
    for (const d of v.diagnoses) body.append(dl([["symptom", d.symptom], ["cause", d.source], ["response", d.response]]));
    // 6. raw
    body.append(el("h3", "", "6 · Raw"));
    const raw = el("details", "raw"); raw.append(el("summary", "", "contract JSON")); const pre = el("pre", "", JSON.stringify(T, null, 2)); raw.append(pre); body.append(raw);
    renderActions(T, v);
  }

  function raceTimeline(trace) {
    // two-warp timeline from the replayed interleaving: columns are steps, barriers span both warps
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
        const td = el("td", s.warp === w ? (s.stale ? "stale" : s.kind) : "", s.warp === w ? s.text + (s.stale ? " ⚠ stale" : "") : "");
        tr.append(td);
      });
      tbl.append(tr);
    });
    const wrap = el("div"); wrap.style.overflowX = "auto"; wrap.append(tbl);
    const note = el("div", "muted", "Read left to right: each column is one step of the interleaving the checker found; a red cell is a read of a slot that was already overwritten (the race).");
    const box = el("div"); box.append(note, wrap); return box;
  }

  function renderActions(T, v) {
    actions.innerHTML = "";
    const c = S.state.controller, w = c.waiting;
    const here = S.state.live && w && w.pending === T.id;
    if (here && w.checkpoint === "proposed") {
      const edit = el("button", "primary", "edit the proposal"), ins = el("button", "", "insert from pool");
      edit.onclick = () => editor(T);
      ins.onclick = () => insertFromPool(T);
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
    ok.onclick = () => { try { const contract = JSON.parse(ta.value); S.send({ op: "edit", contract, reason: reason.value }); toast("edit sent; press Continue when ready"); open(T.id, current.iteration); } catch (e) { toast("invalid JSON: " + e.message, true); } };
    cancel.onclick = () => open(T.id, current.iteration);
    actions.append(ta, reason, ok, cancel);
  }
  function insertFromPool(T) {
    const pop = (S.state.summary.population || []);
    if (!pop.length) return toast("the pool has no accepted templates yet", true);
    actions.innerHTML = "";
    const sel = document.createElement("select");
    for (const p of pop) { const o = document.createElement("option"); o.value = p.id; const e = S.state.story.find(x => x.template && x.template.id === p.id); o.textContent = `${e && e.template ? e.template.short : p.id} · ${p.refines} · ${p.workload}`; sel.append(o); }
    const ok = el("button", "primary", "insert"), cancel = el("button", "", "cancel");
    ok.onclick = () => { S.send({ op: "inject", template: sel.value, reason: "architect: insert from pool" }); toast("inserted; press Continue when ready"); open(T.id, current.iteration); };
    cancel.onclick = () => open(T.id, current.iteration);
    actions.append(sel, ok, cancel);
  }
  function toast(msg, bad) { S.emit("toast", msg, bad); }
  S.on("controller", () => { if (current && !pane.hidden) open(current.id, current.iteration); });
  return { open, close };
})();
