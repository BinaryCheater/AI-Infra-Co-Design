/* The architect console. Everything shown comes from the ledger (via /api) and the controller
   snapshot; commands go over the WebSocket. Vanilla JS, no build step. */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v; else if (k === "html") n.innerHTML = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v); else if (v != null) n.setAttribute(k, v);
    }
    for (const k of kids.flat()) if (k != null) n.append(k.nodeType ? k : document.createTextNode(String(k)));
    return n;
  };
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const fmt = (x, d = 2) => x == null || Number.isNaN(x) ? "–" : (typeof x === "number" ? x.toFixed(d) : String(x));
  const pct = x => x == null ? "–" : (x * 100).toFixed(x === 0 ? 0 : 2) + "%";

  const state = { run: null, runs: [], controller: {}, timeline: [], summary: {}, events: [], selected: null,
                  view: null, catalog: [] };
  let ws, refreshTimer;

  // ------------------------------------------------------------------ transport
  function connect() {
    ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
    ws.onopen = () => setStatus("connected");
    ws.onclose = () => { setStatus("disconnected · retrying", "error"); setTimeout(connect, 1500); };
    ws.onmessage = m => {
      const msg = JSON.parse(m.data);
      if (msg.type === "hello") { applyState(msg.state); loadEvents(); }
      else if (msg.type === "event") { if (!state.run || msg.event.run === state.run) { state.events.push(msg.event); appendLedger(msg.event); } scheduleRefresh(); }
      else if (msg.type === "controller") { state.controller = msg.controller; renderStatus(); renderActions(); }
      else if (msg.type === "error") toast("error: " + msg.error);
      else if (msg.type === "ok") { state.controller = msg.controller; renderStatus(); renderActions(); }
    };
  }
  const send = msg => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(msg)); else toast("not connected"); };
  const api = async (path) => (await fetch("/api/" + path + (path.includes("?") ? "&" : "?") + "run=" + encodeURIComponent(state.run || ""))).json();

  function scheduleRefresh() { clearTimeout(refreshTimer); refreshTimer = setTimeout(refresh, 120); }
  async function refresh() {
    const s = await api("state");
    applyState(s);
    if (state.selected) selectContract(state.selected, true);
  }
  function applyState(s) {
    const runChanged = s.run !== state.run;
    Object.assign(state, { run: s.run, runs: s.runs, controller: s.controller, timeline: s.timeline, summary: s.summary });
    renderRuns(); renderStatus(); renderTimeline(); renderStores(); renderActions();
    if (runChanged) loadEvents();
    // an edit or inject replaced the selected proposal: follow it to the new pending contract
    if (state.selected && !state.timeline.some(r => r.contract === state.selected)) {
      const row = state.timeline.find(r => r.human.some(h => h.from === state.selected || h.replaced === state.selected));
      if (row && row.contract) selectContract(row.contract, true);
    }
  }
  async function loadEvents() {
    if (!state.run) return;
    state.events = await api("events");
    $("#ledger").innerHTML = "";
    state.events.forEach(appendLedger);
  }
  async function loadCatalog() { state.catalog = await (await fetch("/api/catalog")).json(); }

  // ------------------------------------------------------------------ header
  function renderRuns() {
    const sel = $("#run-select");
    sel.innerHTML = "";
    for (const r of state.runs) sel.append(el("option", { value: r, selected: r === state.run ? "" : null }, r));
    if (!state.runs.length) sel.append(el("option", {}, "(no runs yet)"));
  }
  function setStatus(text, cls = "") { const s = $("#status"); s.textContent = text; s.className = "status " + cls; }
  function renderStatus() {
    const c = state.controller || {};
    for (const b of $("#mode-seg").children) b.classList.toggle("on", b.dataset.mode === c.mode);
    $("#btn-start").disabled = !!c.live;
    $("#btn-stop").disabled = !c.live;
    $("#btn-pause").disabled = !c.live || c.paused;
    $("#btn-step").disabled = !c.live;
    $("#btn-replay").disabled = !c.live;
    $("#btn-step").textContent = c.mode === "observe" && !c.paused ? "running ▸" : "continue ▸";
    if (c.error) setStatus("run error: " + c.error, "error");
    else if (c.waiting) setStatus(`waiting at ${c.waiting.checkpoint} · iteration ${c.waiting.iteration} · checkpoint #${c.waiting.seq}` + (c.paused ? " · paused" : ""), "waiting");
    else if (c.live) setStatus("live · " + c.mode + (c.paused ? " · pausing at next checkpoint" : ""), "live");
    else if (c.finished) setStatus("run finished");
    else setStatus(state.run ? "idle · " + state.run : "no run");
    for (const tr of $("#timeline tbody").children) tr.classList.toggle("waiting", !!(c.waiting && +tr.dataset.it === c.waiting.iteration));
  }

  // ------------------------------------------------------------------ timeline
  function gateTag(v) {
    if (!v) return el("span", { class: "tag neutral" }, "–");
    return el("span", { class: "tag " + (v === "proved" ? "ok" : v === "open" ? "open" : "bad") }, v);
  }
  function renderTimeline() {
    const tb = $("#timeline tbody");
    tb.innerHTML = "";
    for (const r of state.timeline) {
      const strength = r.strength_final || r.strength;
      const regime = el("span", {}, r.regime_assumed || "–");
      if (r.flip) regime.append(" ", el("span", { class: "tag flip" }, "flip → " + r.regime_actual));
      const outcome = el("span", {}, r.outcome);
      if (r.human.length) outcome.append(" ", el("span", { class: "tag human", title: r.human.map(h => h.kind).join(", ") }, "human ×" + r.human.length));
      if (r.replayed) outcome.append(" ", el("span", { class: "tag neutral" }, "replayed"));
      const measured = r.measured == null ? "–" : `${fmt(r.measured)} (${r.discrepancy >= 0 ? "+" : ""}${pct(r.discrepancy)})`;
      const tr = el("tr", { "data-it": r.iteration, "data-contract": r.contract || "", onclick: () => r.contract && selectContract(r.contract) },
        el("td", { class: "num" }, r.iteration),
        el("td", { class: "num" }, r.level ?? "–"),
        el("td", {}, r.refines || "–"),
        el("td", { class: "mono" }, r.workload || "–"),
        el("td", {}, strength ? el("span", { class: "mk " + strength, title: strength }) : null, r.structure ? `${r.structure} (${r.bucket})` : "–"),
        el("td", {}, regime),
        el("td", { class: "num" }, r.bound == null ? "–" : `${fmt(r.bound)} / ${pct(r.gap)} / ${fmt(r.solver_time, 3)} s`),
        el("td", {}, gateTag(r.family_verdict), " ", gateTag(r.instance_verdict)),
        el("td", { class: "num" }, measured),
        el("td", {}, outcome));
      if (r.contract === state.selected) tr.classList.add("sel");
      tb.append(tr);
    }
    renderStatus();
  }

  // ------------------------------------------------------------------ template view
  async function selectContract(id, silent) {
    state.selected = id;
    for (const tr of $("#timeline tbody").children) tr.classList.toggle("sel", tr.dataset.contract === id);
    const v = await api("template/" + id);
    state.view = v;
    renderTemplate(v);
    if (!silent) $("#template-pane").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function dl(pairs) {
    const d = el("dl");
    for (const [k, v] of pairs) d.append(el("dt", {}, k), el("dd", {}, v));
    return d;
  }
  function renderTemplate(v) {
    const T = v.contract, s = T.structure;
    $("#template-pane").classList.remove("empty");
    $("#tv-empty").hidden = true; $("#tv-body").hidden = false;
    $("#tv-id").textContent = T.id;
    const adm = v.admitted;
    $("#tv-status").textContent = adm ? `admitted · certificate ${adm.certificate}` : (v.verdicts.some(x => x.verdict === "refuted") ? "refuted" : "proposed");
    $("#why-name").textContent = s.name; $("#why-bucket").textContent = s.bucket; $("#why-regime").textContent = s.regime;
    $("#why-move").textContent = s.move; $("#why-obl").textContent = s.obligation;
    const strengthNow = adm?.strength || s.strength;
    $("#why-strength").innerHTML = `<span class="mk ${strengthNow}"></span>${strengthNow.toUpperCase()}` + (strengthNow !== s.strength ? ` <span class="dim">(proposed as ${s.strength})</span>` : "");
    $("#why-gate").textContent = strengthNow === "proved" ? "gate (may prune)" : "steer (may reorder, not prune)";
    $("#why-rationale").textContent = T.rationale || "(none)";
    $("#why-evidence").textContent = (s.evidence || []).length ? "[" + s.evidence.join(", ") + "]" : "";

    // formulation face
    const ff = $("#face-form"); ff.innerHTML = "";
    const decisions = el("table", {}, el("tr", {}, el("th", {}, "X"), el("th", {}, "domain"), el("th", {}, "meaning")));
    for (const [k, d] of Object.entries(T.decisions)) decisions.append(el("tr", {}, el("td", { class: "mono" }, k), el("td", { class: "mono" }, "{" + d.domain.join(", ") + "}"), el("td", {}, d.meaning || "")));
    const cons = el("table", {}, el("tr", {}, el("th", {}, "C"), el("th", {}, "expr"), el("th", {}, "")));
    for (const c of T.constraints) cons.append(el("tr", {}, el("td", { class: "mono" }, c.id), el("td", { class: "mono" }, c.expr), el("td", { class: "dim" }, c.meaning || "")));
    ff.append(dl([["level", `${T.level} · ${T.object}`], ["template", T.template], ["scope", `${T.scope.workload.name} ${T.scope.workload.m}×${T.scope.workload.n}×${T.scope.workload.k} · ${T.scope.chip.name}`],
      ["structural", Object.entries(T.scope.structure).map(([k, x]) => `${k}=${x}`).join(", ")]]),
      el("h3", {}, "decisions X"), decisions, el("h3", {}, "constraints C"), cons,
      el("h3", {}, "cost Ĵ"), el("div", { class: "mono" }, T.cost.model), el("h3", {}, "precondition Pre"),
      el("div", { class: "mono" }, T.preconditions.map(p => `${p.id}: ${p.expr}  [${p.mode}]`).join("\n")),
      el("h3", {}, "lineage"), el("div", { class: "mono dim" }, v.lineage.length ? v.lineage.map(l => `${l.id} (${l.refines || "?"}, ${l.workload})`).join(" → ") : "root (no parent)"),
      el("div", { class: "dim" }, "provenance: " + Object.entries(T.provenance).filter(([, x]) => x != null).map(([k, x]) => `${k}=${x}`).join(" · ")));

    // optimization face
    const fo = $("#face-opt"); fo.innerHTML = "";
    const x = v.assignments[v.assignments.length - 1];
    if (!x) fo.append(el("div", { class: "dim" }, "not solved (family certification did not pass, or not reached)"));
    else {
      const inc = Object.values(state.summary.incumbents || {}).find(i => i.contract === T.id || i.key.startsWith(T.scope.workload.name + "|"));
      fo.append(dl([["assignment x*", el("span", { class: "mono" }, Object.entries(x.values).map(([k, y]) => `${k}=${y}`).join("  "))],
        ["objective", fmt(x.objective) + " µs"], ["bound", fmt(x.bound) + " µs"], ["gap", pct(x.gap)], ["solver time", fmt(x.solver_time, 3) + " s"], ["status", x.status],
        ["diagnostics", el("span", { class: "mono" }, Object.entries(x.diagnostics || {}).map(([k, y]) => `${k}=${y}`).join(" "))],
        ["incumbent", inc ? `${fmt(inc.cost)} µs by ${inc.contract === T.id ? "this template" : inc.contract} (iteration ${inc.iteration})` + (inc.contract !== T.id && x.objective ? ` · this: ${((x.objective / inc.cost - 1) * 100).toFixed(2)}%` : "") : "–"]]));
    }

    // verification face
    const fv = $("#face-ver"); fv.innerHTML = "";
    const fam = v.verdicts.filter(w => w.check === "family").pop();
    const inst = v.verdicts.filter(w => w.check === "instance").pop();
    // a family certificate sweeps all 24 schedule points; only the ones the precondition claims must pass
    const obSummary = w => { const os = w.obligations.filter(o => w.check !== "family" || o.claimed_by_precondition !== false); const n = os.length, bad = os.filter(o => !/^(proved|agrees)/.test(o.status)).length, reused = os.filter(o => o.reused).length; return `${n - bad}/${n} obligations` + (w.check === "family" ? ` on ${new Set(w.obligations.filter(o => o.family_point && o.claimed_by_precondition).map(o => JSON.stringify(o.family_point))).size} claimed of 24 schedule points` : "") + (reused ? `, ${reused} reused` : ""); };
    if (!fam) fv.append(el("div", { class: "dim" }, "not certified yet"));
    else {
      fv.append(dl([["K1–K3 (family)", el("span", {}, gateTag(fam.verdict), " ", obSummary(fam), ` · ${fmt(fam.checker_time, 3)} s`)],
        ["precondition", fam.provenance.precondition_is_weakest == null ? (fam.provenance.reused_from ? "reused certificate " + fam.provenance.reused_from : "–") : (fam.provenance.precondition_is_weakest ? "weakest on the family (exact)" : "sound but not the weakest")],
        ["instance", inst ? el("span", {}, gateTag(inst.verdict), " ", obSummary(inst), ` · ${(inst.checker_time * 1000).toFixed(0)} ms`) : "–"]]));
      const cex = (inst && inst.counterexample) || (fam && fam.counterexample);
      if (cex) {
        fv.append(el("h3", {}, "counterexample · " + cex.obligation));
        fv.append(el("div", { class: "mono dim" }, "design: " + Object.entries(cex.design).map(([k, y]) => `${k}=${y}`).join(" ")));
        const rp = cex.replay || {};
        if (rp.trace) {
          const pre = el("pre", { html: rp.trace.map(t => t.includes("STALE") ? `<span class="stale">${esc(t)}</span>` : esc(t)).join("\n") });
          fv.append(el("div", { class: "dim" }, "replayed interleaving (admissible: " + rp.admissible + "; agrees with reference: " + rp.agrees_with_reference + ")"), pre);
        } else if (rp.differences_sample) {
          fv.append(el("div", { class: "dim" }, `replay on shape ${JSON.stringify(rp.shape)}: ${rp.differences_total} differences`), el("pre", {}, JSON.stringify(rp.differences_sample, null, 1)));
        } else fv.append(el("pre", {}, JSON.stringify(cex.model, null, 1)));
      }
      const obl = el("details", {}, el("summary", { class: "dim" }, "all obligations"));
      const t = el("table", {}, el("tr", {}, el("th", {}, "obligation"), el("th", {}, "status"), el("th", {}, "s")));
      for (const w of [fam, inst].filter(Boolean)) for (const o of w.obligations) t.append(el("tr", {}, el("td", { class: "mono" }, (w.check === "family" && o.family_point ? `S=${o.family_point.S} P=${o.family_point.P} β=${o.family_point.beta} ` : "") + o.obligation), el("td", {}, o.status + (o.reused ? " (reused)" : "")), el("td", { class: "num" }, o.seconds == null ? "" : o.seconds.toFixed(3))));
      obl.append(t); fv.append(obl);
    }
    for (const d of v.diagnoses) fv.append(el("div", { class: "tag bad", style: "margin-top:6px;white-space:normal" }, `diagnosis → ${d.source}: ${d.response}`));

    // evidence strip
    const m = v.measurements[v.measurements.length - 1];
    $("#evidence-body").innerHTML = m
      ? `measured <b>${fmt(m.measured_cost)}</b> vs predicted <b>${fmt(m.predicted_cost)}</b> µs (${m.discrepancy >= 0 ? "+" : ""}${pct(m.discrepancy)}) · utilizations ${Object.entries(m.utilizations).map(([k, y]) => `${k} ${pct(y)}`).join(", ")} · dominant <b>${m.dominant_resource}</b> · <span class="dim">${esc(m.simulator)}</span>`
      : '<span class="dim">no measurement</span>';
    const flipEv = v.events.filter(e => ["regime_bound", "skip_regime", "probe_result", "split"].includes(e.kind));
    for (const e of flipEv) $("#evidence-body").append(el("div", { class: "dim" }, `${e.kind}: ${Object.entries(e.payload).filter(([k]) => !["children"].includes(k)).map(([k, y]) => `${k}=${typeof y === "object" ? JSON.stringify(y) : typeof y === "number" ? fmt(y) : y}`).join(" · ")}`));
    renderActions();
  }

  // ------------------------------------------------------------------ actions
  function renderActions() {
    const c = state.controller || {}, v = state.view;
    if (!v) return;
    const T = v.contract;
    const waiting = c.waiting;
    const pendingHere = waiting && waiting.pending === T.id;
    const atProposed = pendingHere && waiting.checkpoint === "proposed";
    $("#act-edit").disabled = !atProposed;
    $("#act-inject").disabled = !atProposed;
    $("#act-override").disabled = !waiting;
    const row = state.timeline.find(r => r.contract === T.id);
    const cp = row && row.checkpoints.find(k => k.name === "proposed");
    $("#act-rewind").disabled = !c.live || !cp;
    $("#act-rewind-fresh").disabled = !c.live || !cp;
    $("#act-hint").textContent = !c.live ? "start a run to act" : atProposed ? "the loop is waiting on this proposal: edit or inject replaces it, then continue" : waiting ? `loop waiting at ${waiting.checkpoint} of iteration ${waiting.iteration}; edit/inject only at a proposed checkpoint` : "pause or use step/drive mode to act on a checkpoint";
  }
  function openEditor(title, formNodes, text, onSubmit) {
    $("#editor").hidden = false; $("#editor-title").textContent = title;
    const f = $("#editor-form"); f.innerHTML = ""; f.append(...formNodes);
    const ta = $("#editor-text"); ta.hidden = text == null; ta.value = text || "";
    $("#editor-submit").onclick = () => { try { onSubmit(); $("#editor").hidden = true; } catch (e) { toast(String(e)); } };
    $("#editor-cancel").onclick = () => { $("#editor").hidden = true; };
    $("#editor").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  $("#act-edit").onclick = () => {
    const T = state.view.contract;
    const reason = el("input", { type: "text", placeholder: "why (recorded in the ledger)" });
    openEditor("edit the pending contract (the verifier still has to certify it)", [el("label", {}, "reason", reason)], JSON.stringify(T, null, 2), () => {
      const contract = JSON.parse($("#editor-text").value);
      send({ op: "edit", contract, reason: reason.value });
      toast("edit queued; press continue");
    });
  };
  $("#act-inject").onclick = () => {
    const pop = state.summary.population || [];
    if (!pop.length) return toast("population is empty");
    const sel = el("select", {}, ...pop.map(p => el("option", { value: p.id }, `${p.id} · it ${p.iteration} · ${p.refines} · ${p.workload}`)));
    const reason = el("input", { type: "text", placeholder: "why" });
    openEditor("inject a certified template from the population in place of the pending proposal", [el("label", {}, "template", sel), el("label", {}, "reason", reason)], null, () => {
      send({ op: "inject", template: sel.value, reason: reason.value }); toast("inject queued; press continue");
    });
  };
  $("#act-override").onclick = () => {
    const target = el("select", {}, el("option", { value: "regime" }, "regime decision (performance → steer)"), el("option", { value: "verdict" }, "verifier verdict (correctness → will be refused)"));
    const decision = el("select", {}, el("option", { value: "keep" }, "keep"), el("option", { value: "descend" }, "descend"), el("option", { value: "accept" }, "accept"));
    const reason = el("input", { type: "text", placeholder: "why" });
    openEditor("override: performance obligations steer, correctness obligations gate", [el("label", {}, "target", target), el("label", {}, "decision", decision), el("label", {}, "reason", reason)], null, () => {
      send({ op: "override", target: target.value, decision: decision.value, reason: reason.value }); toast("override queued");
    });
  };
  const rewind = fresh => () => {
    const row = state.timeline.find(r => r.contract === state.view.contract.id);
    const cp = row && row.checkpoints.find(k => k.name === "proposed");
    if (!cp) return toast("no proposed checkpoint for this contract");
    send({ op: "rewind", to: cp.seq, fresh });
    toast(`rewinding to checkpoint #${cp.seq}` + (fresh ? " with a fresh proposal" : ""));
  };
  $("#act-rewind").onclick = rewind(false);
  $("#act-rewind-fresh").onclick = rewind(true);

  // header buttons
  for (const b of $("#mode-seg").children) b.onclick = () => send({ op: "mode", mode: b.dataset.mode });
  $("#btn-start").onclick = () => { const name = prompt("run name", `run-${(state.runs.length + 1).toString().padStart(3, "0")}`); if (name) send({ op: "start", run: name, mode: state.controller.mode || "step" }); };
  $("#btn-pause").onclick = () => send({ op: "pause" });
  $("#btn-step").onclick = () => send({ op: "step" });
  $("#btn-replay").onclick = () => send({ op: "replay" });
  $("#btn-stop").onclick = () => send({ op: "stop" });
  $("#run-select").onchange = e => { send({ op: "select", run: e.target.value }); state.selected = null; state.view = null; $("#template-pane").classList.add("empty"); $("#tv-body").hidden = true; $("#tv-empty").hidden = false; scheduleRefresh(); };

  // ------------------------------------------------------------------ stores + ledger
  function renderStores() {
    const s = state.summary || {}, box = $("#stores"); box.innerHTML = "";
    const pop = s.population || [];
    const d1 = el("details", { open: "" }, el("summary", {}, `Template population · ${pop.length}`), el("ul", {}, ...pop.map(p => el("li", {}, el("a", { onclick: () => selectContract(p.id) }, p.id), ` it ${p.iteration} · ${p.refines} · `, el("span", { class: "dim" }, p.workload)))));
    const inc = Object.values(s.incumbents || {});
    const d2 = el("details", { open: "" }, el("summary", {}, `Design archive · incumbents ${inc.length}`), el("ul", {}, ...inc.map(i => el("li", {}, el("b", {}, i.key.split("|")[0]), ` ${fmt(i.cost)} µs · `, el("span", { class: "mono dim" }, Object.entries(i.design).map(([k, y]) => `${k}=${y}`).join(" ")), " · ", el("a", { onclick: () => selectContract(i.contract) }, "it " + i.iteration)))));
    const d3 = el("details", {}, el("summary", {}, `Proof library · ${s.proofs ?? 0} family certificates`));
    const d4 = el("details", { open: "" }, el("summary", {}, `Regime map · ${(s.flips || []).length} flips`), el("ul", {}, ...(s.flips || []).map(f => el("li", {}, `it ${f.iteration}: assumed ${f.assumed}, measured ${f.actual}`))));
    const d5 = el("details", {}, el("summary", {}, `Human interventions · ${(s.overrides || []).length}`), el("ul", {}, ...(s.overrides || []).map(o => el("li", {}, `it ${o.iteration} ${o.kind}` + (o.reason ? ": " + o.reason : "")))));
    box.append(d1, d2, d3, d4, d5);
  }
  function appendLedger(e) {
    const box = $("#ledger");
    const cls = ["checkpoint"].includes(e.kind) ? "checkpoint" : e.kind.startsWith("human") || e.kind === "rewind" ? "human" : ["rejected", "diagnosis"].includes(e.kind) || (e.payload && e.payload.verdict === "refuted") ? "bad" : "";
    const brief = Object.entries(e.payload).filter(([k]) => !["cursor", "structure", "workload", "values", "utilizations", "params", "from_cursor", "restored_cursor"].includes(k)).slice(0, 5).map(([k, y]) => `${k}=${typeof y === "object" ? JSON.stringify(y).slice(0, 60) : typeof y === "number" && !Number.isInteger(y) ? fmt(y, 3) : y}`).join(" ");
    const n = el("div", { class: "ev " + cls }, `#${e.seq} it${e.iteration} `, el("span", { class: "k" }, e.kind), " ", e.artifact ? el("a", { onclick: () => showArtifact(e.artifact) }, e.artifact) : "", " ", el("span", { class: "dim" }, brief));
    box.append(n);
    $("#ledger-count").textContent = "· " + e.seq + " events";
    if ($("#ledger-follow").checked) box.scrollTop = box.scrollHeight;
  }
  async function showArtifact(id) {
    if (id.startsWith("tmpl_")) return selectContract(id);
    const a = await api("artifact/" + id);
    openEditor("artifact " + id + " (read-only)", [], JSON.stringify(a, null, 2), () => {});
    $("#editor-submit").hidden = true; setTimeout(() => { $("#editor-submit").hidden = false; }, 0);
    $("#editor-text").readOnly = true; $("#editor-cancel").onclick = () => { $("#editor").hidden = true; $("#editor-text").readOnly = false; };
  }
  function toast(msg) { const t = el("div", { class: "toast" }, msg); document.body.append(t); setTimeout(() => t.remove(), 3500); }

  loadCatalog(); connect();
})();
