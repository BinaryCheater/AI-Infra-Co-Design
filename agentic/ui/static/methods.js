/* The optimization methods level by level (agents/formulator/methods.yaml through /api/methods): the domain know-how the
   proposer chooses from. The level ladder on the map shows, per level, how many of the methods that apply to the workload
   the run's templates use; clicking a level opens the list, each method with its status in this run. */
window.Methods = (() => {
  const S = Session;
  const $ = id => document.getElementById(id);
  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
  let data = null, openLevel = null;
  const CHIP = { used: "used", available: "template exists", missing: "no template yet", not_applicable: "does not apply", fixed: "fixed hardware" };

  async function load() {
    data = await S.api("methods");
    for (const lv of data.levels) {
      const g = document.querySelector(`#map .ladder .level[data-level="${lv.level}"]`);
      if (!g) continue;
      const c = g.querySelector("text.count");
      if (c) c.textContent = lv.fixed ? "fixed" : (lv.applicable ? `${lv.used}/${lv.applicable}` : "–");
      g.classList.toggle("fixed", !!lv.fixed);
    }
    if (openLevel) render(openLevel);
  }

  function section(title, content) {
    const box = el("div", "m-sec");
    box.append(el("div", "m-label", title));
    if (Array.isArray(content)) { const ul = el("ul"); for (const x of content) ul.append(el("li", "", x)); box.append(ul); }
    else box.append(el("div", "", content));
    return box;
  }

  function render(level) {
    const lv = data && data.levels.find(x => x.level === level);
    if (!lv) return;
    $("mp-title").textContent = `Level ${lv.level} · ${lv.name}`;
    const body = $("mp-body"); body.innerHTML = "";
    const intro = lv.fixed ? "The hardware is fixed in this study: these are listed so it is clear they were considered."
      : lv.applicable ? `What can be optimized at this level. ${lv.used} of the ${lv.applicable} methods that apply to this workload are used by this run's templates.`
      : "Nothing at this level applies to this workload.";
    body.append(el("p", "m-intro", intro));
    for (const m of lv.methods) {
      const card = el("div", "m-card " + m.status);
      const head = el("div", "m-head");
      head.append(el("span", "m-name", m.name), el("span", "m-chip " + m.status, CHIP[m.status] || m.status));
      card.append(head, el("div", "m-status", m.status_words), el("p", "m-what", m.what));
      card.append(section("Helps when", m.helps_when));
      card.append(section("Costs", m.costs));
      if (m.choices && m.choices.length) card.append(section("Choices it adds", m.choices.map(c => `${c.name}: ${c.domain.join(", ")}`)));
      if (m.checks && m.checks.length) card.append(section("Correctness checks", m.checks));
      if (m.relies_on && m.relies_on.length) card.append(section("Relies on these measurements", m.relies_on));
      card.append(section("Template", m.template ? m.template : "none yet"));
      body.append(card);
    }
  }

  function open(level) { openLevel = level; $("methods-panel").hidden = false; if (data) render(level); else load(); }
  function close() { openLevel = null; $("methods-panel").hidden = true; }
  $("mp-close").onclick = close;
  S.on("state", load); S.on("story", load);
  load();
  return { open, close, load };
})();
