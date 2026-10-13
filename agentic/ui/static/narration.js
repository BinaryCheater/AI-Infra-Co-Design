/* Narration: the story entry's clauses, revealed up to the current visit of the play clock. */
window.Narration = (() => {
  const S = Session;
  const ORDER = ["formulator", "verifier", "solver", "gate", "evaluator", "orchestrator"];
  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };

  function render() {
    const { it, v } = S.state.clock;
    const e = S.state.storyByIt[it];
    const title = document.getElementById("narration-title"), box = document.getElementById("narration");
    box.innerHTML = "";
    if (!e) { title.textContent = "No iteration"; box.append(el("span", "pending", S.state.run ? "The run has not proposed anything yet." : "Open a run.")); return; }
    title.textContent = e.title;
    const visited = e.visits.slice(0, v);
    const maxStation = visited.length ? Math.max(...visited.map(x => ORDER.indexOf(x.station))) : -1;
    const shown = e.clauses.filter(c => ORDER.indexOf(c.station) <= maxStation || (c.station === "formulator" && v >= 1));
    for (const c of shown) {
      const cls = "clause" + (c.text.startsWith("Rejected") || c.text.startsWith("Not allowed") ? " rejected" : "") + (c.text.startsWith("You ") ? " human" : "");
      box.append(el("span", cls, c.text + " "));
    }
    if (!shown.length) box.append(el("span", "pending", "Waiting for the agent to propose…"));
    const c = S.state.controller;
    if (S.state.live && c.waiting && c.waiting.iteration === it && v >= e.visits.length) box.append(el("span", "paused", " — paused here (" + c.waiting.checkpoint + ")"));
    else if (S.state.live && e.outcome === "in_progress" && v >= e.visits.length && !S.state.detached) box.append(el("span", "pending", " …"));
    if (window.Glossary) Glossary.apply(box);
  }
  S.on("clock", render); S.on("story", render); S.on("controller", render); S.on("glossary", render);
  return { render };
})();
