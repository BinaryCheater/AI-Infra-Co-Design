/* The pool: one card per template proposed in the run, newest on top, up to the play clock's iteration. */
window.Pool = (() => {
  const S = Session;
  const box = document.getElementById("pool"), count = document.getElementById("pool-count");
  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };

  function render() {
    const { it, v } = S.state.clock;
    const entries = S.state.story.filter(e => e.template && (e.iteration < it || (e.iteration === it && v >= 1)));
    box.innerHTML = "";
    let solid = 0, red = 0;
    for (const e of [...entries].reverse()) {
      const t = e.template, done = e.iteration < it || v >= e.visits.length;
      const status = e.outcome === "accepted" && done ? "accepted" : e.outcome === "rejected" && done ? "rejected" : "proposed";
      if (status === "accepted") solid++; if (status === "rejected") red++;
      const card = el("div", "card " + status + (e.iteration === it ? " new" : ""));
      card.dataset.id = t.id;
      const chip = el("span", "chip", status === "accepted" ? "accepted" : status === "rejected" ? "rejected" : e.visits.some(x => x.station === "verifier" && x.ok) && e.iteration === it ? "proved safe" : "proposed");
      card.append(chip);
      const head = el("div");
      head.append(el("span", "short", t.short), " ", el("span", "", t.workload.replace(/_.*/, "")));
      if (e.pool_delta.incumbent && status === "accepted") head.append(" ", el("span", "star", "★ best so far " + (e.pool_delta.incumbent_cost != null ? e.pool_delta.incumbent_cost.toFixed(2) + " µs" : "")));
      card.append(head);
      card.append(el("div", "meta", t.refines || ""));
      if (e.pool_delta.reused_from_short) card.append(el("div", "link", "↳ proof reused from " + e.pool_delta.reused_from_short));
      if (t.parent_short) card.append(el("div", "link", "↑ refines " + t.parent_short));
      if (e.human && e.human.some(h => h.kind === "human_edit")) card.append(el("div", "link", "☺ you changed this proposal"));
      if (e.human && e.human.some(h => h.kind === "human_inject")) card.append(el("div", "link", "☺ you inserted this from the pool"));
      card.onclick = () => Inspector.open(t.id, e.iteration);
      box.append(card);
    }
    count.textContent = entries.length ? `${entries.length} templates · ${solid} accepted · ${red} rejected` : "";
  }
  S.on("clock", render); S.on("story", render); S.on("summary", render);
  return { render };
})();
