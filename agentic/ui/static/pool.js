/* The pool: one card per template proposed in the run, newest on top, up to the play clock's
   iteration. Each card shows the describer's headline and its "what's different from the parent"
   line (round 3); the formulator's `refines` shorthand is never shown. */
window.Pool = (() => {
  const S = Session;
  const box = document.getElementById("pool"), count = document.getElementById("pool-count");
  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };

  function render() {
    const { it, v } = S.state.clock;
    const entries = S.state.story.filter(e => e.template && (e.iteration < it || (e.iteration === it && v >= 1)));
    const D = S.state.describe || {};
    box.innerHTML = "";
    let solid = 0, red = 0;
    for (const e of [...entries].reverse()) {
      const t = e.template, d = D[t.id], done = e.iteration < it || v >= e.visits.length;
      const status = e.outcome === "accepted" && done ? "accepted" : e.outcome === "rejected" && done ? "rejected" : "proposed";
      if (status === "accepted") solid++; if (status === "rejected") red++;
      const card = el("div", "card " + status + (e.iteration === it ? " new" : ""));
      card.dataset.id = t.id;
      const chipText = status === "accepted" ? "accepted" : status === "rejected" ? "rejected" : e.visits.some(x => x.station === "verifier" && x.ok) && e.iteration === it ? "proved safe" : "proposed";
      card.append(el("span", "chip", chipText));
      // headline: "{short} · {what it decides} · {status} · {outcome}"; the status is the chip, so show the rest
      const head = el("div", "headline");
      if (d) {
        const parts = d.headline.split(" · ");
        head.append(el("span", "short", parts[0]), " ", el("span", "what", parts[1] || ""));
        const outcome = status === "proposed" ? null : parts.slice(3).join(" · ");
        if (outcome) head.append(el("div", "outcome" + (d.is_best && status === "accepted" ? " star" : ""), (d.is_best && status === "accepted" ? "★ " : "") + outcome));
      } else head.append(el("span", "short", t.short), " ", el("span", "what", t.workload));
      card.append(head);
      if (d && d.delta_from_parent) card.append(el("div", "delta", d.delta_from_parent));
      if (e.pool_delta.reused_from_short) card.append(el("div", "link", "↳ proof reused from " + e.pool_delta.reused_from_short));
      if (e.human && e.human.some(h => h.kind === "human_edit")) card.append(el("div", "link", "☺ you changed this proposal"));
      if (e.human && e.human.some(h => h.kind === "human_inject")) card.append(el("div", "link", "☺ you inserted this from the pool"));
      card.onclick = () => Inspector.open(t.id, e.iteration);
      box.append(card);
    }
    count.textContent = entries.length ? `${entries.length} templates · ${solid} accepted · ${red} rejected` : "";
    if (window.Glossary) Glossary.apply(box);
  }
  S.on("clock", render); S.on("story", render); S.on("summary", render); S.on("describe", render);
  return { render };
})();
