/* The workload bar: which matrix multiplication the next run works on. The list and the choice live on the server
   (/api/workloads, the "workload" command), so every open console shows the same choice. */
window.Workload = (() => {
  const S = Session;
  const $ = id => document.getElementById(id);
  let list = [], chosen = null;
  const fmt = n => n.toLocaleString("en-US");

  function describe(w) {
    if (!w) return "Not chosen yet. Pick the matrix multiplication the optimizer will work on.";
    return `C = A × B, with A of ${fmt(w.m)} × ${fmt(w.k)} and B of ${fmt(w.k)} × ${fmt(w.n)}, so C is ${fmt(w.m)} × ${fmt(w.n)}, ` +
           `${w.dtype} (single precision). ${w.why}`;
  }
  function render() {
    const sel = $("wl-select"); sel.innerHTML = "";
    const none = document.createElement("option"); none.value = ""; none.textContent = "— choose —"; sel.append(none);
    for (const w of list) {
      const o = document.createElement("option"); o.value = w.name;
      o.textContent = `${w.title}: ${fmt(w.m)} × ${fmt(w.k)} times ${fmt(w.k)} × ${fmt(w.n)}`;
      o.selected = w.name === chosen; sel.append(o);
    }
    const w = list.find(x => x.name === chosen);
    $("wl-desc").textContent = describe(w);
    $("workload-bar").classList.toggle("chosen", !!w);
    sel.disabled = !!S.state.live;
  }
  async function load() {
    const r = await (await fetch("/api/workloads")).json();
    list = r.workloads; chosen = r.chosen; render();
  }
  $("wl-select").onchange = e => {
    S.send({ op: "workload", name: e.target.value || null });
    setTimeout(load, 150);
  };
  S.on("controller", () => { load(); });
  load();
  return { load, chosen: () => chosen };
})();
