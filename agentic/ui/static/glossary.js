/* Glossary: hover glosses for terms in any lens text, matched by exact word (no NLP).
   Longer terms are matched first so "weakest precondition" wins over "precondition". */
window.Glossary = (() => {
  let terms = null, regex = null;
  const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  async function load() {
    terms = await (await fetch("glossary.json")).json();
    const keys = Object.keys(terms).sort((a, b) => b.length - a.length);
    regex = new RegExp("(^|[^\\w-])(" + keys.map(esc).join("|") + ")(?![\\w-])", "g");
    Session.emit("glossary");
  }
  function apply(root) {
    if (!regex) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: n => (n.parentElement.closest("textarea, pre, .gloss, .mono, .race, .short, button, select, option") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT) });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      const text = node.nodeValue;
      regex.lastIndex = 0;
      if (!regex.test(text)) continue;
      regex.lastIndex = 0;
      const frag = document.createDocumentFragment();
      let last = 0, m;
      while ((m = regex.exec(text))) {
        const start = m.index + m[1].length;
        frag.append(document.createTextNode(text.slice(last, start)));
        const span = document.createElement("span");
        span.className = "gloss"; span.textContent = m[2]; span.title = terms[m[2]]; span.dataset.gloss = terms[m[2]];
        frag.append(span);
        last = start + m[2].length;
      }
      frag.append(document.createTextNode(text.slice(last)));
      node.parentNode.replaceChild(frag, node);
    }
  }
  function gloss(term) { return terms ? terms[term] : null; }
  load();
  return { apply, gloss, terms: () => terms };
})();
