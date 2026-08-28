/* The in-page panel. Deliberately self-contained: no framework, no build step,
   and every style scoped under #researchos-panel so it cannot collide with the
   host page - which may be arXiv, a journal site, or a PDF viewer. */

(() => {
  if (window.__researchosInjected) return;
  window.__researchosInjected = true;

  let panel = null;

  function ensurePanel() {
    if (panel && document.body.contains(panel)) return panel;
    panel = document.createElement("div");
    panel.id = "researchos-panel";
    panel.innerHTML = `
      <header>
        <span class="ro-brand">⎎ ResearchOS</span>
        <button class="ro-close" aria-label="Close">&times;</button>
      </header>
      <div class="ro-body"></div>`;
    panel.querySelector(".ro-close").addEventListener("click", hide);
    document.body.appendChild(panel);
    return panel;
  }

  function hide() {
    panel?.classList.remove("ro-open");
  }

  function show(html) {
    const node = ensurePanel();
    node.querySelector(".ro-body").innerHTML = html;
    node.classList.add("ro-open");
  }

  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
    );

  /** Minimal markdown: **bold**, paragraphs, and [arXiv-id] links. */
  function render(text) {
    return String(text || "")
      .split("\n")
      .filter((l) => l.trim())
      .map((line) => {
        let safe = esc(line);
        safe = safe.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
        safe = safe.replace(
          /\[(\d{4}\.\d{4,5})([^\]]*)\]/g,
          '<a href="https://arxiv.org/abs/$1" target="_blank" rel="noopener">[$1]</a>',
        );
        return `<p>${safe}</p>`;
      })
      .join("");
  }

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === "researchos:loading") {
      show(`<div class="ro-loading"><div class="ro-dots"><i></i><i></i><i></i></div>
        <p class="ro-muted">Looking up <em>${esc(msg.query.slice(0, 90))}</em></p></div>`);
    }

    if (msg.type === "researchos:error") {
      show(`<div class="ro-error"><strong>Could not explain that</strong>
        <p>${esc(msg.message)}</p></div>`);
    }

    if (msg.type === "researchos:result") {
      const d = msg.data || {};
      if (!d.resolved && !d.explanation) {
        const alts = (d.candidates || [])
          .slice(0, 4)
          .map((c) => `<li>${esc(c.title)}</li>`)
          .join("");
        show(`<div class="ro-error"><strong>Could not identify that paper</strong>
          <p>Try selecting just the title, or the arXiv id.</p>
          ${alts ? `<p class="ro-muted">Did you mean:</p><ul>${alts}</ul>` : ""}</div>`);
        return;
      }

      const paper = d.paper;
      const head = paper
        ? `<div class="ro-paper">
             <h2>${esc(paper.title)}</h2>
             <p class="ro-meta">${esc((paper.authors || []).slice(0, 3).join(", "))}${
               (paper.authors || []).length > 3 ? " et al." : ""
             }${paper.published ? " · " + esc(paper.published) : ""}${
               paper.arxiv_url
                 ? ` · <a href="${esc(paper.arxiv_url)}" target="_blank" rel="noopener">arXiv ↗</a>`
                 : ""
             }</p>
           </div>`
        : "";

      const sources = (d.passages || [])
        .slice(0, 3)
        .map(
          (p) =>
            `<li><span class="ro-score">${(p.score ?? 0).toFixed(2)}</span> ${esc(
              (p.section || "") + " — " + p.text.slice(0, 150),
            )}…</li>`,
        )
        .join("");

      show(`${head}<div class="ro-answer">${render(d.explanation)}</div>
        ${sources ? `<details class="ro-sources"><summary>Passages used</summary><ul>${sources}</ul></details>` : ""}`);
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") hide();
  });
})();
