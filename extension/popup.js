const DEFAULTS = { apiBase: "http://127.0.0.1:8100", apiToken: "", webApp: "http://localhost:3000" };
const $ = (id) => document.getElementById(id);

chrome.storage.sync.get(DEFAULTS).then((s) => {
  $("base").value = s.apiBase;
  $("token").value = s.apiToken;
  $("webapp").value = s.webApp;
});

$("save").addEventListener("click", async () => {
  const apiBase = ($("base").value || DEFAULTS.apiBase).replace(/\/+$/, "");
  const apiToken = $("token").value.trim();
  const webApp = ($("webapp").value || DEFAULTS.webApp).replace(/\/+$/, "");
  await chrome.storage.sync.set({ apiBase, apiToken, webApp });

  const status = $("status");
  status.textContent = "Testing…";
  status.className = "status";
  try {
    const res = await fetch(`${apiBase}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const h = await res.json();
    status.textContent = `Connected · ${h.papers.toLocaleString()} papers${
      h.generation_enabled ? "" : " · no API key, passages only"
    }`;
    status.className = "status ok";
  } catch {
    status.textContent = "Cannot reach the backend. Is it running?";
    status.className = "status bad";
  }
});
