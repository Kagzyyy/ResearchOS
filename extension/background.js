/* Context menu + keyboard shortcut. The heavy lifting is the backend's; this
   only decides what text to send and where to show the result. */

const DEFAULTS = { apiBase: "http://127.0.0.1:8100", apiToken: "" };

async function settings() {
  return { ...DEFAULTS, ...(await chrome.storage.sync.get(DEFAULTS)) };
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "researchos-explain",
    title: 'Explain "%s" with ResearchOS',
    contexts: ["selection"],
  });
});

async function explain(tabId, text) {
  const trimmed = (text || "").trim();
  if (trimmed.length < 3) return;

  // Show the panel immediately so the wait is visible, not silent.
  chrome.tabs.sendMessage(tabId, { type: "researchos:loading", query: trimmed });

  const { apiBase, apiToken } = await settings();
  const headers = { "Content-Type": "application/json" };
  if (apiToken) headers["Authorization"] = `Bearer ${apiToken}`;

  try {
    const res = await fetch(`${apiBase}/explain`, {
      method: "POST",
      headers,
      body: JSON.stringify({ citation: trimmed, generate: true }),
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(
        res.status === 401
          ? "Access token rejected — check it in the extension popup."
          : res.status === 429
            ? "Rate limited. Wait a minute and try again."
            : `Request failed (${res.status}). ${detail.slice(0, 160)}`,
      );
    }
    chrome.tabs.sendMessage(tabId, { type: "researchos:result", data: await res.json() });
  } catch (err) {
    const offline = String(err).includes("Failed to fetch");
    chrome.tabs.sendMessage(tabId, {
      type: "researchos:error",
      message: offline
        ? "Cannot reach ResearchOS. Is the backend running on port 8100?"
        : String(err.message || err),
    });
  }
}

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "researchos-explain" && tab?.id) {
    explain(tab.id, info.selectionText);
  }
});

chrome.commands.onCommand.addListener(async (command) => {
  if (command !== "explain-selection") return;
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return;
  const [{ result } = {}] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => window.getSelection()?.toString() ?? "",
  });
  explain(tab.id, result);
});
