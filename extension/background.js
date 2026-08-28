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

/** Send to the in-page panel; report whether it was actually received.
 *
 *  Chrome does not inject content scripts into its built-in PDF viewer, and
 *  papers are mostly read as PDFs. Without this check the panel silently never
 *  appears on exactly the pages the extension exists for.
 */
async function tell(tabId, message) {
  try {
    await chrome.tabs.sendMessage(tabId, message);
    return true;
  } catch {
    return false;
  }
}

/** Fallback when no panel can be injected: hand the query to the web app. */
async function openInWebApp(query) {
  const { webApp } = { webApp: "http://localhost:3000", ...(await chrome.storage.sync.get({ webApp: "http://localhost:3000" })) };
  await chrome.tabs.create({ url: `${webApp}/?explain=${encodeURIComponent(query)}` });
}

async function explain(tabId, text) {
  const trimmed = (text || "").trim();
  if (trimmed.length < 3) return;

  // Show the panel immediately so the wait is visible, not silent.
  const reachable = await tell(tabId, { type: "researchos:loading", query: trimmed });
  if (!reachable) {
    // A PDF viewer, a chrome:// page, or a tab loaded before install.
    await openInWebApp(trimmed);
    return;
  }

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
    await tell(tabId, { type: "researchos:result", data: await res.json() });
  } catch (err) {
    const offline = String(err).includes("Failed to fetch");
    await tell(tabId, {
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
