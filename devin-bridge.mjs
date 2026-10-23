#!/usr/bin/env node
// Local stdio MCP proxy for the Devin MCP server.
// Forwards newline-delimited JSON-RPC on stdin to https://mcp.devin.ai/mcp
// (Streamable HTTP) and writes responses to stdout. Reads DEVIN_API_KEY
// (service-user cog_ key) from its own environment — no header quoting
// through cmd/npx/mcp-remote layers. DEVIN_ORG_ID is optional.

import { execSync } from "node:child_process";

// Read env vars from the registry first — the parent app's environment may
// predate a setx, while HKCU\Environment always holds the current values.
function registryEnv(name) {
  try {
    const out = execSync(`reg query HKCU\\Environment /v ${name}`,
      { encoding: "utf8", stdio: ["pipe", "pipe", "pipe"] });
    const m = out.match(/REG_(?:SZ|EXPAND_SZ)\s+(.+)/);
    return m ? m[1].trim() : null;
  } catch {
    return null;
  }
}

function envVar(name) {
  return registryEnv(name) || process.env[name] || null;
}

const API_KEY = envVar("DEVIN_API_KEY") || envVar("DEVIN_REST_API_KEY");
const ORG_ID = envVar("DEVIN_ORG_ID");
const UPSTREAM = process.env.DEVIN_MCP_URL || "https://mcp.devin.ai/mcp";

function log(...a) {
  process.stderr.write(`[devin-bridge] ${a.join(" ")}\n`);
}

log(`key fingerprint: ${API_KEY ? `${API_KEY.slice(0, 6)}… len=${API_KEY.length}` : "NONE"}`
  + ` (source: ${registryEnv("DEVIN_API_KEY") ? "registry" : "process.env/fallback"})`
  + ` org_id=${ORG_ID || "none"}`);

let sessionId = null;

// Extract the JSON-RPC response for `wantId` from an SSE body.
function parseSSE(text, wantId) {
  const msgs = [];
  for (const block of text.split(/\r?\n\r?\n/)) {
    const data = block
      .split(/\r?\n/)
      .filter(l => l.startsWith("data:"))
      .map(l => l.slice(5).trimStart())
      .join("\n");
    if (!data) continue;
    try { msgs.push(JSON.parse(data)); } catch { /* ignore non-JSON events */ }
  }
  if (!msgs.length) return null;
  return msgs.find(m => m.id === wantId) ?? msgs[msgs.length - 1];
}

async function sendOne(msg) {
  const headers = {
    Authorization: `Bearer ${API_KEY}`,
    "Content-Type": "application/json",
    Accept: "application/json, text/event-stream",
  };
  if (ORG_ID) headers["X-Org-Id"] = ORG_ID;
  if (sessionId) headers["mcp-session-id"] = sessionId;

  let resp;
  try {
    resp = await fetch(UPSTREAM, { method: "POST", headers, body: JSON.stringify(msg) });
  } catch (e) {
    throw new Error(`upstream unreachable: ${e.message}`);
  }

  const newSid = resp.headers.get("mcp-session-id");
  if (newSid) sessionId = newSid;

  // Notifications are acknowledged with 202/204 and no body.
  if (resp.status === 202 || resp.status === 204) return null;

  const ct = resp.headers.get("content-type") || "";
  let out = null;
  const text = await resp.text();
  if (ct.includes("text/event-stream")) {
    out = parseSSE(text, msg.id);
  } else if (text) {
    try { out = JSON.parse(text); } catch { out = null; }
  }

  if (!resp.ok && (!out || out.error === undefined)) {
    out = {
      jsonrpc: "2.0",
      id: msg.id ?? null,
      error: { code: resp.status, message: `HTTP ${resp.status}: ${text.slice(0, 400)}` },
    };
  }
  return out;
}

async function dispatch(msg) {
  const items = Array.isArray(msg) ? msg : [msg];
  const results = [];
  for (const item of items) {
    try {
      const r = await sendOne(item);
      if (r !== null) results.push(r);
    } catch (e) {
      if (item.id !== undefined && item.id !== null) {
        results.push({ jsonrpc: "2.0", id: item.id, error: { code: -32603, message: e.message } });
      } else {
        log(`notification failed: ${e.message}`);
      }
    }
  }
  if (!results.length) return;
  const out = Array.isArray(msg) ? results : results[0];
  process.stdout.write(JSON.stringify(out) + "\n");
}

let buf = "";
process.stdin.on("data", chunk => {
  buf += chunk;
  let idx;
  while ((idx = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, idx).trim();
    buf = buf.slice(idx + 1);
    if (!line) continue;
    let msg;
    try { msg = JSON.parse(line); } catch { continue; }
    dispatch(msg).catch(e => log(`dispatch error: ${e.message}`));
  }
});
// No explicit exit on stdin end: the process exits naturally once stdin
// closes and pending upstream fetches settle.

if (!API_KEY) {
  log("DEVIN_API_KEY not set — upstream calls will fail auth");
}
