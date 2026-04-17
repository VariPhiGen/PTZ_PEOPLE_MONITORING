/**
 * fanOutToNodes — parallel request dispatch to a client's nodes with
 * configurable result merging.
 *
 * Merge strategies
 * ────────────────
 *  union          Concatenate .items arrays from all successful responses.
 *  first          Return the first successful response verbatim.
 *  merge_journey  Merge + sort journey events by first_seen (cross-node).
 *  sum            Add .total / .count numeric fields across responses.
 *  latest         Return the response with the largest .updated_at field.
 */
import type { Env, FanOutResult, MergeStrategy, NodeRow } from "../types.js";

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Query D1 for all ONLINE nodes serving `clientId`, fan out `path` in
 * parallel, and merge the results according to `strategy`.
 *
 * @param env           Worker environment bindings
 * @param clientId      Client UUID (routes to only this client's nodes)
 * @param path          Path + query string to forward (e.g. "/api/cameras")
 * @param init          Fetch options (method, headers, body …)
 * @param strategy      How to combine results from multiple nodes
 */
export async function fanOutToNodes(
  env:       Env,
  clientId:  string,
  path:      string,
  init:      RequestInit,
  strategy:  MergeStrategy,
): Promise<FanOutResult> {
  const nodes = await getClientNodes(env.DB, clientId);

  if (nodes.length === 0) {
    return { data: null, nodeCount: 0, errors: ["no_nodes_available"], partial: true };
  }

  const results = await Promise.allSettled(
    nodes.map((node) => fetchNode(node.api_endpoint, path, init)),
  );

  const successes: { json: unknown; node: NodeRow }[] = [];
  const errors:    string[]                           = [];

  for (let i = 0; i < results.length; i++) {
    const r = results[i];
    if (r.status === "fulfilled" && r.value.ok) {
      try {
        successes.push({ json: await r.value.json(), node: nodes[i] });
      } catch {
        errors.push(`${nodes[i].node_id}: bad_json`);
      }
    } else {
      const msg =
        r.status === "rejected"
          ? String(r.reason)
          : `${nodes[i].node_id}: http_${r.value.status}`;
      errors.push(msg);
    }
  }

  const partial = errors.length > 0 && successes.length > 0;

  if (successes.length === 0) {
    return { data: null, nodeCount: 0, errors, partial: false };
  }

  const data = merge(successes.map((s) => s.json), strategy);
  return { data, nodeCount: successes.length, errors, partial };
}

/**
 * Proxy to a single specific node by its api_endpoint.
 */
export async function proxyToNode(
  endpoint:  string,
  path:      string,
  init:      RequestInit,
): Promise<Response> {
  return fetchNode(endpoint, path, init);
}

/**
 * Return the least-loaded ONLINE node for a client
 * (lowest ratio of active_cameras / max_cameras).
 */
export async function leastLoadedNode(
  db:       D1Database,
  clientId: string,
): Promise<NodeRow | null> {
  const nodes = await getClientNodes(db, clientId);
  if (nodes.length === 0) return null;

  return nodes.reduce((best, n) => {
    const loadA = n.active_cameras / (n.max_cameras || 1);
    const loadB = best.active_cameras / (best.max_cameras || 1);
    return loadA < loadB ? n : best;
  });
}

/**
 * Return any single ONLINE node for a client (first in the list).
 */
export async function anyClientNode(
  db:       D1Database,
  clientId: string,
): Promise<NodeRow | null> {
  const nodes = await getClientNodes(db, clientId);
  return nodes[0] ?? null;
}

/**
 * Return any single ONLINE node regardless of client (for admin/auth routes).
 */
export async function anyNode(db: D1Database): Promise<NodeRow | null> {
  const row = await db
    .prepare("SELECT * FROM nodes WHERE status = 'ONLINE' ORDER BY RANDOM() LIMIT 1")
    .first<NodeRow>();
  return row ?? null;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

async function getClientNodes(db: D1Database, clientId: string): Promise<NodeRow[]> {
  const { results } = await db
    .prepare(`
      SELECT DISTINCT n.*
      FROM nodes n
      JOIN camera_assignments ca ON ca.node_id = n.node_id
      WHERE ca.client_id = ?
        AND n.status = 'ONLINE'
      ORDER BY n.active_cameras ASC
    `)
    .bind(clientId)
    .all<NodeRow>();
  return results;
}

async function fetchNode(
  apiEndpoint: string,
  path:        string,
  init:        RequestInit,
): Promise<Response> {
  const url = `${apiEndpoint.replace(/\/$/, "")}${path}`;
  return fetch(url, { ...init, redirect: "follow" });
}

// ── Merge strategies ──────────────────────────────────────────────────────────

function merge(responses: unknown[], strategy: MergeStrategy): unknown {
  if (responses.length === 1) return responses[0];

  switch (strategy) {
    case "union":         return mergeUnion(responses);
    case "first":         return responses[0];
    case "merge_journey": return mergeJourney(responses);
    case "sum":           return mergeSum(responses);
    case "latest":        return mergeLatest(responses);
    default:              return responses[0];
  }
}

/** Concatenate .items arrays; sum .total. */
function mergeUnion(responses: unknown[]): unknown {
  const items:  unknown[] = [];
  let   total             = 0;
  const extra: Record<string, unknown> = {};

  for (const r of responses) {
    if (!isRecord(r)) continue;
    if (Array.isArray(r["items"])) items.push(...(r["items"] as unknown[]));
    if (typeof r["total"] === "number") total += r["total"];
    // Preserve top-level non-items fields from first response
    for (const [k, v] of Object.entries(r)) {
      if (k !== "items" && k !== "total" && !(k in extra)) extra[k] = v;
    }
  }

  return { ...extra, total, items };
}

/** Merge journey `.events` arrays, sort by first_seen, deduplicate by source_id. */
function mergeJourney(responses: unknown[]): unknown {
  const seen   = new Set<string>();
  const events: unknown[] = [];
  let   base: Record<string, unknown> = {};

  for (const r of responses) {
    if (!isRecord(r)) continue;
    if (Object.keys(base).length === 0) base = { ...r };
    const evts = r["events"];
    if (!Array.isArray(evts)) continue;
    for (const e of evts) {
      if (!isRecord(e)) continue;
      const id = String(e["source_id"] ?? "");
      if (!seen.has(id)) {
        seen.add(id);
        events.push(e);
      }
    }
  }

  // Sort by first_seen ascending
  events.sort((a, b) => {
    const fa = isRecord(a) ? Number(a["first_seen"] ?? 0) : 0;
    const fb = isRecord(b) ? Number(b["first_seen"] ?? 0) : 0;
    return fa - fb;
  });

  return { ...base, events, total: events.length };
}

/** Sum numeric .total / .count at the top level. */
function mergeSum(responses: unknown[]): unknown {
  const totals: Record<string, number> = {};
  const base = isRecord(responses[0]) ? { ...responses[0] } : {};

  for (const r of responses) {
    if (!isRecord(r)) continue;
    for (const [k, v] of Object.entries(r)) {
      if (typeof v === "number") totals[k] = (totals[k] ?? 0) + v;
    }
  }
  return { ...base, ...totals };
}

/** Return the response with the largest .updated_at value. */
function mergeLatest(responses: unknown[]): unknown {
  let best   = responses[0];
  let bestTs = isRecord(best) ? Number(best["updated_at"] ?? 0) : 0;

  for (const r of responses.slice(1)) {
    const ts = isRecord(r) ? Number(r["updated_at"] ?? 0) : 0;
    if (ts > bestTs) { best = r; bestTs = ts; }
  }
  return best;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}
