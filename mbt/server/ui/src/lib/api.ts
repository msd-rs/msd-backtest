import type {
  CurveQueryResult,
  RunManifest,
  RunMode,
  RunSummaryRecord,
  ServerConfigData,
  ShortcutRecord,
  StrategyRecord,
  SystemDefaults,
  TableQueryResult,
} from "@/types/api";

async function apiFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`;
    try {
      const body = await resp.json();
      if (body && typeof body.detail === "string") {
        detail = body.detail;
      } else if (body && body.detail) {
        detail = JSON.stringify(body.detail);
      }
    } catch {
      const txt = await resp.text().catch(() => "");
      if (txt) detail = txt;
    }
    throw new Error(detail);
  }
  const ct = resp.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    return (await resp.json()) as T;
  }
  return (await resp.text()) as unknown as T;
}

export const api = {
  getHealth: () => apiFetch<string>("/api/health"),
  getConfig: () => apiFetch<ServerConfigData>("/api/config"),
  updateConfig: (payload: {
    server?: { host: string; port: number };
    defaults: SystemDefaults;
  }) =>
    apiFetch<ServerConfigData>("/api/config", {
      method: "PUT",
      body: JSON.stringify(payload),
    }),

  listStrategies: () => apiFetch<StrategyRecord[]>("/api/strategies"),
  registerStrategy: (payload: { path: string; name?: string; notes?: string }) =>
    apiFetch<StrategyRecord>("/api/strategies", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updateStrategy: (id: string, payload: { name?: string; notes?: string }) =>
    apiFetch<StrategyRecord>(`/api/strategies/${id}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  refreshStrategy: (id: string) =>
    apiFetch<StrategyRecord>(`/api/strategies/${id}/refresh`, { method: "POST" }),
  deleteStrategy: (id: string) =>
    apiFetch<{ deleted: boolean; id: string }>(`/api/strategies/${id}`, {
      method: "DELETE",
    }),

  createShortcut: (
    strategyId: string,
    payload: { name: string; mode: RunMode; params: Record<string, unknown> },
  ) =>
    apiFetch<ShortcutRecord>(`/api/strategies/${strategyId}/shortcuts`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updateShortcut: (
    shortcutId: string,
    payload: { name?: string; mode?: RunMode; params?: Record<string, unknown> },
  ) =>
    apiFetch<ShortcutRecord>(`/api/shortcuts/${shortcutId}`, {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
  duplicateShortcut: (shortcutId: string, name?: string) =>
    apiFetch<ShortcutRecord>(`/api/shortcuts/${shortcutId}/duplicate`, {
      method: "POST",
      body: JSON.stringify(name ? { name } : {}),
    }),
  deleteShortcut: (shortcutId: string) =>
    apiFetch<{ deleted: boolean; id: string }>(`/api/shortcuts/${shortcutId}`, {
      method: "DELETE",
    }),

  startRun: (payload: {
    strategy_id?: string;
    shortcut_id?: string;
    mode?: RunMode;
    overrides?: Record<string, unknown>;
  }) =>
    apiFetch<RunSummaryRecord>("/api/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getRun: (runId: string) => apiFetch<RunSummaryRecord>(`/api/runs/${runId}`),
  cancelRun: (runId: string) =>
    apiFetch<RunSummaryRecord>(`/api/runs/${runId}/cancel`, { method: "POST" }),
  getRunLogs: (runId: string) => apiFetch<string>(`/api/runs/${runId}/logs`),
  getRunSummary: (runId: string) => apiFetch<RunManifest>(`/api/runs/${runId}/summary`),
  queryTable: (
    runId: string,
    table: string,
    params: {
      page?: number;
      page_size?: number;
      sort_by?: string;
      sort_dir?: "asc" | "desc";
      q?: string;
      factor?: string;
      period?: number;
    },
  ) => {
    const qs = new URLSearchParams();
    if (params.page) qs.set("page", String(params.page));
    if (params.page_size) qs.set("page_size", String(params.page_size));
    if (params.sort_by) qs.set("sort_by", params.sort_by);
    if (params.sort_dir) qs.set("sort_dir", params.sort_dir);
    if (params.q) qs.set("q", params.q);
    if (params.factor) qs.set("factor", params.factor);
    if (params.period !== undefined) qs.set("period", String(params.period));
    return apiFetch<TableQueryResult>(
      `/api/runs/${runId}/tables/${encodeURIComponent(table)}?${qs.toString()}`,
    );
  },
  queryCurve: (
    runId: string,
    params: {
      obj?: string;
      factor?: string;
      period?: number;
      start_idx?: number;
      end_idx?: number;
      max_points?: number;
    },
  ) => {
    const qs = new URLSearchParams();
    if (params.obj) qs.set("obj", params.obj);
    if (params.factor) qs.set("factor", params.factor);
    if (params.period !== undefined) qs.set("period", String(params.period));
    if (params.start_idx !== undefined) qs.set("start_idx", String(params.start_idx));
    if (params.end_idx !== undefined) qs.set("end_idx", String(params.end_idx));
    if (params.max_points !== undefined) qs.set("max_points", String(params.max_points));
    return apiFetch<CurveQueryResult>(`/api/runs/${runId}/curve?${qs.toString()}`);
  },
  exportUrl: (runId: string, format: "csv" | "xlsx" | "json" | "parquet", table?: string) => {
    const qs = new URLSearchParams({ format });
    if (table) qs.set("table", table);
    return `/api/runs/${runId}/export?${qs.toString()}`;
  },
};
