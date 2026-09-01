import type {
  BorrowerDetail,
  BorrowerListResponse,
  DashboardStats,
  ExplanationResponse,
  HealthResponse,
  ModelInfoResponse,
  ModelMetrics,
  RecoveryQueueResponse,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  modelInfo: () => request<ModelInfoResponse>("/api/v1/model/info"),
  modelMetrics: () => request<ModelMetrics>("/api/v1/model/metrics"),
  dashboardStats: () => request<DashboardStats>("/api/v1/dashboard/stats"),

  listBorrowers: (params: {
    search?: string;
    risk_level?: string;
    priority?: string;
    sort_by?: string;
    order?: string;
    limit?: number;
    offset?: number;
  }) => {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") q.set(k, String(v));
    });
    return request<BorrowerListResponse>(`/api/v1/borrowers?${q.toString()}`);
  },

  getBorrower: (id: string) =>
    request<BorrowerDetail>(`/api/v1/borrowers/${encodeURIComponent(id)}`),

  recoveryQueue: (limit = 100) =>
    request<RecoveryQueueResponse>(`/api/v1/recovery/priority?limit=${limit}`),

  explain: (payload: Record<string, unknown>) =>
    request<ExplanationResponse>("/api/v1/explain", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
