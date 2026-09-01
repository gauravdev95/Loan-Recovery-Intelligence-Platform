import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import type { DashboardStats, ModelInfoResponse, ModelMetrics } from "../api/types";
import { StatCard } from "../components/StatCard";
import { ErrorState, Spinner } from "../components/states";

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [model, setModel] = useState<ModelInfoResponse | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const [s, mInfo, mMetrics] = await Promise.all([
        api.dashboardStats(),
        api.modelInfo().catch(() => null),
        api.modelMetrics().catch(() => null),
      ]);
      setStats(s);
      setModel(mInfo);
      setMetrics(mMetrics);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load dashboard");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  if (loading) return <Spinner label="Loading dashboard…" />;
  if (error) return <ErrorState message={error} onRetry={load} />;
  if (!stats) return <ErrorState message="No dashboard data available." onRetry={load} />;

  const distribution = [
    { name: "Low", value: stats.low_risk, fill: "#10b981" },
    { name: "Medium", value: stats.medium_risk, fill: "#f59e0b" },
    { name: "High", value: stats.high_risk, fill: "#f97316" },
    { name: "Critical", value: stats.critical_risk, fill: "#ef4444" },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-slate-900">Dashboard</h2>
        <p className="text-sm text-slate-500">
          Portfolio risk overview and model health
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-5">
        <StatCard label="Total borrowers" value={stats.total_borrowers} />
        <StatCard
          label="Critical risk"
          value={stats.critical_risk}
          tone={stats.critical_risk > 0 ? "bad" : "default"}
        />
        <StatCard label="High risk" value={stats.high_risk} tone="warn" />
        <StatCard label="Medium risk" value={stats.medium_risk} tone="warn" />
        <StatCard
          label="Avg default prob"
          value={(stats.avg_default_probability * 100).toFixed(1) + "%"}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm lg:col-span-2">
          <h3 className="mb-4 text-sm font-semibold text-slate-700">
            Borrowers by risk level
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={distribution} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="value" name="Borrowers" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h3 className="mb-4 text-sm font-semibold text-slate-700">
            Best model — ROC-AUC
          </h3>
          <p className="text-4xl font-bold text-brand-700">
            {metrics?.roc_auc ? metrics.roc_auc.toFixed(3) : "—"}
          </p>
          <p className="mt-2 text-xs text-slate-500">
            {model ? `${model.model_name} (v${model.model_version})` : "Model info unavailable"}
          </p>
          {metrics?.pr_auc !== undefined && (
            <div className="mt-4 space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-500">PR-AUC</span>
                <span className="font-mono">{metrics.pr_auc.toFixed(3)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">F1</span>
                <span className="font-mono">{metrics?.f1?.toFixed(3)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Recall</span>
                <span className="font-mono">{metrics?.recall?.toFixed(3)}</span>
              </div>
            </div>
          )}
          <Link
            to="/performance"
            className="mt-4 inline-block text-sm font-medium text-brand-600 hover:underline"
          >
            View model performance →
          </Link>
        </div>
      </div>
    </div>
  );
}
