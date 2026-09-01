import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ModelInfoResponse, ModelMetrics } from "../api/types";
import {
  EMPTY_METRICS,
  MODEL_COMPARISON,
  TUNED_COMPARISON,
} from "../data/comparison";
import { ErrorState, Spinner } from "../components/states";

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

export default function ModelPerformance() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [m, i] = await Promise.all([
          api.modelMetrics().catch(() => EMPTY_METRICS),
          api.modelInfo().catch(() => null),
        ]);
        setMetrics(m);
        setInfo(i);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load performance");
      } finally {
        setLoading(false);
      }
    }
    void load();
  }, []);

  if (loading) return <Spinner label="Loading model performance…" />;
  if (error) return <ErrorState message={error} />;

  const cm = metrics?.confusion_matrix;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-slate-900">Model Performance</h2>
        <p className="text-sm text-slate-500">
          {info ? `${info.model_name} — v${info.model_version}` : "Final model"}
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
        {[
          ["ROC-AUC", metrics?.roc_auc],
          ["PR-AUC", metrics?.pr_auc],
          ["F1", metrics?.f1],
          ["Precision", metrics?.precision],
          ["Recall", metrics?.recall],
          ["Accuracy", metrics?.accuracy],
        ].map(([label, value]) => (
          <div
            key={String(label)}
            className="rounded-xl border border-slate-200 bg-white p-4 text-center shadow-sm"
          >
            <p className="text-xs uppercase tracking-wide text-slate-500">{label}</p>
            <p className="mt-1 text-2xl font-bold text-brand-700">
              {value !== undefined ? Number(value).toFixed(3) : "—"}
            </p>
          </div>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm lg:col-span-2">
          <h3 className="mb-3 text-sm font-semibold text-slate-700">
            Model comparison (validation)
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] text-right text-sm">
              <thead className="border-b border-slate-200 text-xs uppercase text-slate-500">
                <tr>
                  <th className="py-2 text-left">Model</th>
                  <th>Acc</th>
                  <th>Prec</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>ROC-AUC</th>
                  <th>PR-AUC</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {MODEL_COMPARISON.map((row) => (
                  <tr key={row.model} className="hover:bg-slate-50">
                    <td className="py-2 text-left font-medium">
                      {row.model}
                      {row.note && (
                        <span className="ml-2 rounded bg-brand-50 px-1.5 py-0.5 text-[10px] text-brand-700">
                          {row.note}
                        </span>
                      )}
                    </td>
                    <td>{pct(row.accuracy)}</td>
                    <td>{row.precision.toFixed(3)}</td>
                    <td>{row.recall.toFixed(3)}</td>
                    <td>{row.f1.toFixed(3)}</td>
                    <td>{row.roc_auc.toFixed(3)}</td>
                    <td>{row.pr_auc.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-slate-400">
            Values are actual measured validation-split results; final test-set
            metrics are shown in the cards above.
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h3 className="mb-3 text-sm font-semibold text-slate-700">
            Confusion matrix (test)
          </h3>
          {cm ? (
            <div className="grid grid-cols-2 gap-2 text-center">
              <div className="rounded-md bg-emerald-50 p-4">
                <p className="text-2xl font-bold text-emerald-700">{cm.tn}</p>
                <p className="text-xs text-emerald-600">True negative</p>
              </div>
              <div className="rounded-md bg-red-50 p-4">
                <p className="text-2xl font-bold text-red-600">{cm.fp}</p>
                <p className="text-xs text-red-500">False positive</p>
              </div>
              <div className="rounded-md bg-amber-50 p-4">
                <p className="text-2xl font-bold text-amber-700">{cm.fn}</p>
                <p className="text-xs text-amber-600">False negative</p>
              </div>
              <div className="rounded-md bg-emerald-50 p-4">
                <p className="text-2xl font-bold text-emerald-700">{cm.tp}</p>
                <p className="text-xs text-emerald-600">True positive</p>
              </div>
            </div>
          ) : (
            <p className="text-sm text-slate-400">No confusion matrix available.</p>
          )}
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <h3 className="mb-3 text-sm font-semibold text-slate-700">
          Hyperparameter-tuned candidates
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[460px] text-right text-sm">
            <thead className="border-b border-slate-200 text-xs uppercase text-slate-500">
              <tr>
                <th className="py-2 text-left">Model</th>
                <th>F1</th>
                <th>ROC-AUC</th>
                <th>PR-AUC</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {TUNED_COMPARISON.map((row) => (
                <tr key={row.model} className="hover:bg-slate-50">
                  <td className="py-2 text-left font-medium">{row.model}</td>
                  <td>{row.f1.toFixed(3)}</td>
                  <td>{row.roc_auc.toFixed(3)}</td>
                  <td>{row.pr_auc.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
