import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { BorrowerRow } from "../api/types";
import { PriorityBadge, Probability, RiskBadge } from "../components/badges";
import { EmptyState, ErrorState, Spinner } from "../components/states";

export default function RecoveryQueue() {
  const [items, setItems] = useState<BorrowerRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const res = await api.recoveryQueue(100);
      setItems(res.items);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load recovery queue");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-xl font-semibold text-slate-900">Recovery Queue</h2>
        <p className="text-sm text-slate-500">
          Borrowers sorted by recovery priority (decision-support only)
        </p>
      </div>

      {loading ? (
        <Spinner label="Loading recovery queue…" />
      ) : error ? (
        <ErrorState message={error} onRetry={load} />
      ) : items.length === 0 ? (
        <EmptyState message="No borrowers in the recovery queue yet. Run predictions first." />
      ) : (
        <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
          <table className="w-full min-w-[640px] text-left text-sm">
            <thead className="border-b border-slate-200 bg-slate-50 text-xs uppercase text-slate-500">
              <tr>
                <th className="px-4 py-3">Priority</th>
                <th className="px-4 py-3">Borrower</th>
                <th className="px-4 py-3">Default prob</th>
                <th className="px-4 py-3">Risk</th>
                <th className="px-4 py-3">Recommended action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {items.map((b) => (
                <tr key={b.borrower_id} className="hover:bg-slate-50">
                  <td className="px-4 py-3">
                    <PriorityBadge priority={b.recovery_priority} />
                  </td>
                  <td className="px-4 py-3">
                    <Link
                      to={`/borrowers/${b.borrower_id}`}
                      className="font-medium text-brand-600 hover:underline"
                    >
                      {b.borrower_id}
                    </Link>
                  </td>
                  <td className="px-4 py-3">
                    <Probability value={b.default_probability} />
                  </td>
                  <td className="px-4 py-3">
                    <RiskBadge level={b.risk_level} />
                  </td>
                  <td className="px-4 py-3 text-xs text-slate-600">
                    {b.recommended_action ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
